#!/usr/bin/env python3
"""00_preprocess.py — Prepare real Fleck et al. data for hgx modeling.

Reads Zenodo processed files and produces modeling-ready numpy arrays.
Run once before analysis scripts.

Usage:
    python scripts/00_preprocess.py
    python scripts/00_preprocess.py --num-bins 20
    python scripts/00_preprocess.py --data-dir /workspace/benchmark/data
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEY_TFS = ["GLI3", "FOXG1", "TBR1", "DLX1", "DLX2", "EMX1", "EOMES", "NEUROD6"]
LINEAGES = ["telencephalon", "early", "nt"]
FATES = ["DF", "VF", "MH"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timer(msg: str):
    """Context manager that prints elapsed time for a step."""

    class _T:
        def __init__(self, msg):
            self.msg = msg

        def __enter__(self):
            self.t0 = time.perf_counter()
            print(f"[step] {self.msg} ...", flush=True)
            return self

        def __exit__(self, *exc):
            dt = time.perf_counter() - self.t0
            print(f"        done in {dt:.1f}s", flush=True)

    return _T(msg)


def _detect_data_dir() -> Path:
    """Auto-detect data directory: prefer /workspace/benchmark/data (DGX Spark
    inside Docker), fall back to ../data relative to this script."""
    dgx = Path("/workspace/benchmark/data")
    if dgx.is_dir():
        return dgx
    local = Path(__file__).resolve().parent.parent / "data"
    if local.is_dir():
        return local
    raise FileNotFoundError(
        "Cannot find data directory. Pass --data-dir explicitly."
    )


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------


def main(data_dir: Path, num_bins: int = 10, feature_dim: int = 16) -> None:
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory : {data_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Bins: {num_bins}  |  Feature dim: {feature_dim}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load GRN and build gene universe
    # ------------------------------------------------------------------
    with _timer("Step 1 — Load GRN (coefs.tsv)"):
        coefs_path = data_dir / "pando" / "coefs.tsv"
        coefs = pd.read_csv(coefs_path, sep="\t")
        print(f"        coefs shape: {coefs.shape}")

        all_tfs = sorted(coefs["tf"].unique())
        all_targets = sorted(coefs["target"].unique())
        grn_genes = sorted(set(all_tfs) | set(all_targets))
        print(f"        unique TFs: {len(all_tfs)}")
        print(f"        unique targets: {len(all_targets)}")
        print(f"        GRN gene universe: {len(grn_genes)}")

    # ------------------------------------------------------------------
    # Step 2: Load count matrix and intersect with GRN genes
    # ------------------------------------------------------------------
    with _timer("Step 2 — Load count matrix and intersect"):
        counts_path = data_dir / "zenodo" / "data_matrices" / "counts.mtx.gz"
        features_path = data_dir / "zenodo" / "data_matrices" / "features.tsv.gz"
        barcodes_path = data_dir / "zenodo" / "data_matrices" / "barcodes.tsv.gz"

        # Read gene names and barcodes
        features_df = pd.read_csv(features_path, sep="\t", header=None)
        feature_names: list[str] = features_df.iloc[:, 0].astype(str).tolist()
        barcodes_df = pd.read_csv(barcodes_path, sep="\t", header=None)
        barcodes: list[str] = barcodes_df.iloc[:, 0].astype(str).tolist()
        print(f"        features: {len(feature_names)}  barcodes: {len(barcodes)}")

        # Read sparse count matrix (genes x cells)
        counts_raw = sio.mmread(counts_path)  # returns coo
        counts_raw = sp.csc_matrix(counts_raw)  # csc for efficient column slicing
        print(f"        raw counts shape: {counts_raw.shape}")

        # Intersect GRN genes with expression feature names
        feature_set = set(feature_names)
        gene_names = sorted(g for g in grn_genes if g in feature_set)
        print(f"        GRN-expression intersection: {len(gene_names)} genes")

        # Build index mapping: position in feature_names for each kept gene
        feature_name_to_idx = {n: i for i, n in enumerate(feature_names)}
        gene_row_indices = np.array(
            [feature_name_to_idx[g] for g in gene_names], dtype=np.intp
        )

        # Subset count matrix to intersection genes (keep sparse)
        counts_sub = counts_raw[gene_row_indices, :]  # (n_genes, n_cells)
        n_genes = len(gene_names)
        n_cells_expr = counts_sub.shape[1]
        print(f"        subset counts: {counts_sub.shape}")

    # ------------------------------------------------------------------
    # Step 3: Load metadata and match cells
    # ------------------------------------------------------------------
    with _timer("Step 3 — Load metadata and match cells"):
        meta_path = data_dir / "zenodo" / "data_matrices" / "meta.tsv.gz"
        meta = pd.read_csv(meta_path, sep="\t")
        print(f"        metadata shape: {meta.shape}")

        # Match barcodes
        barcode_to_col = {bc: i for i, bc in enumerate(barcodes)}
        meta_cellids = meta["cellID"].astype(str).tolist()

        matched_meta_rows: list[int] = []
        matched_col_indices: list[int] = []
        for mi, cid in enumerate(meta_cellids):
            if cid in barcode_to_col:
                matched_meta_rows.append(mi)
                matched_col_indices.append(barcode_to_col[cid])

        matched_col_indices_arr = np.array(matched_col_indices, dtype=np.intp)
        meta_matched = meta.iloc[matched_meta_rows].reset_index(drop=True)
        n_cells = len(matched_col_indices)
        print(f"        matched cells: {n_cells}")

        # Subset count matrix to matched cells
        counts_matched = counts_sub[:, matched_col_indices_arr]  # (n_genes, n_cells)

        pseudotime = meta_matched["velocity_pseudotime"].values.astype(np.float32)
        lineage = meta_matched["lineage"].values
        stage = meta_matched["stage_manual"].values

    # ------------------------------------------------------------------
    # Step 4: Normalize expression
    # ------------------------------------------------------------------
    with _timer("Step 4 — Normalize expression"):
        # Library size normalization (per cell)
        counts_csc = sp.csc_matrix(counts_matched, dtype=np.float64)
        lib_sizes = np.array(counts_csc.sum(axis=0)).ravel()  # (n_cells,)
        lib_sizes[lib_sizes == 0] = 1.0  # avoid division by zero

        # Multiply each column by 1e4 / lib_size (CPM-like)
        inv_lib = sp.diags(1e4 / lib_sizes)  # (n_cells, n_cells) diagonal
        normalized = counts_csc @ inv_lib  # (n_genes, n_cells)

        # Log1p transform and densify (subset is ~2700 x 34k = manageable)
        log_expr = np.log1p(normalized.toarray()).astype(np.float64)

        # Z-score per gene (row-wise)
        means = log_expr.mean(axis=1, keepdims=True)
        stds = log_expr.std(axis=1, keepdims=True)
        stds[stds < 1e-8] = 1.0
        scaled = ((log_expr - means) / stds).astype(np.float32)
        print(f"        scaled expression: {scaled.shape}  dtype={scaled.dtype}")

    # ------------------------------------------------------------------
    # Step 5: Compute PCA features for nodes
    # ------------------------------------------------------------------
    with _timer("Step 5 — PCA node features (PPCA dimensionality estimation)"):
        rng = np.random.default_rng(42)
        n_subset = min(5000, n_cells)
        subset_idx = rng.choice(n_cells, size=n_subset, replace=False)
        subset = scaled[:, subset_idx].astype(np.float64)  # (n_genes, n_subset)

        # Gene-gene covariance + full eigendecomposition
        cov = np.cov(subset)  # (n_genes, n_genes)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals_desc = eigvals[::-1]  # descending order

        # --- PPCA dimensionality estimation (Minka/MELODIC, from neurojax) ---
        d_max = len(eigvals_desc)
        lambdas = np.maximum(eigvals_desc, 1e-19)
        log_lam = np.log(lambdas)
        cs_log = np.cumsum(log_lam)
        cs_lam = np.cumsum(lambdas)
        sum_noise = cs_lam[-1] - cs_lam

        all_k = np.arange(1, d_max)
        m_noise = d_max - all_k
        v = np.maximum(sum_noise[all_k - 1] / m_noise, 1e-19)
        log_lik = -(n_subset / 2) * (cs_log[all_k - 1] + m_noise * np.log(v))
        m_params = d_max * all_k - 0.5 * all_k * (all_k + 1)

        aic = log_lik - m_params
        bic = log_lik - 0.5 * m_params * np.log(n_subset)
        k_aic = int(np.argmax(aic) + 1)
        k_bic = int(np.argmax(bic) + 1)
        k_consensus = int(round((k_aic + k_bic) / 2))

        # Explained variance
        total_var = eigvals_desc.sum()
        cumvar = np.cumsum(eigvals_desc) / total_var
        k_90 = int(np.searchsorted(cumvar, 0.90) + 1)
        k_95 = int(np.searchsorted(cumvar, 0.95) + 1)

        print(f"        Eigenspectrum (PPCA/MELODIC):")
        print(f"          AIC optimal: k={k_aic}")
        print(f"          BIC optimal: k={k_bic}")
        print(f"          Consensus:   k={k_consensus}")
        print(f"          90% var:     k={k_90},  95% var: k={k_95}")

        # Use consensus if default, otherwise user's choice
        if feature_dim == 16:
            effective_dim = max(2, min(k_consensus, 64))
            print(f"          Using consensus: {effective_dim}")
        else:
            effective_dim = feature_dim
            print(f"          Using user-specified: {effective_dim}")

        node_features = eigvecs[:, -effective_dim:].astype(np.float32)
        print(f"        node_features shape: {node_features.shape}")

        np.save(out_dir / "eigenvalues.npy", eigvals_desc.astype(np.float32))
        np.save(out_dir / "node_features_pca.npy", node_features)

    # ------------------------------------------------------------------
    # Step 6: Pseudotime-binned temporal expression
    # ------------------------------------------------------------------
    with _timer(f"Step 6 — Temporal expression ({num_bins} bins)"):
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        temporal_expression = np.zeros((num_bins, n_genes), dtype=np.float32)
        lineage_fractions = np.zeros((num_bins, 3), dtype=np.float32)
        pseudotime_centers = np.zeros(num_bins, dtype=np.float32)

        for b in range(num_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b < num_bins - 1:
                mask = (pseudotime >= lo) & (pseudotime < hi)
            else:
                # Last bin includes right edge
                mask = (pseudotime >= lo) & (pseudotime <= hi)

            if mask.sum() == 0:
                print(f"        WARNING: bin {b} [{lo:.2f}, {hi:.2f}) has 0 cells")
                continue

            temporal_expression[b] = scaled[:, mask].mean(axis=1)
            pseudotime_centers[b] = pseudotime[mask].mean()

            bin_lineage = lineage[mask]
            n_bin = mask.sum()
            for li, lname in enumerate(LINEAGES):
                lineage_fractions[b, li] = (bin_lineage == lname).sum() / n_bin

        print(f"        temporal_expression: {temporal_expression.shape}")
        print(f"        lineage_fractions: {lineage_fractions.shape}")

        np.save(out_dir / "temporal_expression.npy", temporal_expression)
        np.save(out_dir / "pseudotime_centers.npy", pseudotime_centers)
        np.save(out_dir / "lineage_fractions.npy", lineage_fractions)

    # ------------------------------------------------------------------
    # Step 7: Load fate probabilities from velocity h5ad
    # ------------------------------------------------------------------
    with _timer("Step 7 — Fate probabilities from h5ad"):
        h5ad_path = data_dir / "zenodo" / "RNA_all_velo.h5ad"
        adata = ad.read_h5ad(h5ad_path, backed="r")
        obs = adata.obs

        # Extract fate columns
        fate_cols_present = [f for f in FATES if f in obs.columns]
        if len(fate_cols_present) != 3:
            print(f"        WARNING: only found fate columns {fate_cols_present}")
        fate_df = obs[fate_cols_present].copy()
        fate_df.index = fate_df.index.astype(str)

        # Match cells by barcode with metadata-matched cells
        matched_barcodes = meta_matched["cellID"].astype(str).tolist()
        h5ad_barcodes = set(fate_df.index)

        cell_fate_list: list[np.ndarray] = []
        cell_fate_mask: list[bool] = []  # which metadata cells have h5ad fate data
        for bc in matched_barcodes:
            if bc in h5ad_barcodes:
                row = fate_df.loc[bc, fate_cols_present].values.astype(np.float32)
                cell_fate_list.append(row)
                cell_fate_mask.append(True)
            else:
                cell_fate_mask.append(False)

        cell_fate_probs = np.array(cell_fate_list, dtype=np.float32)  # (n_matched, 3)
        cell_fate_mask_arr = np.array(cell_fate_mask)
        n_matched_fates = cell_fate_probs.shape[0]
        print(f"        cells with fate data: {n_matched_fates}")

        np.save(out_dir / "cell_fate_probs.npy", cell_fate_probs)

        # Per-bin mean fates
        fate_probabilities = np.zeros((num_bins, 3), dtype=np.float32)
        for b in range(num_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b < num_bins - 1:
                pt_mask = (pseudotime >= lo) & (pseudotime < hi)
            else:
                pt_mask = (pseudotime >= lo) & (pseudotime <= hi)
            combined = pt_mask & cell_fate_mask_arr
            if combined.sum() > 0:
                # Get indices of matched-fate cells within this bin
                fate_indices = []
                fate_counter = 0
                for ci in range(n_cells):
                    if cell_fate_mask_arr[ci]:
                        if pt_mask[ci]:
                            fate_indices.append(fate_counter)
                        fate_counter += 1
                if fate_indices:
                    fate_probabilities[b] = cell_fate_probs[fate_indices].mean(axis=0)

        print(f"        fate_probabilities: {fate_probabilities.shape}")
        np.save(out_dir / "fate_probabilities.npy", fate_probabilities)

        # Close backed h5ad
        adata.file.close()

    # ------------------------------------------------------------------
    # Step 8: Build incidence matrix and module labels
    # ------------------------------------------------------------------
    with _timer("Step 8 — Incidence matrix and module labels"):
        gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}

        # Build TF list: only TFs that appear in gene_names and have at least
        # one significant target in gene_names
        sig_coefs = coefs[coefs["padj"] < 0.05].copy()
        tf_names: list[str] = []
        tf_target_map: dict[str, list[int]] = {}

        for tf in sorted(sig_coefs["tf"].unique()):
            if tf not in gene_name_to_idx:
                continue
            targets = sig_coefs.loc[sig_coefs["tf"] == tf, "target"].unique()
            target_indices = [gene_name_to_idx[t] for t in targets if t in gene_name_to_idx]
            if len(target_indices) == 0:
                continue
            tf_names.append(tf)
            tf_target_map[tf] = target_indices

        n_edges = len(tf_names)
        print(f"        TFs with significant targets in gene set: {n_edges}")

        incidence = np.zeros((n_genes, n_edges), dtype=np.float32)
        for ei, tf in enumerate(tf_names):
            # TF itself is in the hyperedge
            tf_idx = gene_name_to_idx[tf]
            incidence[tf_idx, ei] = 1.0
            # All significant targets
            for ti in tf_target_map[tf]:
                incidence[ti, ei] = 1.0

        # Module labels: argmax of incidence row; -1 if gene is in no regulon
        row_sums = incidence.sum(axis=1)
        module_labels = np.full(n_genes, -1, dtype=np.int32)
        assigned_mask = row_sums > 0
        module_labels[assigned_mask] = incidence[assigned_mask].argmax(axis=1).astype(
            np.int32
        )
        n_assigned = assigned_mask.sum()
        print(f"        incidence: {incidence.shape}  |  assigned genes: {n_assigned}")

        np.save(out_dir / "incidence.npy", incidence)
        np.save(out_dir / "module_labels.npy", module_labels)

    # ------------------------------------------------------------------
    # Step 9: Perturbation data from GRN structure
    # ------------------------------------------------------------------
    with _timer("Step 9 — Perturbation data for key TFs"):
        available_key_tfs = [tf for tf in KEY_TFS if tf in gene_name_to_idx]
        K = len(available_key_tfs)
        print(f"        key TFs available: {K} / {len(KEY_TFS)}")

        perturbation_masks = np.zeros((K, n_genes), dtype=bool)
        perturbation_effects = np.zeros((K, n_genes), dtype=np.float32)
        perturbation_fates = np.zeros((K, 3), dtype=np.float32)

        # Precompute per-gene mean expression (un-scaled) for quartile analysis
        mean_expr_per_cell = log_expr.astype(np.float32)  # (n_genes, n_cells)

        for ki, tf in enumerate(available_key_tfs):
            tf_idx = gene_name_to_idx[tf]
            perturbation_masks[ki, tf_idx] = True

            # Expression effect from GRN coefficients
            tf_coefs = sig_coefs[sig_coefs["tf"] == tf].copy()
            for _, row in tf_coefs.iterrows():
                target = row["target"]
                if target in gene_name_to_idx:
                    tidx = gene_name_to_idx[target]
                    # Knockout = remove TF -> reverse the estimated effect
                    perturbation_effects[ki, tidx] = -float(row["estimate"])

            # Normalize effects to unit variance (if nonzero)
            eff = perturbation_effects[ki]
            eff_std = eff.std()
            if eff_std > 1e-8:
                perturbation_effects[ki] = eff / eff_std

            # Fate shift: compare top vs bottom quartile of TF expression
            tf_expr = mean_expr_per_cell[tf_idx]  # (n_cells,)
            q25 = np.percentile(tf_expr, 25)
            q75 = np.percentile(tf_expr, 75)
            low_mask = tf_expr <= q25
            high_mask = tf_expr >= q75

            # Average fates in each quartile (use cells that have fate data)
            combined_low = low_mask & cell_fate_mask_arr
            combined_high = high_mask & cell_fate_mask_arr

            if combined_low.sum() > 0 and combined_high.sum() > 0:
                # Map to cell_fate_probs indices
                low_fate_idx = []
                high_fate_idx = []
                fate_counter = 0
                for ci in range(n_cells):
                    if cell_fate_mask_arr[ci]:
                        if low_mask[ci]:
                            low_fate_idx.append(fate_counter)
                        if high_mask[ci]:
                            high_fate_idx.append(fate_counter)
                        fate_counter += 1

                mean_low = cell_fate_probs[low_fate_idx].mean(axis=0) if low_fate_idx else np.zeros(3, dtype=np.float32)
                mean_high = cell_fate_probs[high_fate_idx].mean(axis=0) if high_fate_idx else np.zeros(3, dtype=np.float32)
                # Knockout removes the TF, so the shift is from high->low
                perturbation_fates[ki] = mean_low - mean_high

        print(f"        perturbation_masks: {perturbation_masks.shape}")
        print(f"        perturbation_effects: {perturbation_effects.shape}")
        print(f"        perturbation_fates: {perturbation_fates.shape}")

        np.save(out_dir / "perturbation_masks.npy", perturbation_masks)
        np.save(out_dir / "perturbation_effects.npy", perturbation_effects)
        np.save(out_dir / "perturbation_fates.npy", perturbation_fates)

    # ------------------------------------------------------------------
    # Step 10: TF index mappings
    # ------------------------------------------------------------------
    with _timer("Step 10 — TF index mappings"):
        tf_gene_indices = {tf: gene_name_to_idx[tf] for tf in tf_names if tf in gene_name_to_idx}
        key_tf_indices = {tf: gene_name_to_idx[tf] for tf in KEY_TFS if tf in gene_name_to_idx}

        print(f"        tf_gene_indices: {len(tf_gene_indices)} entries")
        print(f"        key_tf_indices: {key_tf_indices}")

        with open(out_dir / "tf_gene_indices.json", "w") as f:
            json.dump(tf_gene_indices, f, indent=2)
        with open(out_dir / "key_tf_indices.json", "w") as f:
            json.dump(key_tf_indices, f, indent=2)

    # ------------------------------------------------------------------
    # Step 11: Save gene/TF name lists and summary
    # ------------------------------------------------------------------
    with _timer("Step 11 — Save names and summary"):
        with open(out_dir / "gene_names.json", "w") as f:
            json.dump(gene_names, f)
        with open(out_dir / "tf_names.json", "w") as f:
            json.dump(tf_names, f)

        summary = {
            "n_genes": n_genes,
            "n_edges": n_edges,
            "n_cells": n_cells,
            "n_cells_matched_fates": n_matched_fates,
            "num_bins": num_bins,
            "feature_dim": feature_dim,
            "key_tfs": KEY_TFS,
            "key_tfs_available": available_key_tfs,
            "lineages": LINEAGES,
            "fates": FATES,
            "fate_mapping": {"DF": "cortical", "VF": "GE", "MH": "neural_tube"},
            "stages": ["iPSC", "nect_nepi", "npc", "neuron"],
            "data_dir": str(data_dir),
            "grn_genes_total": len(grn_genes),
            "grn_expression_intersection": n_genes,
            "n_tfs_total": len(all_tfs),
            "n_tfs_with_sig_targets": n_edges,
        }

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"        summary: {summary}")

    print()
    print("=" * 60)
    print("Preprocessing complete. Outputs saved to:")
    print(f"  {out_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Fleck et al. 2023 data for hgx benchmark"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root data directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Number of pseudotime bins (default: 10)",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=16,
        help="PCA feature dimension per gene (default: 16)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else _detect_data_dir()
    main(data_dir=data_dir, num_bins=args.num_bins, feature_dim=args.feature_dim)
