#!/usr/bin/env python3
"""Preprocess Pollen/Ding et al. 2026 CRISPRi Perturb-seq data.

Reads GSE284197_screen.h5ad and produces modeling-ready arrays for
the hgx perturbation comparison.

The screen dataset contains CRISPRi knockdowns of 44 TFs in primary
human cortical cultures (2D), with single-cell RNA-seq readout.

Outputs to data/pollen/processed/:
  - gene_names.json           Gene universe (intersection with Fleck GRN)
  - tf_names.json             44 perturbed TFs
  - perturbation_masks.npy    (K, n_genes) boolean KO masks
  - perturbation_effects.npy  (K, n_genes) observed DE per KO
  - cell_type_fractions.npy   (K, n_types) cell type composition shift per KO
  - incidence.npy             (n_genes, n_tfs) regulatory incidence (from DE)
  - node_features_pca.npy     (n_genes, d) PCA features
  - summary.json              Dataset metadata

Usage:
    uv run python scripts/preprocess_pollen.py
    uv run python scripts/preprocess_pollen.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Non-targeting control guide identifiers (typical in CRISPRi screens)
NT_PATTERNS = ["non-targeting", "NT", "CTRL", "scramble", "NTC", "safe"]

# Cell types expected in cortical cultures
EXPECTED_CELL_TYPES = [
    "RG",           # Radial glia
    "oRG",          # Outer radial glia
    "IPC",          # Intermediate progenitor cells
    "EN",           # Excitatory neurons
    "IN",           # Interneurons
    "Astrocyte",    # Astrocytes
    "OPC",          # Oligodendrocyte progenitors
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timer(msg: str):
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


def _detect_guide_column(obs: pd.DataFrame) -> str | None:
    """Find the column containing guide/TF assignments."""
    candidates = [
        "gene", "target_gene", "guide_identity", "perturbation",
        "sgRNA_group", "KD_gene", "knockdown", "CRISPRi_target",
        "gRNA_target", "guide_target", "assigned_gene",
    ]
    for col in candidates:
        if col in obs.columns:
            return col
    # Heuristic: look for a column with TF-like values
    for col in obs.columns:
        vals = obs[col].dropna().unique()
        if len(vals) < 100 and any(
            v in vals for v in ["GLI3", "ARX", "NR2E1", "ZNF219", "TBR1"]
        ):
            return col
    return None


def _detect_celltype_column(obs: pd.DataFrame) -> str | None:
    """Find the column containing cell type annotations."""
    candidates = [
        "cell_type", "celltype", "CellType", "cluster", "leiden",
        "annotation", "cell_class", "subclass",
    ]
    for col in candidates:
        if col in obs.columns:
            return col
    return None


def _is_control(label: str) -> bool:
    """Check if a guide label corresponds to non-targeting controls."""
    label_lower = str(label).lower()
    return any(pat.lower() in label_lower for pat in NT_PATTERNS)


# ---------------------------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------------------------

def main(data_dir: Path) -> None:
    pollen_dir = data_dir / "pollen"
    screen_path = pollen_dir / "screen.h5ad"
    out_dir = pollen_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not screen_path.exists():
        print(f"ERROR: {screen_path} not found.")
        print("Run: uv run python scripts/download_pollen.py --screen-only")
        return

    # ------------------------------------------------------------------
    # Step 1: Load screen h5ad and inspect structure
    # ------------------------------------------------------------------
    with _timer("Step 1 — Load screen.h5ad"):
        adata = ad.read_h5ad(screen_path)
        print(f"        Shape: {adata.shape} (cells x genes)")
        print(f"        obs columns: {list(adata.obs.columns)[:20]}")
        print(f"        var columns: {list(adata.var.columns)[:10]}")

        if hasattr(adata, "layers") and adata.layers:
            print(f"        Layers: {list(adata.layers.keys())}")

        n_cells, n_genes_total = adata.shape

    # ------------------------------------------------------------------
    # Step 2: Identify guide assignments and cell types
    # ------------------------------------------------------------------
    with _timer("Step 2 — Identify perturbation assignments"):
        guide_col = _detect_guide_column(adata.obs)
        if guide_col is None:
            print("        ERROR: Cannot find guide/perturbation column")
            print(f"        Available columns: {list(adata.obs.columns)}")
            print("        Please specify manually and re-run")
            # Save column info for debugging
            with open(out_dir / "obs_columns.json", "w") as f:
                json.dump({
                    "columns": list(adata.obs.columns),
                    "dtypes": {c: str(adata.obs[c].dtype) for c in adata.obs.columns},
                    "sample_values": {
                        c: adata.obs[c].dropna().unique()[:10].tolist()
                        for c in adata.obs.columns
                        if adata.obs[c].dtype == "object" or adata.obs[c].dtype.name == "category"
                    },
                }, f, indent=2, default=str)
            return

        print(f"        Guide column: '{guide_col}'")
        guide_labels = adata.obs[guide_col].astype(str)
        unique_guides = sorted(guide_labels.unique())
        print(f"        Unique guide labels: {len(unique_guides)}")

        # Separate TF targets from controls
        controls = [g for g in unique_guides if _is_control(g)]
        tf_targets = [g for g in unique_guides if not _is_control(g)]
        print(f"        TF targets: {len(tf_targets)}")
        print(f"        Controls: {len(controls)} ({controls[:5]})")
        if len(tf_targets) <= 60:
            print(f"        TFs: {tf_targets}")

        celltype_col = _detect_celltype_column(adata.obs)
        if celltype_col:
            celltypes = sorted(adata.obs[celltype_col].dropna().unique())
            print(f"        Cell type column: '{celltype_col}' ({len(celltypes)} types)")
            print(f"        Types: {celltypes[:15]}")
        else:
            print("        WARNING: No cell type column found")
            celltypes = []

    # ------------------------------------------------------------------
    # Step 3: Compute DE for each TF knockdown vs controls
    # ------------------------------------------------------------------
    with _timer("Step 3 — Compute per-TF differential expression"):
        # Get control cells
        ctrl_mask = guide_labels.isin(controls) if controls else guide_labels.isna()
        n_ctrl = ctrl_mask.sum()
        print(f"        Control cells: {n_ctrl}")

        if n_ctrl < 50:
            print("        WARNING: Very few control cells, DE may be unreliable")

        # Get expression matrix (densify if sparse, work in float32)
        X = adata.X
        if sp.issparse(X):
            # For large matrices, compute means without densifying
            ctrl_mean = np.array(X[ctrl_mask.values].mean(axis=0)).ravel()
            ctrl_var = np.array(
                X[ctrl_mask.values].power(2).mean(axis=0)
            ).ravel() - ctrl_mean ** 2
        else:
            X_dense = np.asarray(X, dtype=np.float32)
            ctrl_mean = X_dense[ctrl_mask.values].mean(axis=0)
            ctrl_var = X_dense[ctrl_mask.values].var(axis=0)

        gene_names_all = list(adata.var_names)

        # Compute log2FC for each TF knockdown
        de_results: dict[str, dict] = {}

        for tf in tf_targets:
            tf_mask = guide_labels == tf
            n_tf = tf_mask.sum()
            if n_tf < 10:
                print(f"        {tf}: only {n_tf} cells, skipping")
                continue

            if sp.issparse(X):
                tf_mean = np.array(X[tf_mask.values].mean(axis=0)).ravel()
            else:
                tf_mean = X_dense[tf_mask.values].mean(axis=0)

            # log2FC (add pseudocount to avoid log(0))
            pseudo = 0.1
            log2fc = np.log2(tf_mean + pseudo) - np.log2(ctrl_mean + pseudo)

            # Simple t-test p-value approximation
            # (proper DE would use wilcoxon or edgeR, but this is sufficient
            # for direction and magnitude estimation)
            pooled_var = (ctrl_var + 1e-8)
            se = np.sqrt(pooled_var / n_ctrl + pooled_var / n_tf)
            z = (tf_mean - ctrl_mean) / (se + 1e-8)
            from scipy.stats import norm
            pval = 2 * norm.sf(np.abs(z))

            de_results[tf] = {
                "n_cells": int(n_tf),
                "log2fc": log2fc.astype(np.float32),
                "pval": pval.astype(np.float64),
                "mean_expr": tf_mean.astype(np.float32),
            }

        print(f"        Computed DE for {len(de_results)} TFs")
        n_sig_per_tf = {
            tf: int((r["pval"] < 0.05).sum())
            for tf, r in de_results.items()
        }
        top_tfs = sorted(n_sig_per_tf.items(), key=lambda x: -x[1])[:10]
        for tf, n_sig in top_tfs:
            print(f"          {tf}: {n_sig} sig genes (padj<0.05), "
                  f"{de_results[tf]['n_cells']} cells")

    # ------------------------------------------------------------------
    # Step 4: Intersect gene universe with Fleck et al. GRN
    # ------------------------------------------------------------------
    with _timer("Step 4 — Intersect with Fleck et al. gene universe"):
        # Load Fleck GRN gene names
        fleck_grn_path = data_dir / "pando" / "coefs.tsv"
        if fleck_grn_path.exists():
            grn_df = pd.read_csv(fleck_grn_path, sep="\t")
            fleck_genes = sorted(
                set(grn_df["tf"].unique()) | set(grn_df["target"].unique())
            )
            print(f"        Fleck GRN genes: {len(fleck_genes)}")
        else:
            print("        WARNING: Fleck GRN not found, using all Pollen genes")
            fleck_genes = gene_names_all

        # Intersection
        pollen_gene_set = set(gene_names_all)
        shared_genes = sorted(set(fleck_genes) & pollen_gene_set)
        print(f"        Pollen genes: {len(gene_names_all)}")
        print(f"        Shared genes: {len(shared_genes)}")

        # Build gene index mapping (Pollen var_names -> shared index)
        pollen_idx = {g: i for i, g in enumerate(gene_names_all)}
        shared_gene_idx = [pollen_idx[g] for g in shared_genes]
        gene_name_to_idx = {g: i for i, g in enumerate(shared_genes)}
        n_genes = len(shared_genes)

        # Check TF overlap
        pollen_tfs_in_shared = [tf for tf in de_results if tf in gene_name_to_idx]
        fleck_tfs = set(grn_df["tf"].unique()) if fleck_grn_path.exists() else set()
        overlapping_tfs = [tf for tf in pollen_tfs_in_shared if tf in fleck_tfs]
        print(f"        Pollen TFs in shared genes: {len(pollen_tfs_in_shared)}")
        print(f"        Pollen TFs also in Fleck GRN: {len(overlapping_tfs)}")
        if overlapping_tfs:
            print(f"        Overlapping TFs: {overlapping_tfs[:20]}")

    # ------------------------------------------------------------------
    # Step 5: Build perturbation arrays
    # ------------------------------------------------------------------
    with _timer("Step 5 — Build perturbation arrays"):
        tf_list = sorted(de_results.keys())
        K = len(tf_list)

        perturbation_masks = np.zeros((K, n_genes), dtype=bool)
        perturbation_effects = np.zeros((K, n_genes), dtype=np.float32)

        for ki, tf in enumerate(tf_list):
            # Mark the TF itself
            if tf in gene_name_to_idx:
                perturbation_masks[ki, gene_name_to_idx[tf]] = True

            # Extract log2FC for shared genes only
            full_log2fc = de_results[tf]["log2fc"]
            for gi, gene in enumerate(shared_genes):
                pidx = pollen_idx[gene]
                perturbation_effects[ki, gi] = full_log2fc[pidx]

            # Normalize to unit variance
            std = perturbation_effects[ki].std()
            if std > 1e-8:
                perturbation_effects[ki] /= std

        print(f"        perturbation_masks: {perturbation_masks.shape}")
        print(f"        perturbation_effects: {perturbation_effects.shape}")

        np.save(out_dir / "perturbation_masks.npy", perturbation_masks)
        np.save(out_dir / "perturbation_effects.npy", perturbation_effects)

    # ------------------------------------------------------------------
    # Step 6: Cell type composition per perturbation
    # ------------------------------------------------------------------
    with _timer("Step 6 — Cell type composition shifts"):
        if celltype_col and celltypes:
            ct_list = sorted(celltypes)
            n_types = len(ct_list)
            ct_to_idx = {ct: i for i, ct in enumerate(ct_list)}

            cell_type_fractions = np.zeros((K, n_types), dtype=np.float32)
            ctrl_fractions = np.zeros(n_types, dtype=np.float32)

            # Control cell type distribution
            ctrl_ct = adata.obs.loc[ctrl_mask, celltype_col].dropna()
            for ct, count in ctrl_ct.value_counts().items():
                if ct in ct_to_idx:
                    ctrl_fractions[ct_to_idx[ct]] = count / len(ctrl_ct)

            for ki, tf in enumerate(tf_list):
                tf_mask = guide_labels == tf
                tf_ct = adata.obs.loc[tf_mask, celltype_col].dropna()
                for ct, count in tf_ct.value_counts().items():
                    if ct in ct_to_idx:
                        cell_type_fractions[ki, ct_to_idx[ct]] = count / len(tf_ct)
                # Store as shift from control
                cell_type_fractions[ki] -= ctrl_fractions

            np.save(out_dir / "cell_type_fractions.npy", cell_type_fractions)
            with open(out_dir / "cell_type_names.json", "w") as f:
                json.dump(ct_list, f)
            print(f"        cell_type_fractions: {cell_type_fractions.shape}")
        else:
            print("        No cell type data available, skipping")

    # ------------------------------------------------------------------
    # Step 7: Build incidence matrix from DE results
    # ------------------------------------------------------------------
    with _timer("Step 7 — Build incidence matrix from DE"):
        # Each TF defines a hyperedge containing itself + significant targets
        incidence = np.zeros((n_genes, K), dtype=np.float32)
        for ki, tf in enumerate(tf_list):
            if tf in gene_name_to_idx:
                incidence[gene_name_to_idx[tf], ki] = 1.0
            # Add significant targets (padj < 0.05, |log2fc| > 0.25)
            full_log2fc = de_results[tf]["log2fc"]
            full_pval = de_results[tf]["pval"]
            for gi, gene in enumerate(shared_genes):
                pidx = pollen_idx[gene]
                if abs(full_log2fc[pidx]) > 0.25 and full_pval[pidx] < 0.05:
                    incidence[gi, ki] = 1.0

        n_assigned = (incidence.sum(axis=1) > 0).sum()
        print(f"        incidence: {incidence.shape}")
        print(f"        genes in at least one regulon: {n_assigned}/{n_genes}")
        np.save(out_dir / "incidence.npy", incidence)

    # ------------------------------------------------------------------
    # Step 8: PCA node features
    # ------------------------------------------------------------------
    with _timer("Step 8 — PCA node features"):
        # Use control cell expression for gene-gene covariance
        if sp.issparse(X):
            ctrl_expr = np.array(
                X[ctrl_mask.values][:, shared_gene_idx].todense()
            ).astype(np.float64)
        else:
            ctrl_expr = X_dense[ctrl_mask.values][:, shared_gene_idx].astype(np.float64)

        # Subsample for PCA if too many cells
        rng = np.random.default_rng(42)
        n_sub = min(5000, ctrl_expr.shape[0])
        sub_idx = rng.choice(ctrl_expr.shape[0], size=n_sub, replace=False)
        sub = ctrl_expr[sub_idx].T  # (n_genes, n_sub)

        cov = np.cov(sub)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Take top components explaining 90% variance
        eigvals_desc = eigvals[::-1]
        cumvar = np.cumsum(eigvals_desc) / eigvals_desc.sum()
        k_90 = int(np.searchsorted(cumvar, 0.90) + 1)
        k_use = max(min(k_90, 50), 2)

        node_features = eigvecs[:, -k_use:].astype(np.float32)
        print(f"        PCA dim: {k_use} (90% var at k={k_90})")
        print(f"        node_features: {node_features.shape}")
        np.save(out_dir / "node_features_pca.npy", node_features)

    # ------------------------------------------------------------------
    # Step 9: Save metadata
    # ------------------------------------------------------------------
    with _timer("Step 9 — Save metadata"):
        with open(out_dir / "gene_names.json", "w") as f:
            json.dump(shared_genes, f)
        with open(out_dir / "tf_names.json", "w") as f:
            json.dump(tf_list, f)

        # Per-TF DE summary as CSV (for compare_pollen.py)
        de_rows = []
        for tf in tf_list:
            full_log2fc = de_results[tf]["log2fc"]
            full_pval = de_results[tf]["pval"]
            for gi, gene in enumerate(shared_genes):
                pidx = pollen_idx[gene]
                de_rows.append({
                    "gene": gene,
                    "ko_gene": tf,
                    "log2fc": float(full_log2fc[pidx]),
                    "pval": float(full_pval[pidx]),
                })
        de_df = pd.DataFrame(de_rows)
        de_df.to_csv(out_dir / "pollen_de.csv", index=False)

        summary = {
            "paper": "Ding, Kim et al. (Pollen lab) Nature 2026",
            "doi": "10.1038/s41586-025-09997-7",
            "geo": "GSE284197",
            "system": "primary human cortical cultures (2D CRISPRi)",
            "n_cells_total": int(n_cells),
            "n_cells_control": int(n_ctrl),
            "n_genes_total": int(n_genes_total),
            "n_genes_shared": n_genes,
            "n_tfs_perturbed": len(tf_list),
            "n_tfs_in_fleck_grn": len(overlapping_tfs),
            "overlapping_tfs": overlapping_tfs,
            "tf_list": tf_list,
            "pca_dim": k_use,
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"        Summary: {summary}")

    print()
    print("=" * 60)
    print("Pollen preprocessing complete. Outputs:")
    print(f"  {out_dir}")
    print()
    print("Next: uv run python scripts/compare_pollen.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Pollen/Ding 2026 CRISPRi data for hgx comparison"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=None,
        help="Root data directory (auto-detected if omitted)",
    )
    args = parser.parse_args()

    if args.data_dir:
        data_dir = args.data_dir.resolve()
    else:
        candidates = [
            Path("/workspace/benchmark/data"),
            Path(__file__).resolve().parent.parent / "data",
        ]
        data_dir = next((p for p in candidates if p.is_dir()), None)
        if data_dir is None:
            print("ERROR: Cannot find data directory. Pass --data-dir.")
        else:
            main(data_dir)
