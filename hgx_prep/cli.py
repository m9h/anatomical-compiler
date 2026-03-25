"""hgx-prep CLI: download, preprocess, and standardize GRN datasets."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from . import download, grn, io, normalize, pca, perturbation, registry, temporal


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
    for col in obs.columns:
        vals = obs[col].dropna().unique()
        if len(vals) < 100 and any(
            v in vals for v in ["GLI3", "ARX", "NR2E1", "ZNF219", "TBR1"]
        ):
            return col
    return None


def _is_control(label: str, patterns: list[str]) -> bool:
    label_lower = str(label).lower()
    return any(pat.lower() in label_lower for pat in patterns)


# -------------------------------------------------------------------------
# Pipeline for h5ad Perturb-seq data (like Pollen)
# -------------------------------------------------------------------------

def _run_h5ad_pipeline(
    h5ad_path: Path,
    out_dir: Path,
    config: registry.DatasetConfig,
    args: argparse.Namespace,
) -> None:
    """Process a Perturb-seq h5ad file."""

    with _timer("Load h5ad"):
        adata = ad.read_h5ad(h5ad_path)
        print(f"        Shape: {adata.shape} (cells x genes)")
        n_cells, n_genes_total = adata.shape

    with _timer("Identify perturbations"):
        guide_col = args.guide_col or config.guide_column or _detect_guide_column(adata.obs)
        if guide_col is None:
            print("ERROR: Cannot detect guide column. Use --guide-col.")
            print(f"  Available: {list(adata.obs.columns)}")
            sys.exit(1)
        print(f"        Guide column: '{guide_col}'")

        guide_labels = adata.obs[guide_col].astype(str).values
        unique_guides = sorted(set(guide_labels))
        controls = [g for g in unique_guides if _is_control(g, config.control_patterns)]
        tf_targets = [g for g in unique_guides if not _is_control(g, config.control_patterns)]
        print(f"        TFs: {len(tf_targets)}, Controls: {len(controls)}")

    with _timer("Compute gene universe"):
        gene_names_all = list(adata.var_names)

        # Optionally intersect with a reference GRN
        if args.intersect_grn:
            ref_df = pd.read_csv(args.intersect_grn, sep="\t")
            ref_genes = sorted(set(ref_df["tf"].unique()) | set(ref_df["target"].unique()))
            gene_names = sorted(set(ref_genes) & set(gene_names_all))
            print(f"        Intersected: {len(gene_names)} / {len(gene_names_all)}")
        else:
            gene_names = sorted(gene_names_all)

        pollen_idx = {g: i for i, g in enumerate(gene_names_all)}
        gene_idx_in_source = {g: pollen_idx[g] for g in gene_names if g in pollen_idx}
        n_genes = len(gene_names)

    with _timer("Compute perturbation effects (observed DE)"):
        shared_col_idx = [pollen_idx[g] for g in gene_names]

        # Build DE results for the GRN module
        X = adata.X
        ctrl_mask = np.isin(guide_labels, controls)

        if sp.issparse(X):
            ctrl_mean = np.asarray(X[ctrl_mask].mean(axis=0)).ravel()
            ctrl_var = np.asarray(X[ctrl_mask].power(2).mean(axis=0)).ravel() - ctrl_mean ** 2
        else:
            X_dense = np.asarray(X, dtype=np.float32)
            ctrl_mean = X_dense[ctrl_mask].mean(axis=0)
            ctrl_var = X_dense[ctrl_mask].var(axis=0)

        n_ctrl = ctrl_mask.sum()
        de_results: dict[str, dict] = {}

        for tf in tf_targets:
            tf_mask = guide_labels == tf
            n_tf = tf_mask.sum()
            if n_tf < args.min_cells:
                continue
            if sp.issparse(X):
                tf_mean = np.asarray(X[tf_mask].mean(axis=0)).ravel()
            else:
                tf_mean = X_dense[tf_mask].mean(axis=0)

            pseudo = 0.1
            log2fc = np.log2(tf_mean + pseudo) - np.log2(ctrl_mean + pseudo)
            pooled_var = ctrl_var + 1e-8
            se = np.sqrt(pooled_var / n_ctrl + pooled_var / n_tf)
            z = (tf_mean - ctrl_mean) / (se + 1e-8)
            from scipy.stats import norm
            pval = 2 * norm.sf(np.abs(z))

            de_results[tf] = {
                "log2fc": log2fc.astype(np.float32),
                "pval": pval.astype(np.float64),
                "n_cells": int(n_tf),
            }
        print(f"        Computed DE for {len(de_results)} TFs")

    with _timer("Build perturbation arrays"):
        tf_list = sorted(de_results.keys())
        K = len(tf_list)
        gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}

        p_masks = np.zeros((K, n_genes), dtype=bool)
        p_effects = np.zeros((K, n_genes), dtype=np.float32)

        for ki, tf in enumerate(tf_list):
            if tf in gene_name_to_idx:
                p_masks[ki, gene_name_to_idx[tf]] = True
            full_log2fc = de_results[tf]["log2fc"]
            for gi, gene in enumerate(gene_names):
                pidx = pollen_idx.get(gene)
                if pidx is not None:
                    p_effects[ki, gi] = full_log2fc[pidx]
            std = p_effects[ki].std()
            if std > 1e-8:
                p_effects[ki] /= std

    grn_method = args.grn
    if grn_method == "de":
        with _timer("Build incidence from DE"):
            grn_result = grn.build_incidence_from_de(
                de_results, gene_names,
                log2fc_threshold=0.25, pval_threshold=args.padj,
                gene_idx_in_source=gene_idx_in_source,
            )
    elif grn_method == "supplied":
        with _timer("Load supplied GRN"):
            grn_result = grn.load_supplied_grn(
                Path(args.grn_file), gene_names, padj_threshold=args.padj,
            )
    elif grn_method == "pando":
        with _timer("Load Pando GRN"):
            grn_result = grn.load_pando_grn(
                Path(args.grn_file), gene_names, padj_threshold=args.padj,
            )
    else:
        print(f"ERROR: Unknown --grn method: {grn_method}")
        sys.exit(1)

    with _timer("PCA node features"):
        # Use control cells for gene-gene covariance
        if sp.issparse(X):
            ctrl_expr = np.asarray(X[ctrl_mask][:, shared_col_idx].todense(), dtype=np.float64)
        else:
            ctrl_expr = X_dense[ctrl_mask][:, shared_col_idx].astype(np.float64)

        pca_result = pca.compute_pca(
            ctrl_expr.T,  # (n_genes, n_cells)
            dim_method=args.dim_method,
            dim=args.feature_dim,
            variance_threshold=0.90,
        )
        print(f"        PCA dim: {pca_result.dim} ({pca_result.method})")

    with _timer("Write output"):
        io.write_processed(
            out_dir,
            gene_names=gene_names,
            tf_names=grn_result.tf_names,
            incidence=grn_result.incidence,
            node_features_pca=pca_result.features,
            perturbation_masks=p_masks,
            perturbation_effects=p_effects,
            eigenvalues=pca_result.eigenvalues,
            module_labels=grn_result.module_labels,
            source_doi=config.doi,
            source_geo=config.geo,
            grn_method=grn_method,
            pca_details=pca_result.details,
            extra_metadata={
                "paper": config.name,
                "n_cells_total": n_cells,
                "n_cells_control": int(n_ctrl),
                "n_tfs_perturbed": K,
            },
        )


# -------------------------------------------------------------------------
# Pipeline for MatrixMarket + metadata (like Fleck)
# -------------------------------------------------------------------------

def _run_mtx_pipeline(
    data_dir: Path,
    out_dir: Path,
    config: registry.DatasetConfig,
    args: argparse.Namespace,
) -> None:
    """Process MatrixMarket count data with Pando GRN."""

    with _timer("Load GRN"):
        coefs_path = data_dir / "pando" / "coefs.tsv"
        if args.grn_file:
            coefs_path = Path(args.grn_file)
        coefs = pd.read_csv(coefs_path, sep="\t")
        all_tfs = sorted(coefs["tf"].unique())
        all_targets = sorted(coefs["target"].unique())
        grn_genes = sorted(set(all_tfs) | set(all_targets))
        print(f"        GRN: {len(all_tfs)} TFs, {len(grn_genes)} genes")

    with _timer("Load count matrix"):
        counts_path = data_dir / "zenodo" / "data_matrices" / "counts.mtx.gz"
        features_path = data_dir / "zenodo" / "data_matrices" / "features.tsv.gz"
        barcodes_path = data_dir / "zenodo" / "data_matrices" / "barcodes.tsv.gz"

        features_df = pd.read_csv(features_path, sep="\t", header=None)
        feature_names = features_df.iloc[:, 0].astype(str).tolist()
        barcodes_df = pd.read_csv(barcodes_path, sep="\t", header=None)
        barcodes = barcodes_df.iloc[:, 0].astype(str).tolist()

        counts_raw = sio.mmread(counts_path)
        counts_raw = sp.csc_matrix(counts_raw)

        feature_set = set(feature_names)
        gene_names = sorted(g for g in grn_genes if g in feature_set)
        feature_name_to_idx = {n: i for i, n in enumerate(feature_names)}
        gene_row_indices = np.array([feature_name_to_idx[g] for g in gene_names], dtype=np.intp)
        counts_sub = counts_raw[gene_row_indices, :]
        n_genes = len(gene_names)
        print(f"        {n_genes} genes x {counts_sub.shape[1]} cells")

    with _timer("Load metadata"):
        meta_path = data_dir / "zenodo" / "data_matrices" / "meta.tsv.gz"
        meta = pd.read_csv(meta_path, sep="\t")
        barcode_to_col = {bc: i for i, bc in enumerate(barcodes)}
        meta_cellids = meta["cellID"].astype(str).tolist()

        matched_meta_rows, matched_col_indices = [], []
        for mi, cid in enumerate(meta_cellids):
            if cid in barcode_to_col:
                matched_meta_rows.append(mi)
                matched_col_indices.append(barcode_to_col[cid])

        matched_col_arr = np.array(matched_col_indices, dtype=np.intp)
        meta_matched = meta.iloc[matched_meta_rows].reset_index(drop=True)
        counts_matched = counts_sub[:, matched_col_arr]
        n_cells = len(matched_col_indices)
        print(f"        Matched: {n_cells} cells")

        pseudotime = meta_matched[config.pseudotime_column].values.astype(np.float32)
        lineage_vals = meta_matched[config.lineage_column].values if config.lineage_column else None

    with _timer("Normalize expression"):
        scaled = normalize.normalize_expression(counts_matched)
        log_expr = np.log1p(
            sp.csc_matrix(counts_matched, dtype=np.float64)
            @ sp.diags(1e4 / np.maximum(
                np.asarray(counts_matched.sum(axis=0)).ravel(), 1.0
            ))
        ).toarray().astype(np.float32)
        print(f"        Scaled: {scaled.shape}")

    with _timer("PCA"):
        pca_result = pca.compute_pca(
            scaled,
            dim_method=args.dim_method,
            dim=args.feature_dim,
        )
        print(f"        PCA dim: {pca_result.dim}")

    with _timer("Build GRN incidence"):
        grn_result = grn.load_pando_grn(
            coefs_path, gene_names, padj_threshold=args.padj,
        )
        print(f"        {len(grn_result.tf_names)} TFs with targets")

    temporal_result = None
    if config.has_pseudotime and not args.skip_temporal:
        with _timer(f"Temporal expression ({args.num_bins} bins)"):
            temporal_result = temporal.bin_temporal(
                scaled, pseudotime,
                num_bins=args.num_bins,
                lineage=lineage_vals,
                lineage_names=config.lineage_names or None,
            )

    # Fate probabilities (optional)
    fate_probs = None
    if config.has_fates and not args.skip_fates:
        with _timer("Fate probabilities"):
            h5ad_path = data_dir / "zenodo" / "RNA_all_velo.h5ad"
            if h5ad_path.exists():
                adata = ad.read_h5ad(h5ad_path, backed="r")
                fate_df = adata.obs[config.fate_columns].copy()
                fate_df.index = fate_df.index.astype(str)

                matched_barcodes = meta_matched["cellID"].astype(str).tolist()
                h5ad_bcs = set(fate_df.index)
                fate_list = []
                fate_mask = []
                for bc in matched_barcodes:
                    if bc in h5ad_bcs:
                        fate_list.append(fate_df.loc[bc].values.astype(np.float32))
                        fate_mask.append(True)
                    else:
                        fate_mask.append(False)

                cell_fate_probs = np.array(fate_list, dtype=np.float32)
                fate_mask_arr = np.array(fate_mask)

                # Per-bin fates
                bin_edges = np.linspace(0, 1, args.num_bins + 1)
                fate_probs = np.zeros((args.num_bins, len(config.fate_columns)), dtype=np.float32)
                for b in range(args.num_bins):
                    lo, hi = bin_edges[b], bin_edges[b + 1]
                    if b < args.num_bins - 1:
                        pt_mask = (pseudotime >= lo) & (pseudotime < hi)
                    else:
                        pt_mask = (pseudotime >= lo) & (pseudotime <= hi)
                    combined = pt_mask & fate_mask_arr
                    if combined.sum() > 0:
                        fate_indices = []
                        counter = 0
                        for ci in range(n_cells):
                            if fate_mask_arr[ci]:
                                if pt_mask[ci]:
                                    fate_indices.append(counter)
                                counter += 1
                        if fate_indices:
                            fate_probs[b] = cell_fate_probs[fate_indices].mean(axis=0)

                adata.file.close()
                print(f"        {cell_fate_probs.shape[0]} cells with fate data")

    with _timer("Perturbation effects"):
        gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
        perturb_result = perturbation.compute_grn_perturbations(
            coefs, gene_name_to_idx, config.key_tfs, n_genes,
        )
        print(f"        {len(perturb_result.tf_list)} TFs")

    with _timer("Write output"):
        io.write_processed(
            out_dir,
            gene_names=gene_names,
            tf_names=grn_result.tf_names,
            incidence=grn_result.incidence,
            node_features_pca=pca_result.features,
            perturbation_masks=perturb_result.masks,
            perturbation_effects=perturb_result.effects,
            eigenvalues=pca_result.eigenvalues,
            module_labels=grn_result.module_labels,
            temporal_expression=(
                temporal_result.expression if temporal_result else None
            ),
            pseudotime_centers=(
                temporal_result.pseudotime_centers if temporal_result else None
            ),
            lineage_fractions=(
                temporal_result.lineage_fractions if temporal_result else None
            ),
            fate_probabilities=fate_probs,
            source_doi=config.doi,
            grn_method="pando",
            pca_details=pca_result.details,
            extra_metadata={
                "paper": config.name,
                "n_cells": n_cells,
                "key_tfs": config.key_tfs,
                "lineages": config.lineage_names,
                "fates": config.fate_columns,
            },
        )


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="hgx-prep",
        description="Download, preprocess, and standardize GRN datasets for hypergraph analysis.",
    )

    # Data source
    src = parser.add_argument_group("data source")
    src.add_argument("--doi", type=str, help="Paper DOI")
    src.add_argument("--geo", type=str, help="GEO accession (e.g. GSE284197)")
    src.add_argument("--h5ad", type=str, help="Path to local h5ad file (skip download)")
    src.add_argument("--data-dir", type=str, help="Path to local data directory (for mtx pipelines)")

    # GRN
    grn_group = parser.add_argument_group("GRN construction")
    grn_group.add_argument("--grn", choices=["pando", "de", "supplied"], default=None,
                          help="GRN inference method (default: from registry or 'de')")
    grn_group.add_argument("--grn-file", type=str, help="Path to GRN edge list TSV (for --grn supplied/pando)")
    grn_group.add_argument("--padj", type=float, default=0.05, help="Adjusted p-value threshold")
    grn_group.add_argument("--intersect-grn", type=str, help="Intersect gene universe with this GRN TSV")

    # Features
    feat = parser.add_argument_group("feature extraction")
    feat.add_argument("--dim-method", choices=["ppca", "variance", "fixed"], default="ppca",
                      help="PCA dimensionality method (default: ppca)")
    feat.add_argument("--feature-dim", type=int, default=None, help="Fixed PCA dim (for --dim-method fixed)")

    # Perturbation
    pert = parser.add_argument_group("perturbation")
    pert.add_argument("--guide-col", type=str, help="Override guide column name in h5ad")
    pert.add_argument("--min-cells", type=int, default=10, help="Min cells per TF for DE")

    # Temporal
    temp = parser.add_argument_group("temporal")
    temp.add_argument("--num-bins", type=int, default=10, help="Pseudotime bins")
    temp.add_argument("--skip-temporal", action="store_true", help="Skip temporal outputs")
    temp.add_argument("--skip-fates", action="store_true", help="Skip fate probability outputs")

    # Output
    parser.add_argument("--out", type=str, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--list-datasets", action="store_true", help="List known datasets")

    args = parser.parse_args(argv)

    if args.list_datasets:
        print("Known datasets:")
        for ds in registry.list_known():
            print(f"  {ds.name}")
            if ds.doi:
                print(f"    DOI: {ds.doi}")
            if ds.geo:
                print(f"    GEO: {ds.geo}")
            print(f"    GRN: {ds.grn_default}  Format: {ds.expression_format}")
            print()
        return

    if not args.out:
        parser.error("--out is required")

    # Resolve dataset config
    identifier = args.doi or args.geo
    config = registry.lookup(identifier) if identifier else None

    if config is None:
        config = registry.DatasetConfig(
            name="Custom dataset",
            doi=args.doi,
            geo=args.geo,
        )

    if args.grn is None:
        args.grn = config.grn_default

    out_dir = Path(args.out)

    print("=" * 65)
    print(f"  hgx-prep: {config.name}")
    print(f"  Output: {out_dir}")
    print(f"  GRN method: {args.grn}")
    print("=" * 65)
    print()

    if args.dry_run:
        print("  --dry-run: would process with above settings")
        return

    # Dispatch to appropriate pipeline
    if args.h5ad:
        _run_h5ad_pipeline(Path(args.h5ad), out_dir, config, args)
    elif args.data_dir and config.expression_format == "mtx_gz":
        _run_mtx_pipeline(Path(args.data_dir), out_dir, config, args)
    elif args.geo:
        # Download then process
        dl_dir = out_dir / "raw"
        main_file = config.download_files.get("screen")
        if main_file:
            downloaded = download.download_geo(args.geo, dl_dir, [main_file])
            if downloaded:
                _run_h5ad_pipeline(downloaded[0], out_dir, config, args)
            else:
                print("ERROR: Download failed")
                sys.exit(1)
        else:
            print("ERROR: No known files for this GEO accession.")
            print("  Use --h5ad to point to a local file.")
            sys.exit(1)
    else:
        print("ERROR: Specify --h5ad, --data-dir, or --geo")
        sys.exit(1)

    print()
    print("=" * 65)
    print(f"  Done. Output: {out_dir}")
    valid, missing = io.validate_processed(out_dir)
    if valid:
        print("  Validation: PASS (all required files present)")
    else:
        print(f"  Validation: FAIL (missing: {missing})")
    print("=" * 65)


if __name__ == "__main__":
    main()
