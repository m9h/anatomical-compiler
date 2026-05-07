import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from hgx_prep import grn, pca, normalize

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

def main():
    parser = argparse.ArgumentParser(description="Preprocess ZSCAPE zebrafish perturbation atlas")
    parser.add_argument("--input", type=str, required=True, help="Input zesta.h5ad file")
    parser.add_argument("--out-dir", type=str, default="data/zscape/processed", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with _timer("Loading ZSCAPE data (backed)"):
        adata = ad.read_h5ad(args.input, backed='r')
        n_cells, n_genes = adata.shape
        gene_names = adata.var["gene_short_name"].tolist()
        
    # 1. Identify conditions
    with _timer("Identifying conditions"):
        conditions = adata.obs["gene_target"].unique().tolist()
        ctrl_cond = "control_control"
        if ctrl_cond not in conditions:
            # Try to find control
            ctrl_cond = [c for c in conditions if "control" in c][0]
            
        # Focus on single knockouts for the TF list
        # Condition format: tf_control or tf1_tf2
        single_kos = [c for c in conditions if c.endswith("_control") and c != ctrl_cond]
        tf_list = sorted([c.replace("_control", "") for c in single_kos])
        print(f"  Found {len(tf_list)} single TF knockouts")

    # 2. Compute mean expression per condition
    # To do this efficiently on 2.6M cells, we process in chunks or use a sparse sum
    with _timer("Computing condition means"):
        # Load indices for each condition
        cond_to_idx = {c: np.where(adata.obs["gene_target"] == c)[0] for c in single_kos + [ctrl_cond]}
        
        # We only have 2000 genes, so we can load the full matrix into memory if needed
        # 2.6M * 2000 * 4 bytes = 20.8 GB. This should fit on most high-RAM machines.
        # If not, we do it cell-type by cell-type or in gene chunks.
        # Let's try loading the whole thing first for speed.
        print("  Loading expression matrix into memory...")
        X = adata.X[:] # This might take 20GB
        if hasattr(X, "toarray"):
            X = X.toarray()
            
        means = {}
        for cond, idx in cond_to_idx.items():
            means[cond] = X[idx].mean(axis=0)
            
        ctrl_mean = means[ctrl_cond]
        
    # 3. Compute DE and Abundance Shifts
    with _timer("Computing DE and abundance"):
        de_results = {}
        abundance_shifts = [] # (K, n_types)
        
        cell_types = sorted(adata.obs["cell_type_broad"].dropna().unique().tolist())
        ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}
        n_types = len(cell_types)
        
        ctrl_counts = adata.obs[adata.obs["gene_target"] == ctrl_cond]["cell_type_broad"].value_counts()
        ctrl_fractions = np.zeros(n_types)
        for ct, count in ctrl_counts.items():
            ctrl_fractions[ct_to_idx[ct]] = count / ctrl_counts.sum()
            
        for tf in tf_list:
            cond = f"{tf}_control"
            # Log2FC (assuming data is already log1p)
            log2fc = means[cond] - ctrl_mean
            
            # Placeholder p-value (we'd need a proper test for real p-vals, 
            # but for benchmarking we can use a magnitude threshold)
            # In a real run, we'd use t-test or Wilcoxon.
            pval = np.zeros(n_genes) # dummy
            
            de_results[tf] = {"log2fc": log2fc, "pval": pval}
            
            # Abundance shift
            tf_counts = adata.obs[adata.obs["gene_target"] == cond]["cell_type_broad"].value_counts()
            tf_fractions = np.zeros(n_types)
            for ct, count in tf_counts.items():
                if ct in ct_to_idx:
                    tf_fractions[ct_to_idx[ct]] = count / tf_counts.sum()
            abundance_shifts.append(tf_fractions - ctrl_fractions)
            
        abundance_shifts = np.array(abundance_shifts, dtype=np.float32)

    # 4. Build Hypergraph
    with _timer("Building hypergraph"):
        # We use a simple magnitude threshold for 'DE' edges since we didn't compute p-vals
        # Actually, let's use hgx_prep.grn logic
        grn_res = grn.build_incidence_from_de(
            de_results,
            gene_names,
            log2fc_threshold=0.1, # sensitive for ZSCAPE
            pval_threshold=1.1    # disable pval filter for now
        )
        
    # 5. PCA Node Features
    with _timer("Computing PCA features"):
        # Use the control mean as a base feature or the full matrix
        # compute_pca expects (genes, cells)
        pca_res = pca.compute_pca(X.T, dim=64, dim_method="fixed")
        node_features_pca = pca_res.features

    # 6. Save outputs
    with _timer("Saving arrays"):
        np.save(out_dir / "incidence.npy", grn_res.incidence)
        np.save(out_dir / "node_features_pca.npy", node_features_pca)
        
        # Perturbation effects (K, n_genes)
        pert_effects = np.array([de_results[tf]["log2fc"] for tf in tf_list], dtype=np.float32)
        np.save(out_dir / "perturbation_effects.npy", pert_effects)
        
        # Cell type fractions (K, n_types)
        np.save(out_dir / "cell_type_fractions.npy", abundance_shifts)
        
        # Metadata
        with open(out_dir / "gene_names.json", "w") as f:
            json.dump(gene_names, f)
        with open(out_dir / "tf_names.json", "w") as f:
            json.dump(tf_list, f)
        with open(out_dir / "cell_type_names.json", "w") as f:
            json.dump(cell_types, f)
            
        summary = {
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_tfs": len(tf_list),
            "n_cell_types": n_types,
            "conditions": conditions
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Preprocessing complete. Results in {out_dir}")

if __name__ == "__main__":
    main()
