import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def load_fleck_grn():
    data_dir = PROJECT_ROOT / "data/processed"
    with open(data_dir / "gene_names.json") as f: genes = json.load(f)
    with open(data_dir / "tf_names.json") as f: tfs = json.load(f)
    incidence = np.load(data_dir / "incidence.npy")
    return genes, tfs, incidence

def main():
    print("Loading Fleck Organoid GRN...")
    f_genes, f_tfs, f_inc = load_fleck_grn()
    
    print("Loading Yan et al. (2024) Bioprinted Brain data...")
    # Preprocessing might still be running or finished
    adata_path = Path("data/bioprinting/yan_2024_processed.h5ad")
    if not adata_path.exists():
        print("  Error: Yan dataset not preprocessed yet.")
        return
        
    adata = sc.read_h5ad(adata_path, backed='r')
    
    # 1. Gene Mapping
    y_genes = adata.var_names.astype(str).str.upper().tolist()
    shared_genes = sorted(set(f_genes) & set(y_genes))
    print(f"Shared genes: {len(shared_genes)}")
    
    f_idx = [f_genes.index(g) for g in shared_genes]
    y_idx = [y_genes.index(g) for g in shared_genes]
    
    # 2. Extract Yan Correlation-based Topology
    # We'll use the top neurogenic TFs from Fleck
    target_tfs = ["FOXG1", "TBR1", "DLX1", "DLX2", "EMX1", "EOMES", "NEUROD6", "NR2E1"]
    available_tfs = [tf for tf in target_tfs if tf in y_genes and tf in f_tfs]
    print(f"TFs for comparison: {available_tfs}")
    
    # Load expression for shared genes (subsample for speed)
    print("Loading Yan expression data...")
    # Take 5000 cells for correlation calculation
    n_cells = min(adata.shape[0], 5000)
    X_y = adata[:n_cells, y_idx].to_memory().X
    if hasattr(X_y, "toarray"): X_y = X_y.toarray()
    
    # Standardize
    X_y = (X_y - X_y.mean(axis=0)) / (X_y.std(axis=0) + 1e-8)
    
    results = []
    for tf in available_tfs:
        print(f"  Analysing {tf}...")
        # TF expression in Yan
        tf_expr_sparse = adata[:n_cells, y_genes.index(tf)].to_memory().X
        tf_expr = tf_expr_sparse.toarray().flatten() if hasattr(tf_expr_sparse, "toarray") else tf_expr_sparse.flatten()
        tf_expr = (tf_expr - tf_expr.mean()) / (tf_expr.std() + 1e-8)
        
        # Correlation in Yan
        corr = (X_y.T @ tf_expr) / X_y.shape[0]
        # Top 5% as regulon
        threshold = np.percentile(np.abs(corr), 95)
        obs_y = (np.abs(corr) >= threshold).astype(float)
        
        # Fleck Prediction
        fi = f_tfs.index(tf)
        pred_f = f_inc[f_idx, fi]
        
        # Overlap
        n_shared = len(shared_genes)
        n_human_f = int(pred_f.sum())
        n_yan = int(obs_y.sum())
        n_both = int((pred_f * obs_y).sum())
        
        # Fisher test
        table = [[n_both, n_human_f - n_both], [n_yan - n_both, n_shared - n_human_f - n_yan + n_both]]
        _, pval = stats.fisher_exact(table, alternative="greater")
        
        jaccard = n_both / (n_human_f + n_yan - n_both) if (n_human_f + n_yan - n_both) > 0 else 0
        
        results.append({"tf": tf, "jaccard": jaccard, "p": pval, "n_fleck": n_human_f, "n_yan": n_yan, "n_both": n_both})
        print(f"    {tf:<10}: Jaccard {jaccard:>6.4f}, p {pval:.2e}, both {n_both:>2}")

    # 3. Plot
    plt.figure(figsize=(10, 6))
    tfs = [r["tf"] for r in results]
    jaccards = [r["jaccard"] for r in results]
    plt.bar(tfs, jaccards, color="royalblue")
    plt.ylabel("Jaccard Index (Regulon Overlap)")
    plt.title("Fidelity of Bioprinted Functional Brain Tissue (Yan 2024)\nvs Organoid GRN (Fleck 2023)")
    
    for i, r in enumerate(results):
        plt.text(i, jaccards[i] + 0.001, f"p={r['p']:.1e}", ha="center", fontsize=9)
        
    out_path = Path("figures/bioprinting_yan_fidelity.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    
    # Save results
    with open("figures/bioprinting_yan_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
