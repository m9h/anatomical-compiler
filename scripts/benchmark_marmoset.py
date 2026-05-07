import argparse
import json
import time
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

def load_fleck():
    data_dir = PROJECT_ROOT / "data/processed"
    d = {}
    with open(data_dir / "gene_names.json") as f: d["gene_names"] = json.load(f)
    with open(data_dir / "tf_names.json") as f: d["tf_names"] = json.load(f)
    d["incidence"] = np.load(data_dir / "incidence.npy")
    return d

def main():
    print("Loading datasets...")
    fleck = load_fleck()
    marmoset = sc.read_h5ad(PROJECT_ROOT / "data/krienen/cortex_GAD_annotated.h5ad", backed='r')
    
    # 1. Gene mapping
    # Marmoset data was annotated with scTab, so it has 'sctab_cell_type'
    # We want to check interneuron (GAD) regulons.
    
    # Common genes (symbols)
    # Marmoset var has 'feature_name' (human symbols from scTab mapping)
    m_genes = marmoset.var['feature_name'].astype(str).str.upper().tolist()
    f_genes = [g.upper() for g in fleck["gene_names"]]
    shared_genes = sorted(set(f_genes) & set(m_genes))
    print(f"Shared genes: {len(shared_genes)}")
    
    f_idx = [f_genes.index(g) for g in shared_genes]
    m_idx = [m_genes.index(g) for g in shared_genes]
    
    # 2. Extract Marmoset correlation-based topology
    # We'll pick a few interneuron TFs shared between human and marmoset
    m_tfs_all = sorted(marmoset.obs['sctab_cell_type'].unique())
    print(f"Marmoset scTab types: {m_tfs_all[:5]}...")
    
    # Let's use specific GAD-relevant TFs: DLX1, DLX2, ASCL1, SOX2
    target_tfs = ["DLX1", "DLX2", "ASCL1", "SOX2"]
    available_tfs = [tf for tf in target_tfs if tf in m_genes and tf in fleck["tf_names"]]
    print(f"TFs for comparison: {available_tfs}")
    
    # Compute correlation in Marmoset (chunked)
    print("Computing Marmoset correlations...")
    # Load shared gene expression for all cells
    # (cells, shared_genes)
    X_m = marmoset[:, m_idx].to_memory().X
    if hasattr(X_m, "toarray"): X_m = X_m.toarray()
    
    # Standardize
    X_m = (X_m - X_m.mean(axis=0)) / (X_m.std(axis=0) + 1e-8)
    
    results = []
    for tf in available_tfs:
        # TF expression
        tf_expr_sparse = marmoset[:, m_genes.index(tf)].to_memory().X
        tf_expr = tf_expr_sparse.toarray().flatten() if hasattr(tf_expr_sparse, "toarray") else tf_expr_sparse.flatten()
        tf_expr = (tf_expr - tf_expr.mean()) / (tf_expr.std() + 1e-8)
        
        # Correlation with shared genes
        corr = (X_m.T @ tf_expr) / X_m.shape[0]
        # Use top 5% of genes by correlation as the regulon
        threshold = np.percentile(np.abs(corr), 95)
        obs_m = (np.abs(corr) >= threshold).astype(float)
        
        # Human prediction (topology)
        fi = fleck["tf_names"].index(tf)
        pred_f = fleck["incidence"][f_idx, fi]
        
        # Overlap analysis
        n_shared = len(shared_genes)
        n_human = int(pred_f.sum())
        n_marm = int(obs_m.sum())
        n_both = int((pred_f * obs_m).sum())
        
        # Fisher test
        # [[both, human_only], [marm_only, neither]]
        table = [[n_both, n_human - n_both], [n_marm - n_both, n_shared - n_human - n_marm + n_both]]
        _, pval = stats.fisher_exact(table, alternative="greater")
        
        jaccard = n_both / (n_human + n_marm - n_both) if (n_human + n_marm - n_both) > 0 else 0
        
        results.append({"tf": tf, "jaccard": jaccard, "p": pval, "n_human": n_human, "n_marm": n_marm, "n_both": n_both})
        print(f"  {tf:<10}: Human {n_human:>4}, Marm {n_marm:>4}, Both {n_both:>2}, Jaccard {jaccard:>6.4f}, p {pval:.2e}")

    # 3. Plot
    if results:
        plt.figure(figsize=(10, 6))
        tfs = [r["tf"] for r in results]
        jaccards = [r["jaccard"] for r in results]
        plt.bar(tfs, jaccards, color="orchid")
        plt.ylabel("Jaccard Index (Neighbor Overlap)")
        plt.title("Evolutionary Regulon Overlap: Human (Organoid) vs. Marmoset (Interneuron)")
        # Add labels for p-values
        for i, r in enumerate(results):
            plt.text(i, jaccards[i] + 0.001, f"p={r['p']:.1e}", ha="center", fontsize=9)

if __name__ == "__main__":
    main()
