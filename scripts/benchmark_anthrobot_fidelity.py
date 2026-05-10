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
import seaborn as sns
from scipy import stats

def main():
    print("Loading Anthrobots (2023) data...")
    adata_path = Path("data/bioprinting/anthro_2023_processed.h5ad")
    if not adata_path.exists():
        print(f"  Error: {adata_path} not found.")
        return
        
    adata = sc.read_h5ad(adata_path)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.log1p(adata)
    
    # 1. Identify Motility/Cilia Regulons
    # Master regulators of multiciliated cells
    motility_genes = ["FOXJ1", "MCIDAS", "CCNO", "DNAI1", "DNAH5", "DYX1C1"]
    present_genes = [g for g in motility_genes if g in adata.var_names]
    print(f"Motility genes found: {present_genes}")
    
    # 2. Score Motility Pattern
    sc.tl.score_genes(adata, present_genes, score_name="motility_score")
    
    # 3. Compare samples (S1 vs S2)
    # S1 and S2 might represent different stages or conditions
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=adata.obs, x="sample", y="motility_score", palette="muted")
    plt.title("Anthrobot Motility Program Activity (FOXJ1/Cilia Regulon)")
    plt.ylabel("Motility Score (Gene Score)")
    
    out_path = Path("figures/anthrobot_motility_benchmark.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Figure saved to {out_path}")
    
    # 4. Modularity Analysis (Hodge Laplacian snippet)
    # We'll just compute a basic connectivity score for now
    print("Computing structural modularity...")
    # (Using a simpler proxy since Hodge is intensive)
    sc.pp.highly_variable_genes(adata, n_top_genes=500)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.5)
    
    n_clusters = len(adata.obs['leiden'].unique())
    print(f"  Detected {n_clusters} regulatory modules.")
    
    # Save results
    results = {
        "n_cells": adata.shape[0],
        "n_genes": adata.shape[1],
        "motility_mean": float(adata.obs['motility_score'].mean()),
        "n_modules": n_clusters,
        "samples": adata.obs['sample'].unique().tolist()
    }
    
    with open("figures/anthrobot_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved to figures/anthrobot_results.json")

if __name__ == "__main__":
    main()
