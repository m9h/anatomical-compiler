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

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def main():
    print("Loading Fleck Organoid RNA...")
    fleck_path = Path("data/zenodo/RNA_all_velo.h5ad")
    if not fleck_path.exists():
        print(f"  Error: {fleck_path} not found.")
        return
    
    # Load backed to save memory, but we need to compute scores
    adata = sc.read_h5ad(fleck_path, backed='r')
    
    # Take a 5000 cell subset for benchmark
    print("  Subsampling 5000 cells...")
    adata_sub = adata[:5000].to_memory()
    sc.pp.log1p(adata_sub)
    
    print("Loading Neocortex Atlas loadings (Sonthalia 2026)...")
    loadings = pd.read_csv("data/SonthaliaEtAl_SupplTable1_p7CtxDev_GeneLoadings.txt", sep="\t")
    
    # 1. Align genes
    shared_genes = sorted(set(adata_sub.var_names) & set(loadings.GeneSymbol))
    print(f"  Shared genes: {len(shared_genes)}")
    
    # Filter loadings and data
    loadings = loadings[loadings.GeneSymbol.isin(shared_genes)].set_index("GeneSymbol")
    loadings = loadings.loc[shared_genes] # Match order
    
    # Patterns p1-p7
    pattern_cols = [c for c in loadings.columns if c.startswith('p')]
    W = loadings[pattern_cols].values # (n_shared_genes, 7)
    
    # 2. Project data: Score = X @ W
    # adata.X is (n_cells, n_genes)
    X = adata_sub[:, shared_genes].X
    if hasattr(X, "toarray"): X = X.toarray()
    
    scores = X @ W # (n_cells, 7)
    
    # Add to adata.obs
    for i, p in enumerate(pattern_cols):
        # Human-friendly name mapping
        p_name = p.split('.')[0]
        adata_sub.obs[p_name] = scores[:, i]

    # 3. Visualization
    # Use existing UMAP if available
    if "X_umap" not in adata_sub.obsm:
        print("  Computing UMAP...")
        sc.pp.highly_variable_genes(adata_sub, n_top_genes=2000)
        sc.pp.pca(adata_sub)
        sc.pp.neighbors(adata_sub)
        sc.tl.umap(adata_sub)
        
    print("  Generating figures...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Annotations based on top genes
    annotations = {
        "p1": "Translation/Growth",
        "p2": "Mature Neurons",
        "p3": "Signaling/Arrest",
        "p4": "IPCs/Early Neuro",
        "p5": "Progenitors/Cycling",
        "p6": "Young Neurons",
        "p7": "Excitatory Neurons"
    }
    
    for i, p in enumerate(pattern_cols):
        p_id = p.split('.')[0]
        sc.pl.umap(adata_sub, color=p_id, ax=axes[i], show=False, title=f"{p_id}: {annotations[p_id]}", cmap="magma")
        
    # Remove last empty axe
    fig.delaxes(axes[-1])
    
    out_path = Path("figures/neocortex_atlas_fidelity.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"  Figure saved to {out_path}")
    
    # 4. Correlation with Lineage
    if "lineage" in adata_sub.obs.columns:
        print("  Analyzing pattern alignment with lineage...")
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=adata_sub.obs, x="lineage", y="p7")
        plt.title("Pattern 7 (Excitatory) activity across organoid lineages")
        plt.savefig("figures/neocortex_atlas_lineage_p7.png", dpi=200)

    # Save summary
    results = {
        "shared_genes": len(shared_genes),
        "patterns": pattern_cols,
        "mean_scores": adata_sub.obs[ [c.split('.')[0] for c in pattern_cols] ].mean().to_dict()
    }
    with open("figures/neocortex_atlas_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
