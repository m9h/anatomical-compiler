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
    print("Loading Gartner 2021 4D Bioprinting data...")
    adata = sc.read_h5ad("data/bioprinting/gartner_2021_processed.h5ad")
    
    # 1. Maturity Markers for Kidney
    # From Combes/Lindstrom fetal reference
    markers = {
        "Podocyte": ["NPHS1", "NPHS2", "MAFB"],
        "Proximal Tubule": ["LRP2", "CUBN", "HNF4A"],
        "Loop of Henle": ["SLC12A1", "UMOD"],
        "Stroma": ["COL1A1", "PDGFRB"]
    }
    
    sc.pp.log1p(adata)
    
    results = []
    for lineage, genes in markers.items():
        present = [g for g in genes if g in adata.var_names]
        if not present: continue
        
        # Score cells
        sc.tl.score_genes(adata, present, score_name=f"{lineage}_score")
        
        # Aggregate by sample
        for sample in ["man", "r0", "r40"]:
            score = adata[adata.obs['sample'] == sample].obs[f"{lineage}_score"].mean()
            results.append({
                "lineage": lineage,
                "sample": sample,
                "is_4d": sample != "man",
                "score": float(score)
            })

    df = pd.DataFrame(results)
    
    # 2. Plot: 4D Improvement (r40 vs man)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="lineage", y="score", hue="sample", palette="Set2")
    plt.title("4D Bioprinting Fidelity: Gartner 2021 conformation analysis\n(r40 lines vs manual dots)")
    plt.ylabel("Lineage Maturity Score")
    
    out_path = Path("figures/gartner_4d_fidelity.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Figure saved to {out_path}")
    
    # 3. Structural integration (proxy)
    # We'll calculate cluster modularity as a proxy for structural complexity
    print("Computing structural complexity proxy...")
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.5)
    
    modularity = []
    for sample in ["man", "r0", "r40"]:
        sub = adata[adata.obs['sample'] == sample]
        n_mod = len(sub.obs['leiden'].unique())
        modularity.append({"sample": sample, "n_modules": n_mod})
        
    print(pd.DataFrame(modularity))
    
    # Save summary
    with open("figures/gartner_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
