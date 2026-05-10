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

def main():
    print("Loading Sonthalia 2026 patterns (p7)...")
    loadings = pd.read_csv("data/SonthaliaEtAl_SupplTable1_p7CtxDev_GeneLoadings.txt", sep="\t")
    
    # We focus on p2 (Mature Neurons) and p7 (Excitatory Neurons)
    pattern_cols = ["p2.Shared.Mammal_subcomp.2", "p5.Shared.Mammal_subcomp.5", "p7.Shared.Mammal_subcomp.7"]
    loadings_sub = loadings[["GeneSymbol"] + pattern_cols].set_index("GeneSymbol")
    
    datasets = {
        "Organoid (Fleck)": "data/zenodo/RNA_all_velo.h5ad",
        "Bioprinted Brain (Tang)": "data/bioprinting/tang_2020_processed.h5ad",
        "Bioprinted Kidney (Lawlor)": "data/bioprinting/lawlor_2021_processed.h5ad",
        "Bioprinted Liver (Zhang)": "data/bioprinting/zhang_2025_processed.h5ad"
    }
    
    results = []
    
    for name, path in datasets.items():
        print(f"Processing {name}...")
        if not Path(path).exists():
            print(f"  Skipping: {path} not found.")
            continue
            
        adata = sc.read_h5ad(path)
        if name == "Organoid (Fleck)":
            # Use same 5k subset as before for speed
            adata = adata[:5000].to_memory()
            
        sc.pp.log1p(adata)
        
        # Align genes
        shared = sorted(set(adata.var_names) & set(loadings_sub.index))
        W = loadings_sub.loc[shared].values
        X = adata[:, shared].X
        if hasattr(X, "toarray"): X = X.toarray()
        
        # Project
        scores = X @ W # (n_cells, 3)
        mean_scores = scores.mean(axis=0)
        
        results.append({
            "dataset": name,
            "p2_mature": float(mean_scores[0]),
            "p5_progenitor": float(mean_scores[1]),
            "p7_excitatory": float(mean_scores[2])
        })

    df = pd.DataFrame(results)
    print("\nMaturity Scores (p2 = Mature, p5 = Progenitor, p7 = Excitatory):")
    print(df)
    
    # 1. Maturity Ratio: p2 / p5
    df['maturity_ratio'] = df['p2_mature'] / (df['p5_progenitor'] + 1e-8)
    
    # 2. Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="dataset", y="maturity_ratio", palette="viridis")
    plt.title("System Maturity Score: Bioprinting vs. Self-Organized Organoids\n(Ratio of Neocortex Atlas Pattern 2 vs Pattern 5)")
    plt.ylabel("Maturity Ratio (Mature/Progenitor)")
    plt.xticks(rotation=45)
    
    out_path = Path("figures/system_maturity_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n  Figure saved to {out_path}")
    
    # Save results
    with open("figures/system_maturity_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
