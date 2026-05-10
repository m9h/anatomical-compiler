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
    print("Loading Sonthalia 2026 p40 loadings...")
    loadings = pd.read_excel("data/Sonthalia_Table3.xlsx", sheet_name="GeneLoadings")
    
    # Target Patterns:
    # p30: Synapse (ASD-enriched)
    # p27: Outer Radial Glia (oRG)
    # p2: Mature Neurons (Mammalian Shared)
    pattern_map = {
        "p2.Shared.Mammal_subcomp.2": "Mature Neuron",
        "p27.Shared.Mammal_subcomp.27": "Outer Radial Glia",
        "p30.Shared.Mammal_subcomp.30": "Synaptic (ASD)"
    }
    
    loadings_sub = loadings[["GeneSymbol"] + list(pattern_map.keys())].set_index("GeneSymbol")
    
    datasets = {
        "Organoid (Fleck)": "data/zenodo/RNA_all_velo.h5ad",
        "Bioprinted Brain (Tang)": "data/bioprinting/tang_2020_processed.h5ad",
        "Bioprinted Liver (Zhang)": "data/bioprinting/zhang_2025_processed.h5ad"
    }
    
    results = []
    
    for name, path in datasets.items():
        print(f"Processing {name}...")
        if not Path(path).exists():
            continue
            
        adata = sc.read_h5ad(path)
        if name == "Organoid (Fleck)":
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
        
        for i, (p_id, p_name) in enumerate(pattern_map.items()):
            results.append({
                "dataset": name,
                "pattern": p_name,
                "score": float(mean_scores[i])
            })

    df = pd.DataFrame(results)
    
    # Normalize by Organoid score for each pattern to see relative improvement
    org_scores = df[df.dataset == "Organoid (Fleck)"].set_index("pattern")["score"]
    df['relative_fidelity'] = df.apply(lambda row: row['score'] / org_scores[row['pattern']], axis=1)
    
    # 2. Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="pattern", y="relative_fidelity", hue="dataset")
    plt.axhline(1.0, color="red", linestyle="--", label="Organoid Baseline")
    plt.title("Advanced Fidelity Analysis: Bioprinting vs. Self-Organized Organoids\n(Relative to Sonthalia 2026 Primary Brain Patterns)")
    plt.ylabel("Relative Activity (1.0 = Organoid)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    out_path = Path("figures/advanced_fidelity_bioprinting.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n  Figure saved to {out_path}")
    
    # Save summary
    with open("figures/advanced_fidelity_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
