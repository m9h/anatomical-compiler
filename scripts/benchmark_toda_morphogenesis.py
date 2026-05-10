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
    print("Loading Toda 2020 Synthetic Morphogen data...")
    adata = sc.read_h5ad("data/bioprinting/toda_2020_processed.h5ad")
    
    # 1. Map to primary patterns (Sonthalia 2026)
    print("Loading Neocortex Atlas patterns...")
    loadings = pd.read_csv("data/SonthaliaEtAl_SupplTable1_p7CtxDev_GeneLoadings.txt", sep="\t")
    
    # Standardize to uppercase for matching
    adata.var.index = adata.var.GeneSymbol_orig.astype(str).str.upper()
    loadings.GeneSymbol = loadings.GeneSymbol.astype(str).str.upper()
    
    shared = sorted(set(adata.var.index) & set(loadings.GeneSymbol))
    print(f"Shared genes: {len(shared)}")
    
    # 2. Score patterns in synthetic data
    pattern_cols = ["p1.Shared.Mammal_subcomp.1", "p3.Shared.Mammal_subcomp.3"]
    loadings_sub = loadings[loadings.GeneSymbol.isin(shared)].set_index("GeneSymbol")[pattern_cols]
    
    # Ensure order matches
    X = adata[:, shared].X
    W = loadings_sub.loc[shared].values
    
    scores = X @ W

    # Plot Logic: Growth vs Arrest in synthetic circuits
    plt.figure(figsize=(10, 6))
    plt.scatter(scores[:, 0], scores[:, 1], c="purple", alpha=0.6)
    plt.xlabel("Pattern 1 (Growth) Score")
    plt.ylabel("Pattern 3 (Arrest) Score")
    plt.title("Synthetic Morphology Logic: Programming Growth vs Arrest\n(Toda 2020 SynNotch circuits projected into Neocortex Atlas)")
    
    out_path = Path("figures/toda_morphogenesis_logic.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Figure saved to {out_path}")
    
    # Save results
    with open("figures/toda_results.json", "w") as f:
        json.dump({"n_shared": len(shared), "scores": scores.tolist()}, f, indent=2)

if __name__ == "__main__":
    main()
