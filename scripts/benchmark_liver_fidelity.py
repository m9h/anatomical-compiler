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

def main():
    print("Loading Liver Bioprinting data...")
    adata = sc.read_h5ad("data/bioprinting/zhang_2025_processed.h5ad")
    
    # Pre-process: Log transform
    sc.pp.log1p(adata)
    
    # 1. Calculate log2FC (3D_HHO vs 2D_hiHeps)
    hho = adata[adata.obs['culture'] == '3D_HHO'].X.mean(axis=0)
    hiheps = adata[adata.obs['culture'] == '2D_hiHeps'].X.mean(axis=0)
    
    log2fc = hho - hiheps
    
    genes = adata.var_names.tolist()
    
    # 2. Define key Liver Regulons (Master Regulators)
    # Based on Zhang et al. 2025 and literature
    regulons = {
        "HNF4A": ["ALB", "APOA1", "APOB", "CYP3A4", "HNF1A", "F9", "F10", "SERPINC1"],
        "FOXA2": ["HNF4A", "AFP", "TTR", "ADIRF", "APOA2"],
        "HHEX": ["PROX1", "GATA4", "TBX3", "FOXA2"]
    }
    
    results = []
    print("Validating Liver Regulons...")
    for tf, targets in regulons.items():
        # Check induction of targets in HHO
        shared_targets = [t for t in targets if t in genes]
        if not shared_targets: continue
        
        t_idx = [genes.index(t) for t in shared_targets]
        target_fc = log2fc[t_idx]
        
        # Mean induction of regulon
        mean_fc = np.mean(target_fc)
        
        # Significance (vs all genes)
        _, pval = stats.ttest_1samp(target_fc, 0.5, alternative="greater") # Null: no induction (>0.5)
        # Or better: Wilcoxon rank-sum vs background
        all_other_fc = np.delete(log2fc, t_idx)
        _, p_mw = stats.mannwhitneyu(target_fc, all_other_fc, alternative="greater")
        
        results.append({
            "tf": tf,
            "mean_log2fc": float(mean_fc),
            "p_val": float(p_mw),
            "n_targets": len(shared_targets)
        })
        print(f"  {tf:<6}: Mean log2FC {mean_fc:>6.4f}, p {p_mw:.2e}, targets {len(shared_targets)}")

    # 3. Plot
    plt.figure(figsize=(10, 6))
    tfs = [r["tf"] for r in results]
    fcs = [r["mean_log2fc"] for r in results]
    plt.bar(tfs, fcs, color="forestgreen")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Mean log2FC (3D HHO vs 2D hiHeps)")
    plt.title("Liver Fidelity: Master Regulator Induction in 3D Bioprinted Tissue")
    
    for i, r in enumerate(results):
        plt.text(i, fcs[i] + 0.05, f"p={r['p_val']:.1e}", ha="center")
        
    out_path = Path("figures/liver_fidelity_zhang.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    
    # Save results
    with open("figures/liver_fidelity_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
