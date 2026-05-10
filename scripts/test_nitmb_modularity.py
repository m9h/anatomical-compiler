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
import jax.numpy as jnp
from scipy import stats

def calculate_modularity_identifiability(ev0):
    """
    Computes a 'Modularity Identifiability' score inspired by NITMB.
    Higher score = modules are more distinct and identifiable.
    Score = Mean(Spectral Gaps) / Variance(Eigenvalues)
    """
    if len(ev0) < 2: return 0
    gaps = np.diff(ev0)
    identifiability = np.mean(gaps[:10]) / (np.std(ev0[:10]) + 1e-8)
    return float(identifiability)

def main():
    print("NITMB Modularity Testing: Quantifying Identifiability across systems...")
    
    # Load previously computed results if available, otherwise we use the summary
    results_path = Path("figures/kidney_modularity_results.json")
    if not results_path.exists():
        print("  Error: Kidney modularity results not found. Run benchmark_kidney_modularity.py first.")
        return
        
    with open(results_path) as f:
        data = json.load(f)
        
    final_report = []
    
    # We analyze the L0 spectra from:
    # 1. Bioprinted Kidney (Constructed)
    # 2. Brain Organoid (Self-organized)
    # 3. Fetal Kidney (Blueprint)
    
    systems = {
        "Bioprinted Kidney": data["kidney"]["ev0"],
        "Brain Organoid": data["brain"]["ev0"] if data["brain"] else [],
        "Fetal Kidney (Ref)": data["kidney_ref"]["ev0"] if data["kidney_ref"] else []
    }
    
    for name, ev0 in systems.items():
        if not ev0: continue
        score = calculate_modularity_identifiability(np.array(ev0))
        final_report.append({"system": name, "identifiability_score": score})
        print(f"  {name:<20}: Identifiability Score = {score:.4f}")

    df = pd.DataFrame(final_report)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns_colors = ["teal", "firebrick", "forestgreen"]
    plt.bar(df["system"], df["identifiability_score"], color=sns_colors[:len(df)])
    plt.title("NITMB Metric: Module Identifiability\n(Quantifying how distinct regulatory modules are within the system)")
    plt.ylabel("Identifiability Score (Spectral Gap / Std)")
    
    out_path = Path("figures/nitmb_modularity_test.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n  Figure saved to {out_path}")
    
    # Save NITMB Report
    with open("figures/nitmb_modularity_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

if __name__ == "__main__":
    main()
