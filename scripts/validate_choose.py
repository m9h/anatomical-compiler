import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import hgx

def load_dataset(data_dir: Path):
    d = {}
    with open(data_dir / "gene_names.json") as f: d["gene_names"] = json.load(f)
    with open(data_dir / "tf_names.json") as f: d["tf_names"] = json.load(f)
    d["incidence"] = np.load(data_dir / "incidence.npy")
    d["node_features_pca"] = np.load(data_dir / "node_features_pca.npy")
    d["perturbation_effects"] = np.load(data_dir / "perturbation_effects.npy")
    return d

def main():
    fleck_dir = PROJECT_ROOT / "data/processed"
    choose_dir = PROJECT_ROOT / "data/choose/processed_real"
    
    with open(fleck_dir / "gene_names.json") as f: fleck_genes = json.load(f)
    with open(fleck_dir / "tf_names.json") as f: fleck_tfs = json.load(f)
    fleck_inc = np.load(fleck_dir / "incidence.npy")
    
    choose = load_dataset(choose_dir)
    choose_genes = choose["gene_names"]
    choose_tfs = choose["tf_names"]
    
    print(f"Fleck: {len(fleck_genes)} genes, {len(fleck_tfs)} TFs")
    print(f"CHOOSE: {len(choose_genes)} genes, {len(choose_tfs)} TFs")
    
    # 1. Map shared genes
    f_genes_u = [g.upper() for g in fleck_genes]
    c_genes_u = [g.upper() for g in choose_genes]
    shared_genes = sorted(set(f_genes_u) & set(c_genes_u))
    print(f"Shared genes: {len(shared_genes)}")
    
    f_idx = [f_genes_u.index(g) for g in shared_genes]
    c_idx = [c_genes_u.index(g) for g in shared_genes]
    
    # 2. Benchmark: Fleck Human Prediction -> CHOOSE Real Validation
    # We'll check if CHD4/ADNP targets in human match observed effects in organoids
    print("\nBenchmark: Fleck Prediction -> CHOOSE Real (CHD4/ADNP)")
    
    # Note: CHD4 is not a TF in Fleck, but CHD3 and CHD7 are. 
    # ADNP2 is in Fleck. Let's use ADNP2 as a proxy for ADNP.
    
    results = []
    for tf_sym in ["ADNP", "CHD3", "CHD7"]:
        if tf_sym in choose_tfs or (tf_sym == "ADNP" and "ADNP" in choose_tfs):
            c_tf = "ADNP" if tf_sym == "ADNP" else tf_sym
            # Fleck prediction (incidence)
            if tf_sym in fleck_tfs:
                fi = fleck_tfs.index(tf_sym)
                pred_f = fleck_inc[f_idx, fi]
            else:
                # Try ADNP2 as proxy
                if tf_sym == "ADNP" and "ADNP2" in fleck_tfs:
                    fi = fleck_tfs.index("ADNP2")
                    pred_f = fleck_inc[f_idx, fi]
                else:
                    continue
            
            # CHOOSE observation (DE)
            ci = choose_tfs.index("ADNP" if tf_sym == "ADNP" else tf_sym)
            obs_c = choose["perturbation_effects"][ci, c_idx]
            
            mask = (pred_f != 0)
            if mask.any():
                concord = (np.sign(obs_c[mask]) == -1).mean() # Assume incidence is 'activating'
                print(f"  {tf_sym:<10}: {mask.sum()} human targets, Concordance {concord:>6.1%} (downregulated in KO)")
                results.append({"tf": tf_sym, "concord": float(concord)})

    # 3. Plot
    if results:
        plt.figure(figsize=(8, 5))
        tfs = [r["tf"] for r in results]
        concords = [r["concord"] for r in results]
        plt.bar(tfs, concords, color="gold")
        plt.axhline(y=0.5, color="gray", linestyle="--")
        plt.ylabel("Target Downregulation in KO (%)")
        plt.title("Experimental Validation: Fleck prediction vs. CHOOSE CRISPR")
        plt.savefig(PROJECT_ROOT / "figures/choose_benchmark.png")
        print(f"Figure saved to figures/choose_benchmark.png")

if __name__ == "__main__":
    main()
