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

def benchmark_park_maturation(micro_dir, fig_dir):
    print("\n--- Benchmarking Park 2023 (Microglia Addition) ---")
    adata_path = micro_dir / "park_2023_processed.h5ad"
    if not adata_path.exists(): return None
    
    # Load backed to save memory
    adata = sc.read_h5ad(adata_path, backed='r')
    
    # We use Sonthalia Pattern 2 (Mature Neuron) to score maturation
    loadings = pd.read_csv("data/SonthaliaEtAl_SupplTable1_p7CtxDev_GeneLoadings.txt", sep="\t")
    p2_genes = loadings.nlargest(100, "p2.Shared.Mammal_subcomp.2").GeneSymbol.tolist()
    
    # Subsample cells for scoring (1000 per condition)
    conditions = adata.obs['condition'].unique()
    adatas_sub = []
    for cond in conditions:
        idx = np.where(adata.obs['condition'] == cond)[0]
        sub_idx = np.random.choice(idx, min(len(idx), 1000), replace=False)
        adatas_sub.append(adata[sub_idx].to_memory())
    
    adata_mem = sc.concat(adatas_sub)
    sc.pp.log1p(adata_mem)
    
    present = [g for g in p2_genes if g in adata_mem.var_names]
    sc.tl.score_genes(adata_mem, present, score_name="maturity_score")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=adata_mem.obs, x="condition", y="maturity_score", palette="viridis")
    plt.title("Park 2023: Impact of Microglia on Organoid Maturation\n(Sonthalia Pattern 2 Score)")
    plt.savefig(fig_dir / "microglia_park_maturation.png", dpi=200)
    
    # Statistics: Co-culture vs Organoid
    ctrl = adata_mem.obs[adata_mem.obs['condition'] == 'Organoid']['maturity_score']
    test = adata_mem.obs[adata_mem.obs['condition'] == 'Co-culture']['maturity_score']
    t_stat, p_val = stats.ttest_ind(test, ctrl)
    
    print(f"  Maturity Shift: {test.mean() - ctrl.mean():.4f}, p={p_val:.2e}")
    
    return {"park_maturity_shift": float(test.mean() - ctrl.mean()), "park_p": float(p_val)}

def benchmark_favuzzi_pruning(micro_dir, fig_dir):
    print("\n--- Benchmarking Favuzzi 2021 (Synaptic Pruning) ---")
    adata_path = micro_dir / "favuzzi_2021_processed.h5ad"
    if not adata_path.exists(): return None
    
    adata = sc.read_h5ad(adata_path)
    sc.pp.log1p(adata)
    
    # GABA receptors: GABBR1, GABBR2
    gaba_genes = ["GABBR1", "GABBR2", "GAD1", "GAD2"]
    present = [g for g in gaba_genes if g in adata.var_names]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=adata.obs, x="condition", y=adata[:, "GABBR1"].X.mean(axis=0).tolist()[0] if "GABBR1" in adata.var_names else 0)
    plt.title("Favuzzi 2021: GABA-R KO Validation")
    plt.savefig(fig_dir / "microglia_favuzzi_validation.png", dpi=200)
    
    return {"favuzzi_n_cells": adata.shape[0]}

def benchmark_popova_inflammation(micro_dir, fig_dir):
    print("\n--- Benchmarking Popova 2021 (LPS Activation) ---")
    # This is small, we can load it directly
    adata_path = micro_dir / "popova_2021_processed.h5ad"
    if not adata_path.exists(): return None
    
    adata = sc.read_h5ad(adata_path)
    sc.pp.log1p(adata)
    
    # Inflammatory markers: TNF, IL1B, CCL2
    markers = ["TNF", "IL1B", "CCL2", "CXCL8"]
    present = [g for g in markers if g in adata.var_names]
    
    # This is a cell x gene matrix where cells are samples
    # We don't have metadata for LPS +/- in the processed H5AD yet
    # but we can look at the variance
    print(f"  Top Variable Inflammatory Genes: {adata[:, present].X.var(axis=0)}")
    
    return {"popova_n_samples": adata.shape[0]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/bioprinting/microglia")
    parser.add_argument("--fig-dir", type=str, default="figures")
    args = parser.parse_args()
    
    micro_dir = Path(args.data_dir)
    fig_dir = Path(args.fig_dir)
    
    results = {}
    
    res_park = benchmark_park_maturation(micro_dir, fig_dir)
    if res_park: results.update(res_park)
    
    res_fav = benchmark_favuzzi_pruning(micro_dir, fig_dir)
    if res_fav: results.update(res_fav)
    
    res_pop = benchmark_popova_inflammation(micro_dir, fig_dir)
    if res_pop: results.update(res_pop)
    
    with open(fig_dir / "microglia_track_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
