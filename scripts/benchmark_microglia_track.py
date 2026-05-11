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
    
    # 1. Map to Pruning Regulon
    # Genes involved in microglial pruning and GABA sensing
    pruning_genes = ["GABBR1", "GABBR2", "TREM2", "TYROBP", "C1QA", "C3"]
    present = [g for g in pruning_genes if g in adata.var_names]
    
    sc.tl.score_genes(adata, present, score_name="pruning_score")
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=adata.obs, x="condition", y="pruning_score", palette="coolwarm")
    plt.title("Favuzzi 2021: Pruning Regulome in GABAB-R KO Microglia")
    plt.savefig(fig_dir / "microglia_favuzzi_pruning.png", dpi=200)
    
    # Compare KO vs WT
    ctrl = adata.obs[adata.obs['condition'] == 'WT']['pruning_score']
    test = adata.obs[adata.obs['condition'] == 'KO']['pruning_score']
    t_stat, p_val = stats.ttest_ind(test, ctrl)
    
    print(f"  Pruning Shift: {test.mean() - ctrl.mean():.4f}, p={p_val:.2e}")
    
    return {"favuzzi_pruning_shift": float(test.mean() - ctrl.mean()), "favuzzi_p": float(p_val)}

def benchmark_popova_modularity(micro_dir, fig_dir):
    print("\n--- Benchmarking Popova 2021 (Chimeric Modularity) ---")
    adata_path = micro_dir / "popova_2021_processed.h5ad"
    if not adata_path.exists(): return None
    
    adata = sc.read_h5ad(adata_path)
    sc.pp.log1p(adata)
    
    # We use Cluster Modularity (n_modules) as a proxy for regulatory stabilization
    # Samples: GSM5478754 (img), GSM5478755 (batch1), GSM5478756 (batch2)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.5)
    
    results = []
    for gsm in adata.obs['sample'].unique():
        sub = adata[adata.obs['sample'] == gsm]
        n_mod = len(sub.obs['leiden'].unique())
        results.append({"gsm": gsm, "n_modules": n_mod})
        print(f"  {gsm:<10}: {n_mod} modules")
        
    return {"popova_modularity": results}

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
    
    res_pop = benchmark_popova_modularity(micro_dir, fig_dir)
    if res_pop: results.update(res_pop)
    
    with open(fig_dir / "microglia_track_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
