import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

def main():
    print("Loading Fleck Brain GRN...")
    data_dir = Path("data/processed")
    with open(data_dir / "gene_names.json") as f: f_genes = json.load(f)
    with open(data_dir / "tf_names.json") as f: f_tfs = json.load(f)
    f_incidence = np.load(data_dir / "incidence.npy") # (n_genes, n_tfs)

    print("Loading Tang et al. Bioprinting data...")
    adata = sc.read_h5ad("data/bioprinting/tang_2020_processed.h5ad")
    
    # 1. Focus on GSC-CW468 cells
    gsc = adata[adata.obs['cell_type'] == 'GSC'].copy()
    
    # Split 2D (Sphere) vs 3D (Tetraculture)
    # Culture types: ['Sphere' 'Tetraculture' '2D' '3D']
    gsc_2d = gsc[gsc.obs['culture'] == 'Sphere'].X.mean(axis=0)
    gsc_3d = gsc[gsc.obs['culture'] == 'Tetraculture'].X.mean(axis=0)
    
    # log2FC (3D vs 2D)
    # Data is already log-transformed in the XLSX (values like 5.39)
    log2fc = gsc_3d - gsc_2d
    
    # 2. Map shared genes
    t_genes = adata.var_names.tolist()
    shared_genes = sorted(set(f_genes) & set(t_genes))
    print(f"Shared genes: {len(shared_genes)}")
    
    f_idx = [f_genes.index(g) for g in shared_genes]
    t_idx = [t_genes.index(g) for g in shared_genes]
    
    log2fc_shared = log2fc[t_idx]
    
    # 3. Validation: Does Fleck GRN predict 3D-induced genes?
    # We'll check the top TFs from our Pollen analysis
    top_tfs = ["NR2E1", "ARX", "MEIS2", "SOX2", "ASCL1"]
    
    results = []
    for tf in top_tfs:
        if tf not in f_tfs: continue
        
        fi = f_tfs.index(tf)
        # Fleck regulon (incidence)
        f_reg = f_incidence[f_idx, fi]
        
        # Check induction in 3D bioprinted GSCs
        # We define "induced" as log2FC > 0.5
        is_induced = (log2fc_shared > 0.5).astype(float)
        
        # Overlap analysis
        n_both = (f_reg * is_induced).sum()
        n_fleck = f_reg.sum()
        n_induced = is_induced.sum()
        n_total = len(shared_genes)
        
        # Fisher test
        table = [[n_both, n_fleck - n_both], [n_induced - n_both, n_total - n_fleck - n_induced + n_both]]
        _, pval = stats.fisher_exact(table, alternative="greater")
        
        jaccard = n_both / (n_fleck + n_induced - n_both) if (n_fleck + n_induced - n_both) > 0 else 0
        
        results.append({"tf": tf, "jaccard": jaccard, "p": pval, "n_both": int(n_both)})
        print(f"  {tf:<10}: Jaccard {jaccard:>6.4f}, p {pval:.2e}, both {n_both:>3.0f}")

    # 4. Global Direction Concordance
    # For all genes in the intersection of Fleck regulon and Tang DE
    # check if the sign matches (assuming Fleck weights were positive for these targets)
    # Actually, Pando gives signed coefficients. We should use those.
    # For now, let's just use the binary incidence.
    
    # 5. Plot
    plt.figure(figsize=(10, 5))
    tfs = [r["tf"] for r in results]
    jaccards = [r["jaccard"] for r in results]
    plt.bar(tfs, jaccards, color="firebrick")
    plt.ylabel("Jaccard Index (Regulon vs 3D-Induced)")
    plt.title("Bioprinting Fidelity: Organoid GRN predicts 3D-induced genes in Glioblastoma")
    
    for i, r in enumerate(results):
        plt.text(i, jaccards[i] + 0.005, f"p={r['p']:.1e}", ha="center")
        
    out_path = Path("figures/bioprinting_tang_validation.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
