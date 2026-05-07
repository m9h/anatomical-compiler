import os
import sys
import json
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import hgx
    import devograph
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

def load_fleck_data():
    fleck_dir = PROJECT_ROOT / "data/processed"
    d = {}
    for fname in ["gene_names", "tf_names"]:
        with open(fleck_dir / f"{fname}.json") as f:
            d[fname] = json.load(f)
    for fname in ["perturbation_effects", "incidence", "node_features_pca"]:
        d[fname] = np.load(fleck_dir / f"{fname}.npy")
    return d

def compute_filtered_de(adata, cell_type, min_cells=20):
    print(f"\nComputing DE for {cell_type}...")
    # Find indices for this cell type
    ct_mask = adata.obs["sctab_cell_type"] == cell_type
    if ct_mask.sum() == 0:
        print(f"  No cells found for {cell_type}")
        return None
        
    subset = adata[ct_mask].to_memory()
    
    # Filter for single targets and NT
    nt_cells = subset[subset.obs["Gene_target_single"] == "non-targeting"]
    if len(nt_cells) < min_cells:
        print(f"  Not enough control cells ({len(nt_cells)})")
        return None
        
    nt_mean = np.array(nt_cells.X.mean(axis=0)).flatten()
    
    tfs = [tf for tf in subset.obs["Gene_target_single"].unique() if tf not in ["non-targeting", "NA", "nan"]]
    
    de_results = {}
    for tf in tfs:
        tf_cells = subset[subset.obs["Gene_target_single"] == tf]
        if len(tf_cells) >= min_cells:
            tf_mean = np.array(tf_cells.X.mean(axis=0)).flatten()
            # Log2FC (assuming log1p data, this is mean difference)
            log2fc = tf_mean - nt_mean
            de_results[tf] = log2fc
            
    return de_results

def main():
    fleck = load_fleck_data()
    fleck_genes = fleck["gene_names"]
    fleck_key_tfs = ["GLI3", "FOXG1", "TBR1", "DLX1", "DLX2", "EMX1", "EOMES", "NEUROD6"]
    all_fleck_tfs = fleck["tf_names"]
    
    print(f"Loading annotated Pollen data...")
    adata = sc.read_h5ad(PROJECT_ROOT / "data/pollen/screen_annotated.h5ad", backed='r')
    
    pollen_genes = adata.var_names.tolist()
    shared_genes = sorted(set(fleck_genes) & set(pollen_genes))
    f_idx = [fleck_genes.index(g) for g in shared_genes]
    p_idx = [pollen_genes.index(g) for g in shared_genes]
    
    target_cell_types = ["oligodendrocyte precursor cell", "neuron", "glutamatergic neuron"]
    
    all_results = {}
    for ct in target_cell_types:
        pollen_de = compute_filtered_de(adata, ct)
        if not pollen_de: continue
        
        results = []
        for tf in all_fleck_tfs:
            if tf in pollen_de:
                p_eff = pollen_de[tf][p_idx]
                if tf in fleck_key_tfs:
                    f_eff = fleck["perturbation_effects"][fleck_key_tfs.index(tf), f_idx]
                else:
                    t_idx = all_fleck_tfs.index(tf)
                    f_eff = fleck["incidence"][f_idx, t_idx]
                
                r, _ = stats.pearsonr(f_eff, p_eff)
                mask = (f_eff != 0) & (p_eff != 0)
                concord = float((np.sign(f_eff[mask]) == np.sign(p_eff[mask])).mean()) if mask.any() else 0.0
                
                results.append({"tf": tf, "r": float(r), "concord": concord})
        
        all_results[ct] = results
        mean_concord = np.mean([r["concord"] for r in results])
        print(f"  Mean concordance for {ct}: {mean_concord:.1%}")

    # Plot multi-panel
    fig, axes = plt.subplots(1, len(all_results), figsize=(15, 4), sharey=True)
    for i, (ct, res) in enumerate(all_results.items()):
        # Sort by concordance
        res_sorted = sorted(res, key=lambda x: x["concord"], reverse=True)
        top_tfs = [r["tf"] for r in res_sorted[:10]]
        top_concords = [r["concord"] for r in res_sorted[:10]]
        
        axes[i].bar(top_tfs, top_concords, color="skyblue")
        axes[i].axhline(y=0.5, color="gray", linestyle="--")
        axes[i].set_title(f"{ct.split()[0]} (Mean: {np.mean([r['concord'] for r in res]):.1%})")
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "figures/pollen_filtered_transfer_multi.png")
    print(f"\nMulti-panel figure saved to figures/pollen_filtered_transfer_multi.png")
    
    # Save JSON
    with open(PROJECT_ROOT / "figures/pollen_filtered_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
