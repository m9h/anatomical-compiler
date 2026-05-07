import os
import sys
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from hgx_prep import pca, grn

def main():
    data_dir = PROJECT_ROOT / "data/choose"
    out_dir = data_dir / "processed_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading CHOOSE real h5ad...")
    adata = sc.read_h5ad(data_dir / "CHOOSE_filtered.h5ad")
    print(f"  Shape: {adata.shape}")
    
    # 1. Cleaning and Normalization
    print("Cleaning and preprocessing...")
    # Filter cells with zero counts
    sc.pp.filter_cells(adata, min_genes=100)
    # Check if data is sparse or dense
    if hasattr(adata.X, "toarray"):
        print("  Converting to sparse matrix for efficiency...")
        
    # Check if already log-transformed by looking at max value
    max_val = adata.X.max()
    if max_val > 50:
        print("  Normalizing and log-transforming...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print(f"  Data already normalized (max={max_val:.2f})")
    
    # Load Fleck genes to intersect
    with open(PROJECT_ROOT / "data/processed/gene_names.json") as f:
        fleck_genes = json.load(f)
    
    # Map CHOOSE mouse symbols to human UPPERCASE
    adata.var["human_symbol"] = adata.var_names.str.upper()
    
    shared_genes = [g for g in fleck_genes if g in adata.var["human_symbol"].values]
    print(f"  Shared genes with Fleck (Human mapped): {len(shared_genes)}")
    
    # Identify indices in adata.var
    var_map = pd.Series(adata.var_names, index=adata.var["human_symbol"]).to_dict()
    shared_mouse_names = [var_map[g] for g in shared_genes]
    
    adata_shared = adata[:, shared_mouse_names].copy()
    # Rename variables to human symbols for consistency
    adata_shared.var_names = shared_genes
    adata_shared.obs_names_make_unique()
    gene_names = adata_shared.var_names.tolist()
    
    # 2. Extract DE for CHD4 and ADNP
    print("Computing DE for CHD4 and ADNP...")
    results_de = {}
    for gene in ["Chd4", "Adnp"]:
        ko_cells = adata_shared[adata_shared.obs['batch'].str.contains(f"{gene}_cKO", case=False)]
        ctrl_cells = adata_shared[adata_shared.obs['batch'] == "E16.5_wt_dorsaltel_1"]
        
        if len(ko_cells) > 0 and len(ctrl_cells) > 0:
            ko_mean = np.array(ko_cells.X.mean(axis=0)).flatten()
            ctrl_mean = np.array(ctrl_cells.X.mean(axis=0)).flatten()
            log2fc = ko_mean - ctrl_mean
            results_de[gene.upper()] = log2fc
            print(f"  {gene}: {len(ko_cells)} KO cells vs {len(ctrl_cells)} control cells")

    # 3. PCA Features
    print("Computing PCA features...")
    # compute_pca expects (genes, cells)
    pca_res = pca.compute_pca(adata_shared.X.T, dim=64, dim_method="fixed")
    
    # 4. GRN Incidence
    print("Building incidence...")
    tfs = list(results_de.keys())
    incidence = np.zeros((len(gene_names), len(tfs)), dtype=np.float32)
    for i, tf in enumerate(tfs):
        # Sensitive threshold for chromatin remodelers
        incidence[:, i] = (np.abs(results_de[tf]) > 0.1).astype(float)
        
    # 5. Save
    print("Saving processed arrays...")
    np.save(out_dir / "incidence.npy", incidence)
    np.save(out_dir / "node_features_pca.npy", pca_res.features)
    np.save(out_dir / "perturbation_effects.npy", np.array(list(results_de.values()), dtype=np.float32))
    
    with open(out_dir / "gene_names.json", "w") as f:
        json.dump(gene_names, f)
    with open(out_dir / "tf_names.json", "w") as f:
        json.dump(tfs, f)
        
    print(f"Done! Results in {out_dir}")

if __name__ == "__main__":
    main()
