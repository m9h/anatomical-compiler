import os
import sys
import json
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from hgx_prep import pca, grn

def main():
    data_dir = PROJECT_ROOT / "data/mouse"
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Loo 2019 Mouse metadata...")
    meta = pd.read_csv(data_dir / "E14_meta.txt.gz", sep="\t")
    meta.columns = ["cell", "cluster"]
    
    print("Loading counts...")
    df = pd.read_csv(data_dir / "E14_counts.txt.gz", sep="\t", index_col=0)
    print(f"  Shape: {df.shape}")
    
    # Fix cell names: counts have e14-WT10, meta has e14.WT10
    df.columns = [c.replace("-", ".") for c in df.columns]
    
    # Align
    shared_cells = sorted(set(df.columns) & set(meta["cell"]))
    print(f"  Shared cells: {len(shared_cells)}")
    
    df = df[shared_cells]
    meta = meta.set_index("cell").loc[shared_cells]
    
    # Create AnnData
    adata = ad.AnnData(X=df.values.T.astype(np.float32), obs=meta, var=pd.DataFrame(index=df.index))
    
    # 1. HVGs and Normalization
    print("Preprocessing with scanpy...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    
    # Subset to HVGs
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    gene_names = adata_hvg.var_names.tolist()
    X = adata_hvg.X.T # (genes, cells)
    
    # 2. PCA Features
    print("Computing PCA features...")
    pca_res = pca.compute_pca(X, dim=64, dim_method="fixed")
    
    # 3. Correlation-based GRN
    print("Building correlation-based GRN...")
    mouse_tfs = ["Pax6", "Eomes", "Tbr1", "Neurod2", "Sox2", "Hes1", "Dlx1", "Dlx2"]
    available_tfs = [tf for tf in mouse_tfs if tf in gene_names]
    print(f"  TFs found in HVGs: {available_tfs}")
    
    # Re-extract full TF expression if not in HVGs (though they should be)
    tf_expr_full = []
    for tf in available_tfs:
        tf_expr_full.append(adata_hvg[:, tf].X.flatten())
    tf_expr_full = np.array(tf_expr_full)
    
    # Standardize for correlation
    X_std = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    TF_std = (tf_expr_full - tf_expr_full.mean(axis=1, keepdims=True)) / (tf_expr_full.std(axis=1, keepdims=True) + 1e-8)
    
    # Correlation matrix (genes, K)
    corr_mat = (X_std @ TF_std.T) / X.shape[1]
    incidence = (np.abs(corr_mat) > 0.3).astype(np.float32)
    
    # 4. Save outputs
    print("Saving processed arrays...")
    np.save(out_dir / "incidence.npy", incidence)
    np.save(out_dir / "node_features_pca.npy", pca_res.features)
    
    with open(out_dir / "gene_names.json", "w") as f:
        json.dump(gene_names, f)
    with open(out_dir / "tf_names.json", "w") as f:
        json.dump(available_tfs, f)
    
    with open(out_dir / "gene_names.json", "w") as f:
        json.dump(gene_names, f)
    with open(out_dir / "tf_names.json", "w") as f:
        json.dump(available_tfs, f)
        
    print(f"Done! Results in {out_dir}")

if __name__ == "__main__":
    main()
