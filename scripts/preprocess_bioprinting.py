import argparse
import os
from pathlib import Path

import pandas as pd
import scanpy as sc
import numpy as np

def preprocess_tang(input_path, output_path):
    print(f"Preprocessing Tang et al. (2020) from {input_path}")
    # Load XLSX (Gene x Sample)
    df = pd.read_excel(input_path, index_col=0)
    
    # Create AnnData (Sample x Gene)
    adata = sc.AnnData(df.T)
    
    # Extract metadata from column names
    # Examples: GSC-CW468-Sphere-1, GSC-CW468-Tetraculture-1, Macrophage-THP1-2D-1
    metadata = []
    for col in df.columns:
        parts = col.split("-")
        cell_type = parts[0]
        line = parts[1]
        culture = parts[2]
        rep = parts[3]
        metadata.append({
            "sample_id": col,
            "cell_type": cell_type,
            "line": line,
            "culture": culture,
            "replicate": rep,
            "is_3d": culture in ["Tetraculture", "3D"]
        })
    
    adata.obs = pd.DataFrame(metadata, index=adata.obs_names)
    
    print(f"  Shape: {adata.shape}")
    print(f"  Culture types: {adata.obs['culture'].unique()}")
    
    adata.write(output_path)
    print(f"  Saved to {output_path}")

def preprocess_lawlor(input_dir, output_path):
    print(f"Preprocessing Lawlor et al. (2021) from {input_dir}")
    # MTX files are named GSM4120180_...
    # scanpy read_10x_mtx expects matrix.mtx, genes.tsv, barcodes.tsv
    # So we'll symlink or rename them temporarily if needed, but better to use read_mtx
    
    prefix = "GSM4120180_"
    adata = sc.read_mtx(input_dir / f"{prefix}matrix.mtx.gz").T
    genes = pd.read_csv(input_dir / f"{prefix}genes.tsv.gz", sep="\t", header=None)
    barcodes = pd.read_csv(input_dir / f"{prefix}barcodes.tsv.gz", sep="\t", header=None)
    
    adata.var_names = genes[1].values # Gene symbols
    adata.var['gene_ids'] = genes[0].values
    adata.obs_names = barcodes[0].values
    
    adata.var_names_make_unique()
    
    # Basic metadata
    adata.obs['study'] = "Lawlor2021"
    adata.obs['system'] = "Kidney Organoid"
    adata.obs['culture'] = "Bioprinted"
    
    print(f"  Shape: {adata.shape}")
    adata.write(output_path)
    print(f"  Saved to {output_path}")

def preprocess_zhang(input_path, output_path):
    print(f"Preprocessing Zhang et al. (2025) from {input_path}")
    # Load DEG list (TSV)
    # Format: #ID HHO_1 HHO_2 HHO_3 hiHeps_1 hiHeps_2 hiHeps_3
    df = pd.read_csv(input_path, sep="\t")
    df = df.rename(columns={"#ID": "gene_symbol"})
    df = df.set_index("gene_symbol")
    
    # Create AnnData (Sample x Gene)
    adata = sc.AnnData(df.T)
    
    # Extract metadata from index
    metadata = []
    for sample in adata.obs_names:
        if "HHO" in sample:
            culture = "3D_HHO"
            is_3d = True
        else:
            culture = "2D_hiHeps"
            is_3d = False
        metadata.append({
            "sample_id": sample,
            "culture": culture,
            "is_3d": is_3d,
            "system": "Liver Hepatorganoid"
        })
    
    adata.obs = pd.DataFrame(metadata, index=adata.obs_names)
    
    print(f"  Shape: {adata.shape}")
    adata.write(output_path)
    print(f"  Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess bioprinting datasets into AnnData")
    parser.add_argument("--data-dir", type=str, default="data/bioprinting", help="Input data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    
    # 1. Tang
    tang_in = data_dir / "tang_2020/GSE147147_processed.xlsx"
    if tang_in.exists():
        preprocess_tang(tang_in, data_dir / "tang_2020_processed.h5ad")
        
    # 2. Lawlor
    lawlor_dir = data_dir / "lawlor_2021"
    if lawlor_dir.exists():
        preprocess_lawlor(lawlor_dir, data_dir / "lawlor_2021_processed.h5ad")

    # 3. Zhang
    zhang_in = data_dir / "zhang_2025/GSE298708_All.DEG_final.txt.gz"
    if zhang_in.exists():
        preprocess_zhang(zhang_in, data_dir / "zhang_2025_processed.h5ad")

if __name__ == "__main__":
    main()
