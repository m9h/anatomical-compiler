import argparse
import os
from pathlib import Path

import pandas as pd
import scanpy as sc
import numpy as np

def preprocess_tang(input_path, output_path):
    print(f"Preprocessing Tang et al. (2020) from {input_path}")
    df = pd.read_excel(input_path, index_col=0)
    adata = sc.AnnData(df.T)
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
    adata.write(output_path)
    print(f"  Saved to {output_path}")

def preprocess_lawlor(input_dir, output_path):
    print(f"Preprocessing Lawlor et al. (2021) from {input_dir}")
    prefix = "GSM4120180_"
    adata = sc.read_mtx(input_dir / f"{prefix}matrix.mtx.gz").T
    genes = pd.read_csv(input_dir / f"{prefix}genes.tsv.gz", sep="\t", header=None)
    barcodes = pd.read_csv(input_dir / f"{prefix}barcodes.tsv.gz", sep="\t", header=None)
    adata.var_names = genes[1].values
    adata.var['gene_ids'] = genes[0].values
    adata.obs_names = barcodes[0].values
    adata.var_names_make_unique()
    adata.obs['study'] = "Lawlor2021"
    adata.obs['system'] = "Kidney Organoid"
    adata.obs['culture'] = "Bioprinted"
    print(f"  Shape: {adata.shape}")
    adata.write(output_path)
    print(f"  Saved to {output_path}")

def preprocess_zhang(input_path, output_path):
    print(f"Preprocessing Zhang et al. (2025) from {input_path}")
    df = pd.read_csv(input_path, sep="\t")
    df = df.rename(columns={"#ID": "gene_symbol"}).set_index("gene_symbol")
    adata = sc.AnnData(df.T)
    metadata = []
    for sample in adata.obs_names:
        culture = "3D_HHO" if "HHO" in sample else "2D_hiHeps"
        metadata.append({
            "sample_id": sample,
            "culture": culture,
            "is_3d": "3D" in culture,
            "system": "Liver Hepatorganoid"
        })
    adata.obs = pd.DataFrame(metadata, index=adata.obs_names)
    print(f"  Shape: {adata.shape}")
    adata.write(output_path)
    print(f"  Saved to {output_path}")

def preprocess_yan(input_dir, output_path):
    print(f"Preprocessing Yan et al. (2024) from {input_dir}")
    mtx_path = input_dir / "GSE234774_rnaseq_filtered_scRNA.mtx.gz"
    if not mtx_path.exists():
        print(f"  MTX file not found: {mtx_path}")
        return
    adata = sc.read_mtx(mtx_path).T
    genes = pd.read_csv(input_dir / "GSE234774_rnaseq_features.txt.gz", sep="\t", header=None)
    barcodes = pd.read_csv(input_dir / "GSE234774_rnaseq_barcodes.txt.gz", sep="\t", header=None)
    meta = pd.read_csv(input_dir / "GSE234774_rnaseq_meta.txt.gz", sep="\t", index_col=0)
    adata.var_names = genes[0].values
    adata.obs_names = barcodes[0].values
    adata.obs = meta
    adata.var_names_make_unique()
    print(f"  Shape: {adata.shape}")
    adata.write(output_path)
    print(f"  Saved to {output_path}")

def preprocess_anthro(input_dir, output_path):
    print(f"Preprocessing Anthrobots (2023) from {input_dir}")
    adatas = []
    for sample in ["S1", "S2"]:
        gsm = "GSM7950242" if sample == "S1" else "GSM7950243"
        mtx = input_dir / f"{gsm}_matrix_{sample}.mtx.gz"
        feat = input_dir / f"{gsm}_features_{sample}.tsv.gz"
        bc = input_dir / f"{gsm}_barcodes_{sample}.tsv.gz"
        if not mtx.exists(): continue
        print(f"  Reading {sample}...")
        adata = sc.read_mtx(mtx).T
        genes = pd.read_csv(feat, sep="\t", header=None)
        barcodes = pd.read_csv(bc, sep="\t", header=None)
        adata.var_names = genes[1].values
        adata.obs_names = barcodes[0].values
        adata.obs['sample'] = sample
        adata.obs['study'] = "Gumuskaya2023"
        adata.obs['system'] = "Anthrobot"
        adata.var_names_make_unique()
        adatas.append(adata)
    if not adatas:
        print("  No Anthrobot files found.")
        return
    print("  Merging samples...")
    adata_full = sc.concat(adatas, index_unique="_")
    print(f"  Shape: {adata_full.shape}")
    adata_full.write(output_path)
    print(f"  Saved to {output_path}")

def preprocess_kidney_ref(input_path, output_path):
    print(f"Preprocessing Human Fetal Kidney reference from {input_path}")
    df = pd.read_csv(input_path, sep="\t", index_col=0)
    adata = sc.AnnData(df.T)
    adata.obs['study'] = "Combes2019"
    adata.obs['system'] = "Human Fetal Kidney"
    adata.obs['stage'] = "16w"
    print(f"  Shape: {adata.shape}")
    adata.write(output_path)
    print(f"  Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess bioprinting datasets into AnnData")
    parser.add_argument("--data-dir", type=str, default="data/bioprinting", help="Input data directory")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    tang_in = data_dir / "tang_2020/GSE147147_processed.xlsx"
    if tang_in.exists(): preprocess_tang(tang_in, data_dir / "tang_2020_processed.h5ad")
    
    lawlor_dir = data_dir / "lawlor_2021"
    if lawlor_dir.exists(): preprocess_lawlor(lawlor_dir, data_dir / "lawlor_2021_processed.h5ad")

    zhang_in = data_dir / "zhang_2025/GSE298708_All.DEG_final.txt.gz"
    if zhang_in.exists(): preprocess_zhang(zhang_in, data_dir / "zhang_2025_processed.h5ad")

    yan_dir = data_dir / "yan_2024"
    if yan_dir.exists(): preprocess_yan(yan_dir, data_dir / "yan_2024_processed.h5ad")

    anthro_dir = data_dir / "anthro_2023"
    if anthro_dir.exists(): preprocess_anthro(anthro_dir, data_dir / "anthro_2023_processed.h5ad")

    kidney_ref = data_dir / "kidney_ref/GSM2741551_counts.tsv.gz"
    if kidney_ref.exists(): preprocess_kidney_ref(kidney_ref, data_dir / "kidney_ref_processed.h5ad")

if __name__ == "__main__":
    main()
