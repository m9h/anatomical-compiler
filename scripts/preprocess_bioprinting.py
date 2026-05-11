import argparse
import os
from pathlib import Path
import json

import pandas as pd
import scanpy as sc
import numpy as np

def preprocess_vassal(input_dir, output_path):
    print(f"Preprocessing Vassal 2024 from {input_dir}")
    files = list(input_dir.glob("GSM*matrix.mtx.gz"))
    adatas = []
    for f in files:
        gsm = f.name.split("_")[0]
        sample_name = f.name.split("_")[1]
        print(f"  Reading {sample_name}...")
        feat_list = list(input_dir.glob(f"{gsm}_*features.tsv.gz")) + list(input_dir.glob(f"{gsm}_*genes.tsv.gz"))
        bc_list = list(input_dir.glob(f"{gsm}_*barcodes.tsv.gz"))
        if not feat_list or not bc_list: continue
        adata = sc.read_mtx(f).T
        genes = pd.read_csv(feat_list[0], sep="\t", header=None)
        barcodes = pd.read_csv(bc_list[0], sep="\t", header=None)
        adata.var_names = genes[1].values if genes.shape[1] > 1 else genes[0].values
        adata.obs_names = barcodes[0].values
        adata.obs['sample'] = sample_name
        adata.obs['study'] = "Vassal2024"
        adata.var_names_make_unique()
        adatas.append(adata)
    if not adatas: return
    adata_full = sc.concat(adatas, index_unique="_")
    adata_full.write(output_path)
    print(f"  Saved to {output_path}")

def preprocess_shin(input_dir, output_path):
    print(f"Preprocessing Tung/Shin 2024 from {input_dir}")
    counts_path = input_dir / "GSE232164_counts.csv.gz"
    meta_path = input_dir / "GSE232164_meta.csv.gz"
    if not counts_path.exists(): return
    
    # Load counts
    df = pd.read_csv(counts_path, index_col=0)
    gene_symbols = df['gene_name'].values
    df = df.drop(columns=['gene_name'])
    
    # Load meta
    meta = pd.read_csv(meta_path, index_col=0)
    
    # Standardize names to match meta: ['BMP4-1', 'BMP4-TGFB2-1', 'NT-1', 'TGFB2-1']
    new_cols = []
    for col in df.columns:
        # bmp4_rep1, bmp4_tgfb2_rep1, nontreated_rep1, tgfb2_rep1
        rep = col.split("_")[-1].replace("rep", "")
        if "nontreated" in col:
            name = "NT"
        elif "bmp4_tgfb2" in col:
            name = "BMP4-TGFB2"
        elif col.startswith("bmp4"):
            name = "BMP4"
        elif col.startswith("tgfb2"):
            name = "TGFB2"
        else:
            name = col.split("_")[0].upper()
        new_cols.append(f"{name}-{rep}")
    
    df.columns = new_cols
    
    # Reorder df to match meta exactly
    df = df[meta.index]
    
    adata = sc.AnnData(df.T)
    adata.var_names = gene_symbols
    adata.obs = meta
    adata.obs['study'] = "TungShin2024"
    adata.var_names_make_unique()
    adata.write(output_path)
    print(f"  Saved to {output_path}")

def preprocess_micro_park(input_dir, output_path):
    print(f"Preprocessing Park 2023 from {input_dir}")
    files = list(input_dir.glob("*_hi_rc.txt.gz"))
    adatas = []
    for f in files:
        sample_name = f.name.split("_")[1]
        print(f"  Reading {sample_name}...")
        df = pd.read_csv(f, sep="\t", index_col=0)
        adata = sc.AnnData(df.T)
        adata.obs['sample'] = sample_name
        adata.obs['condition'] = "Co-culture" if "coculture" in sample_name else "Organoid" if "organoid" in sample_name else "iMac"
        adatas.append(adata)
    if not adatas: return
    adata_full = sc.concat(adatas, index_unique="_")
    adata_full.write(output_path)
    print(f"  Saved to {output_path}")

def preprocess_micro_favuzzi(input_dir, output_path):
    print(f"Preprocessing Favuzzi 2021 from {input_dir}")
    files = list(input_dir.glob("*.h5"))
    adatas = []
    for f in files:
        sample_name = f.stem.split("_", 1)[1]
        print(f"  Reading {sample_name}...")
        adata = sc.read_10x_h5(f)
        adata.obs['sample'] = sample_name
        adata.obs['condition'] = "WT" if "WT" in sample_name else "KO" if "KO" in sample_name else "CTR"
        adata.var_names_make_unique()
        adatas.append(adata)
    if not adatas: return
    adata_full = sc.concat(adatas, index_unique="_")
    adata_full.write(output_path)
    print(f"  Saved to {output_path}")

def preprocess_micro_popova(input_dir, output_path):
    print(f"Preprocessing Popova 2021 from {input_dir}")
    adatas = []
    files = list(input_dir.glob("GSM*"))
    gsm_ids = sorted(set([f.name.split("_")[0] for f in files]))
    for gsm in gsm_ids:
        mtx_list = list(input_dir.glob(f"{gsm}_*matrix.mtx.gz"))
        tsv_list = list(input_dir.glob(f"{gsm}_*matrix.tsv.gz"))
        bc_list = list(input_dir.glob(f"{gsm}_*barcodes.tsv.gz"))
        feat_list = list(input_dir.glob(f"{gsm}_*features.tsv.gz"))
        if not bc_list or not feat_list: continue
        if mtx_list:
            adata = sc.read_mtx(mtx_list[0]).T
        elif tsv_list:
            import gzip
            with gzip.open(tsv_list[0], 'rt') as f: first_line = f.readline()
            if "MatrixMarket" in first_line: adata = sc.read_mtx(tsv_list[0]).T
            else:
                df = pd.read_csv(tsv_list[0], sep="\t", index_col=0)
                adata = sc.AnnData(df.T)
        else: continue
        genes = pd.read_csv(feat_list[0], sep="\t", header=None)
        barcodes = pd.read_csv(bc_list[0], sep="\t", header=None)
        adata.var_names = genes[1].values if genes.shape[1] > 1 else genes[0].values
        adata.obs_names = barcodes[0].values
        adata.obs['sample'] = gsm
        adata.var_names_make_unique()
        adatas.append(adata)
    if not adatas: return
    adata_full = sc.concat(adatas, index_unique="_")
    adata_full.write(output_path)
    print(f"  Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/bioprinting")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    micro_dir = data_dir / "microglia"

    if (data_dir / "vassal_2024").exists():
        preprocess_vassal(data_dir / "vassal_2024", data_dir / "vassal_2024_processed.h5ad")
    if (data_dir / "shin_2024").exists():
        preprocess_shin(data_dir / "shin_2024", data_dir / "shin_2024_processed.h5ad")
    if (micro_dir / "park_2023").exists():
        preprocess_micro_park(micro_dir / "park_2023", micro_dir / "park_2023_processed.h5ad")
    if (micro_dir / "favuzzi_2021").exists():
        preprocess_micro_favuzzi(micro_dir / "favuzzi_2021", micro_dir / "favuzzi_2021_processed.h5ad")
    if (micro_dir / "popova_2021").exists():
        preprocess_micro_popova(micro_dir / "popova_2021", micro_dir / "popova_2021_processed.h5ad")

if __name__ == "__main__":
    main()
