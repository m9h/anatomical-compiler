"""Preprocess synthetic-agency datasets into AnnData."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def preprocess_hrvatin(input_dir, output_path):
    print(f"Preprocessing Hrvatin et al. (2018) from {input_dir}")
    counts_path = input_dir / "GSE102827_merged_all_raw.csv.gz"
    types_path = input_dir / "GSE102827_cell_type_assignments.csv.gz"
    if not counts_path.exists():
        print(f"  Error: {counts_path} not found.")
        return

    print("  Reading counts CSV (this is large — ~1 GB unzipped)...")
    df = pd.read_csv(counts_path, index_col=0)
    adata = sc.AnnData(df.T.astype(np.float32))

    # Cell IDs are bare barcodes like 'x2_35_0_bc0013'; metadata (stim, sample,
    # maintype, celltype, subtype) lives in the separate cell-type-assignments file.
    if types_path.exists():
        types = pd.read_csv(types_path, index_col=0)
        for col in ("stim", "sample", "maintype", "celltype", "subtype"):
            if col in types.columns:
                adata.obs[col] = types[col].reindex(adata.obs_names).values
        adata.obs["stim_duration"] = adata.obs.get("stim", pd.Series(index=adata.obs.index)).astype(str)
        adata.obs["is_stimulated"] = adata.obs["stim_duration"].isin(["1h", "4h"])
    else:
        adata.obs["stim_duration"] = "unknown"
        adata.obs["is_stimulated"] = False

    adata.obs["study"] = "Hrvatin2018"
    adata.obs["system"] = "Mouse_V1"
    adata.obs["track"] = "synthetic_agency"
    adata.var_names_make_unique()
    print(f"  Shape: {adata.shape}")
    print(f"  Stim conditions: {adata.obs['stim_duration'].value_counts().to_dict()}")
    adata.write(output_path)
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/agency")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    hrvatin_dir = data_dir / "hrvatin_2018"
    if hrvatin_dir.exists():
        preprocess_hrvatin(hrvatin_dir, data_dir / "hrvatin_2018_processed.h5ad")
    else:
        print(f"  No agency datasets found in {data_dir}")


if __name__ == "__main__":
    main()
