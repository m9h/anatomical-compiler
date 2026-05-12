"""Preprocess Balzer 2022 (GSE180420) kidney IRI timecourse into AnnData.

The counts matrix ships as a gzipped R RDS file containing a sparse `dgCMatrix`,
which pyreadr cannot decode. We therefore call out to Rscript to export the
counts as Matrix Market (.mtx.gz) + barcodes.tsv + features.tsv (10X format),
then use scanpy.read_mtx like every other dataset in this repo.

Resulting AnnData:
  obs columns: pheno (Control, IRI_short_{1,3,14}, IRI_long_{1,3,14}),
               injury, post_injury_day, study, system, track
  113,579 cells, ~32k mouse genes
"""
import argparse
import gzip
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


RSCRIPT = r"""
suppressMessages({
  library(Matrix)
})
args <- commandArgs(trailingOnly = TRUE)
rds_path <- args[1]
out_dir  <- args[2]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
m <- readRDS(rds_path)
cat(sprintf("Class: %s\n", class(m)[[1]]))
cat(sprintf("Dim: %d x %d\n", nrow(m), ncol(m)))
writeMM(m, file.path(out_dir, "matrix.mtx"))
writeLines(rownames(m), file.path(out_dir, "features.tsv"))
writeLines(colnames(m), file.path(out_dir, "barcodes.tsv"))
"""


def export_rds_to_mtx(rds_gz_path, work_dir):
    """Decompress RDS and export to MatrixMarket via Rscript."""
    rds_plain = work_dir / "counts.rds"
    if not rds_plain.exists():
        print(f"  Decompressing {rds_gz_path} -> {rds_plain}")
        with gzip.open(rds_gz_path, "rb") as f_in, open(rds_plain, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    mtx_dir = work_dir / "10x"
    if (mtx_dir / "matrix.mtx").exists():
        print(f"  10X export already present at {mtx_dir}, reusing.")
        return mtx_dir

    script_path = work_dir / "export.R"
    script_path.write_text(RSCRIPT)
    print("  Running Rscript to export RDS -> MatrixMarket...")
    res = subprocess.run(
        ["Rscript", str(script_path), str(rds_plain), str(mtx_dir)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print("  Rscript failed:")
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError("Rscript export failed")
    print(f"  {res.stdout.strip()}")
    return mtx_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/bioprinting/balzer_2022")
    parser.add_argument("--out", default="data/bioprinting/balzer_2022_processed.h5ad")
    parser.add_argument("--work-dir", default="data/bioprinting/balzer_2022/_export",
                        help="Scratch dir for RDS decompression and MatrixMarket export")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    counts_rds = data_dir / "GSE180420_EXPORT_counts.rds.gz"
    pheno_path = data_dir / "GSE180420_EXPORT_pheno.txt.gz"
    cluster_path = data_dir / "GSE180420_EXPORT_clusters.txt.gz"
    umap_path = data_dir / "GSE180420_EXPORT_umap.txt.gz"

    if not counts_rds.exists():
        print(f"  Error: {counts_rds} not found.")
        return

    mtx_dir = export_rds_to_mtx(counts_rds, work_dir)
    print(f"  Loading {mtx_dir} into AnnData...")
    adata = sc.read_mtx(mtx_dir / "matrix.mtx").T  # MatrixMarket: rows=genes -> transpose
    adata.var_names = pd.read_csv(mtx_dir / "features.tsv", header=None)[0].values
    adata.obs_names = pd.read_csv(mtx_dir / "barcodes.tsv", header=None)[0].values
    adata.var_names_make_unique()

    if pheno_path.exists():
        ph = pd.read_csv(pheno_path, sep="\t", index_col=0)
        ph = ph.set_index("barcodes")
        adata.obs["pheno"] = ph["pheno"].reindex(adata.obs_names).values

    # Parse pheno into structured columns
    pheno = adata.obs.get("pheno", pd.Series(index=adata.obs_names))
    adata.obs["injury"] = pheno.map(
        {"Control": "Control",
         "IRI_short_1": "IRI_short", "IRI_short_3": "IRI_short", "IRI_short_14": "IRI_short",
         "IRI_long_1": "IRI_long",   "IRI_long_3": "IRI_long",   "IRI_long_14": "IRI_long"}
    )
    adata.obs["post_injury_day"] = pheno.str.extract(r"_(\d+)$")[0].astype(float)
    adata.obs["time"] = adata.obs["post_injury_day"].fillna(0)  # Control = day 0
    adata.obs["study"] = "Balzer2022"
    adata.obs["system"] = "Mouse_Kidney_IRI"
    adata.obs["track"] = "self_repair"

    if cluster_path.exists():
        cl = pd.read_csv(cluster_path, sep="\t", index_col=0)
        if "cluster" in cl.columns:
            adata.obs["cluster"] = cl["cluster"].reindex(adata.obs_names).values
    if umap_path.exists():
        um = pd.read_csv(umap_path, sep="\t", index_col=0)
        coord_cols = [c for c in um.columns if c.lower().startswith("umap")]
        if len(coord_cols) >= 2:
            coords = um[coord_cols[:2]].reindex(adata.obs_names).values
            adata.obsm["X_umap"] = coords.astype(np.float32)

    print(f"  Shape: {adata.shape}")
    print(f"  Pheno: {adata.obs['pheno'].value_counts().to_dict()}")
    print(f"  Time: {adata.obs['time'].value_counts(dropna=False).to_dict()}")
    out = Path(args.out)
    adata.write(out)
    print(f"  Saved to {out}")


if __name__ == "__main__":
    main()
