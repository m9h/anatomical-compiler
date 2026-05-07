import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import anndata as ad
import pandas as pd
import scipy.sparse as sp
import numpy as np

def convert_rds_to_h5ad(rds_path, h5ad_path):
    print(f"Reading {rds_path}...")
    # Load RDS
    ro.r(f'data <- readRDS("{rds_path}")')
    
    # Check class
    obj_class = str(ro.r('class(data)')[0])
    print(f"Object class: {obj_class}")
    
    if obj_class == "Seurat":
        print("Detected Seurat object. Extracting counts...")
        ro.r('library(Seurat)')
        ro.r('counts <- GetAssayData(data, assay="RNA", slot="counts")')
        ro.r('meta <- data@meta.data')
    elif obj_class == "dgCMatrix":
        print("Detected sparse matrix.")
        ro.r('counts <- data')
        ro.r('meta <- data.frame(row.names=colnames(data))')
    else:
        print("Unknown class. Trying to treat as matrix...")
        ro.r('counts <- as.matrix(data)')
        ro.r('meta <- data.frame(row.names=colnames(data))')

    # Convert meta to pandas
    with localconverter(ro.default_converter + pandas2ri.converter):
        meta_df = ro.conversion.rpy2py(ro.r('as.data.frame(meta)'))
    
    # Convert sparse matrix to scipy
    # This is tricky via rpy2. Alternative: save as mtx in R and read in python.
    # But since R is not installed, we hope rpy2 works.
    print("Converting counts to scipy sparse matrix...")
    # Get components of dgCMatrix
    ro.r('i <- counts@i')
    ro.r('p <- counts@p')
    ro.r('x <- counts@x')
    ro.r('dims <- dim(counts)')
    ro.r('rn <- rownames(counts)')
    ro.r('cn <- colnames(counts)')
    
    i = np.array(ro.r('i'))
    p = np.array(ro.r('p'))
    x = np.array(ro.r('x'))
    dims = np.array(ro.r('dims'))
    rn = list(ro.r('rn'))
    cn = list(ro.r('cn'))
    
    counts_sparse = sp.csc_matrix((x, i, p), shape=dims)
    
    print(f"Creating AnnData with shape {counts_sparse.shape}...")
    adata = ad.AnnData(X=counts_sparse.T, obs=meta_df)
    adata.var_names = rn
    
    print(f"Saving to {h5ad_path}...")
    adata.write_h5ad(h5ad_path)
    print("Done!")

if __name__ == "__main__":
    convert_rds_to_h5ad("data/choose/CHOOSE_ASD_GE.rds", "data/choose/CHOOSE_ASD_GE.h5ad")
