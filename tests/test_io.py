import numpy as np
import json
import pytest
from pathlib import Path
import anndata as ad
from hgx_prep.io import to_anndata

def test_to_anndata(tmp_path):
    # Setup mock processed directory
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    
    n_genes, n_pcs, n_tfs = 10, 5, 3
    
    # Create mock arrays
    features = np.random.randn(n_genes, n_pcs).astype(np.float32)
    incidence = (np.random.rand(n_genes, n_tfs) > 0.5).astype(np.float32)
    gene_names = [f"Gene{i}" for i in range(n_genes)]
    tf_names = [f"TF{i}" for i in range(n_tfs)]
    
    np.save(proc_dir / "node_features_pca.npy", features)
    np.save(proc_dir / "incidence.npy", incidence)
    with open(proc_dir / "gene_names.json", "w") as f:
        json.dump(gene_names, f)
    with open(proc_dir / "tf_names.json", "w") as f:
        json.dump(tf_names, f)
        
    # Optional files
    module_labels = np.random.randint(0, 2, size=n_genes)
    np.save(proc_dir / "module_labels.npy", module_labels)
    
    # Run conversion
    adata = to_anndata(proc_dir)
    
    # Assertions
    assert isinstance(adata, ad.AnnData)
    assert adata.shape == (n_genes, n_pcs)
    assert list(adata.obs_names) == gene_names
    assert "grn" in adata.uns
    assert np.allclose(adata.uns["grn"]["incidence"], incidence)
    assert adata.uns["grn"]["tf_names"] == tf_names
    assert "module" in adata.obs
