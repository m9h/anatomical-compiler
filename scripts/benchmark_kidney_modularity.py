import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import jax
import jax.numpy as jnp
import hgx

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def compute_hodge_scores(adata, n_hvg=500):
    print(f"Computing Hodge scores for {adata.shape[0]} cells...")
    # 1. HVGs
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    hvg_data = adata[:, adata.var.highly_variable].to_memory()
    
    # 2. Correlation-based incidence
    # (n_cells, n_hvg)
    X = hvg_data.X
    if hasattr(X, "toarray"): X = X.toarray()
    
    # Compute gene-gene correlation
    corr = np.corrcoef(X.T)
    
    # Each HVG's top 10 correlated neighbors forms a hyperedge
    n_genes = corr.shape[0]
    incidence = np.zeros((n_genes, n_genes), dtype=np.float32)
    for i in range(n_genes):
        # Top 10 neighbors
        top_idx = np.argsort(np.abs(corr[i]))[-11:] # self + 10
        incidence[top_idx, i] = 1.0
        
    # 3. hgx Hypergraph
    hg = hgx.from_incidence(jnp.array(incidence))
    
    # 4. Hodge Laplacians
    print("  Computing Hodge Laplacians (L0, L1)...")
    laplacians = hgx.hodge_laplacians(hg)
    L0 = laplacians[0]
    
    # Eigenvalues of L0 (Graph Laplacian)
    ev0 = np.array(jnp.linalg.eigvalsh(L0))
    # Non-zero eigenvalues
    ev0 = ev0[ev0 > 1e-6]
    
    # Algebraic Connectivity (Fiedler value)
    fiedler = ev0[0] if len(ev0) > 0 else 0
    
    # L1 (if available)
    if len(laplacians) > 1:
        L1 = laplacians[1]
        if L1.shape[0] > 0:
            ev1 = np.array(jnp.linalg.eigvalsh(L1))
            ev1 = ev1[ev1 > 1e-6]
        else:
            ev1 = np.array([])
    else:
        ev1 = np.array([])
        
    return {
        "ev0": ev0,
        "ev1": ev1,
        "fiedler": fiedler,
        "n_nodes": n_genes,
        "n_edges": incidence.shape[1]
    }

def main():
    data_dir = Path("data/bioprinting")
    print("Loading Kidney Bioprinting data...")
    kidney = sc.read_h5ad(data_dir / "lawlor_2021_processed.h5ad")
    
    # Basic QC
    sc.pp.filter_cells(kidney, min_genes=200)
    sc.pp.log1p(kidney)
    
    # Compute Hodge scores (reduced HVG for speed and memory)
    k_scores = compute_hodge_scores(kidney, n_hvg=100)
    
    # Comparison: We'll use a synthetic Brain Organoid baseline if Fleck isn't easy to load
    # or just use the Fleck GRN if we can.
    # Actually, let's load a subset of Fleck for a fair comparison (n_hvg=100)
    print("Loading Brain Organoid (Fleck) for comparison...")
    # Use the preprocessed fleck RNA if available
    fleck_path = Path("data/zenodo/RNA_all_velo.h5ad")
    if fleck_path.exists():
        brain = sc.read_h5ad(fleck_path, backed='r')
        # Subsample to match cell count if needed, but HVG is over genes
        # Let's just take a 1k cell subset
        brain_sub = brain[:1000].to_memory()
        b_scores = compute_hodge_scores(brain_sub, n_hvg=100)
    else:
        print("  Fleck RNA not found, skipping comparison.")
        b_scores = None

    # 3. Reference: Human Fetal Kidney
    print("Loading Human Fetal Kidney reference for comparison...")
    k_ref_path = data_dir / "kidney_ref_processed.h5ad"
    if k_ref_path.exists():
        k_ref = sc.read_h5ad(k_ref_path)
        # Taking subset for speed
        k_ref_sub = k_ref[:1000].to_memory()
        sc.pp.log1p(k_ref_sub)
        r_scores = compute_hodge_scores(k_ref_sub, n_hvg=100)
    else:
        print("  Kidney reference not found.")
        r_scores = None

    # 5. Plot
    plt.figure(figsize=(12, 5))
    
    # Panel A: L0 Eigenspectrum
    plt.subplot(1, 2, 1)
    plt.plot(k_scores["ev0"][:100], 'o-', label="Bioprinted Kidney", color="teal")
    if b_scores:
        plt.plot(b_scores["ev0"][:100], 's-', label="Brain Organoid", color="firebrick")
    if r_scores:
        plt.plot(r_scores["ev0"][:100], 'd-', label="Fetal Kidney (Ref)", color="forestgreen")
    plt.ylabel("$L_0$ Eigenvalue")
    plt.xlabel("Index")
    plt.title("Hodge $L_0$ Spectrum (Modularity)")
    plt.legend()
    
    # Panel B: Connectivity Bar
    plt.subplot(1, 2, 2)
    labels = ["Kidney (Bioprinted)"]
    vals = [k_scores["fiedler"]]
    colors = ["teal"]
    if b_scores:
        labels.append("Brain (Organoid)")
        vals.append(b_scores["fiedler"])
        colors.append("firebrick")
    if r_scores:
        labels.append("Kidney (Fetal)")
        vals.append(r_scores["fiedler"])
        colors.append("forestgreen")
        
    plt.bar(labels, vals, color=colors)
    plt.ylabel("Algebraic Connectivity (Fiedler)")
    plt.title("Structural Integration Score")
    
    out_path = Path("figures/kidney_modularity_hodge.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    
    # Save results
    def _sanitize(v):
        if isinstance(v, (np.ndarray, jax.Array)):
            return v.tolist()
        if isinstance(v, (np.float32, np.float64)):
            return float(v)
        return v

    results = {
        "kidney": {k: _sanitize(v) for k, v in k_scores.items()},
        "brain": {k: _sanitize(v) for k, v in b_scores.items()} if b_scores else None,
        "kidney_ref": {k: _sanitize(v) for k, v in r_scores.items()} if r_scores else None
    }
    with open("figures/kidney_modularity_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
