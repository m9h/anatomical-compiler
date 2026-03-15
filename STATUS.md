# Organoid-HGX-Benchmark: Status Report

## Overview

Benchmark demonstrating hgx (JAX hypergraph neural networks) + devograph
(developmental neurobiology extensions) on cerebral organoid gene regulatory
networks from Fleck et al. 2023 (Nature, doi:10.1038/s41586-022-05279-8).

## Infrastructure

- **GitHub repos**: m9h/hgx (core), m9h/devograph (extensions), m9h/organoid-hgx-benchmark
- **Compute**: DGX Spark GB10 (128GB unified, CUDA 13.1) via NGC PyTorch 26.02 container
- **Alternative**: Google Colab A100 (notebook at notebooks/organoid_hgx_colab.ipynb)
- **JAX**: 0.9.1 with CUDA on Spark; version varies on Colab

## Data (25 GB on DGX Spark)

All from Zenodo doi:10.5281/zenodo.5242913 (latest version: records/15371701)

- **grn_modules.tsv**: 74,448 Pando GRN edges, 720 TFs, 2,792 genes
- **RNA_data.h5ad**: 49,718 cells x 33,538 genes
- **RNA_all_velo.h5ad**: 34,088 cells, spliced/unspliced, fate probs (DF/VF/MH), UMAP
- **data_matrices/**: sparse counts + metadata (pseudotime, lineage, stage)
- **seurat_objects.tar.gz**: 17 GB Seurat objects
- **motif2tf.tsv**: 576 TF-motif mappings

## Preprocessing Pipeline (00_preprocess.py)

- PPCA/MELODIC dimensionality estimation: AIC=168, BIC=26, consensus=97
- 2,792 genes x 720 regulon incidence matrix
- 97-dim PCA node features
- 10 pseudotime bins with mean expression, lineage fractions, fate probs
- Perturbation data for 8 key TFs
- Runs in 13 seconds

## Analyses Completed (generate_figures.py)

All 8 figures generated on DGX Spark GPU in 175 seconds:

| # | Analysis | Key Result |
|---|----------|------------|
| 1 | GRN Architecture | 2,792 nodes, 720 regulons, FOXG1 top eigenvector centrality |
| 2 | Module Detection | SheafDiffusion 100% accuracy on 720-class classification |
| 3 | Trajectory | TF expression, lineage, CellRank fates along pseudotime |
| 4 | Eigenspectrum | PPCA consensus k=97, BIC=26, AIC=168 |
| 5 | Spectral | Hypergraph Laplacian eigenvalues, beta_0 computed |
| 6 | Neural ODE/SDE | ODE rollout MSE=0.359, SDE sigma=0.083, 20 trajectories |
| 7 | Perturbation | 720 TFs screened, in-silico KO predictions |
| 8 | Persistence | H0=499, H1=111 topological features |

## Known Issues

- **hgx JAX version pin** (fixed: removed <0.5 cap)
- **hgx \_\_init\_\_.py missing devograph modules** (fixed: reverted, use devograph separately)
- **Betweenness centrality OOM** on dense clique expansion (fixed: k=50 approximation)
- **THNNConv NaN on real data** (needs gradient clipping, not yet fixed)
- **Colab editable install issues** with hatchling (needs `pip install hatchling` first)
- **Perturbation correlations near zero** (needs better training data -- simulated effects too simple)
- **Topology subsample needed** for persistence (500 node max for ripser)

## What's NOT Done Yet

1. **Validation against Pando**: Haven't verified hgx reproduces Pando's known findings
2. **Standard benchmarks**: No comparison against DHG/HyperGCN/AllSet on Cora/Citeseer/Pubmed
3. **Second dataset**: Nature 2026 paper (Pollen lab CRISPRi Perturb-seq) not yet started
4. **Real CROP-seq validation**: GLI3 KO predictions not compared to actual CROP-seq DE
5. **Poincare analysis**: Failed due to dimension mismatch (needs obs_dim=fd not hardcoded)
6. **Cross-species**: C. elegans data loaders work on Spark but not Colab
