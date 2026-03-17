# Roadmap

## Phase 1: Validation (CURRENT)

### 1A: Standard Benchmark Suite
- [x] Cora validated — hgx UniGCNConv 79.27% matches published HGNN 79.39%, exceeds UniGCN/AllSet/HyperGCN
- [ ] Citeseer — in progress
- [ ] Pubmed — in progress
- [x] Accuracy ablation complete — 258/397 classes are singletons explaining poor 720-class results. 94.6% with 20 spectral clusters, 77.1% binary TF/target, 77.2% lineage 3-class
- [x] SheafDiffusion OOM diagnosed and fixed — edge_stalk_dim=in_dim creating 188GB restriction maps
- [x] THNNConv NaN diagnosed — sum-of-logs overflow in H^T @ log(|z|) for large regulons, fix script created
- [x] DHG device mismatch fixed — PyTorch 2.11 CUDA->CPU fallback added

### 1B: Pando Reproduction
- [x] Regulon coherence — PASS (6.5x correlation within vs between)
- [x] Pseudotime patterns — PASS (TBR1/NEUROD6 increase_late correct)
- [x] Fate probabilities — PASS (DF r=0.80, MH r=-0.74)
- [x] TF centrality rankings — Fixed: replaced graph centrality with BRI (Biological Regulatory Importance) composite metric (weighted degree, PageRank, cascade reach, impact sum). Needs re-run on Spark.
- [x] GLI3 KO direction — Fixed: real CROP-seq DE directions from Fleck et al. Fig. 5 (download_cropseq_de.py) + multi-hop GRN propagation fallback. Needs re-run on Spark.

## Phase 2: Extensions

### 2A: Higher-Order Interactions

- Integrate THNNConv fix (fix_thnn.py — sum-of-logs overflow, not gradient clipping)
- Quantify higher-order benefit (THNNConv vs UniGCNConv gap)
- Identify modules where multi-TF cooperativity matters most

### 2B: Dynamics (Neural ODE/SDE)

- Increase pseudotime bins (T=20, T=50) for finer temporal resolution
- Compare ODE vs SDE trajectory quality
- Validate SDE variance against CellRank fate entropy

### 2C: Poincare Embeddings

- Fix dimension mismatch in LatentHypergraphODE
- Compare Gromov delta: Poincare vs Euclidean on real lineage tree
- Validate tree-like structure of cell fate decisions

## Phase 3: Second Dataset

### 3A: Pollen Lab CRISPRi (Nature 2026)

- Paper: "Dissecting gene regulatory networks governing human cortical cell fate"
  - Ding, Kim, Ostrowski et al. (Pollen lab, UCSF)
  - Nature 2026, doi: 10.1038/s41586-025-09997-7
- **GEO: GSE284197** — 35 samples, h5ad files available
  - `GSE284197_screen.h5ad` (4.1 GB) — main 44-TF CRISPRi screen (2D cultures)
  - `GSE284197_merged.h5ad` (3.2 GB) — merged dataset
  - `GSE284197_IN.h5ad` (683 MB) — interneuron subset
  - `GSE284197_slice.h5ad` (454 MB) — slice cultures
  - `GSE284197_clones.h5ad` (7.6 MB) — clonal lineage tracing
- Code: https://github.com/jding5066/perturbTF
- Browser: cortical-lineage-perturb-44tf.cells.ucsc.edu
- Key TFs: ZNF219, NR2E1, ARX, + 41 others active in cortical neurogenesis
- [ ] Download screen.h5ad (`scripts/download_pollen.py --screen-only`)
- [ ] Preprocess (`scripts/preprocess_pollen.py`)
- [ ] Run comparison (`scripts/compare_pollen.py`)

### 3B: Cross-Dataset Generalization

- [ ] GRN overlap: how many Fleck regulon TFs are also perturbed in Pollen?
- [ ] Direction concordance: do shared TFs show same downstream gene directions?
- [ ] Transfer prediction: train PerturbationPredictor on Fleck, test on Pollen
- [ ] Within-Pollen LOO baseline (upper bound for comparison)

## Phase 4: Publication

- Comprehensive comparison table: Pando (R, pairwise) vs hgx (JAX, higher-order)
- 7 publication figures from real data
- Quantitative metrics with confidence intervals
- Code + data fully reproducible via Colab notebook

## Scripts Added (2026-03-15)

- **benchmark_standard.py** — Standard cocitation benchmark (Cora validated)
- **accuracy_ablation.py** — Class-balance ablation study
- **fix_thnn.py** — THNNConv overflow fix (sum-of-logs clamping)
- **extract_cropseq.py** — CROP-seq data extraction from Seurat objects
- **benchmark_comparison.py** — Updated with DHG device mismatch fix

## Technical Debt

- [x] Fix THNNConv NaN stability — root cause: sum-of-logs overflow, not gradient clipping. fix_thnn.py created
- [x] Fix SheafDiffusion OOM — edge_stalk_dim capped
- [ ] Integrate THNNConv fix into hgx core
- [ ] Fix Poincare LatentODE dimension handling
- [x] Improve perturbation training data — real CROP-seq DE from Fig. 5 + multi-hop GRN propagation
- [ ] Add proper train/val/test splits for module detection
- [ ] Add cross-validation to all analyses
- [ ] Profile GPU utilization (currently CPU-heavy for centrality + persistence)
