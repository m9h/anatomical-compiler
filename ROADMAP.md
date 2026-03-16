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
- [ ] TF centrality rankings — FAIL (graph centrality != biological importance, needs new metrics)
- [ ] GLI3 KO direction — FAIL (needs real CROP-seq data from Seurat objects, extract_cropseq.py created)

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

- [ ] Paper: "Dissecting gene regulatory networks governing human cortical cell fate"
- 44 TFs screened with CRISPRi in primary human cortical cultures
- Perturb-seq readout -- perfect for PerturbationPredictor validation
- Data: bioRxiv 10.1101/2025.09.23.678137, GEO accession TBD
- Plan in progress

### 3B: Cross-Dataset Generalization

- Train perturbation predictor on Fleck et al. organoid data
- Test on Pollen et al. primary tissue data
- Evaluate transfer learning across culture systems

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
- [ ] Improve perturbation training data (use real CROP-seq DE, not GRN-simulated)
- [ ] Add proper train/val/test splits for module detection
- [ ] Add cross-validation to all analyses
- [ ] Profile GPU utilization (currently CPU-heavy for centrality + persistence)
