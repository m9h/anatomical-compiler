# Roadmap

## Phase 1: Validation (CURRENT)

### 1A: Standard Benchmark Suite

- Run hgx vs DHG vs HyperGCN on Cora, Citeseer, Pubmed cocitation
- Compare: accuracy, training time, inference time, memory
- Proves hgx is functionally correct AND fast

### 1B: Pando Reproduction

- Verify TF centrality rankings match Pando's reported key regulators
- Verify module structure (within vs between regulon correlation)
- Verify GLI3 KO direction matches CROP-seq observations
- Verify pseudotime expression patterns match known biology

## Phase 2: Extensions

### 2A: Higher-Order Interactions

- Fix THNNConv (gradient clipping)
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
- 44 TFs screened with CRISPRi in primary human cortical cultures
- Perturb-seq readout -- perfect for PerturbationPredictor validation
- Data: bioRxiv 10.1101/2025.09.23.678137, GEO accession TBD

### 3B: Cross-Dataset Generalization

- Train perturbation predictor on Fleck et al. organoid data
- Test on Pollen et al. primary tissue data
- Evaluate transfer learning across culture systems

## Phase 4: Publication

- Comprehensive comparison table: Pando (R, pairwise) vs hgx (JAX, higher-order)
- 7 publication figures from real data
- Quantitative metrics with confidence intervals
- Code + data fully reproducible via Colab notebook

## Technical Debt

- [ ] Fix THNNConv gradient stability
- [ ] Fix Poincare LatentODE dimension handling
- [ ] Improve perturbation training data (use real CROP-seq DE, not GRN-simulated)
- [ ] Add proper train/val/test splits for module detection
- [ ] Add cross-validation to all analyses
- [ ] Profile GPU utilization (currently CPU-heavy for centrality + persistence)
