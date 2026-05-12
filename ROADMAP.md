# Roadmap

## Phase 1: Validation (COMPLETE)

### 1A: Standard Benchmark Suite
- [x] Cora validated — hgx UniGCNConv 78.72% matches published HGNN 79.39%
- [x] Citeseer — 64.80% (7pt gap is architectural, not data)
- [x] Pubmed — 76.10% on 19,717 nodes in 15.5s / 6.7 GB RAM
- [x] Accuracy ablation — 94.6% on 20 spectral clusters, 77.2% lineage 3-class
- [x] SheafDiffusion OOM diagnosed and fixed
- [x] THNNConv NaN diagnosed — fix script created
- [x] DHG device mismatch fixed

### 1B: Pando Reproduction (5/5 PASS — confirmed 2026-03-17)
- [x] TF centrality — PASS (TF-aware ranking, composite precision@100 = 0.62)
- [x] Regulon coherence — PASS (gap = +0.116)
- [x] GLI3 KO direction — PASS (80% via hypergraph signal propagation)
- [x] Pseudotime patterns — PASS (TBR1/NEUROD6 increase_late)
- [x] Fate probabilities — PASS (DF r=0.80, MH r=-0.74)

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

- [x] Fix dimension mismatch in LatentHypergraphODE (fixed in cac6cba)
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

- [x] GRN overlap: 34/44 shared TFs ($p < 10^{-5}$)
- [x] Direction concordance: 82.9% mean accuracy (POU2F1 79.6%, ASCL1 76.7%)
- [x] Transfer prediction: Zero-shot transfer from Fleck -> Pollen validated
- [x] Within-Pollen LOO baseline (mean $r=0.36$)

### 3C: New Regulome Atlas Extensions

- [ ] **Posterior Brain (Azbukina et al., 2025)**: Midbrain/hindbrain regulomes (snRNA/ATAC)
  - DOI: 10.1101/2025.03.20.644368 | Zenodo: 10.5281/zenodo.11203684
  - Target: Identify TF networks for dopaminergic and glycinergic neurons
- [ ] **Retinal Regulomes (Wahle et al., 2023)**: Spatiotemporal retinal organoid atlas
  - DOI: 10.1038/s41587-023-01747-2 | ArrayExpress: E-MTAB-12714
  - Target: Validate hgx on non-brain developmental GRNs (e.g., OTX2 regulon)
- [x] **Simulation-Based Inference (SBI)**:
  - [x] Invert hgx models to estimate GRN posteriors from Pollen/Wahle CRISPRi data
  - [x] Compare hgx-SBI against standard Pando/GRN-VAE methods

### 3D: Foundation Model Benchmarking (Theis Lab)

- [x] **scTab (Cell Type Annotation)**:
  - [x] Standalone scTab inference script implemented (`scripts/annotate_sctab.py`)
  - [x] Standardized cell type labels across Fleck and Pollen datasets.
  - [x] Evaluated population-specific transfer concordance.
- [ ] **Prophet (Perturbation Prediction)**:
  - Benchmark `theislab/Prophet` (Transformer-based) against `hgx.PerturbationPredictor` (Hypergraph-based).
  - Compare Pearson r and direction concordance on Pollen/Ding 2026 CRISPRi data.
  - Analyze "zero-shot" transfer performance: how well does a foundation model (Prophet) perform compared to a task-specific model (hgx)?

### 3E: Evolutionary Benchmark (NHP, Mouse, Ferret)

- [x] **Marmoset Interneurons (Krienen 2021)**:
  - [x] Downloaded and annotated with scTab.
  - [x] Validated DLX1 regulon conservation ($p = 2.12 \times 10^{-3}$).
- [x] **Mouse Neocortex (Loo 2019)**:
  - [x] Preprocessed E14.5 neocortex.
  - [x] Identified EOMES (14.3%) and NEUROD2 (11.6%) as conserved regulators.

### 3F: Experimental Validation (CRISPR Screens)

- [x] **CHOOSE Screen (Sharf/Li 2023)**:
  - [x] Downloaded 4.28 GB h5ad (CHD4/ADNP KOs).
  - [x] Preprocessed with Human-Mouse symbol mapping.
  - [x] Validated ADNP effects in organoids.
- [x] **Real CROP-seq (Fleck 2023)**:
  - [x] Extracted real DE for GLI3, TBR1, NEUROD6, PAX6, HES1 using R/Seurat.
  - [x] Achieved strong validation (GLI3 vs TBR1 r=0.83).

### 3G: Solé Open-Problems Suite (Synthetic Multicellularity)

Mapped against Solé et al. 2024 (npj Systems Biology and Applications):

- [x] **Patterning** — GSE156162 (Toda 2020 synNotch) — `benchmark_toda_morphogenesis.py`
- [x] **Robotics** — GSE249581 (Gumuskaya 2023 Anthrobots) — `benchmark_anthrobot_fidelity.py`
- [x] **Metabolism / Vascularization** — GSE131094 (Shi 2020 vOrganoids), corrected
  from the original GSE131936 misattribution. 57,180 cells (vOrg 31,851 / Org 25,329,
  Day 65 + Day 100). Cross-talk regulons in vOrganoids: **RBPJ +0.072 (p=7.5e-19)**,
  **EPAS1 −0.056 (p=6.3e-17)** — quantitative evidence for hypoxia-relief by vasculature
  (the "metabolic wall"). Pre-normalized log matrix — `benchmark_vorganoid_crosstalk.py`
  auto-detects negative values and skips re-normalization.
- [x] **Agency / Bio-Computing** — pivoted from GSE207577 (Kagan DishBrain, private
  until 2026-12-31, MEA-only) to GSE102827 (Hrvatin 2018 V1 light-stim scRNA-seq).
  65,539 cells × 0h/1h/4h. IEG timecourse: −0.13 → +0.06 → −0.07 (canonical
  immediate-early peak-at-1h dynamics); late genes (BDNF/HOMER1/DUSP1) peak 1–4 h.
- [x] **Robustness / Self-Repair** — GSE138835 (Lawlor) has only 1 sample;
  pivoted to **GSE180420 (Balzer 2022 Nat Commun)** mouse kidney IRI, 113,579
  cells × 4 timepoints (Day 0/1/3/14, IRI_short + IRI_long).
  `benchmark_regenerative_flow.py` fits a Hypergraph Neural ODE in 1.2 s and
  cleanly separates **stable regenerative drivers** (Lhx1 MSE=0.05, Cdh1=0.06,
  Pax8=0.08, Pax2=0.10, Six1/2/Wt1/Foxc2≈0.11) from **dynamic injury markers**
  (Fos=4.4, Jun=1.5, Cd44=0.93, Atf3=0.80) — exactly the homeostatic-vs-stress
  split predicted by the regenerative-flow hypothesis.

Local artifacts: `figures/{vorganoid_crosstalk,learning_regulome,regenerative_flow}.{png,*_results.json}`.
All three benchmarks gracefully fall back to TF-expression scoring when Pando
GRN (`data/zenodo/grn_modules.tsv`, DGX-only) is absent; richer regulon-based
scoring activates automatically on DGX Spark.

## Phase 4: Publication

- Comprehensive comparison table: Pando (R, pairwise) vs hgx (JAX, higher-order)
- 7 publication figures from real data
- Quantitative metrics with confidence intervals
- Code + data fully reproducible via Colab notebook
- [x] Consolidated into a single literate knitr/Sweave manuscript (`publication/paper.Rnw` → `paper.tex` → `paper.pdf`); result tables/values pulled live from `figures/*_results.json` + `data/cropseq/*.csv`

## Phase 5: Network control / the anatomical compiler (jaxctrl)

- [x] `benchmark_network_control.py` — linear network controllability + steer-to-target on the Pando TF co-regulation graph (controllability Gramians, structural driver nodes, LQR)
- [x] `benchmark_anatomical_compiler.py` — nonlinear optimal control on the *learned* Hypergraph Neural ODE (`diffrax` adjoints + Adam; jaxctrl LQR warm-start): target tissue state → TF-actuation schedule
- [ ] `minimum_driver_nodes` via hgx on the (non-uniform) Pando hypergraph — needs a uniform decomposition first
- [ ] richer plants (larger regulome, more timepoints, mechanistically-constrained drift) and richer actuators (the §4.3 bioprinting / optogenetic / dose / bioelectric handles)
- [ ] couple the anatomical compiler to the wet-lab loop (FRESH/SWIFT/PRINTESS print parameters as actuation; model-guided vasculature)

## Educational track (notebooks/)

A planned course (~6–8 lab sessions) walking students through the project: regulomes & hypergraphs
→ fidelity benchmarking → modularity/identifiability → Hypergraph Neural ODEs → control theory on
cellular dynamics (jaxctrl) → the anatomical compiler → synthetic morphology in the wet lab →
(stretch) cancer as loss of module identifiability. See `notebooks/README.md` for the sequence.
Seed material: `notebooks/organoid_hgx_colab.ipynb` + the three `jaxctrl/examples/*.ipynb/.py`.

- [x] Notebook 1 — regulomes and hypergraphs (`notebooks/01_regulomes_and_hypergraphs.ipynb`): the Fleck incidence; why a regulon is a hyperedge not a clique; `hgx` basics; the Laplacian spectrum; exercises
- [ ] Notebook 2 — benchmarking fidelity (organoid → primary CRISPRi; cross-species)
- [ ] Notebook 3 — modularity & identifiability (Hodge Laplacian; the Module Identifiability Index)
- [ ] Notebook 4 — Hypergraph Neural ODEs (latent dynamics; drivers vs stress; attractors)
- [ ] Notebook 5 — control theory on cellular dynamics (reuse the jaxctrl example notebooks)
- [ ] Notebook 6 — the anatomical compiler (optimal control on a learned regulome ODE)
- [ ] Notebooks 7–8 — synthetic morphology in the wet lab; cancer-as-loss-of-identifiability (stretch)

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
- [x] Fix Poincare LatentODE dimension handling (fixed in cac6cba)
- [x] Improve perturbation training data — real CROP-seq DE extracted from Seurat objects.
- [x] Implement robust large file downloader with resume support.
- [ ] Add proper train/val/test splits for module detection
- [ ] Add cross-validation to all analyses
- [ ] Profile GPU utilization (currently CPU-heavy for centrality + persistence)
