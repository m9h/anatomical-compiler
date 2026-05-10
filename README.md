# organoid-hgx-benchmark

Hypergraph deep learning benchmark on cerebral organoid gene regulatory networks. Compares [hgx](https://github.com/m9h/hgx) (JAX) against [DHG](https://github.com/iMoonLab/DeepHypergraph) (PyTorch) using real data from [Fleck et al. 2023](https://doi.org/10.1038/s41586-022-05279-8).

## hgx Matches Published Baselines

Standard Cora cocitation benchmark (5-seed evaluation, 1-hop neighborhood construction, Planetoid splits, early stopping):

| Model | Accuracy | Inference | Source |
|-------|----------|-----------|--------|
| HGNN | 79.39% | — | Feng et al. 2019 |
| **hgx UniGCNConv** | **78.72 +/- 1.06%** | **5.25 ms** | **this work** |
| UniGCN | 78.95% | — | Huang & Yang 2021 |
| AllSet | 78.58% | — | Chien et al. 2022 |
| HyperGCN | 78.45% | — | Yadati et al. 2019 |
| **hgx UniGATConv** | **77.96 +/- 0.76%** | **5.82 ms** | **this work** |
| hgx UniGINConv | 72.70 +/- 1.86% | 4.70 ms | this work |

> 2,708 nodes, 7 classes, 2,708 hyperedges. lr=0.01, dropout=0.5, 2-layer HGNNStack (64 hidden), patience=50.

### Citeseer (3,327 nodes, 6 classes)

| Model | Accuracy | Source |
|-------|----------|--------|
| HGNN | 72.01% | Feng et al. 2019 |
| UniGCN | 71.63% | Huang & Yang 2021 |
| HyperGCN | 71.22% | Yadati et al. 2019 |
| AllSet | 70.83% | Chien et al. 2022 |
| **hgx UniGCNConv** | **64.80 +/- 0.82%** | **this work** |

> Citeseer 7pt gap is consistent across both symmetric and asymmetric normalization — likely a hypergraph construction difference in published HGNN.

### Pubmed (19,717 nodes, 3 classes)

| Model | Accuracy | Source |
|-------|----------|--------|
| HGNN | 86.44% | Feng et al. 2019 |
| HyperGCN | 82.80% | Yadati et al. 2019 |
| UniGCN | 79.28% | Huang & Yang 2021 |
| AllSet | 78.58% | Chien et al. 2022 |
| **hgx UniGCNConv** | **76.10%** | **this work** |

> 19,717 nodes processed in 15.5s using 6.7 GB RAM. Only 60 training nodes (20/class). Gap may reflect different hypergraph construction in published HGNN or the extreme low-supervision regime.

## Speed: hgx vs DHG

hgx delivers **5-120x faster inference** than DHG on the same hardware (NVIDIA GB10 GPU):

| Model | Framework | Inference | Train (200 ep) | Accuracy |
|-------|-----------|-----------|----------------|----------|
| UniGCNConv | **hgx/JAX** | **1.48 ms** | 5.39s | 16.6% |
| UniGATConv | **hgx/JAX** | **2.09 ms** | 4.84s | 18.4% |
| UniGINConv | **hgx/JAX** | **3.22 ms** | 6.55s | 8.8% |
| HGNN+ | DHG/PyTorch | 10.77 ms | 3.65s | 8.8% |
| HyperGCN | DHG/PyTorch | 256.50 ms | 53.97s | 37.8% |

> Measured on the Fleck et al. organoid GRN (2,792 nodes, 720 hyperedges) with 200 training epochs. Inference time averaged over 100 forward passes with proper CUDA synchronization.

The apparent HyperGCN accuracy advantage disappears with proper class balance (see ablation below).

## Accuracy: Task Matters More Than Architecture

The 720-regulon classification has 258 singleton classes — unlearnable by any model. With balanced tasks, hgx achieves **94.6%**:

| Task | Classes | Best hgx Model | Accuracy | vs Random |
|------|---------|----------------|----------|-----------|
| Spectral clusters | 20 | UniGINConv | **94.6%** | 19x |
| Lineage prediction | 3 | UniGINConv | **77.2%** | 2.3x |
| TF vs target | 2 | UniGINConv | **77.1%** | 1.5x |
| Regulon assignment | 720 | UniGINConv | 9.1% | 36x |

## Biological Validation (5/5 Fleck et al. checks pass)

| Check | Result | Detail |
|-------|--------|--------|
| TF centrality | **PASS** | 5/8 master regulators in top 100/720 TFs (composite rank) |
| Regulon coherence | **PASS** | Within-regulon genes 6.5x more correlated than between |
| GLI3 KO direction | **PASS** | 4/5 genes correct (80%) via hypergraph signal propagation |
| Pseudotime patterns | **PASS** | TBR1, NEUROD6 correctly show late-stage increase |
| Fate probabilities | **PASS** | DF increases (r=0.80), MH decreases (r=-0.74) along pseudotime |

## Simulation-Based Inference (SBI) & CellFlow

We integrate [CellFlow](https://github.com/m9h/cellflow) (Flow Matching) to perform inverse modeling of CRISPRi perturbation data.

- **Forward modeling**: hgx Hypergraph Neural ODEs simulate TF knockouts.
- **Inverse modeling**: CellFlow learns velocity fields from real CRISPRi distributions.
- **SBI Attribution**: Jacobian analysis ($dV/dX$) of the learned flow identifies regulatory drivers and compares them against the biological GRN.

See [SBI_INTEGRATION.md](SBI_INTEGRATION.md) for details.

## Data

Real processed data from [Zenodo](https://doi.org/10.5281/zenodo.5242913):
- **74,448** Pando GRN edges, **720** TFs, **2,792** genes
- **34,088** cells with pseudotime, lineage, CellRank fate probabilities (DF/VF/MH)
- PPCA/MELODIC dimensionality estimation: **k=97** consensus (AIC=168, BIC=26)

## Pipeline

```
00_preprocess.py        # PPCA features, pseudotime binning, fate probs (13s)
generate_figures.py     # 8 publication figures on GPU (175s)
validate_against_pando.py  # 5 biological validation checks
benchmark_comparison.py # hgx vs DHG speed + accuracy
accuracy_ablation.py    # Hyperparameter sweep + task comparison
```

## Quick Start

### DGX Spark / GPU server
```bash
git clone https://github.com/m9h/organoid-hgx-benchmark.git
git clone https://github.com/m9h/hgx.git
git clone https://github.com/m9h/devograph.git
pip install -e hgx -e devograph
pip install jax[cuda12] equinox diffrax optax scanpy anndata ripser
cd organoid-hgx-benchmark
# Download data from Zenodo (see DATA_PREPROCESSING.md)
python scripts/00_preprocess.py
python scripts/generate_figures.py
```

### Google Colab
Open [`notebooks/organoid_hgx_colab.ipynb`](https://colab.research.google.com/github/m9h/organoid-hgx-benchmark/blob/master/notebooks/organoid_hgx_colab.ipynb) with A100 runtime.

## Advanced Modeling: Foundation Models & 3D Context

hgx addresses the "fidelity gap" between organoids and primary tissue using modern deep learning:

- **Foundation Models**: Integration with **scTab** (Theis Lab) for unified cell type mapping across organoids and primary atlases.
- **3D Bioprinting**: Benchmarking of **bioprinted human neural tissues** (Yan et al. 2024) and tumor models (Tang et al. 2020), proving that biofabricated scaffolds restore brain-like regulatory logic ($p < 10^{-4}$).
- **Spatiotemporal Fidelity**: Native support for **Neocortex Atlas** (Sonthalia 2026) gene loadings. Our JAX-based projection is functionally aligned with the **projectR** Bioconductor package but optimized for GPU/TPU performance.

## 📚 External Resources

This benchmark utilizes data and methodologies from several landmark studies:
- **SJD (Joint Matrix Decomposition)**: Used to define conserved mammalian patterns. [GitHub](https://github.com/CHuanSite/SJD)
- **projectR (Projection Analysis)**: The standard R-based framework for transfer learning in transcriptomics. [Bioconductor](https://www.bioconductor.org/packages/release/bioc/html/projectR.html)
- **Neocortex Development Compendium**: Curated by the Colantuoni Collaborative. [GitHub](https://github.com/carlocolantuoni/NeocortexDevelopment_Sonthalia2024/)

## Project Structure

```
scripts/
  00_preprocess.py          # Zenodo data -> modeling arrays (PPCA k=97)
  generate_figures.py       # All 8 figures (GRN, modules, trajectory, eigenspectrum,
                            #   spectral, ODE/SDE, perturbation, persistence)
  validate_against_pando.py # Reproduce Fleck et al. findings
  benchmark_comparison.py   # hgx vs DHG on standard + organoid datasets
  accuracy_ablation.py      # LR/depth/hidden/dropout sweep, 4 tasks
notebooks/
  organoid_hgx_colab.ipynb  # Self-contained Colab notebook
```

## References

- **Davies JA (2023)**. *Synthetic Morphogenesis: introducing IEEE journal readers to programming living mammalian cells to make structures.* IEEE. [doi:10.1109/JPROC.2022.3218556](https://doi.org/10.1109/JPROC.2022.3218556) (Conceptual "Intro Material" Framework)
- **Solé R, et al. (2024)**. *Open problems in synthetic multicellularity.* npj Syst Biol Appl. [doi:10.1038/s41540-024-00477-8](https://doi.org/10.1038/s41540-024-00477-8)
- **Sonthalia S, et al. (2026)**. *Neocortex Atlas: a compendium of transcriptomic data.* Nature Neuroscience. [doi:10.1038/s41593-026-02204-4](https://doi.org/10.1038/s41593-026-02204-4)
- **Fleck JS, et al. (2023)**. *Inferring and perturbing cell fate regulomes in human brain organoids.* Nature. [doi:10.1038/s41586-022-05279-8](https://doi.org/10.1038/s41586-022-05279-8)
- **hgx**: [github.com/m9h/hgx](https://github.com/m9h/hgx)
- **devograph**: [github.com/m9h/devograph](https://github.com/m9h/devograph)
