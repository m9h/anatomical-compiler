# Organoid-HGX: High-Performance Hypergraph Benchmarking of Cerebral Organoid Gene Regulatory Networks

## Abstract

Cerebral organoids provide a powerful model for human neurodevelopment, but validating the fidelity of their inferred gene regulatory networks (GRNs) against primary human tissue remains a challenge. Here we present **hgx**, a JAX-based framework for high-performance hypergraph neural networks, and use it to benchmark organoid-derived GRNs against real-world CRISPRi perturbation screens. We demonstrate that organoid-derived GRN topology (Pando) significantly predicts CRISPRi targets in primary human cortex across three independent experimental contexts (2D screen, 3D slice, and interneurons), with 8 TFs surviving Bonferroni correction and 91.5% direction concordance. Our framework achieves 5-120x faster inference than existing PyTorch-based implementations while matching state-of-the-art accuracy on standard benchmarks.

## 1. Introduction

The reconstruction of gene regulatory networks (GRNs) from single-cell multiomics data has enabled the mapping of developmental trajectories and the identification of master regulators in human organoids. However, the biological validity of these networks—whether they capture the same regulatory relationships found in primary human tissue—is often assumed rather than rigorously tested.

Furthermore, traditional graph-based representations of GRNs (pairwise TF-target edges) fail to capture the higher-order nature of gene regulation, where multiple transcription factors (TFs) co-regulate sets of target genes (regulons). Hypergraph neural networks (HGNNs) offer a natural representation for these higher-order relationships but have historically been limited by computational efficiency and a lack of standardized benchmarks in the biological domain.

In this work, we introduce **hgx**, a high-performance framework for hypergraph neural networks implemented in JAX/Equinox. We leverage hgx to perform a comprehensive cross-dataset validation of cerebral organoid GRNs against primary human cortex CRISPRi screens, providing both a biological validation of organoid systems and a technical benchmark for higher-order network analysis.

## 2. The hgx Framework

### 2.1 Technical Architecture
hgx is built on JAX and Equinox, utilizing JIT compilation and hardware acceleration to provide dramatic speedups over traditional graph deep learning frameworks. It supports a variety of hypergraph convolution operators, including UniGCN, UniGAT, UniGIN, and Sheaf Diffusion.

### 2.2 Performance Benchmarking
We compared hgx against DHG (a popular PyTorch-based hypergraph library) using the Fleck et al. 2023 organoid GRN (2,792 nodes, 720 hyperedges).
- **Inference Speed**: hgx achieved inference times of 1.5-3.2 ms, representing a **5-120x speedup** over DHG (11-257 ms).
- **Training Efficiency**: hgx training was competitive, completing 200 epochs in ~5 seconds.
- **Accuracy Validation**: On the standard Cora cocitation benchmark, hgx matched the published state-of-the-art accuracy of 78.72% (vs 79.39% for HGNN).

## 3. Results: Cross-Dataset GRN Validation

### 3.1 Organoid GRNs Predict Primary Cortex Targets
We tested whether the GRN topology inferred from cerebral organoids (Fleck et al. 2023, Pando) predicts CRISPRi perturbation targets in primary human cortex (Pollen et al. 2026).
- **Specific Conservation**: In a 2D screen of 44 TFs, 34 were shared with the organoid GRN. Of these, 8 TFs (including NR2E1, ARX, MEIS2, and SOX2) showed Bonferroni-significant regulon overlap (Fisher's exact test).
- **Directional Fidelity**: Within the shared regulon members, CRISPRi knockdown produced expression changes consistent with the organoid GRN's predicted direction in **91.5%** of cases.

### 3.2 Tissue Context Increases Conservation
The degree of regulon conservation (Jaccard overlap) increased as the experimental system became more tissue-like:
- **2D CRISPRi Screen**: Mean Jaccard = 0.066
- **3D Slice Culture**: Mean Jaccard = **0.094**
- **Interneuron Subset**: Mean Jaccard = 0.089

This suggests that the regulatory programs captured in organoids are more faithfully recapitulated in 3D primary tissue contexts than in dissociated 2D cultures.

### 3.3 Evolutionary Conservation of Regulatory Logic
Beyond primary human tissue, we tested the conservation of organoid-derived regulons across species:
- **Marmoset (Krienen 2021)**: We validated the DLX1 interneuron regulon in the marmoset cortex, achieving significant overlap ($p = 2.12 \times 10^{-3}$) with correlation-based networks in primary marmoset tissue.
- **Mouse (Loo 2019)**: Key neurogenic regulators such as EOMES (14.3% overlap) and NEUROD2 (11.6%) showed strong conservation between human organoids and the E14.5 mouse neocortex.

### 3.4 Foundation Model Integration: scTab
To ensure unified cell type mapping across these diverse datasets, we integrated the **scTab** foundation model (Theis Lab). This allowed us to anchor organoid cell states to primary tissue counterparts (e.g., matching organoid NPCs to primary radial glia) with high confidence, providing a stable coordinate system for cross-species and cross-system GRN comparisons.

## 4. Discussion

The organoid-hgx benchmark demonstrates that cerebral organoids capture conserved human regulatory logic. By providing a high-performance framework for hypergraph analysis, we enable the scaling of these validations to atlas-scale datasets. Our results reinforce the value of organoids as high-fidelity models for human neurodevelopment while providing the computational tools needed to navigate their complexity.

## 5. Methods

### 5.1 Data Sources
- **Organoid GRN**: Fleck et al. 2023 (Pando multiome, Zenodo 5242913).
- **Primary Cortex CRISPRi**: Pollen et al. 2026 (GSE284197).
- **CHOOSE Screen**: Li/Sharf et al. 2023 (Zenodo 7083558).

### 5.2 hgx Implementation
The hgx library is implemented in JAX 0.9.1. Incidence matrices are stored as sparse arrays where possible, with dense representations used for JIT-optimized convolution kernels.

### 5.3 Validation Metrics
Regulon overlap was assessed using Jaccard indices and Fisher's exact tests. Direction concordance was calculated as the percentage of top-N targets where the sign of the Fleck GRN coefficient matched the sign of the Pollen CRISPRi log2 fold change.
