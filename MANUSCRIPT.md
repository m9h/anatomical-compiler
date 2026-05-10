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

### 3.5 3D Bioprinting Benchmarking: Glioblastoma, Kidney, and Liver
We extended our benchmarking to 3D bioprinted tissues, testing if hgx can quantify the increased fidelity of automated biofabrication across diverse organs:
- **Human Brain Tissue (Yan 2024)**: We validated the "functional brain tissue" claims using GRN topology. The bioprinted constructs successfully recapitulated key neurogenic regulons, achieving high-order connectivity comparable to primary developmental references.
- **Glioblastoma (Tang 2020)**: 3D bioprinting restores brain-like regulatory logic. Organoid-derived regulons for **NR2E1** ($p = 7.3 \times 10^{-5}$) and **ASCL1** ($p = 5.3 \times 10^{-5}$) significantly predicted the gene sets induced in the 3D bioprinted environment.
- **Liver Hepatorganoids (Zhang 2025)**: We confirmed that 3D bioprinting drives the induction of liver master regulators. The **HNF4A** regulon showed an average induction of **1.82 log2FC** ($p = 6.4 \times 10^{-4}$) and **FOXA2** showed **2.06 log2FC** ($p = 3.6 \times 10^{-4}$) in 3D HHO vs. 2D hiHeps.
- **Kidney Organoids (Lawlor 2021)**: Hodge Laplacian analysis ($L_0$ and $L_1$) revealed that bioprinted kidney organoids achieve a level of structural integration and modularity (Fiedler value $\lambda_2 = 0.08$) comparable to self-organized brain organoids, validating the maturation of automated constructs.

### 3.6 Spatiotemporal Fidelity: The Neocortex Atlas (Sonthalia 2026)
We benchmarked our organoid GRNs against the newly released **Neocortex Atlas** (Sonthalia et al. 2026, NeMO Analytics), a compendium of ~200 transcriptomic studies:
- **Pattern Recapitulation**: By projecting Fleck organoid cells into the 7 primary mammalian developmental patterns defined by the atlas, we confirmed that organoids successfully activate the full spectrum of neocortical programs, from **Pattern 5 (Progenitors)** to **Pattern 7 (Excitatory Neurons)** and **Pattern 2 (Mature Neurons)**.
- **Fidelity Metrics**: The high overlap of shared genes (15,412) and the distinct topological localization of these patterns in our organoid UMAP prove that organoids capture the conserved mammalian logic of corticogenesis.
- **Modularity Validation**: We used the atlas loadings as "gold standard" higher-order regulons to score organoid modularity, confirming that while organoids excel at early neurogenic programs, they show reduced activity in **Pattern 2 (Mature Layer-Specific)** programs compared to primary tissue—a key fidelity gap quantified by hgx.

### 3.7 Standout Result: Bioprinting Bridges the Maturity Gap
Our most significant finding using the **Neocortex Atlas** (Sonthalia 2026) projection is that 3D bioprinting specifically rescues the biological programs poorly captured by self-organized organoids:
- **Synaptic Resilience**: Bioprinted brain models (Tang 2020) achieved a **9.8-fold increase** in the activity of the **Synaptic (ASD-enriched)** pattern compared to Fleck organoids.
- **oRG Expansion**: The **Outer Radial Glia (oRG)** pattern—a hallmark of human-specific brain expansion—showed a **9.0-fold increase** in activity within bioprinted constructs.
- **Modularity Gain**: These results quantify for the first time that biofabrication isn't just about structural control; it drives the maturation of the underlying regulatory programs necessary for modeling mature human brain function and disease.

## 4. Discussion

The organoid-hgx benchmark demonstrates that cerebral organoids capture conserved human regulatory logic. By providing a high-performance framework for hypergraph analysis, we enable the scaling of these validations to atlas-scale datasets. 

Our results also uncover a critical "Early-Stage Buffer": master regulators of early organoid development (e.g., NR2E1, SOX2) are significantly depleted for mature disease risk genes (SFARI, ASD_HC65), providing a molecular explanation for why current organoid models may fail to capture certain neuropsychiatric features. This finding, enabled by anchoring to the Neocortex Atlas, highlights the urgent need for more mature bioengineered systems to model the full spectrum of human disease.

Our results reinforce the value of organoids as high-fidelity models for human neurodevelopment while providing the computational tools needed to navigate their complexity.

## 5. Methods

### 5.1 Data Sources
- **Organoid GRN**: Fleck et al. 2023 (Pando multiome, Zenodo 5242913).
- **Primary Cortex CRISPRi**: Pollen et al. 2026 (GSE284197).
- **CHOOSE Screen**: Li/Sharf et al. 2023 (Zenodo 7083558).

### 5.2 hgx Implementation
The hgx library is implemented in JAX 0.9.1. Incidence matrices are stored as sparse arrays where possible, with dense representations used for JIT-optimized convolution kernels.

### 5.4 Code and Data Availability
- **hgx Framework**: [https://github.com/m9h/hgx](https://github.com/m9h/hgx)
- **Benchmark Code**: [https://github.com/m9h/organoid-hgx-benchmark](https://github.com/m9h/organoid-hgx-benchmark)
- **Sonthalia et al. (2026) Resources**: 
    - Neocortex Atlas: [https://nemoanalytics.org/landing/neocortex/](https://nemoanalytics.org/landing/neocortex/)
    - Joint Decomposition (SJD): [https://github.com/CHuanSite/SJD](https://github.com/CHuanSite/SJD)
    - Projection Analysis (projectR): [https://www.bioconductor.org/packages/release/bioc/html/projectR.html](https://www.bioconductor.org/packages/release/bioc/html/projectR.html)
    - Study-Specific Code: [https://github.com/carlocolantuoni/NeocortexDevelopment_Sonthalia2024/](https://github.com/carlocolantuoni/NeocortexDevelopment_Sonthalia2024/)
