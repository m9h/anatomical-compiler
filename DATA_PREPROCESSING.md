# Data Preprocessing Plan: Fleck et al. → hgx Modeling

## 1. Available Data (on DGX Spark)

All processed data from [Fleck et al. 2023](https://doi.org/10.1038/s41586-022-05279-8), downloaded from [Zenodo](https://doi.org/10.5281/zenodo.5242913).

| File | Size | Cells | Genes | Key Contents |
|------|------|-------|-------|-------------|
| `data_matrices/counts.mtx.gz` | 480 MB | 34,088 | 33,538 | Sparse count matrix (MatrixMarket), 155M non-zeros (13.6% density) |
| `data_matrices/meta.tsv.gz` | 1.9 MB | 34,088 | — | `velocity_pseudotime` [0,1], `lineage` (telencephalon/early/nt/other), `stage_manual` (iPSC→nect_nepi→npc→neuron), `pt_bin`, cell cycle scores |
| `RNA_all_velo.h5ad` | 5.7 GB | 34,088 | 32,693 | Spliced + unspliced layers, fate probabilities (`DF`/`MH`/`VF` = Dorsal Forebrain / Medial-Hindbrain / Ventral Forebrain), regional probabilities (12 brain regions), UMAP (`X_umap`), `IsTF` annotation (1,631 TFs) |
| `RNA_data.h5ad` | 920 MB | 49,718 | 33,538 | Full scRNA-seq (superset of integrated cells), `age` (days 4–61), `nowakowski_prediction` (IPC/RG/EN/IN), 4 cell lines |
| `grn_modules.tsv` | 11.6 MB | — | 2,792 | Pando GRN: 74,448 TF→target edges, 720 TFs, columns: tf, target, estimate, padj, peaks |
| `seurat_objects.tar.gz` | 17 GB | — | — | Full Seurat R objects (RNA+ATAC integration, module assignments, CROP-seq results) |
| `motif2tf.tsv` | 62 KB | — | — | 576 motif→TF mappings with similarity scores |

### Cell Populations

| Lineage | Count | Description |
|---------|-------|-------------|
| telencephalon | 22,640 | Forebrain (cortical + GE) — primary population |
| early | 7,378 | Progenitors (iPSC, neuroectoderm, neuroepithelium) |
| nt | 3,843 | Neural tube / non-telencephalic |
| other | 227 | Unassigned |

### Developmental Stages (from metadata)

| Stage | Count | Pseudotime Range |
|-------|-------|-----------------|
| iPSC | 3,409 | Early |
| nect_nepi | 10,227 | Neuroectoderm → neuroepithelium |
| npc | 17,044 | Neural progenitor cells |
| neuron | 3,408 | Differentiated neurons |

### Fate Probabilities (from RNA_all_velo.h5ad)

Three CellRank-derived fate probabilities per cell:
- **DF** (Dorsal Forebrain): cortical excitatory neuron fate → maps to `to_ctx`
- **VF** (Ventral Forebrain): ganglionic eminence interneuron fate → maps to `to_ge`
- **MH** (Medial/Hindbrain): non-telencephalic fate → maps to `to_nt`

All range [0,1] and approximately sum to 1.

---

## 2. Preprocessing Steps

### Step 1: Build the Gene Universe

The GRN and expression data use slightly different gene sets. We need a common universe.

```
GRN genes:          2,792 (720 TFs + 2,535 targets, some overlap)
Velocity h5ad genes: 32,693
Count matrix genes:  33,538
```

**Action:** Intersect GRN genes with expression genes. Expected ~2,700 genes in common. This is the modeling gene set — large enough for biological signal, small enough for GPU.

### Step 2: Normalize Expression

The count matrix is raw UMI counts. For hgx node features we need normalized, scaled expression.

1. **Library-size normalization**: Divide each cell's counts by total UMI, multiply by 10,000 (CPM-like)
2. **Log-transform**: `log1p(normalized_counts)`
3. **Gene scaling**: Z-score per gene across cells (zero mean, unit variance)
4. **No HVG filtering**: We keep all GRN genes, not just highly variable ones — the GRN defines our feature set

Tool: `scanpy.pp.normalize_total()`, `scanpy.pp.log1p()`, `scanpy.pp.scale()`

### Step 3: Build Pseudotime-Binned Expression for Temporal Dynamics

For Neural ODE/SDE training, we need expression trajectories over pseudotime.

1. Sort cells by `velocity_pseudotime` from metadata
2. Bin into `T` equal-width bins (T=10 for initial, T=20 for final)
3. For each bin, compute **mean expression** of GRN genes across cells in that bin
4. Result: `(T, n_genes)` matrix — each row is a pseudotime snapshot

**Also compute per-bin:**
- Lineage fractions: proportion of telencephalon/early/nt cells
- Mean fate probabilities: average DF/MH/VF
- Cell count per bin (for weighting)

This gives us the temporal signal that Neural ODE fits to.

### Step 4: Build the Regulatory Hypergraph

Load the Pando GRN into hgx:

```python
hg = hgx.load_pando_modules(
    coef_csv="grn_modules.tsv",
    modules_csv=None,       # group by TF regulon
    padj_threshold=0.05,
)
# Result: 2,792 nodes, 720 hyperedges (one per TF regulon)
```

**Enrich node features** beyond degree vectors:
- Replace 1-dim degree features with pseudotime-binned expression (T-dim per gene)
- Or use the mean expression across all cells as a static feature vector
- Or use PCA of the expression matrix (top 16–32 PCs) as features

**Edge features** from GRN coefficients:
- Mean absolute estimate per regulon
- Regulon size (number of targets)
- Mean correlation (from `corr` column)

### Step 5: Build Temporal Hypergraphs for ODE/SDE

```python
temporal_hgs = hgx.grn_to_temporal_hypergraphs(
    expression_matrix,    # (n_cells, n_genes) — subsetted to GRN genes
    time_labels,          # (n_cells,) — pt_bin or discretized velocity_pseudotime
    incidence,            # (n_genes, n_edges) — from Pando GRN
    num_timepoints=10,
)
temp_hg = hgx.from_snapshots(temporal_hgs, times=pseudotime_centers)
```

Node features at each timepoint = mean expression of GRN genes in that pseudotime bin.
Topology is shared across timepoints (same regulatory wiring).

### Step 6: Prepare Perturbation Data

For the PerturbationPredictor, we need:

**Training data (from CROP-seq or simulated):**
- Perturbation masks: `Bool[Array, "P n"]` — which gene is knocked out
- Expression targets: `Float[Array, "P n d"]` — observed expression change
- Fate targets: `Float[Array, "P 3"]` — observed fate probability shift (DF/MH/VF)

**Sources:**
1. **Seurat objects** may contain CROP-seq differential expression (GLI3, TBR1, EOMES KOs) — extract from `seurat_objects.tar.gz`
2. **Simulated from GRN structure**: zero out a TF's expression, propagate through incidence matrix to predict downstream effects

**GLI3 KO validation data** (from E-MTAB-11997/E-MTAB-12002):
- GLI3 knockout organoids were profiled separately
- Compare wildtype vs GLI3-KO expression to get ground truth expression changes
- Compare fate probability distributions (DF/MH/VF) between conditions

### Step 7: Build Fate-Specific Subhypergraphs for Topology

For persistent homology comparison across fates:

1. Classify each TF regulon as fate-associated based on target gene expression:
   - **DF-associated** (cortical): regulons where targets have mean DF > 0.5
   - **VF-associated** (GE): regulons where targets have mean VF > 0.5
   - **MH-associated** (neural tube): regulons where targets have mean MH > 0.5
2. Build subhypergraphs by selecting relevant columns from the incidence matrix
3. Compute persistence diagrams for each

### Step 8: Prepare Cross-Species Comparison

hgx built-in datasets are ready to use:
- `hgx.load_cell_lineage()` — C. elegans 3-uniform hypergraph
- `hgx.load_devograph()` — C. elegans 3D developmental tracking

For the organoid side, the preprocessed temporal hypergraph from Step 5 serves as the comparison object.

---

## 3. Output Files (preprocessing → modeling)

| Output | Shape | Used By |
|--------|-------|---------|
| `grn_hypergraph.pkl` | Hypergraph: 2,792 nodes × 720 edges | All scripts |
| `gene_names.json` | List of 2,792 gene names (node order) | All scripts |
| `tf_names.json` | List of 720 TF names (edge order) | Perturbation, centrality |
| `temporal_expression.npz` | `(T, n_genes)` pseudotime-binned means | Temporal ODE/SDE |
| `pseudotime_metadata.csv` | Per-bin: centers, lineage fracs, fate probs, cell counts | Temporal, topology |
| `node_features.npy` | `(n_genes, d)` static features (PCA or mean expr) | Module detection, conv comparison |
| `fate_probabilities.csv` | Per-cell DF/MH/VF from velocity h5ad | Perturbation, trajectory |
| `perturbation_masks.npz` | `(P, n_genes)` boolean KO masks | Perturbation predictor |
| `perturbation_targets.npz` | `(P, n_genes)` expression changes + `(P, 3)` fate shifts | Perturbation predictor |
| `fate_subgraph_indices.json` | Edge indices for DF/VF/MH subhypergraphs | Topology |

---

## 4. Key Decisions

### Feature Dimensionality

The GRN has 2,792 genes but hgx convolutions work with arbitrary feature dimensions. Options:

- **d=1**: Just use mean expression (simplest, but limited signal)
- **d=8–16**: PCA of the expression matrix projected onto GRN genes
- **d=T** (e.g. 10): Use the full pseudotime expression trajectory as the feature vector
- **d=32–64**: Learnable embedding (initialize with PCA, fine-tune during training)

Recommendation: Start with **d=16 (PCA)** for module detection and convolution comparison, **d=T** for temporal dynamics.

### Pseudotime Resolution

- T=10 for initial experiments (fast training, ~3,400 cells per bin)
- T=20 for publication figures (finer temporal resolution, ~1,700 cells per bin)
- T=50 for maximum resolution (but noisier per-bin estimates)

### Fate Mapping

The velocity h5ad uses DF/MH/VF while the paper discusses ctx/GE/NT:
- DF (Dorsal Forebrain) ≈ cortical (ctx)
- VF (Ventral Forebrain) ≈ ganglionic eminence (GE)
- MH (Medial/Hindbrain) ≈ neural tube (NT)

### Handling the 34K vs 50K Cell Discrepancy

- `RNA_data.h5ad`: 49,718 cells (full RNA dataset)
- `data_matrices/meta.tsv.gz` + `RNA_all_velo.h5ad`: 34,088 cells (RNA+ATAC integrated subset)

Use the **34,088 integrated cells** for all modeling — they have pseudotime, fate probabilities, and velocity layers. The additional 15,630 cells in `RNA_data.h5ad` lack these annotations.

---

## 5. Preprocessing Script Outline

```python
# scripts/00_preprocess.py
# Reads raw Zenodo files → outputs modeling-ready arrays

def main():
    # 1. Load GRN and build gene universe
    # 2. Load and normalize expression (34,088 cells × GRN genes)
    # 3. Bin by pseudotime → temporal expression matrix
    # 4. Extract fate probabilities (DF/MH/VF)
    # 5. Build hgx hypergraph with enriched node features
    # 6. Build temporal hypergraphs
    # 7. Prepare perturbation data (from Seurat or simulated)
    # 8. Classify fate-specific subhypergraphs
    # 9. Save all outputs
```

This single preprocessing script replaces the synthetic data generation in `01_prepare_data.py` and feeds real data into scripts 02–08 via `data_loader.py`.
