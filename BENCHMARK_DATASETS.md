# Hypergraph Benchmark Datasets for hgx Validation

> Comprehensive catalog of standard hypergraph benchmark datasets for validating the
> [hgx](https://github.com/HGX-Team/hypergraph-generation) library independently of biological data.
>
> Last updated: 2026-03-15

---

## Table of Contents

1. [Quick Reference Table](#quick-reference-table)
2. [Citation / Co-citation Hypergraphs](#1-citation--co-citation-hypergraphs)
3. [Co-authorship Hypergraphs](#2-co-authorship-hypergraphs)
4. [Visual Object Recognition](#3-visual-object-recognition)
5. [Heterophilic Hypergraph Benchmarks](#4-heterophilic-hypergraph-benchmarks)
6. [Social / E-commerce](#5-social--e-commerce)
7. [Other Standard Benchmarks](#6-other-standard-benchmarks)
8. [Fairness-Sensitive Benchmarks (DHG-Bench)](#7-fairness-sensitive-benchmarks-dhg-bench)
9. [Hypergraph-Level Classification Datasets](#8-hypergraph-level-classification-datasets)
10. [Published Baseline Accuracy Results](#published-baseline-accuracy-results)
11. [How to Load Each Dataset](#how-to-load-each-dataset)
12. [Key Benchmark Frameworks and Repositories](#key-benchmark-frameworks-and-repositories)
13. [Top 5 Recommended Datasets for hgx Validation](#top-5-recommended-datasets-for-hgx-validation)
14. [Sources](#sources)

---

## Quick Reference Table

| Dataset | Domain | Nodes | Hyperedges | Features | Classes | Homophily | Task | Homophilic? | Loading |
|---------|--------|------:|----------:|--------:|--------:|----------:|------|:-----------:|---------|
| Cora (co-citation) | Citation | 2,708 | 1,579 | 1,433 | 7 | 0.75 | Node cls. | Yes | DHG built-in |
| Citeseer (co-citation) | Citation | 3,312 | 1,079 | 3,703 | 6 | ~0.72 | Node cls. | Yes | DHG built-in |
| Pubmed (co-citation) | Citation | 19,717 | 7,963 | 500 | 3 | 0.78 | Node cls. | Yes | DHG built-in |
| Cora-CA (co-authorship) | Citation | 2,708 | 1,072 | 1,433 | 7 | 0.78 | Node cls. | Yes | DHG built-in |
| DBLP-CA (co-authorship) | Citation | 41,302 | 22,363 | 1,425 | 6 | 0.87 | Node cls. | Yes | DHG built-in |
| NTU2012 | 3D Visual | 2,012 | 2,012 | 100 | 67 | 0.79 | Node cls. | Yes | AllSet / DHG-Bench |
| ModelNet40 | 3D Visual | 12,311 | 12,311 | 100 | 40 | 0.87 | Node cls. | Yes | AllSet / DHG-Bench |
| Walmart-Trips | E-commerce | 88,860 | 69,906 | 100 | 11 | 0.60 | Node cls. | Borderline | Cornell ARB |
| House-Committees | Political | 1,290 | 341 | -- | 2 | ~0.40 | Node cls. | No | Cornell ARB |
| Senate-Committees | Political | 282 | ~315 | -- | 2 | ~0.35 | Node cls. | No | Cornell ARB |
| Congress-Bills | Political | ~1,718 | ~60,987 | -- | 2 | ~0.45 | Node cls. | No | Cornell ARB |
| Yelp | Reviews | 50,758 | 679,302 | 1,862 | 9 | 0.29 | Node cls. | No | AllSet / DHG-Bench |
| Trivago | Hotel clicks | 172,738 | 233,202 | 300 | 160 | 0.98 | Node cls. | Yes | DHG-Bench |
| Actor | Co-occurrence | 16,255 | 10,164 | 50 | 3 | 0.46 | Node cls. | No | DHG-Bench |
| Amazon-Ratings | E-commerce | 22,299 | 2,090 | 111 | 5 | 0.37 | Node cls. | No | DHG-Bench |
| Twitch-Gamers | Social | 16,812 | 2,627 | 7 | 2 | 0.49 | Node cls. | No | DHG-Bench |
| Pokec | Social | 14,998 | 2,406 | 65 | 2 | 0.45 | Node cls. | No | DHG-Bench |
| Zoo | Biological | 101 | 42 | 16 | 7 | -- | Node cls. | -- | AllSet |
| Mushroom | Biological | 8,124 | 298 | -- | 2 | -- | Node cls. | -- | AllSet |
| 20Newsgroups | Text | 16,242 | 100 | -- | 7 | -- | Node cls. | -- | AllSet |
| Cooking200 | Recipe | ~5,000 | ~200 | -- | ~10 | -- | Node cls. | -- | DHG built-in |

---

## 1. Citation / Co-citation Hypergraphs

In co-citation hypergraphs, a hyperedge connects all documents cited together by a single paper (or author). These are the most widely used benchmarks in hypergraph neural network research.

### Cora (Co-citation)

- **Construction**: All documents cited by the same paper form a hyperedge.
- **Nodes**: 2,708 (papers) | **Hyperedges**: 1,579 | **Features**: 1,433 (BoW) | **Classes**: 7
- **Avg hyperedge size**: 3.03 | **Homophily**: 0.75
- **Task**: Semi-supervised node classification (paper topic prediction)
- **Source**: Originally from [Sen et al. (2008)](https://linqs.org/datasets/#cora-orig); hypergraph version from [Yadati et al. (2019) HyperGCN](https://github.com/malllabiisc/HyperGCN)
- **Download**: `dhg.data.CocitationCora()` auto-downloads; also in [AllSet repo](https://github.com/jianhao2016/AllSet) `data/` folder

### Citeseer (Co-citation)

- **Construction**: Same as Cora -- documents co-cited together form hyperedges.
- **Nodes**: 3,312 | **Hyperedges**: 1,079 | **Features**: 3,703 (BoW) | **Classes**: 6
- **Avg hyperedge size**: 3.2 +/- 2.0 | **Homophily**: ~0.72
- **Task**: Node classification
- **Source**: [Giles et al. (1998)](https://citeseerx.ist.psu.edu/); hypergraph version from HyperGCN
- **Download**: `dhg.data.CocitationCiteseer()` auto-downloads; also in AllSet repo

### Pubmed (Co-citation)

- **Construction**: All documents cited by an author form one hyperedge.
- **Nodes**: 19,717 (papers) | **Hyperedges**: 7,963 | **Features**: 500 (TF-IDF) | **Classes**: 3 (diabetes subtypes)
- **Avg hyperedge size**: 4.35 | **Homophily**: 0.78
- **Task**: Node classification
- **Source**: [PubMed Diabetes dataset (Sen et al.)](https://linqs.org/datasets/#pubmed-diabetes); hypergraph from HyperGCN / AllSet
- **Download**: `dhg.data.CocitationPubmed()` auto-downloads

---

## 2. Co-authorship Hypergraphs

In co-authorship hypergraphs, all papers written by the same author (or equivalently, all co-authors of a single paper) form a hyperedge.

### Cora-CA (Co-authorship)

- **Construction**: All documents co-authored by the same person form a hyperedge.
- **Nodes**: 2,708 | **Hyperedges**: 1,072 | **Features**: 1,433 (BoW) | **Classes**: 7
- **Avg hyperedge size**: 4.28 | **Max hyperedge size**: 43 | **Homophily**: 0.78
- **Task**: Node classification
- **Source**: Derived from Cora citation network
- **Download**: `dhg.data.CoauthorshipCora()` auto-downloads

### DBLP-CA (Co-authorship)

- **Construction**: All publications by the same author form a hyperedge.
- **Nodes**: 41,302 (papers) | **Hyperedges**: 22,363 | **Features**: 1,425 | **Classes**: 6
- **Avg hyperedge size**: 4.45 | **Homophily**: 0.87
- **Task**: Node classification
- **Note**: Largest of the standard citation benchmarks; good for stress-testing.
- **Source**: [DBLP](https://dblp.org/)
- **Download**: `dhg.data.CoauthorshipDBLP()` auto-downloads

---

## 3. Visual Object Recognition

These datasets originate from 3D shape repositories where objects are represented as hypergraphs (nodes = 3D objects, hyperedges connect objects sharing local geometric features via k-NN on MVCNN features).

### ModelNet40

- **Nodes**: 12,311 (3D objects) | **Hyperedges**: 12,311 | **Features**: 100 (MVCNN) | **Classes**: 40
- **Avg hyperedge size**: 5.0 (fixed k-NN) | **Homophily**: 0.87
- **Task**: 3D object classification (node classification on hypergraph)
- **Source**: [Princeton 3D ShapeNets](https://3dshapenets.cs.princeton.edu/) -- originally 12,311 CAD models across 40 categories
- **Download**: Available via AllSet repo data folder; also in DHG-Bench
- **Note**: Hypergraph constructed by treating each object as a node and building hyperedges from MVCNN feature k-NN neighborhoods.

### NTU2012

- **Nodes**: 2,012 (3D shapes) | **Hyperedges**: 2,012 | **Features**: 100 (MVCNN) | **Classes**: 67
- **Avg hyperedge size**: 5.0 (fixed k-NN) | **Homophily**: 0.79
- **Task**: 3D object classification
- **Source**: [NTU 3D Model Database](http://3d.csie.ntu.edu.tw/)
- **Download**: Available via AllSet repo data folder; also in DHG-Bench
- **Note**: Many fine-grained classes (67) make this a challenging multi-class problem.

### Princeton Shape Benchmark (PSB)

- **Description**: 1,814 polygonal 3D models for shape retrieval and classification
- **Source**: [Princeton Shape Benchmark](https://shape.cs.princeton.edu/benchmark/)
- **Note**: Less commonly used in hypergraph neural network papers than ModelNet40 and NTU2012, but available for hypergraph construction via k-NN on shape descriptors.

---

## 4. Heterophilic Hypergraph Benchmarks

These datasets exhibit low homophily -- nodes connected by the same hyperedge often have different labels. They are critical for testing whether HNN methods can handle non-homophilic structure.

### House-Committees

- **Nodes**: 1,290 (US House members) | **Hyperedges**: 341 (committees)
- **Features**: None (or derived) | **Classes**: 2 (Republican / Democrat)
- **Avg hyperedge size**: ~34.8 | **Homophily**: ~0.40 (heterophilic)
- **Task**: Node classification (party affiliation)
- **Source**: [Austin R. Benson, Cornell](https://www.cs.cornell.edu/~arb/data/house-committees/)
- **Download**: Manual download from Cornell ARB data page; also in AllSet and xgi-data

### Senate-Committees

- **Nodes**: ~282 (US Senators) | **Hyperedges**: ~315 (committees)
- **Features**: None (or derived) | **Classes**: 2 (Republican / Democrat)
- **Homophily**: ~0.35 (heterophilic)
- **Task**: Node classification (party affiliation)
- **Source**: [Austin R. Benson, Cornell](https://www.cs.cornell.edu/~arb/data/senate-committees/)
- **Download**: Manual download from Cornell ARB data page; also in xgi-data

### Congress-Bills

- **Nodes**: ~1,718 (US Congresspersons) | **Hyperedges**: ~60,987 (bills)
- **Features**: None (or derived) | **Classes**: 2 (Republican / Democrat)
- **Construction**: Sponsor + co-sponsors of each bill form a hyperedge. Timestamped.
- **Homophily**: ~0.45 (heterophilic)
- **Task**: Node classification (party affiliation)
- **Source**: [Austin R. Benson, Cornell](https://www.cs.cornell.edu/~arb/data/congress-bills/)
- **Download**: Manual download from Cornell ARB data page

### Actor (Co-occurrence)

- **Nodes**: 16,255 (actors) | **Hyperedges**: 10,164 | **Features**: 50 | **Classes**: 3
- **Homophily**: 0.46 (heterophilic)
- **Task**: Node classification
- **Source**: Derived from movie-actor-director-writer network
- **Download**: DHG-Bench (`pip install dhg-bench`)
- **Note**: Most HNNs underperform simple MLP (86.06%) on this dataset, making it a strong test of heterophilic capability.

---

## 5. Social / E-commerce

### Yelp (Restaurant Reviews)

- **Nodes**: 50,758 (businesses) | **Hyperedges**: 679,302 | **Features**: 1,862 | **Classes**: 9
- **Avg hyperedge size**: 6.66 | **Homophily**: 0.29 (strongly heterophilic)
- **Task**: Node classification (business category or star-rating bucket)
- **Source**: [Yelp Open Dataset](https://www.yelp.com/dataset/); hypergraph version from AllSet
- **Download**: AllSet repo data folder; also in DHG-Bench
- **Note**: Largest standard heterophilic hypergraph benchmark. Good stress-test for scalability.

### Amazon-Ratings

- **Nodes**: 22,299 (products) | **Hyperedges**: 2,090 | **Features**: 111 | **Classes**: 5
- **Avg hyperedge size**: 3.10 | **Homophily**: 0.37 (heterophilic)
- **Task**: Node classification (predict average product rating)
- **Source**: [SNAP Amazon co-purchasing metadata](https://snap.stanford.edu/data/); hypergraph version from DHG-Bench
- **Download**: DHG-Bench

### Walmart-Trips

- **Nodes**: 88,860 (products) | **Hyperedges**: 69,906 (shopping trips)
- **Features**: 100 | **Classes**: 11 (product departments)
- **Avg hyperedge size**: 6.59 | **Homophily**: 0.60 (borderline)
- **Task**: Node classification (department prediction)
- **Source**: [Austin R. Benson, Cornell](https://www.cs.cornell.edu/~arb/data/walmart-trips/) -- originally from Kaggle competition
- **Download**: Cornell ARB data page; also in AllSet and DHG-Bench

### Trivago (Hotel Clicks)

- **Nodes**: 172,738 (hotels) | **Hyperedges**: 233,202 (browsing sessions)
- **Features**: 300 | **Classes**: 160 (countries)
- **Avg hyperedge size**: 3.12 | **Homophily**: 0.98 (strongly homophilic)
- **Task**: Node classification (country prediction)
- **Source**: ACM RecSys Challenge 2019; [Cornell ARB](https://www.cs.cornell.edu/~arb/data/trivago-clicks/)
- **Download**: DHG-Bench
- **Note**: Very large-scale benchmark; 160 classes make this challenging.

### Twitch-Gamers

- **Nodes**: 16,812 | **Hyperedges**: 2,627 | **Features**: 7 | **Classes**: 2
- **Homophily**: 0.49 (heterophilic)
- **Task**: Node classification
- **Download**: DHG-Bench

### Pokec

- **Nodes**: 14,998 | **Hyperedges**: 2,406 | **Features**: 65 | **Classes**: 2
- **Homophily**: 0.45 (heterophilic)
- **Task**: Node classification
- **Download**: DHG-Bench

---

## 6. Other Standard Benchmarks

### Zoo (UCI)

- **Nodes**: 101 (animals) | **Hyperedges**: 42 (attributes as hyperedges)
- **Features**: 16 | **Classes**: 7 (animal types)
- **Avg hyperedge size**: 39.93 | **Max hyperedge size**: 93
- **Task**: Node classification
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/111/zoo)
- **Download**: AllSet repo data folder
- **Note**: Tiny dataset -- useful for smoke-testing, not for serious benchmarking.

### Mushroom (UCI)

- **Nodes**: 8,124 (mushrooms) | **Hyperedges**: 298 (categorical attributes as hyperedges)
- **Features**: 22 categorical attributes | **Classes**: 2 (edible / poisonous)
- **Task**: Node classification
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/73/mushroom)
- **Download**: AllSet repo data folder

### 20Newsgroups

- **Nodes**: 16,242 (documents) | **Hyperedges**: 100 (topic clusters)
- **Features**: TF-IDF | **Classes**: 7 (coarsened newsgroup categories)
- **Task**: Node classification (topic prediction)
- **Source**: [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/); hypergraph version from AllSet
- **Download**: AllSet repo data folder
- **Note**: 9.6% minimum error rate due to 10,267/16,242 unique data points.

### Cooking200 (DHG built-in)

- **Nodes**: ~5,000 (dishes) | **Hyperedges**: ~200 (ingredients)
- **Features**: -- | **Classes**: ~10 cuisines (Chinese, Japanese, French, Russian, etc.)
- **Task**: Node classification (cuisine prediction)
- **Source**: [Yummly.com](https://www.yummly.com/)
- **Download**: `dhg.data.Cooking200()` auto-downloads

---

## 7. Fairness-Sensitive Benchmarks (DHG-Bench)

These datasets are designed for evaluating fairness in hypergraph learning, with protected demographic attributes.

| Dataset | Nodes | Hyperedges | Features | Sensitive Attr. | Task |
|---------|------:|----------:|--------:|-----------------|------|
| German | 1,000 | 1,000 | 27 | Gender | Credit risk classification |
| Bail | 18,876 | 18,876 | 18 | Race | Bail decision prediction |
| Credit | 30,000 | 30,000 | 13 | Age | Default prediction |

- **Source**: DHG-Bench ([GitHub](https://github.com/Coco-Hut/DHG-Bench))
- **Download**: `pip install dhg-bench`

---

## 8. Hypergraph-Level Classification Datasets

For graph-level (hypergraph-level) classification tasks rather than node-level tasks.

| Dataset | Num Hypergraphs | Avg Nodes | Avg Hyperedges | Avg Edge Size | Classes |
|---------|----------------:|----------:|--------------:|--------------:|--------:|
| RHG-10 | 2,000 | 31.3 | 29.8 | 5.2 | 10 |
| RHG-3 | 1,500 | 35.5 | 17.9 | 6.9 | 3 |
| IMDB-Dir-Form | 1,869 | 15.7 | 39.2 | 3.7 | 3 |
| IMDB-Dir-Genre | 3,393 | 17.3 | 36.4 | 3.8 | 3 |
| Steam-Player | 2,048 | 13.8 | 46.4 | 4.5 | 2 |
| Twitter-Friend | 1,310 | 21.6 | 84.3 | 4.3 | 2 |

- **Source**: DHG-Bench
- **Download**: `pip install dhg-bench`

---

## Published Baseline Accuracy Results

### Node Classification on Citation/Co-citation Benchmarks

Accuracy (%) reported from key papers. Results are mean +/- std across multiple splits.

| Method | Cora | Citeseer | Pubmed | Cora-CA | DBLP-CA |
|--------|-----:|--------:|-------:|-------:|--------:|
| MLP | 75.33+/-0.88 | -- | -- | -- | -- |
| HGNN (AAAI 2019) | 79.39+/-1.36 | ~72.5 | ~80.1 | ~82.6 | ~90.5 |
| HyperGCN (NeurIPS 2019) | 78.45+/-1.26 | ~71.8 | ~82.8 | ~79.5 | ~89.0 |
| HNHN (ICML-W 2020) | 76.36+/-1.92 | -- | -- | -- | -- |
| UniGCNII (IJCAI 2021) | 78.81+/-1.05 | ~73.1 | ~80.5 | ~84.1 | ~91.2 |
| AllDeepSets (ICLR 2022) | 76.88+/-1.80 | -- | -- | -- | -- |
| AllSetTransformer (ICLR 2022) | 78.58+/-1.47 | -- | -- | -- | -- |
| ED-HNN (ICLR 2023) | 80.31+/-1.35 | -- | -- | -- | -- |
| TF-HNN (DHG-Bench 2025) | 79.47+/-1.31 | -- | -- | -- | -- |

### Node Classification on Visual Benchmarks

| Method | NTU2012 | ModelNet40 |
|--------|--------:|-----------:|
| HGNN | ~87.4 | ~96.7 |
| UniGCNII | 73.05+/-2.21 | 78.81+/-1.05 |
| THNN | ~94.25 | ~78.55 |
| WCRW-MLP (2025) | 91.25 | 98.90 |
| UniG-Encoder | -- | 98.87 |

### Node Classification on Heterophilic Benchmarks

| Method | House | Senate | Walmart | Yelp | Actor |
|--------|------:|-------:|--------:|-----:|------:|
| MLP | ~67 | ~52 | -- | -- | 86.06+/-0.36 |
| HGNN | ~65 | ~50 | -- | -- | -- |
| AllSet | ~70 | ~55 | ~46 | ~32 | -- |
| HyperND | -- | -- | -- | -- | ~84 |
| ED-HNN | ~72 | ~57 | -- | -- | -- |

> **Note**: Exact accuracy values vary by paper, split strategy, and hyperparameter tuning.
> Consult the original papers and DHG-Bench for canonical results with standardized splits.

---

## How to Load Each Dataset

### Method 1: DeepHypergraph (DHG) -- Easiest for citation / co-authorship

```python
# Install: uv pip install dhg
import dhg

# Co-citation hypergraphs
cora_cc = dhg.data.CocitationCora()
citeseer_cc = dhg.data.CocitationCiteseer()
pubmed_cc = dhg.data.CocitationPubmed()

# Co-authorship hypergraphs
cora_ca = dhg.data.CoauthorshipCora()
dblp_ca = dhg.data.CoauthorshipDBLP()

# Recipe hypergraph
cooking = dhg.data.Cooking200()

# Access data
print(cora_cc)  # Shows num_vertices, num_edges, etc.
X = cora_cc["features"]      # Node feature matrix
y = cora_cc["labels"]         # Node labels
hg = cora_cc["edge_list"]     # List of hyperedges (each is a tuple of node indices)
```

### Method 2: DHG-Bench -- Most comprehensive (22 datasets, 17 methods)

```python
# Install: uv pip install dhg-bench
# GitHub: https://github.com/Coco-Hut/DHG-Bench
# Supports: Cora, Pubmed, Cora-CA, DBLP-CA, NTU2012, ModelNet40,
#           Walmart, Trivago, Actor, Amazon-Ratings, Twitch-Gamers,
#           Pokec, Yelp, German, Bail, Credit, and graph-level datasets
```

### Method 3: AllSet Repository -- Original benchmark suite

```bash
# Clone: git clone https://github.com/jianhao2016/AllSet.git
# Data in: AllSet/data/pyg_data/hypergraph_dataset_updated/
# Raw data: AllSet/data/AllSet_all_raw_data/
# Covers: Cora, Citeseer, Pubmed, Cora-CA, DBLP-CA, Zoo, 20News,
#          Mushroom, NTU2012, ModelNet40, Yelp, House, Walmart
```

### Method 4: Cornell ARB Data -- Heterophilic / temporal hypergraphs

```bash
# Download from: https://www.cs.cornell.edu/~arb/data/
# Available datasets: house-committees, senate-committees, senate-bills,
#                     congress-bills, walmart-trips, trivago-clicks
# Format: Timestamped simplices / hyperedges in text files
```

### Method 5: XGI (Comple(X) Group Interactions) -- Standardized JSON format

```python
# Install: uv pip install xgi
import xgi

# Load from xgi-data repository (44 datasets on Zenodo)
# Uses Hypergraph Interchange Format (HIF) JSON
H = xgi.load_xgi_data("senate-committees")
print(H.num_nodes, H.num_edges)
```

### Method 6: PyTorch Geometric -- For GNN integration

```python
# Install: uv pip install torch-geometric
from torch_geometric.datasets import CornellTemporalHyperGraphDataset

# Temporal hypergraph datasets
dataset = CornellTemporalHyperGraphDataset(root='./data', name='congress-bills')
```

---

## Key Benchmark Frameworks and Repositories

| Framework | URL | Datasets | Methods | Notes |
|-----------|-----|---------|---------|-------|
| **DHG-Bench** | [GitHub](https://github.com/Coco-Hut/DHG-Bench) | 22 | 17 | Most comprehensive; PyTorch + PyG based |
| **DeepHypergraph (DHG)** | [GitHub](https://github.com/iMoonLab/DeepHypergraph) | 20+ | Multiple | Auto-download; `pip install dhg` |
| **AllSet** | [GitHub](https://github.com/jianhao2016/AllSet) | 13 | 10+ | ICLR 2022 benchmark suite |
| **XGI / xgi-data** | [GitHub](https://github.com/xgi-org/xgi-data) | 44 | -- | Standardized HIF JSON; structural analysis |
| **Cornell ARB Data** | [Website](https://www.cs.cornell.edu/~arb/data/) | 10+ | -- | Raw temporal hypergraph data |
| **Awesome-Hypergraph-Network** | [GitHub](https://github.com/gzcsudo/Awesome-Hypergraph-Network) | -- | -- | Curated paper list + dataset/tool refs |
| **ED-HNN** | [GitHub](https://github.com/Graph-COM/ED-HNN) | 9 | 10+ | ICLR 2023; equivariant diffusion |
| **PhenomNN** | [GitHub](https://github.com/yxzwang/PhenomNN) | 10+ | 10+ | ICML 2023; energy-based HNNs |

---

## Top 5 Recommended Datasets for hgx Validation

Based on diversity, availability, published baselines, and size range:

### 1. Cora (Co-citation) -- Small, Homophilic Citation Benchmark

- **Why**: The single most widely benchmarked hypergraph dataset. Every HNN paper reports results on Cora, providing the richest set of published baselines for comparison.
- **Size**: 2,708 nodes / 1,579 hyperedges (small -- fast iteration)
- **Loading**: One-line with DHG: `dhg.data.CocitationCora()`
- **Baselines**: HGNN ~79.4%, UniGCNII ~78.8%, ED-HNN ~80.3%
- **Validates**: Basic hypergraph construction, spectral properties, community structure

### 2. DBLP-CA (Co-authorship) -- Medium, Homophilic Citation Benchmark

- **Why**: At 41K nodes, this is the largest standard citation hypergraph. Tests scalability beyond toy datasets while maintaining well-understood structure with strong published baselines.
- **Size**: 41,302 nodes / 22,363 hyperedges (medium)
- **Loading**: `dhg.data.CoauthorshipDBLP()`
- **Baselines**: HGNN ~90.5%, UniGCNII ~91.2%
- **Validates**: Scalability of hgx hypergraph operations, large incidence matrix handling

### 3. House-Committees -- Small, Heterophilic Political Benchmark

- **Why**: Strongly heterophilic (homophily ~0.40) with large, overlapping hyperedges (avg size ~35). Tests whether hgx correctly handles heterophilic structure where standard assumptions break down.
- **Size**: 1,290 nodes / 341 hyperedges (small)
- **Loading**: Cornell ARB data page or XGI: `xgi.load_xgi_data("house-committees")`
- **Baselines**: AllSet ~70%, ED-HNN ~72%, MLP ~67%
- **Validates**: Heterophilic hypergraph properties, large hyperedge handling

### 4. ModelNet40 -- Medium, Visual Object Recognition

- **Why**: From an entirely different domain (3D shapes), providing diversity. The k-NN-based hypergraph construction is fundamentally different from citation/social hypergraphs.
- **Size**: 12,311 nodes / 12,311 hyperedges (medium)
- **Loading**: AllSet data folder or DHG-Bench
- **Baselines**: WCRW-MLP 98.9%, UniG-Encoder 98.87%, HGNN ~96.7%
- **Validates**: Uniform hyperedge sizes, geometric hypergraph structure

### 5. Yelp -- Large, Heterophilic Social/E-commerce Benchmark

- **Why**: The largest standard heterophilic benchmark (679K hyperedges). Tests both scalability and heterophilic handling simultaneously. The strongly heterophilic nature (homophily 0.29) makes it a challenging stress-test.
- **Size**: 50,758 nodes / 679,302 hyperedges (large)
- **Loading**: AllSet data folder or DHG-Bench
- **Baselines**: AllSet improved over prior methods by ~4%
- **Validates**: Large-scale sparse hypergraph operations, memory efficiency, heterophilic structure

### Summary of Top 5

| Priority | Dataset | Size | Type | Homophilic? | Key Validation Goal |
|:--------:|---------|------|------|:-----------:|---------------------|
| 1 | Cora (co-citation) | Small | Citation | Yes | Baseline correctness |
| 2 | DBLP-CA | Medium | Citation | Yes | Scalability |
| 3 | House-Committees | Small | Political | No | Heterophily handling |
| 4 | ModelNet40 | Medium | 3D Visual | Yes | Domain diversity |
| 5 | Yelp | Large | Reviews | No | Stress-test at scale |

---

## Sources

- [AllSet: You are AllSet (ICLR 2022)](https://openreview.net/forum?id=hpBTIv2uy_E) -- [GitHub](https://github.com/jianhao2016/AllSet)
- [UniGNN: Unified Framework for Graph and Hypergraph NNs (IJCAI 2021)](https://arxiv.org/abs/2105.00956)
- [HGNN: Hypergraph Neural Networks (AAAI 2019)](https://github.com/iMoonLab/HGNN)
- [HyperGCN (NeurIPS 2019)](https://ar5iv.labs.arxiv.org/html/1809.02589)
- [ED-HNN: Equivariant Hypergraph Diffusion (ICLR 2023)](https://github.com/Graph-COM/ED-HNN)
- [PhenomNN: Hypergraph Energy Functions to HNNs (ICML 2023)](https://proceedings.mlr.press/v202/wang23d/wang23d.pdf)
- [THNN: Tensorized Hypergraph Neural Networks (SDM 2024)](https://arxiv.org/abs/2306.02560)
- [DHG-Bench: Comprehensive Benchmark for Deep Hypergraph Learning (2025)](https://github.com/Coco-Hut/DHG-Bench) -- [Paper](https://arxiv.org/abs/2508.12244)
- [DeepHypergraph (DHG) Library](https://deephypergraph.readthedocs.io/) -- [GitHub](https://github.com/iMoonLab/DeepHypergraph)
- [When Hypergraph Meets Heterophily (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34022)
- [Austin R. Benson, Cornell -- Higher-Order Datasets](https://www.cs.cornell.edu/~arb/data/)
- [XGI: Comple(X) Group Interactions](https://xgi.readthedocs.io/) -- [xgi-data](https://github.com/xgi-org/xgi-data)
- [Awesome-Hypergraph-Network](https://github.com/gzcsudo/Awesome-Hypergraph-Network)
- [Hypergraph Representation Learning with WCRW (Entropy 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12562878/)
- [Sheaf Hypergraph Networks (2023)](https://arxiv.org/abs/2309.17116)
- [UCI ML Repository -- Zoo](https://archive.ics.uci.edu/dataset/111/zoo)
- [UCI ML Repository -- Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)
- [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)
- [Princeton Shape Benchmark](https://shape.cs.princeton.edu/benchmark/)
- [Yelp Open Dataset](https://www.yelp.com/dataset/)
- [PyTorch Geometric Datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)
