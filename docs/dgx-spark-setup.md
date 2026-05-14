# DGX Spark — real-mode foundation-model execution

*Setup notes for running [`scripts/fm_embed.py`](../scripts/fm_embed.py), [`scripts/fm_edges_seq.py`](../scripts/fm_edges_seq.py), and [`scripts/ablate_edge_priors.py`](../scripts/ablate_edge_priors.py) in **real mode** on the Biopunk DGX Spark (128 GB GPU). Written 2026-05-13.*

The driver: [`scripts/run_fm_real_dgx.sh`](../scripts/run_fm_real_dgx.sh). One command runs all three node-feature models (UCE, Geneformer, scGPT) plus all three edge-prior models (motif, Evo, Borzoi) plus scGPT in-silico KD plus the two ablations (edge priors + perturbation EIG), sequentially, with per-step logs.

---

## 1. Why DGX Spark, not the laptop

| model | params | FP16 GPU peak | fits on |
|---|---|---|---|
| Geneformer       | 95 M – 1.2 B | 2–8 GB    | laptop / DGX |
| scGPT            | ~50 M        | 1–2 GB    | laptop / DGX |
| UCE              | ~150 M       | 4–8 GB    | laptop / DGX (1280-d encoder forward pass) |
| Evo 1 (131k ctx) | 7 B          | 28–32 GB  | **DGX Spark** |
| Evo 2 (when out) | 7 B / 40 B   | 32–80 GB  | **DGX Spark** |
| Borzoi v1        | ~800 M       | 12–16 GB  | DGX preferred (long sequence context) |
| motif (PWM scan) | n/a          | CPU       | anywhere |

The bottleneck is **Evo** and the longer-context **Borzoi** runs. Geneformer / scGPT / UCE / motif all fit on a workstation GPU; if your laptop's GPU is ≥ 16 GB it can do all of them except the long-context Evo / Borzoi passes. The DGX Spark's 128 GB headroom means you can run them all serialised without paging, and the longer Evo 2 contexts (when released) drop in without re-architecting.

---

## 2. One-time install

Assuming a clean Ubuntu environment with `uv` and CUDA drivers in place:

```bash
# Project deps (existing):
cd ~/dev/anatomical-compiler
uv sync

# Foundation-model deps (real-mode only — stubs need nothing extra):
uv pip install \
    geneformer        \
    'evo-model >= 1.1' \
    borzoi-pytorch    \
    biopython         \
    transformers      \
    'torch >= 2.4'

# scGPT is an editable install from the lab's GitHub:
uv pip install 'git+https://github.com/bowang-lab/scGPT'

# UCE is also editable (CZI release):
uv pip install 'git+https://github.com/snap-stanford/UCE'
```

### Motif database (one-time, for `motif --mode real`)

```bash
# JASPAR 2024 vertebrate core (~150 MB):
mkdir -p /opt/jaspar
curl -L 'https://jaspar.genereg.net/download/data/2024/CORE/JASPAR2024_CORE_non-redundant_pfms_meme.txt' \
     -o /opt/jaspar/JASPAR2024_CORE_vertebrates_non-redundant.meme

# Set so fm_edges_seq.py finds it:
export JASPAR_MEME=/opt/jaspar/JASPAR2024_CORE_vertebrates_non-redundant.meme
echo 'export JASPAR_MEME=/opt/jaspar/JASPAR2024_CORE_vertebrates_non-redundant.meme' >> ~/.bashrc
```

### Pre-fetched model weights (~50 GB total)

Run once to populate the HuggingFace cache before going offline:

```bash
huggingface-cli download ctheodoris/Geneformer
huggingface-cli download subercui/scGPT
huggingface-cli download chanzuckerberg/uce
huggingface-cli download togethercomputer/evo-1-131k-base
huggingface-cli download calico/borzoi-v1
```

If pulling Evo 2 when released, swap the Evo handle in [`scripts/fm_edges_seq.py`](../scripts/fm_edges_seq.py)'s `MODELS["evo"]["checkpoint"]` and re-fetch.

---

## 3. Inputs the real-mode pipeline expects

The driver takes three files on the command line:

| arg | what | how to produce |
|---|---|---|
| `<h5ad>` | a single-cell expression matrix | `scanpy.read_h5ad` compatible; the project's Pollen brain-organoid h5ad fits, or any Biopunk wet-lab capstone output |
| `<edges.csv>` | candidate regulome edges | `tf,target` columns; produced by Pando's output table, or by `scripts/02_pando_import.py` |
| `<promoters.fa>` | FASTA of promoter sequences, keyed by gene symbol | `bedtools getfasta` on a ±2 kb window around each gene's TSS in your reference (hg38 / mm10) |
| `<tfs.txt>` *(optional)* | one TF symbol per line, the candidates for step 4 | typically the unique TFs in `edges.csv` first column; `awk -F, 'NR>1 {print $1}' edges.csv \| sort -u > tfs.txt` |

Promoter extraction one-liner (assumes hg38 + GENCODE GTF):

```bash
awk '$3=="gene"' gencode.v45.basic.annotation.gtf \
  | awk 'BEGIN{OFS="\t"} {match($0, /gene_name "([^"]+)"/, m);
                          if ($7=="+") s=$4-2000; else s=$5-1; e=s+2000;
                          if (s<0) s=0;
                          print $1, s, e, m[1], 0, $7}' \
  | bedtools getfasta -fi hg38.fa -bed - -nameOnly -s > promoters_hg38_2kb.fa
```

---

## 4. One-command real-mode run

```bash
./scripts/run_fm_real_dgx.sh \
    data/pollen.h5ad \
    data/fleck_edges.csv \
    data/promoters_hg38_2kb.fa \
    cache/dgx_real_pollen
```

Expected wall-clock on DGX Spark:
- Geneformer extract: ~2 min for 10 k cells
- scGPT extract: ~1 min
- UCE extract: ~5 min
- motif scan: ~2 min for 100 k edges
- Evo scoring: ~15 min for 100 k edges (the bottleneck; chunk-size tunable)
- Borzoi scoring: ~30 min for 100 k edges (two forward passes per edge)
- ablation: ~30 s

Total: ~1 hour for a typical Pollen-scale run.

Outputs land in `cache/dgx_real_pollen/`:
```
pollen_uce.npy                 (10000, 1280)  float32
pollen_uce_manifest.json
pollen_geneformer.npy          (50000, 512)
pollen_geneformer_manifest.json
pollen_scgpt.npy               (10000, 512)
pollen_scgpt_manifest.json
fleck_edges_motif.npy          (100000,)      float32
fleck_edges_motif_manifest.json
fleck_edges_evo.npy            (100000,)
fleck_edges_evo_manifest.json
fleck_edges_borzoi.npy         (100000,)
fleck_edges_borzoi_manifest.json
ablation.json
ablation.md
logs/
```

---

## 5. Consuming the cache from notebooks

[`notebooks/11_foundation_model_pipeline.ipynb`](../notebooks/11_foundation_model_pipeline.ipynb) loads `extract()` in-process by default for tutorial use. To consume a real-mode cache directory instead, replace the relevant cells with:

```python
emb_uce        = np.load("cache/dgx_real_pollen/pollen_uce.npy")
emb_geneformer = np.load("cache/dgx_real_pollen/pollen_geneformer.npy")
edge_scores    = np.load("cache/dgx_real_pollen/fleck_edges_motif.npy")
```

The downstream code (predictor fitting, ablation, plotting) is **unchanged** — that's the integration-contract point of [`docs/foundation-models.md`](foundation-models.md): one numpy array per step, stub or real.

---

## 6. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `RuntimeError: evo real-mode requires the 'evo' package` | dep not installed | `uv pip install evo-model` |
| `RuntimeError: motif real-mode needs JASPAR MEME file at /opt/jaspar/...` | env var or file missing | export `JASPAR_MEME` to your MEME file |
| Borzoi OOM on long contexts | sequence longer than checkpoint context | chunk the FASTA into ≤196 kb segments |
| Geneformer batch effect dominates | your h5ad has unmixed batches | run `scanpy.pp.combat` or `scvi-tools` first, then re-extract |
| HF cache not found offline | weights not pre-fetched | run the `huggingface-cli download` block in §2 once with internet |

---

## 7. What to measure once a real-mode cache exists

The honest test the project's been waiting on. Pick a target benchmark and re-run with the FM priors swapped in:

| target | what changes | success metric |
|---|---|---|
| [Lab 3](../notebooks/03_benchmarking_fidelity.ipynb) fidelity-triple transfer-r ≈ 0.13 | use Geneformer + scGPT priors on the perturbation predictor | transfer-r on Pollen-test set |
| [Lab 4](../notebooks/04_modularity_identifiability.ipynb) MII gap | rebuild the regulome graph with sequence-edge priors blended in | MII separation between organoid / blueprint / bioprinted |
| [Lab 6](../notebooks/06_control_theory.ipynb) high-leverage TFs | combine scGPT in-silico KD prior with the controllability ranking | agreement on top-10 TFs across methods |
| [`scripts/ablate_edge_priors.py`](../scripts/ablate_edge_priors.py) | swap stub for real motif/Evo/Borzoi | F1 lift over Pando alone on the real Fleck edges |
| [`scripts/ablate_perturb_eig.py`](../scripts/ablate_perturb_eig.py) | swap stub for real scGPT KD on a real perturbation dataset | Spearman ρ recovery vs known perturbation responses (e.g. CHOOSE / Replogle perturb-seq) |
| [`docs/wetlab-program.md`](wetlab-program.md) capstone cycle | use the EIG-rank ranking from step 4 to pick the synNotch / KO cycle TFs | resolved ranking vs the wet-lab readout |

These are the **measurements** that will answer "do FMs make a difference on the *project's* numbers" with real evidence — the stub-mode results in [`notebooks/11_foundation_model_pipeline.ipynb`](../notebooks/11_foundation_model_pipeline.ipynb), [`figures/edge_prior_ablation.md`](../figures/edge_prior_ablation.md), and [`figures/perturb_eig_ablation.md`](../figures/perturb_eig_ablation.md) are floors / sensitivities, not predictions for the real benchmarks.
