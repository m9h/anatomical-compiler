# Foundation models for the anatomical compiler

*Catalogue of biological foundation models that could plug into this project, organised by institution (UCSF / Stanford / Berkeley / Toronto and adjacent), with one concrete end-to-end pipeline example. Written 2026-05-13. Companion to [`docs/computational-roadmap.md`](computational-roadmap.md) §1 (the highest-leverage missing direction) and to [`docs/wetlab-program.md`](wetlab-program.md) (the closed-loop context that makes a *cheap* zero-shot prior useful).*

---

## Why this matters here

[Lab 3](../notebooks/03_benchmarking_fidelity.ipynb)'s fidelity-triple transfer-r ≈ 0.13 ceiling is the project's most empirically exposed weakness. Foundation models pretrained on 30–100 M cells give a *zero-shot* prior that routinely beats task-specific models on the kinds of perturbation-prediction benchmarks Lab 3 runs. None of this requires us to train a foundation model — only to *consume* released checkpoints as embedding extractors / perturbation priors. That keeps the dry-side budget at "extract once, cache, use everywhere" rather than "rent an H100 cluster".

The educational track already calls out scGPT, Geneformer, Prophet, scVI/scANVI, CellOT, CellFlow in [`notebooks/README.md`](../notebooks/README.md) and the roadmap. This doc adds the institutional map and a runnable pipeline pattern.

---

## Catalogue by institution

### UCSF / Gladstone / CZ Biohub SF

| model | citation / handle | what it gives us | substrate |
|---|---|---|---|
| **Geneformer** | Theodoris et al. 2023, *Nature* (Gladstone/UCSF) — `ctheodoris/Geneformer` on HF | ~512-d *gene* embeddings; rank-value pretrained on ~30 M cells; zero-shot perturbation prediction; in-silico TF knockdown scoring | gene-level transformer on ranked expression |
| **Pollen-lab atlas** | Pollen et al. (UCSF) | not a model — the *data* the project's `compare_pollen.py` already consumes; substrate for fine-tuning above | brain-organoid scRNA-seq |
| **scVI / scANVI / totalVI** | Yosef lab (started Berkeley, scVI-tools maintained at Weizmann); the SBI integration TODO targets this | probabilistic cell-state VAE with NB / ZINB / NB-mixture heads; latent space and *per-cell* density estimate | cell-level VAE on raw counts |
| **CZ CELLxGENE Discover** | CZI (Mission Bay) | 100 M+ cells, harmonised; the universe Geneformer/UCE/scGPT pretrain on; queryable by gene/disease/tissue | reference atlas |

UCSF-adjacent (Lim Lab / Gartner Lab) are reagent partners, not foundation models — see [REFERENCES.md](../REFERENCES.md) and the wet-lab programme.

### Stanford / Arc Institute / Stanford-CZI

| model | citation / handle | what it gives us | substrate |
|---|---|---|---|
| **UCE — Universal Cell Embedding** | Rosen, Quake, Leskovec et al. 2024 (Stanford+CZI) — `chanzuckerberg/uce` | 1280-d *cell* embeddings, species-agnostic, no retraining needed; encodes any new h5ad zero-shot | transformer over genes-as-tokens, pretrained on ~36 M cells |
| **HyenaDNA** | Stanford (Ermon, Ré) — `LongSafari/hyenadna-*` | sequence-level embeddings up to 1 M bp; useful for *cis-regulatory* priors on regulome edges | DNA-language model |
| **Evo / Evo 2** | Arc Institute (Stanford-adjacent) — Hie, Nguyen et al. — `togethercomputer/evoX-…` / Arc release | DNA language model, single-nucleotide resolution, up to 1 Mb context; scores TF-binding likelihood and de-novo regulatory-element design | hybrid Hyena/Transformer DNA model |
| **Borzoi** | Linder, Kelley et al. (Google + Kundaje Stanford) | sequence → per-cell-type expression prediction; the strongest current sequence-to-expression model | DNA-to-RNA-seq predictor |
| **Tabula Sapiens** | Quake et al. (Stanford / CZ Biohub) | 500 K cells across ~25 tissues; substrate, not a model | reference atlas |
| **scimilarity** | Heimberg et al. (Genentech, Stanford-affiliated) | fast nearest-neighbour search over 23 M cells; gives a learned-distance metric over cell states | contrastive cell embedder |

### Berkeley

| model | citation / handle | what it gives us | substrate |
|---|---|---|---|
| **scVI-tools (origin)** | Lopez et al. 2018 — Yosef lab origin at Berkeley | see UCSF row — same software family | — |
| **BAIR / Berkeley-DeepMind sequence work** | various | sequence-to-function priors complementary to Evo/Borzoi | — |

Berkeley's foundation-model contributions here mostly flow through CZ Biohub joint affiliations and scVI's origins; less institutionally distinct than Toronto or UCSF.

### Toronto / Vector Institute

| model | citation / handle | what it gives us | substrate |
|---|---|---|---|
| **scGPT** | Cui et al. 2024, *Nat Methods* (Bo Wang lab, Toronto/Vector) — `subercui/scGPT` on HF | the canonical scRNA-seq transformer; 33 M cells; perturbation prediction, batch correction, cell-type annotation; *in-silico* perturbation API | gene-token transformer |
| **scTab / scFormer** | Wang lab follow-ons | tabular cells, missing-data robust; complementary to scGPT | — |
| **Deep Genomics / SpliceAI lineage** | Frey lab (UofT) | splice-site and variant-effect prediction; mostly orthogonal but relevant if we ever need cis-element priors on regulome edges | sequence-to-splicing |
| **PathFinder / Goldenberg lab** | SickKids / Vector | patient-state embeddings; relevant if the project ever moves from organoid to clinical | — |

### Elsewhere (worth mentioning so they're not invisible)

- **scFoundation** — Hao et al. 2024 (Tsinghua + Mila): another scRNA-seq transformer, ~50 M cells, gene-context pretraining.
- **CellPLM** — Wen et al. 2024 (Texas A&M): a *cell*-language model rather than gene-language.
- **Prophet** — Theis Lab (Helmholtz Munich): the most directly *perturbation-prediction-shaped* model; already a `ROADMAP.md` TODO.
- **Nicheformer** — Schaar et al. 2024 (Theis): spatial transcriptomics foundation model; will matter when [computational-roadmap §3](computational-roadmap.md) (spatial / differentiable Cellular Potts) gets touched.
- **TranscriptFormer** — CZI: closely related to UCE.
- **ESM3 / RFDiffusion** — protein/structure; for [computational-roadmap §10](computational-roadmap.md) when synNotch receptor design needs novel scaffolds.

---

## Concrete pipeline example — Lab 11 design

A new `notebooks/11_foundation_model_pipeline.ipynb`, slotting into the existing flow:

```
   raw h5ad (Pollen / CHOOSE / Biopunk capstone)
            │
            ▼
   scanpy normalise + HVG selection                              (existing)
            │
            ├──────────────┐
            ▼              ▼
        UCE          Geneformer
   (Stanford+CZI)   (UCSF/Gladstone)
   1280-d *cell*    512-d *gene*
   embeddings       embeddings
            │              │
            │              ▼
            │      embed regulome nodes (Pando-built graph)
            │      → init for hgx node features
            ▼              │
   cell-state latent ──────┤
   → init state for        │
   Hypergraph Neural ODE   │
   (Lab 5)                 │
                           ▼
              ┌──── scGPT (Toronto/Vector) ────┐
              │  in-silico perturbation prior  │
              │  P(post-state | pre-state, KD) │
              └────────────────┬───────────────┘
                               │
                               ▼
              hgx PerturbationPredictor with FM prior
              (Lab 3 fidelity-triple track 2)
                               │
                               ▼
              Bayesian active-learning loop
              (computational-roadmap §2, now urgent)
                               │
                               ▼
              Biopunk wet-lab cycle (wetlab-program.md)
                               │
                               ▼
              new h5ad → re-embed → close loop
```

**Edges of the regulome get sequence-level priors from Evo 2 / Borzoi** (Arc / Google-Stanford): for each (TF, target) pair, score the cis-regulatory likelihood from the promoter sequence; use as a prior weight on the Pando-inferred edge. That is the *causal-inference complement* called out in [computational-roadmap §1](computational-roadmap.md) — sequence-grounded edges resist the associative-vs-causal failure mode that Lab 3's transfer-r diagnoses.

### What each FM replaces / augments

| pipeline stage | currently | with FMs | gain |
|---|---|---|---|
| cell-state init | mean-centred normalised expression | **UCE** 1280-d cell embedding | species-agnostic, batch-corrected zero-shot |
| node features (regulome) | one-hot gene index | **Geneformer** 512-d gene embedding | gene-context-aware, transferable across tissues |
| edge priors (regulome) | Pando GLM weights only | + **Evo 2** / **Borzoi** sequence scores | causal grounding from cis-regulatory grammar |
| perturbation prediction | hgx PerturbationPredictor trained from scratch | + **scGPT** / **Prophet** zero-shot baseline as feature, prior, or distillation target | the project's most exposed empirical weakness (Lab 3 transfer-r ≈ 0.13) |
| per-cell density | empirical | **scVI** NB-head latent for SBI inverse | makes [Lab 8](../notebooks/08_anatomical_compiler.ipynb)'s SBI step well-posed |

None of these requires fine-tuning. **Extract once, cache as `.npy`, then run the JAX pipeline on the cached arrays.** Total compute: a single A100 day per dataset, end-to-end.

---

## Integration plan — what to build first

In priority order, each a focused PR:

1. **Embedding-extractor utility** — `scripts/fm_embed.py` with subcommands `uce`, `geneformer`, `scgpt`, taking an h5ad in and writing `.npy` arrays out. ~1 day. Establishes the cache contract every downstream lab uses.
2. **Lab 11 notebook** — load cached embeddings, swap them into the existing benchmarks (Lab 3 fidelity, Lab 4 modularity), show before/after on the project's canonical headline numbers. ~3 days. This is the empirical test of whether FMs actually help on *this* project's questions.
3. **Sequence-edge prior** — Evo-2-or-Borzoi scoring of regulome edges; ablation on Pando edges vs Pando+sequence-prior edges. ~1 week. Tests the causal-grounding hypothesis.
4. **scGPT in-silico KD as Lab 6 / Lab 8 prior** — score the project's "high-leverage TF" predictions (Lab 6) against scGPT's zero-shot KD predictions; use disagreement as an EIG signal for the [wetlab-program.md](wetlab-program.md) BO loop. ~1 week. Direct bridge to the wet-lab side.

Total: **~2 weeks** for steps 1–2 (the empirical test), **~1 month** if (3) and (4) also land.

---

## What stays the project's own

Foundation models give *priors*; the project's contributions remain:

- **MII / Hodge spectral diagnostics** (Lab 4) — none of these FMs compute it; it's a *structural* readout *on* the regulome.
- **Identifiability geometry** (Lab 7) — FM embeddings don't change the structural-identifiability of the dynamics on top of them.
- **Control / anatomical compiler** (Lab 6 / Lab 8) — FMs predict states; this project *steers* them.
- **Bioelectric coupling** ([BETSE-JAX](../README.md#bioelectric-layer-companion)) — no FM covers this layer.
- **Closed-loop wet-lab integration** ([wetlab-program.md](wetlab-program.md)) — FMs accelerate the prediction half; the wet-lab half is the empirical ground truth.

The mental model is: FMs are *better priors and better embeddings*; the project's stack is the *control- and identifiability-aware dynamical-systems layer on top*. They compose.

---

## Cross-references

- [`docs/computational-roadmap.md`](computational-roadmap.md) §1 — the strategic framing this doc operationalises.
- [`docs/wetlab-program.md`](wetlab-program.md) — the closed-loop context that makes a zero-shot prior an active-learning input.
- [`ROADMAP.md`](../ROADMAP.md) §3D — the existing Prophet TODO this doc subsumes.
- [`REFERENCES.md`](../REFERENCES.md) — bibitems to add: Theodoris 2023 (Geneformer), Cui 2024 (scGPT), Rosen 2024 (UCE), Nguyen 2024 (Evo), Linder 2025 (Borzoi), Hao 2024 (scFoundation), Wen 2024 (CellPLM). Add when (1) above lands.
