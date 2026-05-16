# Lab 3 — fidelity-triple transfer-r with FM priors

**Mode: real.**

| arm | transfer-r |
|---|---:|
| baseline (one-hot / learned-from-scratch) | +0.542 |
| FM-augmented (Geneformer features) | +0.411 |
| **Δ** | **-0.131** |

_Fidelity-triple transfer-r — train predictor on cross-condition response in context A, test on context B. Compare baseline (gene one-hot features) vs FM (Geneformer-derived per-gene features). Lab 3 published baseline ≈ 0.13._

## Interpretation

The published Lab 3 baseline is **transfer-r ≈ 0.13** (`notebooks/03_benchmarking_fidelity.ipynb`). The FM-augmented arm using Geneformer gene embeddings (and optionally UCE cell embeddings) should lift this if the 30 M-cell pretraining carries co-regulation structure that the one-hot baseline can't access. A positive Δ in real-mode is the empirical answer to _'do FMs make a difference on the project's actual numbers'_.

See also: [`docs/dgx-verifier-runbook.md`](../docs/dgx-verifier-runbook.md) Tier 4; [`docs/foundation-models.md`](../docs/foundation-models.md) step 5; [`notebooks/11_foundation_model_pipeline.ipynb`](../notebooks/11_foundation_model_pipeline.ipynb).
