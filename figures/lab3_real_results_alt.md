# Lab 3 — fidelity-triple transfer-r with FM priors

**Mode: real.**

| arm | transfer-r |
|---|---:|
| baseline (one-hot / learned-from-scratch) | +0.189 |
| FM-augmented (Geneformer features) | +0.058 |
| **Δ** | **-0.131** |

_Real-mode on pollen_slice_geneformer.h5ad (37344 genes, 13548+3983 cells across GW19/GW22). R_train computed on context GW19, R_test on GW22. Lab 3 baseline anchor: transfer-r ≈ 0.13 — delta should be read against that._

## Interpretation

The published Lab 3 baseline is **transfer-r ≈ 0.13** (`notebooks/03_benchmarking_fidelity.ipynb`). The FM-augmented arm using Geneformer gene embeddings (and optionally UCE cell embeddings) should lift this if the 30 M-cell pretraining carries co-regulation structure that the one-hot baseline can't access. A positive Δ in real-mode is the empirical answer to _'do FMs make a difference on the project's actual numbers'_.

See also: [`docs/dgx-verifier-runbook.md`](../docs/dgx-verifier-runbook.md) Tier 4; [`docs/foundation-models.md`](../docs/foundation-models.md) step 5.
