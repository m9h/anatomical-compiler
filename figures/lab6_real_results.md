# Lab 6 — high-leverage TF ranking vs scGPT in-silico KD

**Mode: real.**

Two independent rankings of the same TF set, scored by Spearman ρ and top-K Jaccard overlap:

| metric | value |
|---|---:|
| Spearman(jac, scgpt) | **+0.024** |
| top-10 Jaccard overlap | 0.111 |
| top-20 Jaccard overlap | 0.250 |

## EIG-rank wet-lab queries

The top-K TFs by *disagreement* between the two predictors. These are the highest-expected-information-gain wet-lab queries — running them resolves the ambiguity that the disagreement encodes.

**Top 5 (real):**

- `EMX1`
- `NR4A2`
- `PHF21A`
- `TCF12`
- `SOX6`

**Top 10 (real):**

- `EMX1`
- `NR4A2`
- `PHF21A`
- `TCF12`
- `SOX6`
- `TFAP2C`
- `TCF7L1`
- `SOX2`
- `NFIB`
- `SOX9`

_Real-mode on screen_annotated.h5ad (44/44 TFs survived intersection with adata.var.index). Perturb signal from 'measured' (screen_annotated_measured_perturb.npy). The EIG-rank top-5/10 are the wet-lab BO loop's next-query candidates (per docs/wetlab-program.md). Anchor: stub-mode realistic-regime ρ ≈ 0.56 (see ablate_perturb_eig.py)._

## Interpretation

Lab 6's controllability ranking is *linear* (Jacobian of the LTI surrogate around an attractor; ‖B‖₂ per-TF). scGPT's in-silico KD is *nonlinear* (a 30 M-cell-pretrained transformer predicting post-perturbation state). Where they agree (high Spearman), confidence is high. Where they disagree, the wet lab resolves the inconsistency — that's the BO acquisition signal `scripts/ablate_perturb_eig.py` quantified in stub-mode (EIG-rank beats GREEDY by +0.029 Spearman in the realistic regime). This script is the real-mode answer.

See also: [`docs/dgx-verifier-runbook.md`](../docs/dgx-verifier-runbook.md) Tier 4; [`docs/wetlab-program.md`](../docs/wetlab-program.md) — the cycles these EIG queries inform; [`scripts/ablate_perturb_eig.py`](../scripts/ablate_perturb_eig.py) — the stub.
