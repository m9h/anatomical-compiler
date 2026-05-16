# Edge-prior ablation — Pando alone vs Pando + sequence prior (SYNTHETIC CEILING)

> **Methodology note (2026-05-15):** this synthetic ablation is a *ceiling*, not a
> validation. Real-data behavior is in [`edge_prior_ablation_real.md`](edge_prior_ablation_real.md);
> per CURE-Validation principles, only real-data results count as evidence for or
> against the FM-prior hypothesis. The synthetic +0.109 lift here did **not**
> transfer to the real Pollen regulome.

Synthetic truth: 30 TFs × 300 genes, 15 edges/TF.
Sequence-prior stub: model=motif, fpr=0.05, fnr=0.4.

| method | F1 |
|---|---|
| Pando alone (standardised) | **0.607** |
| Pando alone (raw coefficients) | 0.584 |
| Sequence prior alone | 0.473 |
| Combined (best α=0.30) | **0.716** |

**Δ over Pando (matched, standardised):** +0.109

**Verdict:** KEEP — sequence prior is adding signal beyond Pando

## F1 vs α curve

Combined score = (1 − α) · stdz(W_pando) + α · stdz(W_seq). α=0 row is the
fair Pando-alone baseline; α=1 row is the seq-prior alone.

| α | F1 |
|---|---|
| 0.00 | 0.607 |
| 0.10 | 0.656 |
| 0.20 | 0.689 |
| 0.30 | 0.716 |
| 0.40 | 0.711 |
| 0.50 | 0.662 |
| 0.60 | 0.616 |
| 0.70 | 0.591 |
| 0.80 | 0.569 |
| 0.90 | 0.513 |
| 1.00 | 0.473 |
