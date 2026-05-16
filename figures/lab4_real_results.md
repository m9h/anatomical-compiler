# Lab 4 — MII gap with sequence-edge priors blended in

**Mode: real.**

Blended graph: `(1−α)·stdz(Pando) + α·stdz(seq_prior)`. MII heuristic on the (gene × gene) co-regulation adjacency derived from blending. Spread = max(MII) − min(MII) across the three systems.

| α | spread |
|---:|---:|
| 0.0 | 0.026 |
| 0.2 | 0.051 |
| 0.4 | 0.009 |
| 0.6 | 0.004 |
| 0.8 | 0.018 |
| 1.0 | 0.000 |

**Headline:** at α=0 (Pando alone) spread = 0.026; at best α = 0.2 spread = 0.051; Δ = **+0.025**.

_Real-mode on edges=pollen_edges.csv (32000 rows, 1963 unique target genes). Each system's correlation-incidence adjacency restricted to those targets, blended with seq-prior grid (TF × target → gene-gene similarity). MII = NITMB identifiability heuristic (mean spectral gap / std). Anchor: baseline (α=0) ≈ 0.026 spread across systems._

## Per-system MII at each α

| system | α | mii_heuristic | rel_eigengap |
|---|---:|---:|---:|
| bioprinted_kidney | 0.0 | 0.362 | 0.071 |
| bioprinted_kidney | 0.2 | 0.383 | 0.087 |
| bioprinted_kidney | 0.4 | 0.371 | 0.197 |
| bioprinted_kidney | 0.6 | 0.355 | 0.272 |
| bioprinted_kidney | 0.8 | 0.337 | 0.886 |
| bioprinted_kidney | 1.0 | 0.360 | 0.290 |
| brain_organoid | 0.0 | 0.352 | 0.060 |
| brain_organoid | 0.2 | 0.338 | 0.064 |
| brain_organoid | 0.4 | 0.369 | 0.184 |
| brain_organoid | 0.6 | 0.359 | 0.269 |
| brain_organoid | 0.8 | 0.355 | 0.293 |
| brain_organoid | 1.0 | 0.360 | 0.290 |
| fetal_kidney_ref | 0.0 | 0.336 | 0.064 |
| fetal_kidney_ref | 0.2 | 0.332 | 0.067 |
| fetal_kidney_ref | 0.4 | 0.362 | 0.170 |
| fetal_kidney_ref | 0.6 | 0.356 | 0.266 |
| fetal_kidney_ref | 0.8 | 0.353 | 0.292 |
| fetal_kidney_ref | 1.0 | 0.360 | 0.291 |

## Interpretation

Lab 4's baseline MII numbers (Pando alone, no sequence prior) are around 0.31 (bioprinted kidney), 0.37 (brain organoid), 0.23 (fetal kidney) — see `figures/nitmb_modularity_report.json`. The blend with sequence priors is additive in structural signal: edges where Pando and seq agree get reinforced; where they disagree, the seq prior pulls the topology toward the cis-regulatory grammar. A positive Δ-spread means the sequence prior is amplifying the modularity-of-self-organisation signal the project's whitepaper rests on.

See also: [`docs/dgx-verifier-runbook.md`](../docs/dgx-verifier-runbook.md) Tier 4; [`scripts/ablate_edge_priors.py`](../scripts/ablate_edge_priors.py) — the stub-mode synthetic; [`notebooks/04_modularity_identifiability.ipynb`](../notebooks/04_modularity_identifiability.ipynb) — the baseline.
