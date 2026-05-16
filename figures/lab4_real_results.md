# Lab 4 — MII gap with sequence-edge priors blended in

**Mode: real.**

Blended graph: `(1−α)·stdz(Pando) + α·stdz(seq_prior)`. MII heuristic on the (gene × gene) co-regulation adjacency derived from blending. Spread = max(MII) − min(MII) across the three systems.

| α | spread |
|---:|---:|
| 0.0 | 0.012 |
| 0.2 | 0.012 |
| 0.4 | 0.012 |
| 0.6 | 0.012 |
| 0.8 | 0.012 |
| 1.0 | 0.000 |

**Headline:** at α=0 (Pando alone) spread = 0.012; at best α = 0.4 spread = 0.012; Δ = **+0.000**.

_Real-mode on edges=pollen_edges.csv (32000 rows, 1963 unique target genes). Each system's correlation-incidence adjacency restricted to those targets, blended with seq-prior grid (TF × target → gene-gene similarity). MII = NITMB identifiability heuristic (mean spectral gap / std). Anchor: baseline (α=0) ≈ 0.012 spread across systems._

## Per-system MII at each α

| system | α | mii_heuristic | rel_eigengap |
|---|---:|---:|---:|
| fleck_organoid | 0.0 | 0.324 | 0.024 |
| fleck_organoid | 0.2 | 0.324 | 0.024 |
| fleck_organoid | 0.4 | 0.324 | 0.024 |
| fleck_organoid | 0.6 | 0.324 | 0.024 |
| fleck_organoid | 0.8 | 0.324 | 0.024 |
| fleck_organoid | 1.0 | 0.000 | 0.000 |
| pollen_slice | 0.0 | 0.336 | 0.030 |
| pollen_slice | 0.2 | 0.336 | 0.030 |
| pollen_slice | 0.4 | 0.336 | 0.030 |
| pollen_slice | 0.6 | 0.336 | 0.030 |
| pollen_slice | 0.8 | 0.336 | 0.030 |
| pollen_slice | 1.0 | 0.000 | 0.000 |

## Interpretation

Lab 4's baseline MII numbers (Pando alone, no sequence prior) are around 0.31 (bioprinted kidney), 0.37 (brain organoid), 0.23 (fetal kidney) — see `figures/nitmb_modularity_report.json`. The blend with sequence priors is additive in structural signal: edges where Pando and seq agree get reinforced; where they disagree, the seq prior pulls the topology toward the cis-regulatory grammar. A positive Δ-spread means the sequence prior is amplifying the modularity-of-self-organisation signal the project's whitepaper rests on.

See also: [`docs/dgx-verifier-runbook.md`](../docs/dgx-verifier-runbook.md) Tier 4; [`scripts/ablate_edge_priors.py`](../scripts/ablate_edge_priors.py) — the stub-mode synthetic; [`notebooks/04_modularity_identifiability.ipynb`](../notebooks/04_modularity_identifiability.ipynb) — the baseline.
