# Edge-prior ablation (real-mode)

- Truth: 32,000 positive edges / 88,516 negatives (44 TFs × 2,739 targets)
- Motif scored: 97,127/120,516
- Pando scored: 118,140/120,516

## F1 at top-K (K = n_positives)

- F1 (Pando alone)  = 0.3961  (P=0.396, R=0.396)
- F1 (motif alone)  = 0.2698  (P=0.270, R=0.270)
- F1 (best blend)   = 0.3961  at α = 0.0
- Δ over Pando      = +0.0000

**Verdict: NEUTRAL — no meaningful gain or loss from blending motif**

## α sweep

| α | F1 |
|---|---|
| 0.0 | 0.3961 |
| 0.1 | 0.3961 |
| 0.2 | 0.3961 |
| 0.3 | 0.3961 |
| 0.4 | 0.3961 |
| 0.5 | 0.3961 |
| 0.6 | 0.3961 |
| 0.7 | 0.3961 |
| 0.8 | 0.3961 |
| 0.9 | 0.3961 |
| 1.0 | 0.2892 |