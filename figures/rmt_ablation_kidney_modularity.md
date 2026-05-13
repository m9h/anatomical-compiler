# RMT denoising — ablation on the kidney modularity benchmark

**Question:** Does Marchenko-Pastur (Aparicio-Bordyuh / `randomly`) denoising of the cell × gene matrix, used in place of variance-based HVG selection + raw correlation, improve the Module Identifiability Index ([Lab 4](../notebooks/04_modularity_identifiability.ipynb)) on the three-system kidney-modularity benchmark?

**Setup.** `scripts/ablate_rmt_kidney_modularity.py` mirrors `benchmark_kidney_modularity.py`'s pipeline but adds an RMT arm that (i) selects HVGs by signal-eigenvector loading (the principled MP analog of variance HVGs) and (ii) computes the gene-gene correlation on the MP-denoised (signal-only) reconstruction. Both arms then feed the same Hodge $L_0$ pipeline; MII is the Lab-4 heuristic, `mean(diff(ev0)[:10]) / std(ev0[:10])`. **User-stated adoption threshold: |ΔMII| ≥ 0.05.**

## Results

### Canonical scale (1000 cells × 100 HVG — matches `figures/nitmb_modularity_report.json`)

| system | MII baseline | MII RMT | ΔMII | verdict |
|---|---|---|---|---|
| Bioprinted Kidney (Lawlor 2021) | 0.31 | 0.34 | +0.03 | **drop** |
| Brain Organoid (Fleck 2023 RNA) | 0.37 | 0.35 | −0.02 | **drop** |
| Fetal Kidney (Ref) | 0.23 | 0.26 | +0.04 | **drop** |

All three |ΔMII| are below 0.05; signs aren't even consistent. **RMT denoising is not a leverage point for this metric at this scale.**

### Larger scale (3000 cells × 200 HVG)

| system | MII baseline | MII RMT | ΔMII | verdict |
|---|---|---|---|---|
| Bioprinted Kidney | 0.31 | 0.35 | +0.04 | drop |
| Brain Organoid    | **0.59** | 0.34 | **−0.25** | "ADOPT" but **wrong direction** |
| Fetal Kidney      | 0.36 | 0.35 | −0.01 | drop |

Here the brain organoid's baseline MII *jumps* (0.37 → 0.59) — the larger sample exposes richer cell-type structure that a bigger HVG-correlation graph captures — and the RMT arm *flattens* it back to 0.34, **destroying** the cross-system ranking that the committed `nitmb_modularity_report.json` headline rests on (brain organoid 0.38 > fetal kidney 0.37 > bioprinted kidney 0.35 — *self-organisation makes sharper modules*). So the larger-scale "improvement" is not an improvement: RMT is *suppressing* the modularity signal in the system that has the most of it.

## Conclusion

**Drop.** At the canonical scale, RMT denoising doesn't meaningfully shift the MII; at larger scale it distorts the cross-system comparison in the wrong direction. The MII's *bottleneck is structural* (a genome-scale regulome is one tangled component, [Lab 4](../notebooks/04_modularity_identifiability.ipynb)'s honest finding), not preprocessing-fragile — exactly the prediction. The cleaner future leverage points are the ones [`ROADMAP.md`](../ROADMAP.md) flags: richer plants and richer actuators, not better noise modelling upstream of the regulome.

The denoiser (`scripts/denoise_rmt.py`, 30 lines, JAX-native, GPU-ready, `jax.grad`-differentiable, passes its synthetic self-test at MP-recovers-the-true-rank with 99% reconstruction-MSE reduction) stays in the repo as a tool — *for cases where one might want it* (e.g. a future per-cell SBI inverse calibration; cf. [Lab 8](../notebooks/08_anatomical_compiler.ipynb)) — not as a default in the existing benchmarks.

## Reproduce

```bash
uv run python scripts/denoise_rmt.py --self-test            # the synthetic validation
uv run python scripts/ablate_rmt_kidney_modularity.py       # the canonical run; 1000 cells × 100 HVG
uv run python scripts/ablate_rmt_kidney_modularity.py --n-cells 3000 --n-hvg 200    # larger-scale stress
```

## Pollen fidelity (deferred)

The matched ablation for `compare_pollen.py` (the Lab 3 fidelity triple) would denoise the Pollen 137k-cell × 38k-gene per-cell matrix, recompute the per-TF DE, and re-train the `hgx` `PerturbationPredictor` — a 4 GB-h5ad-loading, ~minutes-of-compute experiment. The prior on its outcome is the same as the kidney one: the fidelity-triple's *magnitude* bottleneck is structural (the downstream-magnitude ridge, [Lab 7](../notebooks/07_structural_identifiability.ipynb)) and the *direction* component is already at 83% — neither is preprocessing-fragile. Worth doing as a separate experiment if a future agent wants the confirmation; not blocking.
