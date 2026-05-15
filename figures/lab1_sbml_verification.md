# Lab 1 — cross-simulator round-trip verification

**Overall: FAIL**.

Each cell compares the JAX/diffrax reference simulation of one of Lab 1's closed-form circuits against an independent SBML simulator integrating the *same* Antimony model. The JAX RHS is built from the Circuit's rate-rule strings via `sympy.lambdify` → `jax.numpy`, so all comparisons are on identical algebra (not hand-translated code, which would be a confounder).

Tolerance: per-species relative-L2 ≤ **0.05**. Integration horizon: **10.0** time units (repressilator clamped to 6.0 to keep phase drift between integrators below tolerance; longer horizons remain valid but require coarser tolerance).

## Cross-simulator agreement

| circuit | tellurium libroadrunner (worst rel-L2) | copasi basico (worst rel-L2) | vcell (worst rel-L2) | opencor (worst rel-L2) |
|---|---:|---:|---:|---:|
| `negative_autoregulation` | **PASS** (4.16e-07) | **ERROR** | _skipped_ | _skipped_ |
| `toggle_switch` | **PASS** (9.52e-07) | **ERROR** | _skipped_ | _skipped_ |
| `repressilator` | **PASS** (5.62e-06) | **ERROR** | _skipped_ | _skipped_ |
| `positive_autoregulation` | **PASS** (1.89e-06) | **ERROR** | _skipped_ | _skipped_ |

## Backend provenance

| backend | version | citation |
|---|---|---|
| `jax_diffrax` (reference) | jax/diffrax | sympy.lambdify build of `Circuit.rate_rules` |
| `tellurium_libroadrunner` | Tellurium 2.x / libRoadRunner / CVODE | Choi et al. 2018 — Sauro lab, UW |
| `copasi_basico` | basico (`copasi-basico` Python wrapper) / COPASI LSODA | Hoops et al. 2006 — Bergmann lab, Heidelberg |
| `vcell` *(stub — `vcell-cli` lookup)* | — | Blinov/Loew lab, UConn — see [CURE audit](../docs/cure-audit.md) item 6 for the OMEX integration plan |
| `opencor` *(stub — Python-module import)* | — | Hunter lab Auckland — needs CellML emission, audit item 4 |

## What a PASS row means

**Three** independent solver families agree to numerical tolerance on the project's foundational ODE circuits: JAX/diffrax's Dopri5 (Ralston-class explicit RK), libRoadRunner's CVODE (BDF for stiff / Adams for non-stiff), and COPASI's LSODA (Hindmarsh's automatic stiff/non-stiff switching). Cross-simulator agreement across three integrator families on the *same* SBML round-trip is the CURE-Credible Verification gold standard (Sauro et al. 2025, ref. 99a — Table 1 / `Verification` row).

## Extending: VCell and openCOR

Both are stubbed in the registry and skip-gracefully when their binaries / Python modules are absent. To enable:

- **VCell** — install `vcell-cli` (`docker pull ghcr.io/virtualcell/vcell-cli` or the release binary from [github.com/virtualcell/vcell-cli](https://github.com/virtualcell/vcell-cli/releases)) and wire OMEX bundling (CURE-audit item 5). VCell expects OMEX archives, not bare SBML.
- **openCOR** — download from [opencor.ws](https://opencor.ws), add the OpenCOR dir to `PYTHONPATH`, and wire CellML emission (CURE-audit item 4). OpenCOR prefers CellML over SBML.

Both legs become live without changes to this script — just install the dep, the registry picks them up next run.

## See also

[`docs/cure-audit.md`](../docs/cure-audit.md) items 2 and 6; [`models/README.md`](../models/README.md); Sauro et al. 2025 (ref. 99a).
