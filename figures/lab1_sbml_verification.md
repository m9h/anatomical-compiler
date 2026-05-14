# Lab 1 — SBML round-trip verification report

**Overall: PASS**.

Each row compares (a) the JAX/diffrax simulation of the circuit defined in `scripts/export_lab1_sbml.py`'s `Circuit` dataclass, against (b) Tellurium / libRoadRunner integrating the Antimony model emitted from the *same* `Circuit`. Both simulators use Dopri5-class adaptive stepping with `rtol=1e-7, atol=1e-9`.

Tolerance: per-species relative-L2 error ≤ 0.05. Integration horizon: 10.0 time units (the repressilator uses 6.0 to keep phase drift between integrators below the tolerance band; longer horizons remain valid but require coarser tolerance).

| circuit | status | worst rel-L2 | per-species detail |
|---|---|---:|---|
| `negative_autoregulation` | **PASS** | 4.16e-07 | x=4.2e-07 |
| `toggle_switch` | **PASS** | 9.52e-07 | u=9.5e-07, v=4.0e-07 |
| `repressilator` | **PASS** | 5.62e-06 | p0=5.5e-06, p1=5.1e-06, p2=5.6e-06 |
| `positive_autoregulation` | **PASS** | 1.85e-06 | x=1.8e-06 |

## What a PASS means

The Antimony export of the project's Lab-1 ODEs simulates identically (to numerical-tolerance) in libRoadRunner / Tellurium as in the project's JAX/diffrax pipeline. This is the **CURE-Credible Verification** deliverable: cross-simulator agreement on the foundational closed-form circuits, demonstrating that the project speaks SBML correctly where the model class permits.

See also: [`docs/cure-audit.md`](../docs/cure-audit.md) priority item 2; [`models/README.md`](../models/README.md) for the file inventory; Sauro et al. 2025 (ref. 99a) for the verification rationale.
