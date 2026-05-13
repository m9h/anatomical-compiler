# Differentiable Cellular Potts — planning doc

*Sister-repo planning doc for `cpjax` (parallel to [BETSE-JAX](../README.md#bioelectric-layer-companion) / `~/Workspace/betse-unified`). What it would build, what gradient strategy, what oracles, how to validate, why now-vs-later. Written 2026-05-13.*

> **Status — deferred but stubbed.** Per the post-wet-lab re-rank in [`docs/computational-roadmap.md`](computational-roadmap.md) §3, this is *deferred*: model-side richer-plant work that doesn't have an immediate wet-lab readout. The stub at `~/Workspace/cpjax` exists so the design isn't lost, and so Phase 0–1 (oracle harness + forward parity) can be started without prelude when the wet-lab cycles begin producing shape data. That Phase 0–1 deliverable — a JAX-native CC3D/Morpheus oracle harness with cached outputs — **is useful to this project regardless of whether the differentiable backend ever lands**.

---

## 1. Why differentiable Cellular Potts

The project's anatomical compiler ([Lab 8](../notebooks/08_anatomical_compiler.ipynb)) currently controls **regulome state**. Tissues also have **shape**. The Hypergraph Neural ODE steers gene-expression trajectories; nothing in the current stack puts `jax.grad` on the cell-shape / division / adhesion / movement layer where the *macroscopic morphological outcome* actually lives.

The sister gap to bioelectric: [BETSE-JAX](../README.md#bioelectric-layer-companion) made the *bioelectric* layer differentiable so inverse design could solve for Vₘₑₘ patterns producing a target morphology. **Differentiable Cellular Potts is the same move on the shape layer** — Hamiltonian-driven cell behaviour, differentiable end-to-end, composable with the rest of the JAX stack (`hgx`, `jaxctrl`, BETSE-JAX, the educational track).

It closes the *deepest* gap the project flags: regulome → form ([`docs/computational-roadmap.md`](computational-roadmap.md) §3).

What it unlocks concretely:

- **Anatomical compiler over shape.** Lab 8's OC/SBI machinery generalises to `J(shape | actuation)` with `∇` through the Potts dynamics.
- **§4.3(i) bioprinting maturation as a gradient sweep.** Currently a coarse grid search; with differentiable shape physics it becomes a continuous BO problem with calibrated uncertainty.
- **Bioelectric ↔ shape coupling computable.** `∂(shape)/∂(Vₘₑₘ pattern)` is currently narrative-only in BETSE-JAX; with `cpjax` it's a real Jacobian — Vₘₑₘ modulates Hamiltonian terms (J(Vₘₑₘ), λ(Vₘₑₘ)).
- **Hypergraph Neural ODE → spatial Neural PDE.** [Lab 5](../notebooks/05_hypergraph_neural_odes.ipynb)'s `hgx` generalises onto a spatial cell-graph; `cpjax` provides the underlying physics layer.
- **Closed loop with shape readouts** (light-sheet, brightfield, segmentation). The wet-lab side ([`docs/wetlab-program.md`](wetlab-program.md) Cycle 1: organoid injury timecourse; Cycle 4: PRINTESS sweep) produces shape data that the dry side currently can only consume qualitatively.

---

## 2. The core obstacle and three strategies

A canonical Potts step:

```
1. Pick a random lattice site x with current cell-id σ(x).
2. Pick a neighbouring site y with σ(y) ≠ σ(x).
3. Compute ΔH = H(σ with x → σ(y)) − H(σ).
4. Accept with p = min(1, exp(−ΔH / T)).
```

The **Hamiltonian H is differentiable** in its energy parameters $\theta$ = (adhesion J_{αβ}, area constraint λ_A, perimeter λ_P, chemotaxis, Vₘₑₘ couplings, …). The **accept/reject step is not**, because it's a Bernoulli draw over a discrete cell-id state.

| strategy | what it does | bias | variance | runtime cost | fits which use case |
|---|---|---|---|---|---|
| **(a) Score-function (REINFORCE)** | $\nabla_\theta \mathbb{E}[R(\tau)] = \mathbb{E}[\nabla_\theta \log p(\tau\|\theta) \cdot R(\tau)]$ on the MC trajectory | unbiased | **high** | ~2× forward | exact gradients when you need them; small lattices |
| **(b) Gumbel-Softmax accept** | replace Bernoulli accept by Gumbel-softmax weighted update; temperature τ→0 recovers CP | low (tunable) | low | ~1.2× forward | the pragmatic production version |
| **(c) Soft-lattice** | each lattice site holds a softmax distribution over cell-ids; H continuous in soft assignments | structurally biased (different physics in detail) | very low | ~3× forward (no MC needed; gradient descent on H) | "Neural Cellular Automata with a Potts-style energy"; matches CP in expectation |

The plan is **prototype all three**, validate against CC3D/Morpheus oracles, pick the one with the best (forward-fidelity) × (gradient-quality) on the canonical benchmarks. Likely outcomes:

- **(a)** remains the *correctness reference* — slow but always-correct fallback.
- **(b)** or **(c)** becomes the *production version* because variance-reduced REINFORCE on a 256³ lattice is hard.
- Hybrid: use **(c)** for parameter inference (smooth loss landscape), **(a)** for confidence intervals on the recovered parameters.

---

## 3. Existing vs build

There is **no existing differentiable Cellular Potts framework**. The mature CP simulators are forward-only:

| project | role | why not gradient source |
|---|---|---|
| **CompuCell3D** (Glazier lineage) | C++ core + Python steppables | C++ MC loop opaque to autodiff; Python steppables run per-step, not differentiable through |
| **Morpheus** (Gerisch/Deutsch lineage) | XML model spec (MorpheusML) + C++ engine | XML→C++ pipeline, no autodiff hook |
| **PhysiCell** | agent-based, off-lattice | wrong substrate, non-differentiable |
| **Chaste** | cell-based C++ framework | non-differentiable |
| **jax-md** (Schoenholz & Cubuk 2020) | differentiable MD in JAX | wrong physics (continuous, not lattice MC) — but the *architectural template* is the closest precedent |
| **Growing Neural CA** (Mordvintsev et al. 2020) | learned continuous CA dynamics | learned not parameter-recovering — solves a different inverse problem; useful as a *baseline* for soft-lattice variant (c) |

**Answer: build new, in JAX, validated against existing.** Same pattern as BETSE-JAX:

- *Build* the differentiable engine (`cpjax`), JAX-native, GPU-first, `jax.grad`-able, `lax.scan`-fast.
- *Use existing* CC3D and Morpheus as **oracles** (forward-parity validation, BioModels test cases).
- *Ingest* **MorpheusML** as the import format — every BioModels CP entry becomes a differentiable model. This is the COMBINE-ecosystem play, exactly parallel to BETSE-JAX importing SBML/CellML.

---

## 4. Development path — six phases

Total: **~6–9 months for a usable v0** at one developer-equivalent. Comparable to BETSE-JAX, slightly harder because of the stochastic-MC differentiability problem.

### Phase 0 — Oracle harness *(~2 weeks)*

A standalone Python wrapper around CC3D and Morpheus that runs them on a curated benchmark set and dumps lattice state per timestep to `.npy` / `.zarr`.

- **Benchmark set:** ~5 canonical CP cases drawn from BioModels and the CC3D/Morpheus example libraries:
  1. **Cell sorting** (Steinberg adhesion-based) — the classic Potts test
  2. **Vasculogenesis** (Merks–Glazier endothelial network formation)
  3. **Chemotaxis** (single-cell + collective)
  4. **Gastrulation toy** (Drasdo-style folding) — minimal multi-tissue case
  5. **Tumour-stroma** (heterotypic adhesion, growth, death) — relevant to [Lab 10](../notebooks/10_cancer_module_identifiability.ipynb)
- **Output contract:** for each (seed, params) → trajectory of (lattice_state[t], summary_stats[t]). Cached, hashed, reproducible.
- **Deliverable independent of differentiability:** the harness itself is a useful JAX-callable testbed for downstream labs even without the gradients.

### Phase 1 — Forward-only JAX CP *(~1–2 months)*

JAX-native CP with `lax.scan` over MC sweeps, `vmap` over batched seeds, JIT-compiled inner loop.

- Implement: site picker (random + ordered + checkerboard parallel updates), `ΔH` computer, Bernoulli accept (using `jax.random.uniform`), state update.
- All energy terms: adhesion (J_{αβ}), area constraint (λ_A), perimeter (λ_P), chemotaxis (gradient field), volume conservation, optional field couplings.
- **Validate forward parity** vs CC3D/Morpheus on the 5 benchmark cases — statistical agreement on summary stats (cluster count, mean cluster size, anisotropy, perimeter/area), not byte-identical trajectories.

### Phase 2 — REINFORCE gradient *(~2 months)*

Score-function estimator on accept/reject:

- $\nabla_\theta \log p(\tau \mid \theta) = \sum_t \nabla_\theta \log p(\text{accept}_t \mid H(\sigma_t; \theta))$
- Per-step gradients are closed-form; multiply by trajectory return.
- **Variance reduction:** baseline (subtract a state-value estimate), antithetic sampling (same noise, ±θ pair), control variates (use H itself as a correlated baseline).
- **Inverse benchmark:** recover known adhesion energies J_{αβ} from observed trajectories on Phase-0 benchmarks. Pass criterion: relative error < 10% on the cell-sorting case with 1k MCS of data.

### Phase 3 — Relaxed gradient *(~2 months)*

Two competing relaxations:

- **(b) Gumbel-Softmax** on the accept step. Hyperparameter: temperature τ; anneal during training.
- **(c) Soft-lattice** — each site holds a (softmax over cell-ids) distribution. Hamiltonian terms generalise: J_{αβ} contribution at site x = Σ p(σ(x)=α) Σ_{y∈N(x)} p(σ(y)=β) J_{αβ}.

Benchmark vs Phase 2:
- gradient signal-to-noise ratio
- parameter-recovery error vs ground truth
- wall-clock per gradient step
- forward-fidelity drift relative to CC3D oracle (the bias side of the bias-variance trade)

Pick the winner; document the loser as a reference implementation.

### Phase 4 — MorpheusML import *(~1 month)*

XML → `cpjax` translator. The adoption play: every BioModels CP entry, every Morpheus tutorial, every published Morpheus model becomes a differentiable model for free.

- Read MorpheusML schema (CellPopulation, CellType, Interaction, System, Analysis sections).
- Map each XML element to the corresponding `cpjax` energy term.
- Round-trip validation: read MorpheusML → run in `cpjax` → check forward parity with Morpheus's own run on the same XML.
- Limit scope to the energy-term subset CP-proper covers; leave reaction-diffusion / global-feedback extensions as later work.

### Phase 5 — Couple to BETSE-JAX and the anatomical compiler *(~1 month)*

- BETSE-JAX exposes Vₘₑₘ field as a JAX array; `cpjax` Hamiltonian accepts that as input (`J = J(Vₘₑₘ)`, `λ_A = λ_A(Vₘₑₘ)`).
- End-to-end gradient: target shape → loss → `jax.grad` flows through cpjax → into BETSE-JAX → into actuation (ion-channel modulation).
- **Lab 12 educational notebook** in this repo: "Differentiable shape control — from bioelectric prepattern to target morphology". Self-contained synthetic case + optional larger-scale validation.

---

## 5. Architecture

`cpjax` repo structure (matches the BETSE-JAX / `~/Workspace/betse-unified` template):

```
~/Workspace/cpjax/
├── pyproject.toml          # uv-managed, JAX/Equinox/Diffrax stack
├── README.md
├── LICENSE                 # Apache-2.0
├── src/cpjax/
│   ├── __init__.py
│   ├── lattice.py          # lattice state, neighbourhoods, boundary conditions
│   ├── hamiltonian.py      # energy terms, all differentiable
│   ├── update.py           # three update kernels: (a) REINFORCE / (b) Gumbel / (c) soft-lattice
│   ├── simulate.py         # high-level run loop, lax.scan over MCS
│   ├── io_morpheus.py      # MorpheusML reader (Phase 4)
│   ├── oracles/            # forward-parity wrappers around CC3D and Morpheus
│   │   ├── cc3d.py
│   │   └── morpheus.py
│   ├── benchmarks/         # the 5 canonical CP cases
│   └── inverse.py          # the inverse-design API (param recovery, shape control)
├── examples/               # Jupyter notebooks: cell-sorting, vasculogenesis, etc.
├── tests/
└── docs/
```

Public API sketch:

```python
import cpjax
import jax.numpy as jnp

# forward (Phase 1)
state0 = cpjax.lattice.init(shape=(128, 128), n_cells=64, seed=0)
params = cpjax.Hamiltonian(J={(0, 1): 8.0, (1, 1): 2.0}, lam_A=1.0, target_A=64.0)
trajectory = cpjax.simulate.run(state0, params, n_mcs=1000, kernel="metropolis")

# gradients (Phase 2/3)
def loss(theta):
    params = cpjax.Hamiltonian(**theta)
    traj = cpjax.simulate.run(state0, params, n_mcs=1000, kernel="gumbel", tau=0.5)
    return cpjax.metrics.shape_distance(traj[-1], target_lattice)

grad = jax.grad(loss)(theta0)              # (b) Gumbel relaxation
grad_rf = cpjax.inverse.reinforce_grad(loss, theta0, n_seeds=64)  # (a) exact REINFORCE

# Morpheus interop (Phase 4)
model = cpjax.io_morpheus.read("biomodels/MODEL2003170001.xml")
diff_model = model.to_jax()                # any BioModels CP entry, now differentiable
```

---

## 6. BioModels benchmark set

Phase 0 deliverable, ordered by complexity:

| # | name | source | tests |
|---|---|---|---|
| 1 | **Cell sorting** | Graner & Glazier 1992; in both CC3D and Morpheus examples | adhesion energy recovery (Phase 2 inverse benchmark) |
| 2 | **Vasculogenesis** | Merks & Glazier 2005; CC3D demo | spatial pattern formation; chemotaxis gradients |
| 3 | **Chemotaxis (single + collective)** | various; Morpheus example | gradient-field coupling; collective vs single-cell switch |
| 4 | **Gastrulation toy** | Drasdo & Forgacs 2000; reimplementation | multi-tissue heterotypic adhesion; volume conservation under deformation |
| 5 | **Tumour-stroma** | Szabó & Merks 2013; BioModels entries | growth + death + heterotypic adhesion; relevant to [Lab 10](../notebooks/10_cancer_module_identifiability.ipynb) |

Optional Phase-5 case: **bioelectric-modulated cell sorting** (J = J(Vₘₑₘ)) — built fresh to demonstrate BETSE-JAX ↔ cpjax coupling.

---

## 7. Cost, risk, and what could derail this

**Cost.** Comparable to BETSE-JAX (~6 months single-developer). Spread:

- Phase 0: 2 wk (a Python project, no real JAX yet)
- Phases 1–2: 3–4 months (real engineering; the REINFORCE gradient on stochastic MC has known variance issues)
- Phase 3: 2 months (relaxations + careful benchmarking)
- Phase 4: 1 month (mostly XML parsing + mapping table)
- Phase 5: 1 month (interop + the educational notebook)

**Top risks.**

1. **REINFORCE variance.** Gradient signal-to-noise on a 256³ lattice may be too low for practical inference. Mitigation: rely on (b) or (c) for production; keep REINFORCE as a small-lattice correctness check. This is the most likely outcome — i.e., the soft-lattice variant is the actual working version.
2. **Relaxation bias.** Soft-lattice forward dynamics may drift measurably from true CP, especially on phase-transition behaviour. Mitigation: characterise the bias explicitly in Phase 3; document the regime where each kernel is trustworthy.
3. **MorpheusML coverage.** The XML schema is large; complete coverage is months of work. Mitigation: target the CP-proper energy-term subset for Phase 4; document what's *not* yet imported.
4. **Memory.** Backprop through long MCS trajectories on large lattices needs gradient checkpointing aggressively. Mitigation: `jax.checkpoint`; benchmark early.
5. **The deferral itself.** Per [`docs/computational-roadmap.md`](computational-roadmap.md), this is *deferred* until wet-lab cycles produce shape data. Risk: months of model-side work without a wet-lab readout that pulls on it. Mitigation: Phase 0 (oracle harness) is useful regardless and is the right place to start; do not commit to Phases 1+ until either (a) a wet-lab cycle produces shape data, or (b) the §4.3(i) bioprinting design search becomes a *committed* programme item.

---

## 8. How this fits the rest of the project

| project artefact | how `cpjax` interacts |
|---|---|
| [Lab 4](../notebooks/04_modularity_identifiability.ipynb) (MII) | unchanged — operates on regulome graphs, orthogonal substrate |
| [Lab 5](../notebooks/05_hypergraph_neural_odes.ipynb) (Hypergraph Neural ODE) | future "Lab 12" spatial extension uses `cpjax` as the cell-shape layer; `hgx` operates *on cells*, `cpjax` operates *between* them |
| [Lab 6](../notebooks/06_control_theory.ipynb) (linear control / `jaxctrl`) | `jaxctrl` minimum-energy controller can be lifted onto the `cpjax` plant once the soft-lattice variant exists |
| [Lab 8](../notebooks/08_anatomical_compiler.ipynb) (anatomical compiler) | the *direct* consumer; shape becomes a controllable target |
| [Lab 9](../notebooks/09_synthetic_morphology_wetlab.ipynb) (wet-lab forward programme) | items (i), (ii), and (iv) all benefit from shape gradients |
| [BETSE-JAX](../README.md#bioelectric-layer-companion) | sister differentiable engine; Phase 5 couples them |
| [`docs/foundation-models.md`](foundation-models.md) | orthogonal — FMs give cell-state priors, `cpjax` gives shape dynamics; they compose |
| [`docs/wetlab-program.md`](wetlab-program.md) | Cycle 1 (organoid injury) and Cycle 4 (PRINTESS sweep) produce shape data that `cpjax` consumes as observations for inverse design |

---

## 9. Cross-references

- [`docs/computational-roadmap.md`](computational-roadmap.md) §3 — strategic framing, current "deferred" status.
- [`docs/wetlab-program.md`](wetlab-program.md) — the wet-lab cycles that would un-defer this.
- [`docs/foundation-models.md`](foundation-models.md) — companion direction (FM priors); composes cleanly with `cpjax`.
- [BETSE-JAX](../README.md#bioelectric-layer-companion) (`~/Workspace/betse-unified`) — the architectural precedent; same sister-repo pattern.
- `~/Workspace/cpjax/` — the stub repo (Phase 0–5 placeholders, README, pyproject).
- COMBINE / MorpheusML / BioModels ecosystem refs in [`REFERENCES.md`](../REFERENCES.md).

---

## 10. Decision triggers — when to un-defer

Start Phase 0 when **any** of the following happens:

- A wet-lab cycle ([`docs/wetlab-program.md`](wetlab-program.md)) produces *segmented shape data* (Cycle 1 organoid injury, Cycle 4 PRINTESS sweep) — the dry side needs a CP layer to consume it quantitatively.
- The §4.3(i) bioprinting maturation programme becomes a *committed* experimental track — that's literally the shape inverse problem this engine is built for.
- BETSE-JAX requests a shape-layer coupling that no existing tool can deliver.
- An external collaborator (CC3D / Morpheus team, Lim Lab on synthetic morphogens, Gartner Lab on bioprinting) makes shape inverse design the bottleneck.

Until then: the stub stands; the design is recorded; no new model-side work is committed.
