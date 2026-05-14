# CURE audit — where the project complies, where the gaps are

*Mapping the **anatomical-compiler** project against Sauro et al. 2025 "From FAIR to CURE" (arXiv:2502.15597) — the COMBINE community's guidelines for **C**redible, **U**nderstandable, **R**eproducible, **E**xtensible computational biology models. Written 2026-05-13.*

The CURE pillars are baselines for *publication-grade* model artifacts; the paper's audience is the COMBINE / SBML / CellML / SED-ML / BioModels community. This project sits adjacent to that community — `hgx` regulomes are machine-learned not hand-coded, and the educational track lives in JAX rather than Tellurium/VCell/openCARP — so the audit is **partial alignment**, not "we shipped an SBML file." But many of the baseline checkboxes already tick, and the gaps are tractable.

This audit is also where the project formalises the [cpjax](../README.md#bioelectric-layer-companion)-side **CURE-compliance** provenance-test pattern (per-benchmark `provenance.toml` + SHA-256 manifests + checksum-verified trajectory caches — see `~/Workspace/cpjax/docs/plan.md` Phase 0). The same idea works upstream: every benchmark in this repo should have an explicit provenance record.

---

## 1. Credible — *does the model match reality, and is the construction sound?*

| baseline requirement | status | where it lives |
|---|---|---|
| Define objectives, scope, biological question | ✅ | [`publication/paper.Rnw`](../publication/paper.Rnw) §1 + each [`notebooks/0*–10`](../notebooks/) lab opens with explicit framing |
| Consistent notation across the project | ✅ | The [three-identifiabilities disambiguation](../notebooks/README.md) is the canonical example; structural / module / practical / fidelity are kept distinct everywhere |
| Verification against other simulators | ⚠️ partial | BETSE-JAX validated forward-parity vs original BETSE; lab notebooks not cross-verified against Tellurium / VCell |
| Validation against experimental data | ✅ where claimed | [Lab 3](../notebooks/03_benchmarking_fidelity.ipynb) fidelity-triple on Pollen organoid; [Lab 4](../notebooks/04_modularity_identifiability.ipynb) MII on Fleck regulome; [`figures/*.json`](../figures/) committed |
| Sensitivity / uncertainty analysis | ⚠️ partial | [Lab 7](../notebooks/07_structural_identifiability.ipynb) does structural identifiability; sensitivity analysis is in [Lab 6](../notebooks/06_control_theory.ipynb) for control inputs; no global posterior uncertainty on the Neural ODE yet (`docs/computational-roadmap.md` §5 flags this) |
| Document limitations | ✅ | Every lab's "what this is and isn't" closer; the RMT ablation drop-verdict; the EIG ablation's regime caveats |

**Additional criteria (Table 1 of Sauro 2025):**

| | status |
|---|---|
| Validation | ✅ |
| Verification | ⚠️ partial — JAX/diffrax solver correctness is implicit, no cross-simulator check |
| Uncertainty | ⚠️ partial — no global posterior; some ablation σ via multi-seed |
| **Provenance** | ✅ this is the project's *strongest* CURE alignment — every dataset has its source / cache hash; [`docs/dgx-spark-setup.md`](dgx-spark-setup.md) ties h5ad → cached `.npy` via SHA-8 in manifests |
| Annotation | ⚠️ partial — gene symbols are present, but no MIRIAM / SBO / KiSAO terms |
| Assumptions | ✅ inline in every lab |
| Purpose | ✅ explicit |
| Scope | ✅ explicit |
| Unbiased calibration | ✅ benchmarks declare train/test splits |

**Recommended tools they list, project alignment:**

- **BioSimulations** (cross-simulator verification service) — *gap*; the Hill-ODEs of [Lab 1](../notebooks/01_gene_circuit_dynamics.ipynb) and the repressilator demo could be SBML-ified and run against the BioSimulations service for round-trip verification.
- **MEMOTE** (genome-scale metabolic model QA) — not applicable (no metabolic models).
- **SBML / CellML** for the deep-biophysics layer — *gap*; addressed below.
- **AIC/BIC** for model selection — *gap*; the project uses transfer-r, F1, Spearman ρ instead. Defensible: these are predictive metrics on held-out data, the gold standard. But AIC/BIC reporting would be cheap.

---

## 2. Understandable — *can a reader / downstream user actually grasp the model?*

| baseline requirement | status | where it lives |
|---|---|---|
| Machine-readable + human-readable representation | ⚠️ partial | The `hgx` regulomes are machine-readable (Python objects, exportable); Lab 1's Hill ODEs are Python code, not SBML. The Hypergraph Neural ODE is *neural*, not symbolically expressible by design — this is a structural mismatch with CURE's "explicit representation not intertwined with implementation" expectation; documented honestly in [`docs/computational-roadmap.md`](computational-roadmap.md) §5 |
| Comprehensive documentation | ✅ | [`publication/paper.Rnw`](../publication/paper.Rnw) (canonical), [`ROADMAP.md`](../ROADMAP.md), [`notebooks/README.md`](../notebooks/README.md), per-doc planning files in `docs/` |
| Repository submission | ⚠️ partial | GitHub yes; BioModels / ModelDB no — and probably won't fit those (no SBML), see below |
| Annotation / comments | ✅ | dense in every lab |
| Document assumptions | ✅ | inline in every lab |
| Graphical illustration | ✅ | every lab figure is generated from code, captioned, and committed for the publication PDF |

**Six-level hierarchy of understanding (Figure 1 of Sauro 2025):**

| level | description | project alignment |
|---|---|---|
| 1 | Purpose, objectives, inputs, outputs | ✅ every lab |
| 2 | System components | ✅ |
| 3 | Interactions | ✅ |
| 4 | Mathematical description | ✅ for Labs 1/4/5/6/7/8/10; Hypergraph Neural ODE is a *parameterised* model so the math is the architecture + the loss, not a closed-form ODE |
| 5 | Evaluation methodology | ✅ |
| 6 | General theory | ✅ — the three-identifiabilities + anatomical-compiler framing |

**Gap:** **SBML / Antimony export of Lab 1's Hill ODEs and the repressilator demo.** Cheap, high-leverage CURE move — a single `scripts/export_lab1_sbml.py` that emits Antimony (which round-trips to SBML via Tellurium) would make Lab 1 a BioModels-submittable artifact and demonstrate the project's *capable* of standards alignment when the model is closed-form. The Hypergraph Neural ODE genuinely *isn't* SBML-shaped, and we should say so explicitly — CURE doesn't require what's structurally impossible, it requires that the choice be documented.

---

## 3. Reproducible — *can someone else run this and get the same numbers?*

| baseline requirement | status | where it lives |
|---|---|---|
| Use community standards | ⚠️ partial | uv-managed Python, JAX/diffrax/optax/equinox; no SBML/CellML on the closed-form side yet |
| Open code / data / models | ✅ | GitHub public, MIT-licensed |
| Version specification | ✅ | `pyproject.toml` pins JAX, scanpy, etc.; `uv.lock` is committed |
| Containerization | ❌ gap | No Dockerfile / Apptainer / Nix flake. The DGX-Spark recipe in [`docs/dgx-spark-setup.md`](dgx-spark-setup.md) is a step toward this — could be hardened into a container spec |
| Reproducibility of benchmarks | ✅ | every benchmark in `scripts/*.py` writes its result to `figures/*.{json,md}` with input fingerprints; multi-seed runs reported (e.g. `figures/perturb_eig_ablation.md`, `figures/edge_prior_ablation.md`) |

**Strongest CURE compliance — the extractor manifest pattern.** [`scripts/fm_embed.py`](../scripts/fm_embed.py), [`scripts/fm_edges_seq.py`](../scripts/fm_edges_seq.py), and [`scripts/fm_perturb_scgpt.py`](../scripts/fm_perturb_scgpt.py) all write a `manifest.json` next to every cached `.npy` with: model name, mode (real/stub), input file SHA-8, citation, checkpoint identifier, seed, version. **This is exactly the CURE-style provenance that Sauro 2025 calls "the project's strongest credibility lever."** Same pattern is already operational in the [`cpjax`](../README.md#bioelectric-layer-companion) Phase-0 oracle harness (per-benchmark `provenance.toml` + Manifest dataclass with SHA-256 + checksum-verified trajectory caches).

**Gap to close — Containerisation.** A `Dockerfile` (or `flake.nix`) building the project's Python env + the optional FM dependencies (geneformer, scgpt, evo-model, borzoi-pytorch, biopython, JASPAR) would make the DGX Spark recipe + the project's CI trivially reproducible. Estimated effort: half a day. Highest-leverage single CURE move.

---

## 4. Extensible — *can other people build on this without reinventing the substrate?*

| baseline requirement | status | where it lives |
|---|---|---|
| Open modeling standards | ⚠️ partial | The `hgx` and `jaxctrl` substrates are open and modular; not SBML |
| Separation of model code from runtime | ✅ | `scripts/` is the runtime, `hgx`/`jaxctrl`/`betse-unified`/`cpjax` are model substrates, `notebooks/` is the teaching/inspection layer |
| Open-source licensing | ✅ | MIT for `anatomical-compiler` and `betse-unified`; Apache-2.0 for `cpjax` |
| Component reusability | ✅ | `extract()` / `extract_edge_scores()` / `predict_kd_responses()` are the consumer APIs; downstream code is unchanged between stub and real mode — this is what the CURE paper means by "model as software function callable by other code" |

The project's design — `hgx` (regulome substrate) + `jaxctrl` (control) + `betse-unified` (bioelectric) + `cpjax` (shape, planned) — *is* the extensibility argument. Each is a separately-installable JAX package; each has its own `pyproject.toml`. The anatomical-compiler is the integration layer, not the substrate.

**Gap:** **A `MODEL_CARD.md`** in the spirit of Mitchell et al. 2019 (which CURE-Understandable §1 implicitly endorses) — a per-major-artifact card (Hypergraph Neural ODE, MII regulome, fidelity-triple predictor, FM-prior cache) documenting: intended use, training data, performance metrics, limitations, ethical considerations. The project already has all of this content scattered across notebooks; a `MODEL_CARD.md` is just the index.

---

## Priority list — what to do

Ranked by leverage / cost:

| # | action | effort | CURE pillar | gain | status |
|---|---|---|---|---|---|
| 1 | **Dockerfile** with the project's full env (incl. real-mode FM deps) | half a day | R (containerization) | reproducibility on any machine + DGX Spark + cloud | **✅ landed 2026-05-13** as [`Dockerfile`](../Dockerfile) (two-stage: `baseline` CPU-only + `fm` for DGX), [`.dockerignore`](../.dockerignore); baked-in build-time smoke test runs `ablate_edge_priors.py` + `ablate_perturb_eig.py` so a broken image fails to build |
| 2 | **`scripts/export_lab1_sbml.py`** — Antimony / SBML export of the Hill ODEs and the repressilator demo + a `scripts/verify_against_biosimulations.py` round-trip test | 1 day | C (verification) + U (standards) | demonstrates the project *can* speak SBML when the model is closed-form; "Lab 1 is BioModels-submittable" is a meaningful claim | pending |
| 3 | **`MODEL_CARD.md`** at repo root — index of the major artifacts with intended-use / limitations / metrics | half a day | U (understandability) + E (reuse) | the missing summary table | **✅ landed 2026-05-13** as [`MODEL_CARD.md`](../MODEL_CARD.md); 8 cards (Hypergraph Neural ODE / MII / fidelity-triple predictor / Lab-6 controllability / anatomical compiler / FM-prior caches / BETSE-JAX / cpjax) with intended-use / training data / metrics / limitations / references each |
| 4 | **MIRIAM / SBO annotations** on the `hgx` regulomes — at least the gene-symbol → Ensembl + species → NCBI taxon mappings | 1 day | C (annotation) | makes the regulome substrate semantically queryable | pending |
| 5 | **OMEX / COMBINE-archive bundling** for the published benchmarks — Pollen-fidelity, kidney-modularity, edge-prior ablation each packed as an OMEX | 1 day | R (community standards) | submittable to BioModels for the closed-form parts | pending |
| 6 | **Cross-simulator verification** for Lab 1 — round-trip through Tellurium + VCell + openCOR, confirm trajectories match within numerical tolerance | 1–2 days | C (verification) | the strongest credibility claim available | pending |

Items 1 and 3 are done (the half-day each); items 2, 4, 5, 6 are progressively more invasive but each remains scoped to a single artifact.

---

## What's structurally not CURE-aligned and why that's OK

The Hypergraph Neural ODE ([Lab 5](../notebooks/05_hypergraph_neural_odes.ipynb)) is a *parameterised* model — its weights are learned, not specified as algebraic kinetics. SBML / CellML are *symbolic* model representations; there's no SBML for a 50,000-parameter Equinox module. This is a fundamental shape mismatch, not a compliance failure.

The CURE paper acknowledges this: "Models embedded in executable code (MATLAB, Python) without explicit representation are difficult to reuse." Their answer is "use SBML where possible, document why not where not." The project's answer is the same: SBML for Lab 1; honest documentation for Lab 5; the *output* of Lab 8 (the anatomical compiler) is meant to compile *to* a standard format (PhysiCell grammar / SBML-Qual / a wet-lab cycle ticket — see [`docs/differentiable-cp.md`](differentiable-cp.md) §3, [`docs/wetlab-program.md`](wetlab-program.md)). That's CURE-aligned-by-design.

---

## Cross-references

- **Sauro et al. 2025** — *From FAIR to CURE: Guidelines for Computational Models of Biological Systems*, arXiv:2502.15597. The source paper.
- [`REFERENCES.md`](../REFERENCES.md) — bibitem to add.
- [`docs/dgx-spark-setup.md`](dgx-spark-setup.md) — the closest thing to a CURE-aligned reproducibility recipe the project has today; should grow into a Dockerfile.
- [`docs/foundation-models.md`](foundation-models.md) — the FM-prior pipeline manifests are already CURE-Reproducible-shaped.
- [`docs/computational-roadmap.md`](computational-roadmap.md) §5 — Bayesian Neural ODEs would close the "uncertainty quantification" gap on Lab 5.
- `~/Workspace/cpjax/docs/plan.md` — Phase 0 already has explicit "CURE-compliance" tests on the benchmark `provenance.toml` files; that pattern propagates upstream.
- COMBINE community / BioSimulations / BioModels are the audience this audit positions the project for.
