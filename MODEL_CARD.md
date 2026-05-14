# Model card — anatomical-compiler artifacts

*Index of the major computational artifacts in this repository, with intended use, source data, performance metrics, limitations, and references. Follows the Mitchell et al. 2019 "Model Cards for Model Reporting" template, adapted for scientific computing artifacts (mechanistic models, learned models, and analysis pipelines). Written 2026-05-13.*

Required reading: this card lives next to [`README.md`](README.md), [`ROADMAP.md`](ROADMAP.md), [`REFERENCES.md`](REFERENCES.md), and the [CURE audit](docs/cure-audit.md) — the audit lists where each artifact does and doesn't meet [Sauro et al. 2025's CURE pillars](https://arxiv.org/abs/2502.15597) (ref. 99a). This card is the *index*; the audit is the *gap analysis*.

The project's artifacts split into three classes:

- **Mechanistic / closed-form** — Hill ODEs (Lab 1), the LQR plant (Lab 6) — SBML-shaped in principle, structurally CURE-Understandable; current SBML export is the priority-2 item in the audit.
- **Learned / parameterised** — the Hypergraph Neural ODE (Lab 5), the FM-prior caches (UCE / Geneformer / scGPT / Evo / Borzoi), the perturbation predictor (Lab 3) — these are 10³–10⁵-parameter learned models with no closed-form representation; CURE's recommendation for them is "use SBML where possible, document why not where not."
- **Diagnostic** — the MII regulome metric (Lab 4), structural identifiability (Lab 7), the cancer-as-loss-of-MII signature (Lab 10) — pure analysis on a substrate; provenance and reproducibility are the relevant pillars.

---

## 1. Hypergraph Neural ODE — learned regulatory-flow plant

**Substrate:** [`hgx`](https://github.com/m9h/hgx) (JAX/Equinox hypergraph neural networks).
**Lab:** [Lab 5 — `notebooks/05_hypergraph_neural_odes.ipynb`](notebooks/05_hypergraph_neural_odes.ipynb).
**Pipeline:** [`scripts/organoid_hgx_colab.ipynb`](scripts/organoid_hgx_colab.ipynb) is the end-to-end Colab/GPU reproduction.

**Intended use.** A learned dynamical-system plant over a single-cell regulatory graph; integrated forward via `diffrax` to predict cell-state trajectories under perturbation, used downstream by [Lab 6](notebooks/06_control_theory.ipynb) (controllability / LQR) and [Lab 8](notebooks/08_anatomical_compiler.ipynb) (anatomical compiler / optimal control). **Not** intended as a published mechanistic model; it's the *learned* counterpart to a Hill-ODE system.

**Training data.** The Fleck et al. 2023 cerebral-organoid regulome (Pando-inferred GRN over scRNA-seq + ATAC), the canonical regulome substrate this project uses.

**Performance metrics.** Forward-pass speed ~ms per cell per timestep on GPU; benchmark accuracy reported per perturbation in the [Lab 3](notebooks/03_benchmarking_fidelity.ipynb) fidelity-triple framework (transfer-r ≈ 0.13 on cross-context held-out perturbations — the project's headline imperfect-prior number, calibrated to the regime where the FM-prior pipeline adds value).

**Limitations.**
- **Not SBML-representable** — 10⁴–10⁵-parameter Equinox module; the CURE response is "document why not + emit Lab-8 output in a standard format" (see [`docs/cure-audit.md`](docs/cure-audit.md) §2 and §"what's structurally not CURE-aligned").
- **Point-estimate uncertainty** — no posterior; [`docs/computational-roadmap.md`](docs/computational-roadmap.md) §5 flags Bayesian Neural ODEs as the gap.
- **Tissue specificity** — trained on a specific organoid regulome; transfer to a fetal-kidney context is the [Lab 3](notebooks/03_benchmarking_fidelity.ipynb) transfer-r ≈ 0.13 question.

**References.** Lab 5 notebook; the `hgx` repo; ref. 86 (CHOOSE / Fleck 2023) for the substrate.

---

## 2. MII (Module Identifiability Index) — regulome structural diagnostic

**Substrate:** Hodge $L_0$/$L_1$ Laplacian on the clique-expansion of the regulome.
**Lab:** [Lab 4 — `notebooks/04_modularity_identifiability.ipynb`](notebooks/04_modularity_identifiability.ipynb); the cancer-diagnostic re-use is [Lab 10](notebooks/10_cancer_module_identifiability.ipynb).
**Pipeline:** `scripts/benchmark_kidney_modularity.py`, `scripts/test_nitmb_modularity.py`.

**Intended use.** A structural readout on a regulome's *module identifiability* — how cleanly its TF-target structure separates into orthogonal regulatory programs. Differentiates organoid vs blueprint vs bioprinted tissue (the project's "fidelity-triple" benchmark), and (Lab 10) flags cancer as a loss of MII.

**Source data.** Same Fleck 2023 regulome substrate as the Hypergraph Neural ODE, plus the three kidney-organoid datasets (`r0`, `r40`, fetal kidney — Gartner Lab DPAC + Tang glioblastoma + Lawlor bioprinted kidney).

**Performance metrics.** MII baseline: kidney 0.31, brain organoid 0.37, fetal kidney 0.23 at canonical scale. **Verified robust to RMT preprocessing** ([`figures/rmt_ablation_kidney_modularity.{json,md}`](figures/rmt_ablation_kidney_modularity.md); see [the project's RMT-ablation memo](README.md)).

**Limitations.**
- **One real comparison only** ([Lab 10](notebooks/10_cancer_module_identifiability.ipynb) §"the one real comparison") — the project documents this honestly.
- **Coarse metric** — `mii_heuristic` is rank-friendly but doesn't pick up the cliffs `relative_eigengap` does; both are reported.
- **Structural, not noise-driven** — confirmed by the RMT ablation drop-verdict; this is a *feature*, not a bug.

**References.** Lab 4 / Lab 10 notebooks; the project's NITMB framing (ref. 103); `figures/nitmb_modularity_report.json`.

---

## 3. Fidelity-triple perturbation predictor — Lab 3 baseline

**Substrate:** `hgx`'s `PerturbationPredictor` — a learned regulome-aware perturbation-response model.
**Lab:** [Lab 3 — `notebooks/03_benchmarking_fidelity.ipynb`](notebooks/03_benchmarking_fidelity.ipynb).
**Pipeline:** `scripts/compare_pollen.py`, `scripts/benchmark_advanced_fidelity.py`.

**Intended use.** Given a TF knockdown and a cell-state context, predict the post-perturbation expression response across all genes. Used as the workhorse benchmark for evaluating prior-knowledge interventions (FM priors, sequence-edge priors, RMT denoising) — i.e., it's the *measurement instrument* for "does intervention X help."

**Source data.** Pollen brain-organoid scRNA-seq with TF knockdown labels (per the fidelity-triple framework in the [whitepaper](publication/paper.Rnw) §2.6).

**Performance metrics.** **Transfer-r ≈ 0.13** on cross-context held-out perturbations (the published headline). With FM gene-feature priors in stub mode on the synthetic, the lift is +0.999 (a *floor* demonstration on a rigged held-out-gene task; the real-data lift is step 5 of [`docs/foundation-models.md`](docs/foundation-models.md), gated on DGX Spark execution).

**Limitations.**
- **The 0.13 transfer-r number is *the* exposed weakness** — it's why this project has a step-4 BO/EIG track at all.
- **Cross-context generalisation is the hard problem** — same-context perturbation prediction is much easier (and not the published number).
- The synthetic Lab 11 floor (one-hot 0.000 → Geneformer-stub +0.999) is *not* a prediction for real performance.

**References.** Lab 3 notebook; `figures/{advanced_fidelity,system_maturity}_results.json`; [`docs/foundation-models.md`](docs/foundation-models.md); ref. 86 (CHOOSE).

---

## 4. Linear controllability ranking — Lab 6 / `jaxctrl`

**Substrate:** [`jaxctrl`](https://github.com/m9h/jaxctrl) — differentiable control on inferred regulomes.
**Lab:** [Lab 6 — `notebooks/06_control_theory.ipynb`](notebooks/06_control_theory.ipynb); [Lab 7](notebooks/07_structural_identifiability.ipynb) certifies what the linearisation can identify.

**Intended use.** Rank TFs by their controllability-eigenvector centrality in the linearised regulatory plant. Gives a "high-leverage TF" list — candidates for synNotch / KO-rescue / morphoceutical interventions (the [wet-lab program](docs/wetlab-program.md) cycles).

**Source data.** A learned LTI surrogate of the Hypergraph Neural ODE around a chosen attractor (one of three: bioprinted kidney, fetal kidney, brain organoid).

**Performance metrics.** Reports the controllability Gramian's spectrum + the LQR optimal-input rank order; agreement with FM-based scGPT in-silico KD ranking is the basis of the step-4 EIG-rank acquisition function (`figures/perturb_eig_ablation.{json,md}` — Spearman ρ recovery vs ground truth, +0.029 over GREEDY at median wet-lab budget in the realistic regime).

**Limitations.**
- **Linearisation around one attractor** — different attractors give different rankings; Lab 6 documents this and Lab 7 quantifies the identifiability ridge.
- **No direct biological validation yet** — that's what the wet-lab cycles in [`docs/wetlab-program.md`](docs/wetlab-program.md) and the DGX-Spark real-mode runs are for.

**References.** Lab 6 notebook; the `jaxctrl` repo; `figures/perturb_eig_ablation.md`.

---

## 5. Anatomical compiler — Lab 8 optimal-control inverse

**Substrate:** `jaxctrl` + `hgx` + (planned) `cpjax` + (operational) [BETSE-JAX](https://github.com/m9h/betse-unified) — the JAX-differentiable plant stack.
**Lab:** [Lab 8 — `notebooks/08_anatomical_compiler.ipynb`](notebooks/08_anatomical_compiler.ipynb).

**Intended use.** Given a *target* tissue state (regulome attractor + optionally Vₘₑₘ pattern + optionally shape), solve for the actuation schedule (TF perturbations, bioelectric prepattern, drug-cocktail timeline, eventually CP-Hamiltonian parameters) that drives the system there. The project's *direct-OC* version; SBI inverse is a planned extension.

**Source data.** No training data per se; this is an *inference* artifact running on the plant learned by Lab 5 (Hypergraph Neural ODE) and the differentiable bioelectric solver in [BETSE-JAX](https://github.com/m9h/betse-unified).

**Performance metrics.** [BETSE-JAX inverse design](https://github.com/m9h/betse-unified): 10-cell tissue driven into an alternating ±20 mV / ±80 mV pattern with loss ~9e-4 → 1.36e-10 (`betse.science.jax.inverse.optimize_pattern`). Lab 8's regulome-side inverse design uses the same `jax.grad` machinery; numerical performance reported per benchmark (`figures/anatomical_compiler_*.json`).

**Limitations.**
- **Plant uncertainty propagates** — if the Neural ODE is wrong on the target region, OC produces an actuation that *the simulator* believes works but reality may not.
- **Combinatorial actuator-set selection is not built in** — exercise (e) of Lab 6 raises this; it's [`docs/computational-roadmap.md`](docs/computational-roadmap.md) §6 territory.
- **Shape control is a future direction** — when `cpjax` (see [`docs/differentiable-cp.md`](docs/differentiable-cp.md)) lands, Lab 8 generalises to shape; until then it controls regulome state.
- **Output format**: currently emits JAX arrays; the CURE-aligned upgrade (per [`docs/cure-audit.md`](docs/cure-audit.md)) is to emit **PhysiCell grammar sentences** (Johnson 2025, ref. 77c) for human-readable + standards-aligned downstream consumption.

**References.** Lab 8 notebook; [BETSE-JAX](https://github.com/m9h/betse-unified) sister repo; [`docs/differentiable-cp.md`](docs/differentiable-cp.md).

---

## 6. Foundation-model prior caches — node + edge + perturbation priors

**Substrate:** [`scripts/fm_embed.py`](scripts/fm_embed.py), [`scripts/fm_edges_seq.py`](scripts/fm_edges_seq.py), [`scripts/fm_perturb_scgpt.py`](scripts/fm_perturb_scgpt.py).
**Lab:** [Lab 11 — `notebooks/11_foundation_model_pipeline.ipynb`](notebooks/11_foundation_model_pipeline.ipynb).
**Doc:** [`docs/foundation-models.md`](docs/foundation-models.md), [`docs/dgx-spark-setup.md`](docs/dgx-spark-setup.md).

**Intended use.** Three categories of zero-shot priors over the regulome:

- **Node priors** — UCE cell embeddings (1280-d, Stanford+CZI), Geneformer gene embeddings (512-d, UCSF/Gladstone), scGPT cell embeddings (512-d, Toronto/Vector).
- **Edge priors** — motif (JASPAR PWM), Evo (Arc Institute log-likelihood), Borzoi (Google/Stanford Δ-track).
- **Perturbation priors** — scGPT in-silico KD (cells × genes response matrix).

Each tool ships in three modes — `real` (load HF checkpoint, GPU-required), `stub` (deterministic structure-preserving projection, no GPU), `auto` (try real, fall back to stub). The integration contract: extract once, cache as `.npy` + JSON manifest, consume downstream.

**Source data.** Whatever h5ad / edges CSV / promoters FASTA / TFs list is passed. The DGX-Spark recipe is in [`docs/dgx-spark-setup.md`](docs/dgx-spark-setup.md).

**Performance metrics.**
- Stub-mode floor on the held-out-gene benchmark ([Lab 11](notebooks/11_foundation_model_pipeline.ipynb)): one-hot 0.000 → Geneformer-stub +0.999.
- Stub-mode edge-prior ablation ([`figures/edge_prior_ablation.md`](figures/edge_prior_ablation.md)): Pando-alone F1=0.607 → combined at α=0.30 F1=0.716 (Δ=+0.109; +0.146 ± 0.032 across 5 seeds).
- Stub-mode BO/EIG ablation ([`figures/perturb_eig_ablation.md`](figures/perturb_eig_ablation.md)): EIG-rank vs GREEDY-by-prior, Δ=+0.029 Spearman ρ at median budget in the realistic prior-quality regime.
- **Real-mode numbers on actual Pollen / Biopunk data: not yet measured** — step 5 of the integration plan.

**Limitations.**
- **Stub-mode numbers are floors, not predictions.** The lift on real data depends on real checkpoint behaviour on real h5ad.
- **Stub vs real attribution.** Stubs are structure-matched random projections; they isolate "co-expression-aware features help" from "30 M-cell pretraining helps." Don't conflate.
- **No fine-tuning support** — the project consumes FMs as zero-shot priors; fine-tuning is an explicit non-goal (the budget rationale is in [`docs/foundation-models.md`](docs/foundation-models.md)).

**Provenance / CURE.** Each cache writes a `<stem>_<model>_manifest.json` next to the `.npy` with: model name, mode, dim, citation, checkpoint identifier, input SHA-8, seed, version. This is the project's **strongest CURE-Reproducible** artifact; same pattern is used in `cpjax`'s Phase-0 oracle harness (per-benchmark `provenance.toml` + Manifest dataclass with SHA-256 + checksum-verified caches).

**References.** [`docs/foundation-models.md`](docs/foundation-models.md) for the institutional catalogue; refs. 77, 77a–77d, 78a in [`REFERENCES.md`](REFERENCES.md) for the PhysiCell ecosystem this complements.

---

## 7. BETSE-JAX — differentiable bioelectric layer (sister repo)

**Substrate:** [`~/Workspace/betse-unified`](https://github.com/m9h/betse-unified) — JAX/diffrax rewrite of BETSE / BETSEE (Pietak & Levin 2016/2017).
**Doc:** [`notebooks/README.md`](notebooks/README.md) "Bioelectric-layer companion"; refs. 39a–39e.

**Intended use.** A `jax.grad`-able simulator of Vₘₑₘ, gap-junction, and ion-channel dynamics over a cell cluster. Couples to this project's [Lab 8](notebooks/08_anatomical_compiler.ipynb) anatomical compiler: target Vₘₑₘ pattern in → required ion currents out. Closes the **bioelectric layer** of the regulome ⇄ form coupling.

**Source data.** No training data; pure mechanistic model with hand-set parameters (Pietak 2016 calibrations).

**Performance metrics.** Inverse design loss 9e-4 → 1.36e-10 on a 10-cell alternating-polarity benchmark. Forward stepping ~2.2 µs/step via `lax.scan` whole-step JIT. 100% test coverage (`test_jax_{inverse,xenobot,grn,morphoceutical,conc,gj,integration,performance}.py`).

**Limitations.**
- Hand-set parameters (not learned from per-cell data).
- 2-D + small-cluster regime; 3-D scaling is future work.
- The shape layer (CP-style) is *not* in scope here — that's the future [`cpjax`](docs/differentiable-cp.md) sister repo.

**References.** Refs. 38, 39a–39e in [`REFERENCES.md`](REFERENCES.md); the [BETSE-JAX repo](https://github.com/m9h/betse-unified); [`notebooks/README.md`](notebooks/README.md) bioelectric section.

---

## 8. `cpjax` — differentiable Cellular Potts (planning stub, sister repo)

**Substrate:** [`~/Workspace/cpjax`](https://github.com/m9h/cpjax) — JAX-native CP engine, currently in Phase 0 (oracle harness).
**Doc:** [`docs/differentiable-cp.md`](docs/differentiable-cp.md).

**Intended use.** The shape-layer companion to BETSE-JAX. Three planned update kernels (Metropolis forward / REINFORCE exact / Gumbel-softmax + soft-lattice relaxed). When operational, it closes the **regulome → form** gap — Lab 8 generalises to shape control.

**Source data.** CC3D and Morpheus run as **oracles** on five canonical CP benchmarks (cell sorting, vasculogenesis, chemotaxis, gastrulation toy, tumour-stroma); MorpheusML import target for COMBINE-ecosystem interop.

**Performance metrics.** **Phase 0 in progress, 58 tests green** (per the cpjax `docs/plan.md`). Forward parity vs oracles, REINFORCE / Gumbel / soft-lattice gradient quality benchmarks are Phases 1–3 future work.

**Limitations.**
- Phase 1+ not started — currently a planning stub + Phase 0 oracle harness.
- Deferred per [`docs/computational-roadmap.md`](docs/computational-roadmap.md) §3 until wet-lab cycles produce shape data.

**CURE alignment.** The Phase-0 oracle harness has explicit *CURE-compliance* tests on the per-benchmark `provenance.toml` files (SHA-256-manifested trajectory caches, pre-dispatch manifests for crash safety, checksum-verified cache loaders). The template for upstream provenance work.

**References.** [`docs/differentiable-cp.md`](docs/differentiable-cp.md), [`docs/cure-audit.md`](docs/cure-audit.md), the cpjax sister repo.

---

## Ethical and scientific considerations

- **Use case.** This is a research artifact for developmental biology / synthetic morphology / regenerative-medicine inference. Not clinically deployed; not used to make patient-level decisions.
- **Dual-use surface.** The anatomical compiler is, in principle, a *design tool* for tissues. Bioethics review is the responsibility of the downstream wet-lab consumer; the wet-lab planning doc ([`docs/wetlab-program.md`](docs/wetlab-program.md)) is internal-Biopunk planning with HTGAA capstone framing.
- **Data provenance.** All training / benchmark datasets are public (CHOOSE, Pollen, kidney organoid trio, Toda synNotch, Krienen interneurons) and cited; no patient data; no biospecimen access required for the dry-side.
- **Bias / fairness.** The models do not predict individual outcomes; the population analyses use cell-type abstractions, not demographic features. The CURE-Credible "unbiased calibration" check applies in the benchmark-construction sense (train/test splits documented per benchmark).

---

## How to cite

If you use this project, please cite the whitepaper ([`publication/paper.Rnw`](publication/paper.Rnw)) and the relevant sister tools:

- **hgx** — [github.com/m9h/hgx](https://github.com/m9h/hgx)
- **jaxctrl** — [github.com/m9h/jaxctrl](https://github.com/m9h/jaxctrl)
- **BETSE-JAX (betse-unified)** — [github.com/m9h/betse-unified](https://github.com/m9h/betse-unified)
- **cpjax (planning)** — [github.com/m9h/cpjax](https://github.com/m9h/cpjax)
- **anatomical-compiler** — [github.com/m9h/anatomical-compiler](https://github.com/m9h/anatomical-compiler)

Full bibliography in [`REFERENCES.md`](REFERENCES.md). The CURE-aligned reporting framework: ref. 99a (Sauro et al. 2025).
