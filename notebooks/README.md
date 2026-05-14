# Notebooks — toward a course on computational synthetic morphology

This folder is the seed of an **educational track**: a sequence of notebooks meant to walk a
student (or a class) through the whole arc of the project — from "what is a regulome and why a
hypergraph" to "design an intervention that steers a tissue toward a target state." The idea is
that each notebook is a self-contained lab session, building on the last, ending in a small
exercise; together they cover the methods in `publication/paper.Rnw` and the experimental
programme in its §4.

## Context: Biopunk Lab × HTGAA 2026a

This educational track is run out of **Biopunk Lab**, the **West Coast node** of
[*How to Grow Almost Anything (HTGAA) 2026a*](https://2026a.htgaa.org), and supports the course's
two **Genetic Circuits** modules:

- **[Week 6 — *Genetic Circuits Part I: Assembly Technologies*](https://2026a.htgaa.org/2026a/course-pages/weeks/week-06/index.html)**
  (Doug Densmore, Traci Haddock): the molecular toolkit for *building* circuits — PCR, Gibson
  assembly, restriction digests — and circuit design/simulation on the Asimov Kernel platform; the
  assignment includes **recreating the repressilator**.
- **[Week 7 — *Genetic Circuits Part II: Neuromorphic Circuits*](https://2026a.htgaa.org/2026a/course-pages/weeks/week-07/index.html)**
  (Ron Weiss): genetic circuits that *compute and learn* — intracellular artificial neural networks
  / perceptrons implemented in cells.

Lab 1 below is the **computational companion to Week 6** — same repressilator (and Hill kinetics,
toggle switch, linear stability), in this course's Python toolchain, and then it shows you what to
do *after* you've recreated it: linearize it and *control* it. The hypergraph-neural-network strand
of this track (Labs 2 & 5) is the conceptual sibling of **Week 7** — Week 7 builds neural networks
*as* gene circuits; this track runs neural networks and control theory *on* gene networks. The full
dependency chain: **Elowitz & Bois, *Biological Circuit Design*** (the dynamical-systems
foundations, see below) → HTGAA Weeks 6–7 (assembling, then computing with, genetic circuits) →
**this track** (regulomes as hypergraphs → hypergraph neural networks → modularity/identifiability
metrics → network control / the *anatomical compiler*) → **wet-lab synthetic morphology**
(bioprinting, synthetic morphogenesis).

## What's here now

- **`00_setup.ipynb`** — **Setup** (start here): the **map** (the arc Setup → 1 → 2 → … → 10, one
  line each, with per-lab deps), the **prerequisites** (Python/numpy; **JAX — the differentiable-
  programming mindset, the load-bearing one**; linear algebra / the graph Laplacian; ODEs & dynamical
  systems / attractors; light mol-dev bio; Elowitz & Bois' *Biological Circuit Design* — with Lab 1
  as the bridge — and HTGAA Weeks 6–7 for the wet-lab-circuits context), an **env check** (required:
  `numpy`/`matplotlib`/`jax`/`scipy`; recommended/per-lab: `equinox`/`diffrax`/`optax`/`hgx`/`sympy`,
  optional: `jaxctrl`/`scanpy` — soft-imported, the labs reimplement what they need), the **data
  substrate** (loads the Fleck organoid regulome, or a synthetic fallback), **four toolchain hello-
  worlds** (`jax.grad` incl. *through a loop*; the hypergraph / clique-expansion blow-up; a
  differentiable ODE — RK4 + `jax.grad` through it; a 2-state plant — Kalman controllability + LQR via
  `scipy.linalg.solve_continuous_are`), the **dependency map** + the **three "identifiabilities"**
  disambiguation (module / structural / practical — and fidelity), and a starter exercise ("break one
  hello-world on purpose; then go to Lab 1 or Lab 2"; + an optional pointer into the BETSE-JAX
  companion at `~/Workspace/betse-unified`). Self-contained; runs in seconds.
- **`01_gene_circuit_dynamics.ipynb`** — **Lab 1** (the bridge from *Biological Circuit Design*):
  Hill functions, negative autoregulation (response-time), the toggle switch & bistability (via
  `jax.jacfwd`), the repressilator + the "linearize then LQR-control it" move — all in this course's
  toolchain (`diffrax`, `jax.jacfwd`/`jax.grad`, `jaxctrl`) rather than SciPy/Bokeh. Self-contained.
- **`02_regulomes_and_hypergraphs.ipynb`** — **Lab 2**: what a regulome is; load the Fleck
  incidence matrix; why a regulon is a *hyperedge*, not a clique (the clique-expansion blow-up
  *and* aliasing, worked on a toy example + the full regulome); `hgx` basics (`from_incidence`,
  node/edge degrees, star vs clique expansion, a `UniGCNConv` forward pass); a first structural
  readout (the hypergraph-Laplacian spectrum and its gap — the seed of the Module Identifiability
  Index); exercises (all-pairs Jaccard regulon overlap among the master TFs, heavy-tailedness of
  regulon sizes, graph degree vs hypergraph degree). Self-contained — reads `data/processed/`,
  falls back to a tiny synthetic regulome if absent.
- **`03_benchmarking_fidelity.ipynb`** — **Lab 3**: does the organoid regulome *predict* perturbations?
  The fidelity **triple** — in-domain magnitude r, transfer direction-accuracy, transfer r — instead
  of a single "accuracy". GRN/screen overlap (organoid's 720 regulons vs Pollen/Ding 2026's 44
  CRISPRi'd TFs → 34 shared, hypergeometric p ≈ 10⁻⁵); zero-shot transfer organoid → primary cortex
  (live on the key TFs the regulome ships KO predictions for, plus the precomputed 34-TF benchmark:
  **direction ≈ 83% with a trained `PerturbationPredictor`** — but raw regulon signs only ≈ 60% —
  vs **magnitude r ≈ 0.13** zero-shot, ≈ 0.36 within-primary); the per-cell-type breakdown (raw
  numbers near chance — aggregate accuracy can lie); why magnitude doesn't transfer = the *practical*
  shadow of Lab 7's *structural* result (the downstream-magnitude ridge → the SBI valley of Lab 8);
  what fidelity is *not* (≠ structural identifiability, ≠ module separability, ≠ transfer fidelity);
  exercises (wire in the real CROP-seq from `extract_cropseq.py`; the `advanced_fidelity` pattern
  scores — fidelity of *identity* vs of *perturbation*; cross-species conservation and why overlap is
  the weak test). Self-contained — reads `data/processed/`, `data/pollen/processed/`, `data/cropseq/*.csv`,
  `figures/pollen_*.json`, `figures/advanced_fidelity_results.json`; graceful fallbacks throughout.
  Full pipeline: `scripts/compare_pollen.py`, `scripts/benchmark_advanced_fidelity.py`.
- **`04_modularity_identifiability.ipynb`** — **Lab 4**: does the regulome *decompose into modules*,
  and how cleanly? Builds the **Hodge Laplacian** of the regulon hypergraph by hand ($L_0$ on genes
  = clique-expansion graph Laplacian, $L_1$ on regulons), reads off the harmonic null (connected
  components), the Fiedler value, and the low-lying spectral gap; defines the **Module Identifiability
  Index** two ways — the project's heuristic (mean of the first 10 spectral gaps / std of the first
  10 eigenvalues, *verbatim* from `scripts/test_nitmb_modularity.py`) and a normalised relative
  eigengap ∈ [0,1] — and shows the genome-scale regulome is *one tangled component* with weak module
  identifiability (the index discriminates *between* systems, it doesn't pull a clean module count out
  of a GRN); reproduces the cross-system NITMB ordering live from the committed Hodge spectra
  (**brain organoid > fetal kidney > bioprinted kidney** — self-organisation makes *sharper* modules
  than cell-by-cell printing); a light "neurogenic stop-signal" sweep (MII / #components along
  organoid pseudotime — the late bins fragment); what the MII is *not* (≠ structural/practical
  identifiability, ≠ fidelity, ≠ a unique number, ≠ the module *labels*); exercises ($L_1$ vs $L_0$ —
  the orderings can flip!; the eigengap as a module count + labelling; where multi-TF cooperativity
  lives; cancer as loss of module identifiability — the Lab 10 stretch). Self-contained — reads
  `data/processed/`, `figures/{kidney_modularity,nitmb_modularity}*.json`; synthetic "blocky"
  fallback. Pipeline: `scripts/benchmark_kidney_modularity.py`, `scripts/test_nitmb_modularity.py`,
  `scripts/06_topology.py`.
- **`05_hypergraph_neural_odes.ipynb`** — **Lab 5**: a cell type is a *stable state of the dynamics
  the regulome runs* (Kauffman / Huang — attractors = cell types; Waddington's landscape made
  literal). Fits a **Neural ODE** ($\dot x = f_\theta(x)$, $f_\theta$ a small MLP — a hand-rolled
  RK4 + `jax.grad` + `optax`, so no `diffrax` needed) on the organoid pseudotime timecourse in ~0.4 s;
  uses **per-gene rollout MSE** as a functional classifier — **stable structural drivers** (on the
  slow, identity-defining manifold — low MSE; the fate TFs cluster there) vs **transient stress
  responders** (off-manifold spikes — high MSE; the IEGs Fos/Jun/Egr1 in systems that have them) —
  with the committed kidney-IRI fit (`regenerative_flow_results.json`: Lhx1/Cdh1/Pax8 ≈ 0.05–0.11 vs
  Fos 4.4, Jun 1.5, Cd44 0.93, Atf3 0.80) and the activity-induced IEG timecourse
  (`learning_regulome_results.json`: peak-at-1 h) as the clean illustrations; the **Waddington
  picture** via a bistable toggle-switch demo (two basins, a separatrix, a "knockout" that crosses it
  → switched fate) + the organoid data's own version (`fate_probabilities.npy` along pseudotime;
  `perturbation_fates.npy` — FOXG1 and DLX2 are the biggest single-TF "landscape tilts"); ties to
  Lab 4 (modules dissolve ⇒ driver set shrinks) and Lab 3 (well-fit drivers ⇒ transferable KO
  directions) and forward to Lab 6/6 (control on the *learned* flow); what a Hypergraph Neural ODE is
  and isn't (it's the *flow*, not the parameters — deliberately structurally non-identifiable;
  the "hypergraph" is a weight-sharing *prior*, `hgx.LatentHypergraphODE`); exercises (ODE→SDE +
  CellRank fate entropy; Poincaré/hyperbolic latent space + Gromov δ; finer time T∈{20,50}; the hgx
  regulon-structured field; close the loop with `jaxctrl`). Self-contained — reads
  `data/processed/{temporal_expression,pseudotime_centers,fate_probabilities,perturbation_fates}.npy`
  + `figures/{learning_regulome,regenerative_flow}_results.json`; synthetic toggle-switch fallback.
  Pipeline: `scripts/04_temporal_dynamics.py` (the `diffrax` + `hgx.LatentHypergraphODE` + SDE +
  Poincaré version), `scripts/benchmark_learning_regulome.py`, `scripts/benchmark_regenerative_flow.py`.
- **`06_control_theory.ipynb`** — **Lab 6**: turns the regulome into a steerable plant $\dot x = Ax +
  Bu$ (a linear TF co-regulation network on the top-200 Pando regulons) and runs the network-control
  toolkit — implemented in pure JAX/SciPy so it runs without `jaxctrl`: the controllability matrix &
  **Kalman rank**, the **controllability Gramian** (average vs modal controllability — Gu et al.
  2015), **minimum-energy control**, **LQR** (Riccati), and the **minimum driver-node** count
  (Liu–Slotine–Barabási matching). Headline (the *control* mirror of Lab 7's *observability*
  result, by duality): **the curated master regulators don't control the network** — Kalman rank ≪
  200 from the 7 key TFs (≈9 at float32 tol, ≈30 at float64 — reproduces the committed
  `network_control_results.json` 9/200); **control leverage ≠ identity** — the highest-leverage
  single TFs (FOXC2/PRDM6/FOXC1, reproduced exactly) are *not* the developmental masters (only EOMES
  in the top 20); a single TF's modal controllability ≈ 0 (need a *set*); the early→late
  steer-to-target costs $E^\star\approx586$ / LQR cost-to-go ≈ 101 with full actuation, effectively
  uncontrollable from the masters alone (Gramian $\lambda_{\min}\approx0$). Notes structural ≠ exact
  controllability (LSB's generic $N_D\approx1$ vs the real rank — our $A$ is a Laplacian, non-generic
  weights). § on the three `jaxctrl` worked examples (`repressilator_control_demo.py`,
  `irma_sindy_lqr.ipynb`, `grn_hypergraph_drivers.ipynb` — the nonlinear / SINDy-surrogate /
  hypergraph-driver versions); exercises (linearise the Lab-5 ODE + control it; the SINDy/Koopman
  surrogate route; the unreachable subspace vs Lab 4's diffuse modules & Lab 7's unobservable
  directions; min-driver-nodes on the hypergraph; the control–robustness trade-off — Yan et al.
  2017). Self-contained — reads `data/processed/{incidence,tf_names,gene_names,tf_gene_indices,
  key_tf_indices,temporal_expression}.*` + `figures/network_control_results.json`; tiny synthetic
  2-state fallback. Pipeline: `scripts/benchmark_network_control.py`; the `jaxctrl` example notebooks.
- **`07_structural_identifiability.ipynb`** — **Lab 7**: a `sympy` structural-identifiability
  check (the Taylor-series / observability-rank test — the symbolic generalisation of `jaxctrl`'s
  linear `is_observable`/`observability_gramian`). Textbook sanity checks (one-pool ✓; Bellman &
  Åström's $k_1{+}k_2$ ✗ with the nullspace direction printed); the Lab-1 circuits (negative
  autoregulation, the repressilator — identifiable from a single output only if you also know the
  initial state, identifiable from all outputs regardless); the linear special case = `jaxctrl`'s
  observability rank, tied to the `benchmark_network_control.py` finding (the regulome's linear
  surrogate is not identifiable/controllable from the master regulators alone → the SBI posterior
  is correctly ridge-shaped); exercises (free Hill exponent → the $n$–$K$ confound; the IRMA GRN's
  minimal output set; a COMBOS compartmental example; the SBI loss-valley). Distinguishes the
  *structural*, *practical*, and *module* senses of "identifiability". Self-contained. (One cell —
  the 6-unknown repressilator — takes ~2–3 min.)
- **`08_anatomical_compiler.ipynb`** — **Lab 8** (the one Labs 1–7 build toward): Levin's **anatomical
  compiler** — *target tissue state in → actuation schedule out* — assembled at the level of a learned
  regulome surrogate. The four-stage stack: **plant** = a learned Neural ODE on the organoid timecourse
  (Lab 5, reused — hand-rolled RK4 + `jax.grad`, no `diffrax`); **system-ID** = a linear surrogate
  $\dot z\approx Az+c$ fit by least-squares on the rollout (the SINDy/Koopman slot) — which turns out
  *unstable* (max Re $\lambda\!\approx\!+9$; the committed kidney-IRI run +31.8, `surrogate_reliable=false`):
  a developmental trajectory linearised is a transient, not a stable plant — the *practical* face of
  Lab 7, and why Lab 6 had to *choose* $A=-L_{\rm sym}$; **controller** = an LQR warm-start on the
  clamped surrogate, then **direct optimal control on the *nonlinear* plant** — a piecewise-constant
  $u(t)$ on a chosen actuator set, minimise $\lVert x(T)-x_{\rm target}\rVert^2 + \lambda\lVert u\rVert^2$,
  Adam *through* the rollout; **validation** = re-integrate the learned ODE under $u^\star$. Demonstrated
  as a **knockout-rescue** (in-silico knock down FOXG1 at $t_0$ → the free rollout drifts off → compute
  the actuation that returns it to the wild-type endpoint): final-state error ↓ ~100% on the actuated
  TFs, ↓ ~39% overall — "you steer what you actuate" (Lab 6's controllability gap, again; the committed
  kidney-IRI run: ↓73% actuated, ↓21% all). § on what the compiler is/isn't (it's a *regulome state*,
  not an anatomy — the regulome↔form gap, where the cell-based simulators and the bioelectric layer come
  in; it's the *flow* you control, not $\theta$; the linear surrogate is a warm-start, not the controller;
  the objective/constraints are modelling choices) + "the arc — Labs 2→8 in one line each"; exercises
  (target a *fate basin* not a state vector — a toggle-switch fate-switch starter that crosses the
  separatrix; a SINDy/Koopman surrogate; MPC vs open-loop; couple the bioelectric layer via BETSE-JAX's
  `optimize_pattern`; actuator-set selection). Self-contained — reads `data/processed/{temporal_expression,
  pseudotime_centers}.npy` + key-TF metadata + `figures/anatomical_compiler_results.json`; synthetic
  toggle-switch fallback. Pipeline: `scripts/benchmark_anatomical_compiler.py` (the `hgx` + `diffrax`-adjoint
  + `jaxctrl` production version) and `scripts/benchmark_network_control.py` (the linear warm-up, Lab 6);
  the `jaxctrl` example notebooks; the bioelectric companion at `~/Workspace/betse-unified` (`betse.science.jax.inverse`).
- **`09_synthetic_morphology_wetlab.ipynb`** — **Lab 9** *(stretch)*: the §4.3 wet-lab forward
  programme, reframed so **every modality is one optimal-control problem on the Hypergraph Neural ODE**
  (Lab 8) — what changes is the actuator $B$ and the readout layer: *programmed* (synNotch / synthetic
  morphogens — Morsut 2016, Toda 2018/2020; Lim Lab; readout = the committed `toda_results.json`
  projection), *printed* (conformation / 4-D bioprinting — Feinberg/Lewis/Skylar-Scott/Gartner; readout
  = `gartner_results.json` man/r0/r40 → lineage maturity, and `system_maturity_results.json`),
  *bioelectric* ($V_{\rm mem}$ prepattern — Levin/Mafe; the BETSE-JAX refactor, `optimize_pattern`),
  *agential* (xenobots/anthrobots — Levin/Bongard; Gumuskaya 2024; readout = `anthrobot_results.json`).
  Demonstrates the control side with one shared bistable plant (the Lab-1/4/6 toggle, used four ways
  — a synNotch input that flips the fate; a static "print-geometry" parameter that biases which
  attractor it self-organises into; a toy $V_{\rm mem}$→GRN two-layer steer; a minimal-sender-fraction
  synNotch-circuit starter) and the readout side with the committed benchmark panels; the
  **model-in-the-loop design cycle** (design → simulate → optimise → build → read out → refine; the
  whitepaper §4.3) and what's real (the readouts) vs aspirational (closing the loop; the regulome↔form
  gap). Exercises (end-to-end synNotch circuit design + minimal sender fraction; multi-dim print
  geometry → target lineage mixture; the full two-layer bioelectric→GRN compiler via BETSE-JAX; a
  cross-system "readiness" score from the fidelity triple + MII + driver stability; the closed loop on
  a real dataset). Self-contained — reads `figures/{toda,gartner,anthrobot,advanced_fidelity,
  system_maturity}_results.json`; synthetic fallbacks. Pipeline: `scripts/benchmark_{toda_morphogenesis,
  gartner_4d,anthrobot_fidelity,advanced_fidelity,system_maturity}.py`; `~/Workspace/betse-unified`.
- **`10_cancer_module_identifiability.ipynb`** — **Lab 10** *(stretch, final)*: the course's metrics
  turned **diagnostic**. The TOFT / atavistic register of cancer (Soto & Sonnenschein; Trigos *et al.*;
  Levin 2021 — alongside, not instead of, the mutational view): **cancer ⇒ the regulome's module
  identifiability falls** (the Fiedler-region gap dissolves) and, dynamically, **the homeostatic
  driver set comes loose while the transient/proliferative programs become persistent** — the
  regeneration arrow of [Lab 5] reversed ("the wound that doesn't heal"; the atavistic stress/
  proliferation machinery freed from multicellular constraint). Reuses Lab 4's MII machinery
  (`mii_heuristic`, `relative_eigengap`) on the one real comparison we have (Lab 4's organoid >
  blueprint > bioprinted trio — small differences, the metric is coarse) + a clearly-labelled
  schematic of the spectral signature (clear-cliff → soft-cliff → ramp; the `relative_eigengap` tracks
  it and ranks the cartoon in the predicted order, the coarse `mii_heuristic` doesn't — Lab 4's caveat
  live) + the dynamical face (the committed kidney-IRI driver split, and a toy "cancer flip" of it) +
  the **constructive flip** — the [anatomical compiler] reframing therapy as *steer the cell back into
  a high-MII, differentiated attractor* ("normalise, don't (just) kill" — differentiation therapy;
  the bioelectric instance, repolarisation / the BETSE-JAX `optimize_pattern`; the pharmacological
  one, "morphoceuticals" — Pio-Lopez & Levin 2023) + the **diagnostic flip** (the readout that
  certifies an engineered tissue, run in reverse, flags a diseased one — Davies's "know when you
  have"). Exercises (run the *real* primary→organoid→tumour-organoid→cancer-line gradient — the
  headline open question; which Hanahan–Weinberg hallmark predicts the MII drop; the bioelectric angle
  — $V_{\rm mem}$ vs MII; the **normalisation control problem** — a working two-attractor "differentiated
  ⇄ tumour" toggle + the dose to revert it, with Lab 8's machinery; atavism, spectrally — the
  unreachable subspace vs the unicellular-ancestral gene set). Self-contained — reads
  `figures/{kidney_modularity_results,nitmb_modularity_report,regenerative_flow_results}.json`;
  synthetic fallbacks. Pipeline: `scripts/{benchmark_kidney_modularity,test_nitmb_modularity,
  benchmark_disease_enrichment,benchmark_tf_disease,benchmark_tang_bioprinting,validate_choose}.py`;
  `~/Workspace/betse-unified`.
*(Not a course lab — a reproduction/benchmark artifact, living in `scripts/`:* **`scripts/organoid_hgx_colab.ipynb`** *— the GPU/Colab notebook that runs `hgx` on the Fleck et al. (2023) cerebral-organoid regulome end-to-end (preprocessing → the 7 publication figures → the 5 biological-validation checks → the hgx-vs-DHG speed/accuracy benchmark). The labs reference it where they touch the real pipeline; it's the "run the whole thing once on a GPU" companion to `scripts/run_all_real.py`.)*

- **`05_5_disentangled_neural_ode.ipynb`** — **Lab 5.5** *(add-on, between Lab 5 and Lab 6)*: the DeepMind
  face-patch analogue applied to cell-state. The question: does an unsupervised representation
  learner recover axis-aligned, separable factors of variation in the regulome's latent space,
  or does it entangle them across dimensions? Trains a vanilla AE and a β-VAE on the same
  4-factor synthetic, computes per-(latent-dim, true-factor) |Pearson r| under Hungarian
  permutation alignment, reports the disentanglement gap. β-VAE gives a *modest* gain on
  the clean linear synthetic (≈+0.06 in the diag-vs-off-diag metric) — the lab is honest
  about why (linear AEs settle into PCA-like bases from random init; β is a tiebreaker on
  this regime, not a transformer). The big wins live on nonlinear / sparse-loading / high-dim
  data — the exercises invite the student to find them. Architecturally: a β·KL term on
  Lab 5's `hgx` Hypergraph Neural ODE latent would give an interpretable coordinate system
  for Lab 6's controllability + Lab 8's anatomical compiler. Adds a **fourth identifiability**
  to the project's diagnostic vocabulary (latent-level, distinct from structural / module /
  practical — all three of which concern the *system*'s recoverability from data, where this
  one concerns the *model*'s representation choice on top of it).

- **`11_foundation_model_pipeline.ipynb`** — **Lab 11** *(add-on, post-Lab-10)*: foundation-model priors for
  perturbation prediction. Operationalises [`docs/foundation-models.md`](../docs/foundation-models.md) — the
  catalogue across UCSF (Geneformer, Gladstone), Stanford/CZI/Arc (UCE, Evo 2, Borzoi), Berkeley (scVI
  lineage), Toronto/Vector (scGPT). The integration contract: **extract once via
  [`scripts/fm_embed.py`](../scripts/fm_embed.py), cache as `.npy`, consume downstream as numpy arrays** —
  no fine-tuning, the notebook code is unchanged between `--mode stub` and `--mode real`. Headline
  demonstration: held-out-gene perturbation-response prediction, one-hot baseline (transfer-r = 0,
  zero signal on unseen genes by construction) vs Geneformer-style co-expression-aware embedding
  (transfer-r ≈ +1.0 on the synthetic; the *floor* of what real Geneformer adds, since the stub is a
  structure-matched random projection lacking the 30 M-cell pretraining). Plus a UCE-stub property
  check on the cell-level arm (1280-d cell embedding contains the latent factor structure;
  cross-batch / sparsity / novel-cell-type robustness are properties of the pretrained checkpoint
  the stub can't simulate — flagged honestly). Self-contained — synthetic 4-factor regulome, no
  GPU, runs in seconds. Pipeline: [`scripts/fm_embed.py`](../scripts/fm_embed.py) (CLI subcommands
  `uce` / `geneformer` / `scgpt`; modes `real` / `stub` / `auto` with informative fallback).
  Companion docs: [`docs/foundation-models.md`](../docs/foundation-models.md),
  [`docs/computational-roadmap.md`](../docs/computational-roadmap.md) §1.

Companion worked examples live in the **`jaxctrl`** repo (the control-theory layer):
[`examples/repressilator_control_demo.py`](https://github.com/m9h/jaxctrl/blob/main/examples/repressilator_control_demo.py)
(quench a 3-gene oscillator: linearize → controllability → LQR → quench the nonlinear flow →
`jax.grad` w.r.t. a kinetic parameter),
[`examples/irma_sindy_lqr.ipynb`](https://github.com/m9h/jaxctrl/blob/main/examples/irma_sindy_lqr.ipynb)
(a GRN end-to-end: simulate an IRMA-topology Hill-ODE → `SINDyOptimizer` linear surrogate →
controllability → LQR "drug input" → sensitivity analysis), and
[`examples/grn_hypergraph_drivers.ipynb`](https://github.com/m9h/jaxctrl/blob/main/examples/grn_hypergraph_drivers.ipynb)
(GRN-as-hypergraph: `minimum_driver_nodes`, `controllability_profile`, the `control_energy`
landscape, `HypergraphControlSystem` + LQR — "which TFs must I perturb to control this regulon?").

## Recommended background — and how this differs from "Biological Circuit Design"

*(The full, runnable version of this section — prerequisites, the env check, the toolchain hello-
worlds, the dependency map — is **[Setup](00_setup.ipynb)**. The short of it: you need Python +
numpy/matplotlib, **JAX** (the differentiable-programming mindset — the one that really matters),
linear algebra (the graph Laplacian), and ODEs / dynamical systems (attractors, linear stability);
you do **not** need prior `hgx` / `jaxctrl` / `diffrax` / `sympy`. The gene-circuit-dynamics
foundations are below.)*

If you want the *dynamical-systems-of-gene-circuits* foundations this course leans on but does not
re-derive — Hill kinetics and cooperativity, autoregulation, the toggle switch and bistability,
feed-forward loops, the **repressilator**, ultrasensitivity, exact adaptation, stochastic gene
expression (Gillespie), and the classic patterning circuits (lateral inhibition / Notch–Delta,
Turing, morphogen-gradient scaling) — work through **Elowitz & Bois, *Biological Circuit Design***
([biocircuits.github.io](https://biocircuits.github.io/), Caltech BE 150 / Bi 250b; SciPy/Bokeh +
the `biocircuits` package). It is the natural **prerequisite/sibling** to this track, especially for
Lab 5 (Hypergraph Neural ODEs) and Lab 6 (control theory) — a student who's done it arrives already
fluent in fixed points, linear stability, Hill ODEs, and the repressilator (which reappears here as
[`jaxctrl/examples/repressilator_control_demo.py`](https://github.com/m9h/jaxctrl/blob/main/examples/repressilator_control_demo.py),
now as a *control* target rather than a design exercise).

This course is a **different direction**, not an alternative intro. *Biological Circuit Design* is
about *small, hand-specified* circuits and their dynamics/noise; this track picks up where its
"small circuits" section leaves off and goes toward (i) **genome-scale, data-driven regulomes**
inferred from single-cell multiomics (Pando/SCENIC/CellOracle), (ii) **higher-order (hypergraph)**
representations and **hypergraph neural networks**, (iii) **modularity & identifiability** as
*computed* metrics (the Hodge-Laplacian spectral gap → the Module Identifiability Index), (iv)
**network control theory** — controllability, driver nodes, LQR/MPC via `diffrax` adjoints — i.e.
the **anatomical compiler**, and (v) **engineered tissues**: organoids, 3D/4D/freeform bioprinting,
synthetic morphogenesis, cancer-as-loss-of-module-identifiability. Different toolchain too
(JAX/Equinox/Diffrax + `hgx`/`jaxctrl`/`devograph` vs SciPy/Bokeh + `biocircuits`). Overlap is real
but narrow — it lives almost entirely at the small-circuit end (the repressilator, the toggle
switch, Hill-ODE GRNs in the `jaxctrl` examples), and where it overlaps, *Biological Circuit Design*
goes deeper on the dynamics and noise while this track goes further toward inference, control, and
tissue scale.

## Ecosystem & complementary platforms

This course is one node in a larger systems-/synthetic-biology computing ecosystem; students should
know the rest of it.

- **Community standards & model repositories.** [**COMBINE**](https://co.mbine.org) is the umbrella
  initiative; under it: [**SBML**](https://sbml.org) (the lingua franca for kinetic models), CellML,
  SED-ML (simulation experiments), the COMBINE archive (bundle model + sim + results), MIRIAM
  (annotation), and the multicellular extensions [**MultiCellML**](https://multicellml.org) and
  **MorpheusML** (Morpheus's model format — its models are in BioModels too). The large curated
  store is [**BioModels**](https://biomodels.org) — the shared library the simulators below (and
  COPASI, Tellurium / libRoadRunner) read and write. Most of the small-circuit and signaling ODEs in
  *Biological Circuit Design* (and the MAPK / p53–Mdm2 / NF-κB / cell-cycle models in the whitepaper's
  outlook) live there as SBML — pull one, import it into `diffrax`, and you have a Lab.
- **Cell-based ("virtual tissue") simulators** — the *spatial / mechanics / reaction–diffusion* side
  this track deliberately does not cover. The community hub is [**OpenVT** (Open Virtual Tissue)](https://www.openvt.org)
  — reproducibility/credibility standards for cell-based models, a curated
  [framework-comparisons bibliography](https://www.openvt.org/pages/publications/framework-comparisons/publications-framework-comparisons-search-script-bib-json.html),
  and the [2026 OpenVT × ECMTB/SMB workshop](https://www.openvt.org/pages/events/workshops/2026openvt-ecmtb-smb-workshop.html).
  The simulators themselves: [**CompuCell3D**](https://compucell3d.org) (Cellular-Potts; its
  [Workshop series](https://compucell3d.org/Workshop26) and [nanoHub-hosted model library](https://compucell3d.org/Models-nanoHub)
  are the de-facto student on-ramp), [**Morpheus**](https://morpheus.gitlab.io) (Cellular-Potts +
  reaction–diffusion, GUI, SBML/MorpheusML), [**PhysiCell**](https://physicell.org) (3-D agent-based,
  tumour microenvironments — Ghaffarizadeh 2018, with [**PhysiCell Studio**](https://github.com/MathCancer/PhysiCell-Studio)
  for no-code authoring (Heiland 2024, ref. 77a) and the **PhysiCell grammar** (Johnson 2025, ref. 77c)
  as a human-readable behaviour DSL — a candidate compilation target for [Lab 8](08_anatomical_compiler.ipynb)),
  [**PhysiBoSS 2.0**](https://github.com/PhysiBoSS/PhysiBoSS) (PhysiCell + per-cell Boolean networks
  via MaBoSS; Ponce-de-Leon 2023, ref. 77b — **the closest existing realisation of the regulome ⇄ form
  coupling this project flags as its deepest gap**, but forward-only / non-differentiable, hence the
  `cpjax` plan; tutorial at Ruscone 2024, ref. 77d), **Chaste** ("Cancer, Heart and Soft Tissue
  Environment"). PhysiCell's HPC-active-learning arm — **EMEWS** (Ozik 2019, ref. 78a) — is the
  precedent for the project's [`docs/computational-roadmap.md`](../docs/computational-roadmap.md) §2
  active-learning track and the wet-lab BO loop in [`docs/wetlab-program.md`](../docs/wetlab-program.md).
  These all simulate cells *moving, adhering, dividing, signalling in space*; this track is the
  **regulome / control complement** (`hgx`
  hypergraph neural networks + `jaxctrl` control on inferred genome-scale regulomes), interoperating
  with that world *via SBML* at the subcellular layer. A full virtual-tissue curriculum pairs them:
  CC3D / Morpheus for the morphogenesis-and-mechanics labs, this track for "the regulatory program
  and how to steer it" — and the whitepaper's forward programme (§4.3) is exactly where the two meet
  (a bioprinting / synNotch / optogenetic actuation, a CC3D-style mechanical readout, an
  `hgx`/`jaxctrl` regulatory readout, all in one model-in-the-loop design cycle).
- **Bioelectric-layer simulators.** [**BETSE / BETSEE**](https://gitlab.com/betse/betse) (the
  BioElectric Tissue Simulation Engine, Pietak & Levin 2016/2017) — finite-volume Vmem / gap-junction /
  ion-channel dynamics over a cell cluster. The **differentiable JAX/`diffrax` refactor of BETSE/BETSEE
  is done** (`betse.science.jax.*` at `~/Workspace/betse-unified`; the way `vpjax` modernises `vbjax`'s
  haemodynamics) — the bioelectric-layer companion to `hgx` (regulome) and `jaxctrl` (control), built to
  follow the Levin Lab / Mafe-group post-2018 shift *from passive simulation to agential / top-down
  anatomical control*. All four "bridge" examples are implemented & verified (test files in parens):
  **(i) inverse bioelectric design** (`betse.science.jax.inverse.optimize_pattern`; `test_jax_inverse.py`)
  — target Vmem pattern in → required ion currents out; a 10-cell benchmark drives resting tissue to an
  alternating ±depolarisation pattern, loss 9e-4 → ~1.4e-10 — the bioelectric "anatomical compiler" in
  miniature; **(ii) xenobot motility** (`betse.science.jax.physics.motility`; `test_jax_xenobot.py`) —
  bioelectric force → movement; the Kriegman et al. 2020/2021 design move done by `jax.grad` on the
  cell-collective's force/motility goals rather than by an evolutionary search; **(iii) Vmem→GRN triggering**
  (`betse.science.jax.physics.grn_trigger`; `test_jax_grn.py`) — a bioelectric prepattern biasing a
  secondary transcriptional program (Pietak & Levin 2017; Cervera/Levin/Mafe 2024–2026 "top-down" Vmem↔transcription);
  **(iv) morphoceutical intervention timelines** (`betse.science.jax.physics.intervention`; `test_jax_morphoceutical.py`)
  — temporal drug schedules driving macro-scale regrowth (Murugan et al. 2022; Pio-Lopez & Levin 2023) —
  plus large-scale pattern integration (100 cells, 1000 steps < 0.1 s), a whole-step JIT-fused `step_pure`
  solver (~2.2 µs/step for 50 cells via `lax.scan`), a GPU-accelerated Poisson solver, and 100% test
  coverage on the JAX modules (`docs/JAX_REFAC.md`). Abstractly each bridge is one optimal-control problem
  on a differentiable plant — the bioelectric set-point added to the §4.3 actuator menu (print geometry ·
  synNotch / synthetic morphogens · light/dose · **Vmem / gap-junction state**), now with a real
  differentiable solver behind it. The course's hooks: Lab 8's exercise (d) (the two-layer bioelectric→GRN
  compiler), Lab 9 §4 (bioelectric control) and §5 (biobots), Lab 10 §4 (the "normalise, don't kill"
  reframing — morphoceuticals). Refs 39a–39e in `REFERENCES.md`; the "manifesto" is Levin 2021 (Cell, ref 38).
- **Cellular-engineering research programmes** — the *wet-lab* communities this computational track
  is in dialogue with. The [**Center for Cellular Construction (CCC)**](https://centerforcellularconstruction.org)
  — an NSF Science and Technology Center headquartered at UCSF with the California Academy of Sciences
  (Wallace Marshall, director; Zev Gartner, co-director) — treats the cell as a *smart, reconfigurable
  material*: cytocomputing, cell-shape / organelle engineering, programmed multicellular assembly. The
  [**Gartner Lab**](https://gartnerlab.ucsf.edu) (UCSF) is the tissue-self-organisation / DNA-programmed-
  assembly-of-cells (DPAC) node — and is *already* a benchmark in this repo: the conformation-controlled
  kidney-organoid data (`man`/`r0`/`r40`; `scripts/benchmark_gartner_4d.py`, `figures/gartner_4d_fidelity.png`)
  that the whitepaper's §4 uses for "print geometry → lineage maturity". The natural complement: CCC /
  Gartner provide the *constructible substrate* and the assembly handles (DPAC, conformation control,
  4D matrices), this track provides the *regulatory readout and the controller* (`hgx` / `jaxctrl` on
  the inferred regulome) — together, the §4.3 model-in-the-loop design cycle. The
  [**Lim Lab**](https://limlab.ucsf.edu) (UCSF — also part of UCSF's cell-design ecosystem) is the
  **synNotch / synthetic-development / therapeutic-cell-engineering** node: customisable synthetic
  receptors (Morsut 2016), self-organising multicellular structures and synthetic morphogens (Toda
  2018/2020 — the Toda 2020 dataset is already a benchmark here, `scripts/benchmark_toda_morphogenesis.py`,
  the whitepaper's §4.3 "hybrid programmed-plus-printed tissues"), AND-gate / combinatorial CAR-T
  (Roybal 2016 — the cancer-immunotherapy angle, §1.6 / §4.3(vi)), and a "common molecular algorithms /
  harnessing cellular modularity" framing that is the synthetic-biology sibling of this project's
  modularity & Module-Identifiability-Index theme (Lab 4). So the actuator menu is concrete: print
  geometry (Gartner/Feinberg/Lewis), synNotch / synthetic morphogens (Lim/Morsut), bioelectric
  set-points (Levin), light/dose schedules (optogenetics) — and each is "one optimal-control problem on
  the Hypergraph Neural ODE" (Lab 8). (See also the Levin Lab for the bioelectric layer, and the Lewis /
  Feinberg / Skylar-Scott labs for the bioprinting handles — refs in `REFERENCES.md`.)

> **A note on "identifiability."** This course uses the word in the *modular-structure* sense — the
> Hodge-Laplacian **Module Identifiability Index** (Lab 4) asks whether a regulome *decomposes* into
> stable, distinct modules. That is **not** the same as the **structural identifiability** of
> dynamic-model parameters — "can these rate constants / Hill coefficients be recovered from the
> measured outputs, even with perfect data?" — which is decided *symbolically*, before any fitting,
> by differential-algebra / power-series methods (Bellman & Åström 1970; the program of J. DiStefano
> III and collaborators at UCLA, e.g. the COMBOS web tool — [biocyb0.cs.ucla.edu](http://biocyb0.cs.ucla.edu/wp/)).
> The numerical, local counterpart is the *observability* rank condition, which `jaxctrl` exposes
> directly (`is_observable`, `observability_matrix`, `observability_gramian`). A structural-ID check
> belongs **before** fitting any mechanistic reduced model (the Hill-ODE GRN in the `jaxctrl` IRMA
> example; the linear surrogate in `scripts/benchmark_anatomical_compiler.py`) — see Lab 6's notes —
> and the *practical* (finite/noisy-data) version is the SBI / profile-likelihood story of Lab 8.

## Planned sequence (a ~10-session course)

**Setup** [`00_setup.ipynb`](00_setup.ipynb) — the map, the toolchain, the env check. Prerequisites; a
   runnable check of the stack (`numpy`/`matplotlib`/`jax`/`scipy` required; `hgx`/`equinox`/`diffrax`/`optax`/`sympy`
   recommended; `jaxctrl` optional); four hello-worlds (`jax.grad` incl. through a loop; the hypergraph /
   clique-expansion blow-up; a differentiable ODE; a 2-state controllability + LQR); the dependency map;
   the three "identifiabilities" disambiguation.
1. **Gene-circuit dynamics in a nutshell** *(bridge from* Biological Circuit Design*; skip if you've done
   it)*. Hill functions; negative autoregulation and response time; the toggle switch & bistability; the
   repressilator and linear stability — restated in this course's toolchain (`diffrax`, `jax.jacfwd`/`jax.grad`,
   `jaxctrl`), ending in the "linearize a circuit, then LQR-control it" move that Labs 6-8 scale up.
   *(`notebooks/01_gene_circuit_dynamics.ipynb`; Elowitz & Bois; Alon; Gardner et al. 2000; Elowitz & Leibler 2000.)*
2. **Regulomes and hypergraphs.** Gene regulatory networks; why a regulon is a *hyperedge*, not a clique.
   Build the Fleck incidence matrix; basic hypergraph operations in `hgx`; the Laplacian spectrum.
   *(`notebooks/02_regulomes_and_hypergraphs.ipynb`; Davidson; Fleck et al. 2023; §1.4 / §2.2.)*
3. **Benchmarking fidelity.** Does an organoid regulome predict CRISPRi outcomes in primary cortex? The
   fidelity *triple* (in-domain r / transfer direction / transfer r); regulon-screen overlap; direction-
   vs-magnitude transfer (raw signs ~60% vs a trained predictor ~83%; magnitude r ~0.13); per-cell-type
   disaggregation; what fidelity is *not*; cross-species conservation. *(`notebooks/03_benchmarking_fidelity.ipynb`;
   builds on `scripts/organoid_hgx_colab.ipynb`; Pollen/Ding 2026; `scripts/compare_pollen.py`,
   `scripts/benchmark_advanced_fidelity.py`; §3.1-3.3.)*
4. **Modularity and identifiability.** The Hodge Laplacian (L0/L1) of the regulon hypergraph; the
   spectral gap -> the Module Identifiability Index (the project heuristic + a normalised relative eigengap);
   cross-system ordering (organoid > blueprint > bioprint); "neurogenic stop-signals" along pseudotime;
   what module identifiability is *not* (!= structural/practical ID, != fidelity). *(`notebooks/04_modularity_identifiability.ipynb`;
   `scripts/benchmark_kidney_modularity.py`, `scripts/test_nitmb_modularity.py`, `scripts/06_topology.py`;
   Hartwell 1999; NITMB framing; §2.3 / §3.x.)*
5. **Dynamics: Hypergraph Neural ODEs.** Fit a Neural ODE on the pseudotime timecourse (bare-metal RK4 +
   `jax.grad` + `optax`); per-gene rollout MSE as a classifier — stable structural drivers vs transient
   stress responders; the Waddington/attractor view (a toggle-switch bistability demo + `fate_probabilities.npy`
   / `perturbation_fates.npy`); what a Hypergraph Neural ODE is and isn't (the *flow*, not the parameters).
   *(`notebooks/05_hypergraph_neural_odes.ipynb`; `scripts/04_temporal_dynamics.py`,
   `scripts/benchmark_learning_regulome.py`, `scripts/benchmark_regenerative_flow.py`; Kauffman; Huang et al. 2009.)*
6. **Control theory on cellular dynamics (`jaxctrl`).** The regulome as a steerable plant: controllability
   (Kalman rank — the masters don't control it, dual of Lab 7), Gramians (average vs modal controllability),
   minimum-energy control, LQR, minimum driver nodes; steer-to-target (the linear "anatomical compiler");
   pointers to the three `jaxctrl` example notebooks (SINDy surrogate, hypergraph drivers, repressilator
   quench). *(`notebooks/06_control_theory.ipynb`; `scripts/benchmark_network_control.py`;
   Liu-Slotine-Barabasi 2011; Gu et al. 2015; Yan et al. 2017; Pezzulo & Levin 2016.)*
7. **Is the model even identifiable?** A symbolic *structural*-identifiability check in `sympy` (the
   Taylor-series / observability-rank test — the symbolic generalisation of `jaxctrl`'s linear `is_observable`):
   before you fit a mechanistic model, can its parameters be recovered from what you measure, with perfect
   data? Run on the Lab 1 circuits and the linear surrogate of Lab 8; distinguished from *module*
   identifiability (Lab 4) and *practical* identifiability (Lab 8). *(`notebooks/07_structural_identifiability.ipynb`;
   Bellman & Astrom 1970; DiStefano III / COMBOS; §2 of the whitepaper.)*
8. **The anatomical compiler.** The four-stage stack — learned plant (Lab 5) -> linear surrogate / system-ID
   (Labs 6/7; it's *unstable* — a warm-start, not the controller) -> LQR warm-start -> **direct optimal control
   on the nonlinear plant** (u(t) minimising state-error + control-cost, gradient through the ODE solve)
   -> closed-loop validation. Demonstrated as a knockout-rescue ("you steer what you actuate"); + "the arc,
   Labs 1->8". *(`notebooks/08_anatomical_compiler.ipynb`; `scripts/benchmark_anatomical_compiler.py`;
   Pezzulo & Levin 2016; Levin 2022; `jaxctrl`.)*
9. *(stretch)* **Synthetic morphology in the wet lab.** The §4.3 forward programme — *programmed* (synNotch
   / synthetic morphogens — Lim/Morsut/Toda), *printed* (conformation / 4-D bioprinting — Feinberg/Lewis/
   Skylar-Scott/Gartner), *bioelectric* (V_mem prepattern — Levin/Mafe; the BETSE-JAX refactor —
   inverse bioelectric design, V_mem<->GRN prepatterning, morphoceutical timelines; refs 38, 39a-39e),
   *agential* (xenobots/anthrobots) — each reframed as *one optimal-control problem on the Hypergraph Neural
   ODE*, differing only in the actuator B and the readout layer; the model-in-the-loop design cycle.
   *(`notebooks/09_synthetic_morphology_wetlab.ipynb`; `scripts/benchmark_{toda_morphogenesis,gartner_4d,
   anthrobot_fidelity,...}.py`; `~/Workspace/betse-unified`; Davies 2008; Solé et al. 2024.)*
10. *(stretch, final)* **Cancer as loss of module identifiability.** The TOFT / atavistic register — cancer
   => the MII falls (the Fiedler-region gap dissolves) + the homeostatic driver set comes loose (the
   regeneration arrow of Lab 5 reversed); the anatomical compiler (Lab 8) as the "normalise, don't (just)
   kill" reframing; the diagnostic flip ("know when you have"). A schematic of the spectral signature + the
   real cross-system comparison + a working "differentiated <-> tumour" normalisation demo.
   *(`notebooks/10_cancer_module_identifiability.ipynb`; `scripts/{benchmark_kidney_modularity,
   test_nitmb_modularity,benchmark_disease_enrichment,benchmark_tf_disease}.py`; `~/Workspace/betse-unified`;
   Soto & Sonnenschein; Trigos et al.; Levin 2021; Pio-Lopez & Levin 2023; §1.6 / §4.3(vi).)*

## Contributing a notebook

Keep each notebook self-contained (download/generate its own small data, or read the committed
`figures/*_results.json` / `data/cropseq/*.csv`), resilient (graceful note if an optional dep or
large dataset is absent), and end with a short exercise. Match the style of the `jaxctrl` example
notebooks. If a notebook needs the full pipeline, point at `scripts/` rather than duplicating it.
