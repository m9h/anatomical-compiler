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

Lab 0.5 below is the **computational companion to Week 6** — same repressilator (and Hill kinetics,
toggle switch, linear stability), in this course's Python toolchain, and then it shows you what to
do *after* you've recreated it: linearize it and *control* it. The hypergraph-neural-network strand
of this track (Labs 1 & 4) is the conceptual sibling of **Week 7** — Week 7 builds neural networks
*as* gene circuits; this track runs neural networks and control theory *on* gene networks. The full
dependency chain: **Elowitz & Bois, *Biological Circuit Design*** (the dynamical-systems
foundations, see below) → HTGAA Weeks 6–7 (assembling, then computing with, genetic circuits) →
**this track** (regulomes as hypergraphs → hypergraph neural networks → modularity/identifiability
metrics → network control / the *anatomical compiler*) → **wet-lab synthetic morphology**
(bioprinting, synthetic morphogenesis).

## What's here now

- **`00b_gene_circuit_dynamics.ipynb`** — **Lab 0.5** (the bridge from *Biological Circuit Design*):
  Hill functions, negative autoregulation (response-time), the toggle switch & bistability (via
  `jax.jacfwd`), the repressilator + the "linearize then LQR-control it" move — all in this course's
  toolchain (`diffrax`, `jax.jacfwd`/`jax.grad`, `jaxctrl`) rather than SciPy/Bokeh. Self-contained.
- **`01_regulomes_and_hypergraphs.ipynb`** — **Lab 1**: what a regulome is; load the Fleck
  incidence matrix; why a regulon is a *hyperedge*, not a clique (the clique-expansion blow-up
  *and* aliasing, worked on a toy example + the full regulome); `hgx` basics (`from_incidence`,
  node/edge degrees, star vs clique expansion, a `UniGCNConv` forward pass); a first structural
  readout (the hypergraph-Laplacian spectrum and its gap — the seed of the Module Identifiability
  Index); exercises (all-pairs Jaccard regulon overlap among the master TFs, heavy-tailedness of
  regulon sizes, graph degree vs hypergraph degree). Self-contained — reads `data/processed/`,
  falls back to a tiny synthetic regulome if absent.
- **`02_benchmarking_fidelity.ipynb`** — **Lab 2**: does the organoid regulome *predict* perturbations?
  The fidelity **triple** — in-domain magnitude r, transfer direction-accuracy, transfer r — instead
  of a single "accuracy". GRN/screen overlap (organoid's 720 regulons vs Pollen/Ding 2026's 44
  CRISPRi'd TFs → 34 shared, hypergeometric p ≈ 10⁻⁵); zero-shot transfer organoid → primary cortex
  (live on the key TFs the regulome ships KO predictions for, plus the precomputed 34-TF benchmark:
  **direction ≈ 83% with a trained `PerturbationPredictor`** — but raw regulon signs only ≈ 60% —
  vs **magnitude r ≈ 0.13** zero-shot, ≈ 0.36 within-primary); the per-cell-type breakdown (raw
  numbers near chance — aggregate accuracy can lie); why magnitude doesn't transfer = the *practical*
  shadow of Lab 5.5's *structural* result (the downstream-magnitude ridge → the SBI valley of Lab 6);
  what fidelity is *not* (≠ structural identifiability, ≠ module separability, ≠ transfer fidelity);
  exercises (wire in the real CROP-seq from `extract_cropseq.py`; the `advanced_fidelity` pattern
  scores — fidelity of *identity* vs of *perturbation*; cross-species conservation and why overlap is
  the weak test). Self-contained — reads `data/processed/`, `data/pollen/processed/`, `data/cropseq/*.csv`,
  `figures/pollen_*.json`, `figures/advanced_fidelity_results.json`; graceful fallbacks throughout.
  Full pipeline: `scripts/compare_pollen.py`, `scripts/benchmark_advanced_fidelity.py`.
- **`03_modularity_identifiability.ipynb`** — **Lab 3**: does the regulome *decompose into modules*,
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
  lives; cancer as loss of module identifiability — the Lab 8 stretch). Self-contained — reads
  `data/processed/`, `figures/{kidney_modularity,nitmb_modularity}*.json`; synthetic "blocky"
  fallback. Pipeline: `scripts/benchmark_kidney_modularity.py`, `scripts/test_nitmb_modularity.py`,
  `scripts/06_topology.py`.
- **`04_hypergraph_neural_odes.ipynb`** — **Lab 4**: a cell type is a *stable state of the dynamics
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
  Lab 3 (modules dissolve ⇒ driver set shrinks) and Lab 2 (well-fit drivers ⇒ transferable KO
  directions) and forward to Lab 5/6 (control on the *learned* flow); what a Hypergraph Neural ODE is
  and isn't (it's the *flow*, not the parameters — deliberately structurally non-identifiable;
  the "hypergraph" is a weight-sharing *prior*, `hgx.LatentHypergraphODE`); exercises (ODE→SDE +
  CellRank fate entropy; Poincaré/hyperbolic latent space + Gromov δ; finer time T∈{20,50}; the hgx
  regulon-structured field; close the loop with `jaxctrl`). Self-contained — reads
  `data/processed/{temporal_expression,pseudotime_centers,fate_probabilities,perturbation_fates}.npy`
  + `figures/{learning_regulome,regenerative_flow}_results.json`; synthetic toggle-switch fallback.
  Pipeline: `scripts/04_temporal_dynamics.py` (the `diffrax` + `hgx.LatentHypergraphODE` + SDE +
  Poincaré version), `scripts/benchmark_learning_regulome.py`, `scripts/benchmark_regenerative_flow.py`.
- **`05b_structural_identifiability.ipynb`** — **Lab 5.5**: a `sympy` structural-identifiability
  check (the Taylor-series / observability-rank test — the symbolic generalisation of `jaxctrl`'s
  linear `is_observable`/`observability_gramian`). Textbook sanity checks (one-pool ✓; Bellman &
  Åström's $k_1{+}k_2$ ✗ with the nullspace direction printed); the Lab-0.5 circuits (negative
  autoregulation, the repressilator — identifiable from a single output only if you also know the
  initial state, identifiable from all outputs regardless); the linear special case = `jaxctrl`'s
  observability rank, tied to the `benchmark_network_control.py` finding (the regulome's linear
  surrogate is not identifiable/controllable from the master regulators alone → the SBI posterior
  is correctly ridge-shaped); exercises (free Hill exponent → the $n$–$K$ confound; the IRMA GRN's
  minimal output set; a COMBOS compartmental example; the SBI loss-valley). Distinguishes the
  *structural*, *practical*, and *module* senses of "identifiability". Self-contained. (One cell —
  the 6-unknown repressilator — takes ~2–3 min.)
- **`organoid_hgx_colab.ipynb`** — "Lab 0 / the benchmark": the GPU/Colab notebook running `hgx`
  on the Fleck et al. (2023) cerebral-organoid regulome end-to-end (preprocessing → figures → the
  5 biological-validation checks → the hgx-vs-DHG speed/accuracy benchmark).

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

If you want the *dynamical-systems-of-gene-circuits* foundations this course leans on but does not
re-derive — Hill kinetics and cooperativity, autoregulation, the toggle switch and bistability,
feed-forward loops, the **repressilator**, ultrasensitivity, exact adaptation, stochastic gene
expression (Gillespie), and the classic patterning circuits (lateral inhibition / Notch–Delta,
Turing, morphogen-gradient scaling) — work through **Elowitz & Bois, *Biological Circuit Design***
([biocircuits.github.io](https://biocircuits.github.io/), Caltech BE 150 / Bi 250b; SciPy/Bokeh +
the `biocircuits` package). It is the natural **prerequisite/sibling** to this track, especially for
Lab 4 (Hypergraph Neural ODEs) and Lab 5 (control theory) — a student who's done it arrives already
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
  reaction–diffusion, GUI, SBML/MorpheusML), **PhysiCell** (3-D agent-based, tumour microenvironments),
  **Chaste** ("Cancer, Heart and Soft Tissue Environment"). They simulate cells *moving, adhering,
  dividing, signalling in space*; this track is the **regulome / control complement** (`hgx`
  hypergraph neural networks + `jaxctrl` control on inferred genome-scale regulomes), interoperating
  with that world *via SBML* at the subcellular layer. A full virtual-tissue curriculum pairs them:
  CC3D / Morpheus for the morphogenesis-and-mechanics labs, this track for "the regulatory program
  and how to steer it" — and the whitepaper's forward programme (§4.3) is exactly where the two meet
  (a bioprinting / synNotch / optogenetic actuation, a CC3D-style mechanical readout, an
  `hgx`/`jaxctrl` regulatory readout, all in one model-in-the-loop design cycle).
- **Bioelectric-layer simulators.** [**BETSE / BETSEE**](https://gitlab.com/betse/betse) (the
  BioElectric Tissue Simulation Engine, Pietak & Levin) — finite-volume Vmem / gap-junction / ion-
  channel dynamics over a cell cluster. This is where a "control the bioelectric layer" experiment
  (§4.3(v) of the whitepaper) would actually be simulated; it is also a candidate for a serious
  modernisation (a differentiable / JAX rewrite, the way `vpjax` modernises the hemodynamics in
  `vbjax`) — a natural Biopunk-Lab project, and a good capstone for this course.
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
  the inferred regulome) — together, the §4.3 model-in-the-loop design cycle. (See also the Levin Lab
  for the bioelectric layer, and the Lewis / Feinberg / Skylar-Scott labs for the bioprinting handles —
  refs in `REFERENCES.md`.)

> **A note on "identifiability."** This course uses the word in the *modular-structure* sense — the
> Hodge-Laplacian **Module Identifiability Index** (Lab 3) asks whether a regulome *decomposes* into
> stable, distinct modules. That is **not** the same as the **structural identifiability** of
> dynamic-model parameters — "can these rate constants / Hill coefficients be recovered from the
> measured outputs, even with perfect data?" — which is decided *symbolically*, before any fitting,
> by differential-algebra / power-series methods (Bellman & Åström 1970; the program of J. DiStefano
> III and collaborators at UCLA, e.g. the COMBOS web tool — [biocyb0.cs.ucla.edu](http://biocyb0.cs.ucla.edu/wp/)).
> The numerical, local counterpart is the *observability* rank condition, which `jaxctrl` exposes
> directly (`is_observable`, `observability_matrix`, `observability_gramian`). A structural-ID check
> belongs **before** fitting any mechanistic reduced model (the Hill-ODE GRN in the `jaxctrl` IRMA
> example; the linear surrogate in `scripts/benchmark_anatomical_compiler.py`) — see Lab 5's notes —
> and the *practical* (finite/noisy-data) version is the SBI / profile-likelihood story of Lab 6.

## Planned sequence (a 6–8 session course)

0.5. **Gene-circuit dynamics in a nutshell** *(bridge from* Biological Circuit Design*)*. Hill
   functions; negative autoregulation and response time; the toggle switch & bistability; the
   repressilator and linear stability — restated in this course's toolchain (`diffrax`,
   `jax.jacfwd`/`jax.grad`, `jaxctrl`), ending in the "linearize a circuit, then LQR-control it"
   move that Lab 5 scales up. *(`notebooks/00b_gene_circuit_dynamics.ipynb`; refs: Elowitz & Bois;
   Alon; Gardner et al. 2000; Elowitz & Leibler 2000.)*
1. **Regulomes and hypergraphs.** Gene regulatory networks; why a regulon is a *hyperedge*, not
   a clique. Build the Fleck incidence matrix; basic hypergraph operations in `hgx`. *(Refs: Davidson;
   Fleck et al. 2023; the §1.4 / §2.2 material.)*
2. **Benchmarking fidelity.** Does an organoid regulome predict CRISPRi outcomes in primary cortex?
   The fidelity *triple* (in-domain r / transfer direction / transfer r); regulon–screen overlap;
   direction-vs-magnitude transfer (raw signs ≈ 60% vs a trained predictor ≈ 83%; magnitude r ≈ 0.13);
   per-cell-type disaggregation; what fidelity is *not*; cross-species conservation.
   *(`notebooks/02_benchmarking_fidelity.ipynb`; builds on `organoid_hgx_colab.ipynb`; Pollen/Ding 2026;
   `scripts/compare_pollen.py`, `scripts/benchmark_advanced_fidelity.py`; §3.1–3.3.)*
3. **Modularity and identifiability.** The Hodge Laplacian ($L_0$/$L_1$) of the regulon hypergraph;
   the spectral gap → the Module Identifiability Index (the project heuristic + a normalised relative
   eigengap); cross-system ordering (organoid > blueprint > bioprint); "neurogenic stop-signals" along
   pseudotime; what module identifiability is *not* (≠ structural/practical ID, ≠ fidelity).
   *(`notebooks/03_modularity_identifiability.ipynb`; `scripts/benchmark_kidney_modularity.py`,
   `scripts/test_nitmb_modularity.py`, `scripts/06_topology.py`; Hartwell 1999; NITMB framing; §2.3 / §3.x.)*
4. **Dynamics: Hypergraph Neural ODEs.** Fit a Neural ODE on the pseudotime timecourse (bare-metal
   RK4 + `jax.grad` + `optax`); per-gene rollout MSE as a classifier — stable structural drivers vs
   transient stress responders; the Waddington/attractor view (a toggle-switch bistability demo +
   `fate_probabilities.npy` / `perturbation_fates.npy`); what a Hypergraph Neural ODE is and isn't
   (the *flow*, not the parameters). *(`notebooks/04_hypergraph_neural_odes.ipynb`;
   `scripts/04_temporal_dynamics.py`, `scripts/benchmark_learning_regulome.py`,
   `scripts/benchmark_regenerative_flow.py`; Kauffman; Huang et al. 2009; §2.4 / §3 regenerative-flow.)*
5. **Control theory on cellular dynamics (`jaxctrl`).** Identify a surrogate (SINDy/Koopman) →
   controllability → LQR → driver nodes on a hypergraph. Use the three jaxctrl example notebooks.
5.5. **Is the model even identifiable?** A symbolic *structural* identifiability check in `sympy`
   (the Taylor-series / observability-rank test — the symbolic generalisation of `jaxctrl`'s linear
   `is_observable`): before you fit a mechanistic model, can its parameters be recovered from what
   you measure, with perfect data? Run on the Lab 0.5 circuits and the linear surrogate of Lab 6;
   distinguished from *module* identifiability (Lab 3) and *practical* identifiability (Lab 6).
   *(`notebooks/05b_structural_identifiability.ipynb`; Bellman & Åström 1970; DiStefano III / COMBOS;
   §2 of the whitepaper.)*
6. **The anatomical compiler.** Optimal control on the *learned* Hypergraph Neural ODE: given a
   target tissue state, compute an actuation schedule (`diffrax` adjoints). *(§3 anatomical-compiler;
   `scripts/benchmark_anatomical_compiler.py`; Levin 2022.)*
7. **Synthetic morphology in the wet lab.** Bioprinting (FRESH/SWIFT/PRINTESS), synthetic-morphogen
   circuits, optogenetic morphogenesis, bioelectric control — the forward programme of §4.3, framed
   as control problems with the model in the loop.
8. *(stretch)* **Cancer as loss of module identifiability.** Run the metrics down a
   primary → organoid → tumour-organoid → cancer-line gradient. *(Soto & Sonnenschein; Trigos et al.;
   §1.6 / §4.3(vi).)*

## Contributing a notebook

Keep each notebook self-contained (download/generate its own small data, or read the committed
`figures/*_results.json` / `data/cropseq/*.csv`), resilient (graceful note if an optional dep or
large dataset is absent), and end with a short exercise. Match the style of the `jaxctrl` example
notebooks. If a notebook needs the full pipeline, point at `scripts/` rather than duplicating it.
