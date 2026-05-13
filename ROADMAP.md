# Roadmap

## Phase 1: Validation (COMPLETE)

### 1A: Standard Benchmark Suite
- [x] Cora validated — hgx UniGCNConv 78.72% matches published HGNN 79.39%
- [x] Citeseer — 64.80% (7pt gap is architectural, not data)
- [x] Pubmed — 76.10% on 19,717 nodes in 15.5s / 6.7 GB RAM
- [x] Accuracy ablation — 94.6% on 20 spectral clusters, 77.2% lineage 3-class
- [x] SheafDiffusion OOM diagnosed and fixed
- [x] THNNConv NaN diagnosed — fix script created
- [x] DHG device mismatch fixed

### 1B: Pando Reproduction (5/5 PASS — confirmed 2026-03-17)
- [x] TF centrality — PASS (TF-aware ranking, composite precision@100 = 0.62)
- [x] Regulon coherence — PASS (gap = +0.116)
- [x] GLI3 KO direction — PASS (80% via hypergraph signal propagation)
- [x] Pseudotime patterns — PASS (TBR1/NEUROD6 increase_late)
- [x] Fate probabilities — PASS (DF r=0.80, MH r=-0.74)

## Phase 2: Extensions

### 2A: Higher-Order Interactions

- Integrate THNNConv fix (fix_thnn.py — sum-of-logs overflow, not gradient clipping)
- Quantify higher-order benefit (THNNConv vs UniGCNConv gap)
- Identify modules where multi-TF cooperativity matters most

### 2B: Dynamics (Neural ODE/SDE)

- Increase pseudotime bins (T=20, T=50) for finer temporal resolution
- Compare ODE vs SDE trajectory quality
- Validate SDE variance against CellRank fate entropy

### 2C: Poincare Embeddings

- [x] Fix dimension mismatch in LatentHypergraphODE (fixed in cac6cba)
- Compare Gromov delta: Poincare vs Euclidean on real lineage tree
- Validate tree-like structure of cell fate decisions

## Phase 3: Second Dataset

### 3A: Pollen Lab CRISPRi (Nature 2026)

- Paper: "Dissecting gene regulatory networks governing human cortical cell fate"
  - Ding, Kim, Ostrowski et al. (Pollen lab, UCSF)
  - Nature 2026, doi: 10.1038/s41586-025-09997-7
- **GEO: GSE284197** — 35 samples, h5ad files available
  - `GSE284197_screen.h5ad` (4.1 GB) — main 44-TF CRISPRi screen (2D cultures)
  - `GSE284197_merged.h5ad` (3.2 GB) — merged dataset
  - `GSE284197_IN.h5ad` (683 MB) — interneuron subset
  - `GSE284197_slice.h5ad` (454 MB) — slice cultures
  - `GSE284197_clones.h5ad` (7.6 MB) — clonal lineage tracing
- Code: https://github.com/jding5066/perturbTF
- Browser: cortical-lineage-perturb-44tf.cells.ucsc.edu
- Key TFs: ZNF219, NR2E1, ARX, + 41 others active in cortical neurogenesis
- [ ] Download screen.h5ad (`scripts/download_pollen.py --screen-only`)
- [ ] Preprocess (`scripts/preprocess_pollen.py`)
- [ ] Run comparison (`scripts/compare_pollen.py`)

### 3B: Cross-Dataset Generalization

- [x] GRN overlap: 34/44 shared TFs ($p < 10^{-5}$)
- [x] Direction concordance: 82.9% mean accuracy (POU2F1 79.6%, ASCL1 76.7%)
- [x] Transfer prediction: Zero-shot transfer from Fleck -> Pollen validated
- [x] Within-Pollen LOO baseline (mean $r=0.36$)

### 3C: New Regulome Atlas Extensions

- [ ] **Posterior Brain (Azbukina et al., 2025)**: Midbrain/hindbrain regulomes (snRNA/ATAC)
  - DOI: 10.1101/2025.03.20.644368 | Zenodo: 10.5281/zenodo.11203684
  - Target: Identify TF networks for dopaminergic and glycinergic neurons
- [ ] **Retinal Regulomes (Wahle et al., 2023)**: Spatiotemporal retinal organoid atlas
  - DOI: 10.1038/s41587-023-01747-2 | ArrayExpress: E-MTAB-12714
  - Target: Validate hgx on non-brain developmental GRNs (e.g., OTX2 regulon)
- [x] **Simulation-Based Inference (SBI)**:
  - [x] Invert hgx models to estimate GRN posteriors from Pollen/Wahle CRISPRi data
  - [x] Compare hgx-SBI against standard Pando/GRN-VAE methods

### 3D: Foundation Model Benchmarking (Theis Lab)

- [x] **scTab (Cell Type Annotation)**:
  - [x] Standalone scTab inference script implemented (`scripts/annotate_sctab.py`)
  - [x] Standardized cell type labels across Fleck and Pollen datasets.
  - [x] Evaluated population-specific transfer concordance.
- [ ] **Prophet (Perturbation Prediction)**:
  - Benchmark `theislab/Prophet` (Transformer-based) against `hgx.PerturbationPredictor` (Hypergraph-based).
  - Compare Pearson r and direction concordance on Pollen/Ding 2026 CRISPRi data.
  - Analyze "zero-shot" transfer performance: how well does a foundation model (Prophet) perform compared to a task-specific model (hgx)?

### 3E: Evolutionary Benchmark (NHP, Mouse, Ferret)

- [x] **Marmoset Interneurons (Krienen 2021)**:
  - [x] Downloaded and annotated with scTab.
  - [x] Validated DLX1 regulon conservation ($p = 2.12 \times 10^{-3}$).
- [x] **Mouse Neocortex (Loo 2019)**:
  - [x] Preprocessed E14.5 neocortex.
  - [x] Identified EOMES (14.3%) and NEUROD2 (11.6%) as conserved regulators.

### 3F: Experimental Validation (CRISPR Screens)

- [x] **CHOOSE Screen (Sharf/Li 2023)**:
  - [x] Downloaded 4.28 GB h5ad (CHD4/ADNP KOs).
  - [x] Preprocessed with Human-Mouse symbol mapping.
  - [x] Validated ADNP effects in organoids.
- [x] **Real CROP-seq (Fleck 2023)**:
  - [x] Extracted real DE for GLI3, TBR1, NEUROD6, PAX6, HES1 using R/Seurat.
  - [x] Achieved strong validation (GLI3 vs TBR1 r=0.83).

### 3G: Solé Open-Problems Suite (Synthetic Multicellularity)

Mapped against Solé et al. 2024 (npj Systems Biology and Applications):

- [x] **Patterning** — GSE156162 (Toda 2020 synNotch) — `benchmark_toda_morphogenesis.py`
- [x] **Robotics** — GSE249581 (Gumuskaya 2023 Anthrobots) — `benchmark_anthrobot_fidelity.py`
- [x] **Metabolism / Vascularization** — GSE131094 (Shi 2020 vOrganoids), corrected
  from the original GSE131936 misattribution. 57,180 cells (vOrg 31,851 / Org 25,329,
  Day 65 + Day 100). Cross-talk regulons in vOrganoids: **RBPJ +0.072 (p=7.5e-19)**,
  **EPAS1 −0.056 (p=6.3e-17)** — quantitative evidence for hypoxia-relief by vasculature
  (the "metabolic wall"). Pre-normalized log matrix — `benchmark_vorganoid_crosstalk.py`
  auto-detects negative values and skips re-normalization.
- [x] **Agency / Bio-Computing** — pivoted from GSE207577 (Kagan DishBrain, private
  until 2026-12-31, MEA-only) to GSE102827 (Hrvatin 2018 V1 light-stim scRNA-seq).
  65,539 cells × 0h/1h/4h. IEG timecourse: −0.13 → +0.06 → −0.07 (canonical
  immediate-early peak-at-1h dynamics); late genes (BDNF/HOMER1/DUSP1) peak 1–4 h.
- [x] **Robustness / Self-Repair** — GSE138835 (Lawlor) has only 1 sample;
  pivoted to **GSE180420 (Balzer 2022 Nat Commun)** mouse kidney IRI, 113,579
  cells × 4 timepoints (Day 0/1/3/14, IRI_short + IRI_long).
  `benchmark_regenerative_flow.py` fits a Hypergraph Neural ODE in 1.2 s and
  cleanly separates **stable regenerative drivers** (Lhx1 MSE=0.05, Cdh1=0.06,
  Pax8=0.08, Pax2=0.10, Six1/2/Wt1/Foxc2≈0.11) from **dynamic injury markers**
  (Fos=4.4, Jun=1.5, Cd44=0.93, Atf3=0.80) — exactly the homeostatic-vs-stress
  split predicted by the regenerative-flow hypothesis.

Local artifacts: `figures/{vorganoid_crosstalk,learning_regulome,regenerative_flow}.{png,*_results.json}`.
All three benchmarks gracefully fall back to TF-expression scoring when Pando
GRN (`data/zenodo/grn_modules.tsv`, DGX-only) is absent; richer regulon-based
scoring activates automatically on DGX Spark.

## Phase 4: Publication

- Comprehensive comparison table: Pando (R, pairwise) vs hgx (JAX, higher-order)
- 7 publication figures from real data
- Quantitative metrics with confidence intervals
- Code + data fully reproducible via Colab notebook
- [x] Consolidated into a single literate knitr/Sweave manuscript (`publication/paper.Rnw` → `paper.tex` → `paper.pdf`); result tables/values pulled live from `figures/*_results.json` + `data/cropseq/*.csv`

## Phase 5: Network control / the anatomical compiler (jaxctrl)

- [x] `benchmark_network_control.py` — linear network controllability + steer-to-target on the Pando TF co-regulation graph (controllability Gramians, structural driver nodes, LQR)
- [x] `benchmark_anatomical_compiler.py` — nonlinear optimal control on the *learned* Hypergraph Neural ODE (`diffrax` adjoints + Adam; jaxctrl LQR warm-start): target tissue state → TF-actuation schedule
- [ ] `minimum_driver_nodes` via hgx on the (non-uniform) Pando hypergraph — needs a uniform decomposition first
- [ ] richer plants (larger regulome, more timepoints, mechanistically-constrained drift) and richer actuators (the §4.3 bioprinting / optogenetic / dose / bioelectric handles)
- [ ] couple the anatomical compiler to the wet-lab loop (FRESH/SWIFT/PRINTESS print parameters as actuation; model-guided vasculature)
- [~] **bioelectric-layer companion: a JAX/`diffrax` refactor of BETSE/BETSEE** (active sister project; Pietak & Levin 2016/2017 finite-volume Vmem/gap-junction/ion-channel solver, made differentiable) — follows the Levin Lab / Mafe-group post-2018 shift from passive simulation to *agential design*: (i) inverse bioelectric design (`jax.grad` to a target shape — the xenobot move by gradient descent; Kriegman 2020/2021), (ii) bioelectric prepattern → triggers a secondary GRN (Pietak & Levin 2017; Cervera/Levin/Mafe 2024–2026 top-down Vmem↔transcription), (iii) morphoceutical intervention timelines (drug-cocktail schedule → limb regrowth; Murugan 2022; Pio-Lopez & Levin 2023), (iv) large-scale pattern integration / embryonic-brain prepattern (Manicka, Pai & Levin 2023). Each is one optimal-control problem on a differentiable plant — the bioelectric set-point added to the §4.3 actuator menu, with a real solver behind it. Manifesto: Levin 2021 (Cell). Refs 38, 39a–39e in `REFERENCES.md`; the natural Lab-9 companion (notebooks/README "Bioelectric-layer simulators").
  - [x] **inverse-design prototype implemented & verified** — `betse.science.jax.inverse.optimize_pattern` (gradient descent through differentiable physics solvers, MSE to a target Vmem pattern; SimState made JAX-tracer/pytree-compatible for JIT + autodiff). Verified benchmark `tests/betse_test/a00_unit/science/test_jax_inverse.py`: discover the transmembrane-current profile $J_n$ that drives a resting 10-cell tissue into an alternating depolarised(-20 mV)/hyperpolarised(-80 mV) pattern — loss 9e-4 → **1.36e-10** (near-exact). Run: `cd betse-unified && BETSE_JAX=1 uv run pytest tests/betse_test/a00_unit/science/test_jax_inverse.py -s`. This is the bioelectric instance of "the anatomical compiler" (target pattern in → required inputs out) — the foundation for the xenobot-motility / morphoceutical-timeline bridge examples. Docs: `docs/JAX_REFAC.md`.

## Educational track (notebooks/)

A planned course (~6–8 lab sessions) walking students through the project: regulomes & hypergraphs
→ fidelity benchmarking → modularity/identifiability → Hypergraph Neural ODEs → control theory on
cellular dynamics (jaxctrl) → the anatomical compiler → synthetic morphology in the wet lab →
(stretch) cancer as loss of module identifiability. See `notebooks/README.md` for the sequence.
Run out of **Biopunk Lab** (the West Coast node of [HTGAA 2026a](https://2026a.htgaa.org)); supports
the course's two **Genetic Circuits** modules — [Week 6](https://2026a.htgaa.org/2026a/course-pages/weeks/week-06/index.html)
*Assembly Technologies* (Densmore, Haddock — DNA assembly + the Asimov Kernel design platform;
assignment recreates the repressilator) and [Week 7](https://2026a.htgaa.org/2026a/course-pages/weeks/week-07/index.html)
*Neuromorphic Circuits* (Ron Weiss — intracellular ANNs / perceptrons in cells). Upstream foundations:
Elowitz & Bois, *Biological Circuit Design* (biocircuits.github.io). Complementary student-facing
virtual-tissue platforms (the *spatial/mechanics* side this track does not cover): **CompuCell3D**
(compucell3d.org — its [Workshop26](https://compucell3d.org/Workshop26) series and the
[nanoHub-hosted model library](https://compucell3d.org/Models-nanoHub)), **Morpheus**
(morpheus.gitlab.io), PhysiCell, Chaste — Cellular-Potts / agent-based; this track is the *regulome
/ control* complement (`hgx`/`jaxctrl`). Cellular-engineering programmes on the *wet-lab* side:
the **Center for Cellular Construction** (centerforcellularconstruction.org — NSF STC at UCSF + Cal
Academy; Marshall/Gartner), the **Gartner Lab** (gartnerlab.ucsf.edu — tissue self-organisation /
DPAC; already a benchmark: the conformation-controlled kidney data, `scripts/benchmark_gartner_4d.py`),
and the **Lim Lab** (limlab.ucsf.edu — synNotch / synthetic development / therapeutic-cell engineering;
Morsut 2016, Toda 2018/2020 — the Toda 2020 synNotch dataset is a benchmark, `benchmark_toda_morphogenesis.py`;
its "harnessing cellular modularity" framing ↔ the Module Identifiability Index, Lab 4).
Model standards/repositories the whole ecosystem shares:
**SBML** (sbml.org) + **BioModels** (biomodels.org) — pull a curated SBML model, import it into
`diffrax`, and you have a Lab. (Distinct thread — *structural* identifiability of model parameters:
the differential-algebra/power-series program of J. DiStefano III et al. / the COMBOS tool; the
numerical-local counterpart is `jaxctrl`'s observability functions. See `notebooks/README.md`.)
Seed material: `notebooks/01_gene_circuit_dynamics.ipynb`,
`notebooks/02_regulomes_and_hypergraphs.ipynb`, `notebooks/organoid_hgx_colab.ipynb`, the three
`jaxctrl/examples/*.{ipynb,py}`.

- [x] Setup — orientation (`notebooks/00_setup.ipynb`): the course map (the arc Setup→1→2→…→10, one line each, with per-lab deps); the prerequisites (Python/numpy; JAX — the differentiable-programming mindset; linear algebra / the graph Laplacian; ODEs & dynamical systems / attractors; light mol-dev bio; Elowitz & Bois' *Biological Circuit Design* with Lab 1 as the bridge; HTGAA Weeks 6–7 for context); an env check (required: numpy/matplotlib/jax/scipy; recommended/per-lab: equinox/diffrax/optax/hgx/sympy; optional: jaxctrl/scanpy — soft-imported); the data substrate (loads the Fleck organoid regulome or a synthetic fallback); four toolchain hello-worlds (jax.grad incl. through a loop; the hypergraph / clique-expansion blow-up; a differentiable ODE — RK4 + jax.grad through it; a 2-state plant — Kalman controllability + LQR via scipy.linalg.solve_continuous_are); the dependency map; the three "identifiabilities" disambiguation (module / structural / practical — and fidelity); a starter exercise (+ an optional pointer into the BETSE-JAX companion at ~/Workspace/betse-unified). Self-contained; runs in seconds
- [x] Notebook 1 — gene-circuit dynamics in a nutshell (`notebooks/01_gene_circuit_dynamics.ipynb`): Hill functions, negative autoregulation, the toggle switch & bistability (`jax.jacfwd`), the repressilator + linearize→LQR — the bridge from *Biological Circuit Design* into this course's toolchain
- [x] Notebook 2 — regulomes and hypergraphs (`notebooks/02_regulomes_and_hypergraphs.ipynb`): the Fleck incidence; why a regulon is a hyperedge not a clique; `hgx` basics; the Laplacian spectrum; exercises
- [x] Notebook 3 — benchmarking fidelity (`notebooks/03_benchmarking_fidelity.ipynb`): the fidelity triple (in-domain r / transfer direction / transfer r); organoid-720 vs Pollen-44 regulon–screen overlap (34 shared, hypergeometric p ≈ 1e-5); zero-shot organoid → primary-cortex transfer (raw regulon signs ≈ 60% direction vs a trained `PerturbationPredictor` ≈ 83%; magnitude r ≈ 0.13 vs ≈ 0.36 within-primary LOO); per-cell-type disaggregation (raw numbers near chance — aggregate accuracy can lie); fidelity ≠ structural identifiability (Lab 7) ≠ module separability (Lab 4) ≠ transfer fidelity; exercises (real CROP-seq via `extract_cropseq.py`; the `advanced_fidelity` pattern scores — fidelity of identity vs perturbation; cross-species conservation). Self-contained; pipeline = `scripts/compare_pollen.py`, `scripts/benchmark_advanced_fidelity.py`
- [x] Notebook 4 — modularity & identifiability (`notebooks/04_modularity_identifiability.ipynb`): builds the Hodge Laplacian (L0 = clique-expansion graph Laplacian on genes; L1 on regulons) by hand — harmonic null / connected components, Fiedler value, low-lying spectral gap; the Module Identifiability Index two ways (the `test_nitmb_modularity.py` heuristic mean(gaps[:10])/std(eigs[:10]) verbatim + a normalised relative eigengap ∈[0,1]); shows the genome-scale regulome is one tangled component with weak module identifiability (the index discriminates between systems, doesn't pull a module count out of a GRN); reproduces the cross-system NITMB ordering live (brain organoid 0.381 > fetal kidney 0.367 > bioprinted kidney 0.353); a light stop-signal sweep along pseudotime (MII / #components — late bins fragment); module ID ≠ structural ID (Lab 7) ≠ practical ID (Lab 8) ≠ fidelity (Lab 3) ≠ a unique number ≠ the module labels; exercises (L1 vs L0 — orderings can flip; eigengap → module count + labelling; multi-TF cooperativity; cancer-as-loss-of-module-identifiability, the Lab 10 stretch). Self-contained; pipeline = `scripts/benchmark_kidney_modularity.py`, `scripts/test_nitmb_modularity.py`, `scripts/06_topology.py`
- [x] Notebook 5 — Hypergraph Neural ODEs (`notebooks/05_hypergraph_neural_odes.ipynb`): cell identity as an attractor (Kauffman/Huang; Waddington); fits a Neural ODE (ẋ = f_θ(x), small MLP — hand-rolled RK4 + jax.grad + optax, no diffrax) on the organoid pseudotime timecourse in ~0.4 s; per-gene rollout MSE as a functional classifier — stable structural drivers (on the slow manifold, low MSE — fate TFs) vs transient stress responders (off-manifold spikes, high MSE — IEGs Fos/Jun/Egr1) with the committed kidney-IRI fit (regenerative_flow: Lhx1/Cdh1/Pax8 ≈0.05-0.11 vs Fos 4.4, Jun 1.5, Cd44 0.93, Atf3 0.80) and the activity-induced IEG timecourse (learning_regulome: peak-at-1h) as clean illustrations; the Waddington picture via a bistable toggle-switch demo (two basins, a separatrix, a KO that crosses it → switched fate) + fate_probabilities.npy / perturbation_fates.npy (FOXG1 and DLX2 the biggest single-TF landscape tilts); ties to Lab 4 (modules dissolve → driver set shrinks) and Lab 3 (well-fit drivers → transferable KO directions) and forward to Lab 6/6; what a Hypergraph Neural ODE is/isn't (the flow not the parameters — deliberately structurally non-identifiable; "hypergraph" = a weight-sharing prior, hgx.LatentHypergraphODE); exercises (ODE→SDE + CellRank fate entropy; Poincaré latent + Gromov δ; finer time T∈{20,50}; the hgx structured field; close the loop with jaxctrl). Self-contained; pipeline = `scripts/04_temporal_dynamics.py`, `scripts/benchmark_learning_regulome.py`, `scripts/benchmark_regenerative_flow.py`
- [x] Notebook 6 — control theory on cellular dynamics (`notebooks/06_control_theory.ipynb`): the regulome as a steerable plant ẋ = Ax + Bu (a linear TF co-regulation network on the top-200 Pando regulons, A = -(L_sym+εI), Hurwitz — mirrors `benchmark_network_control.py`); the network-control toolkit in pure JAX/SciPy (controllability matrix & Kalman rank; controllability Gramian → average vs modal controllability, Gu et al. 2015; minimum-energy control; LQR via solve_continuous_are; minimum driver nodes via Liu–Slotine–Barabási matching). Headline (the control mirror of Lab 7's observability result, by duality): the curated master regulators don't control the network — Kalman rank ≪ 200 from the 7 key TFs (≈9 at float32 tol matching the committed network_control_results.json, ≈30 at float64; the rank is tolerance-sensitive — singular values decay smoothly); control leverage ≠ identity — the top-leverage single TFs (FOXC2/PRDM6/FOXC1, reproduced exactly) aren't the developmental masters (only EOMES in the top-20); a single TF's modal controllability ≈ 0 (need a set/full actuation); steer x0(early)→xf(late) costs E* ≈ 586 / LQR cost-to-go ≈ 101 with full actuation vs effectively uncontrollable from the masters (Gramian λ_min ≈ 0); structural ≠ exact controllability (LSB's generic N_D ≈ 1 vs the real rank — our A is a Laplacian, non-generic weights). § on the three jaxctrl worked examples (repressilator_control_demo.py, irma_sindy_lqr.ipynb, grn_hypergraph_drivers.ipynb); exercises (linearise the Lab-5 ODE + control it; SINDy/Koopman surrogate; the unreachable subspace vs Lab 4's diffuse modules & Lab 7's unobservable directions; min-driver-nodes on the hypergraph; control–robustness trade-off, Yan 2017). Self-contained; pipeline = `scripts/benchmark_network_control.py` + the jaxctrl example notebooks
- [x] Notebook 7 — structural identifiability (`notebooks/07_structural_identifiability.ipynb`): a sympy Taylor-series/observability-rank check; textbook cases (one-pool, Bellman); the Lab-1 circuits; the linear special case = jaxctrl's observability rank, tied to benchmark_network_control.py; structural ≠ module ≠ practical identifiability; exercises (free Hill exponent, IRMA minimal output set, COMBOS example, SBI loss-valley)
- [x] Notebook 8 — the anatomical compiler (`notebooks/08_anatomical_compiler.ipynb`): Levin's anatomical compiler (target tissue state in → actuation schedule out) assembled at the level of a learned regulome surrogate — the four-stage stack: plant = a learned Neural ODE on the organoid timecourse (Lab 5, reused; hand-rolled RK4 + jax.grad, no diffrax needed); system-ID = a linear surrogate ż ≈ Az+c fit by least-squares on the rollout — which is *unstable* (max Re λ ≈ +9; the committed kidney-IRI run +31.8, surrogate_reliable=false): a developmental trajectory linearised is a transient, not a stable plant — the practical face of Lab 7, and why Lab 6 *chose* A=-L_sym; controller = an LQR warm-start on the clamped surrogate, then direct optimal control on the nonlinear plant (piecewise-constant u(t) on a chosen actuator set, minimise ‖x(T)-x_target‖² + λ‖u‖², Adam through the rollout — diffrax adjoints in the production version); validation = re-integrate the learned ODE under u*. Demonstrated as a knockout-rescue (in-silico knock down FOXG1 at t0 → free rollout drifts off → compute the actuation returning it to the wild-type endpoint): final-state error ↓~100% on the actuated TFs, ↓~39% overall — "you steer what you actuate" (the committed kidney-IRI run: ↓73% actuated, ↓21% all). § on what the compiler is/isn't (regulome state ≠ anatomy — the regulome↔form gap, where the cell-based simulators & the bioelectric layer come in; the flow not θ; the surrogate is a warm-start not the controller) + "the arc, Labs 2→8"; exercises (target a fate basin not a state — a toggle-switch fate-switch starter crossing the separatrix; SINDy/Koopman surrogate; MPC vs open-loop; couple the bioelectric layer via BETSE-JAX's optimize_pattern; actuator-set selection). Self-contained; pipeline = `scripts/benchmark_anatomical_compiler.py` + `scripts/benchmark_network_control.py` + the jaxctrl example notebooks + `~/Workspace/betse-unified` (betse.science.jax.inverse)
- [x] Notebook 9 — synthetic morphology in the wet lab (`notebooks/09_synthetic_morphology_wetlab.ipynb`, stretch): the §4.3 forward programme reframed so every modality is one optimal-control problem on the Hypergraph Neural ODE (Lab 8) — what changes is the actuator B and the readout layer: programmed (synNotch / synthetic morphogens — Morsut 2016, Toda 2018/2020; readout = the committed toda_results.json projection), printed (conformation / 4-D bioprinting — Feinberg/Lewis/Skylar-Scott/Gartner; readout = gartner_results.json man/r0/r40 → lineage maturity + system_maturity_results.json), bioelectric (V_mem prepattern — Levin/Mafe; the BETSE-JAX optimize_pattern), agential (xenobots/anthrobots — Gumuskaya 2024; readout = anthrobot_results.json). Demonstrates the control side with one shared bistable plant (the Lab-1/4/6 toggle used four ways — a synNotch input that flips the fate; a static print-geometry parameter biasing which attractor it self-organises into; a toy V_mem→GRN two-layer steer; a minimal-sender-fraction synNotch-circuit starter), the readout side with the committed benchmark panels, plus the model-in-the-loop design cycle (design→simulate→optimise→build→read out→refine) and what's real (the readouts) vs aspirational (closing the loop; the regulome↔form gap). Exercises (end-to-end synNotch circuit design + minimal sender fraction; multi-dim print geometry → target lineage mixture; the full two-layer bioelectric→GRN compiler via BETSE-JAX; a cross-system readiness score from the fidelity triple + MII + driver stability; the closed loop on a real dataset). Self-contained; pipeline = `scripts/benchmark_{toda_morphogenesis,gartner_4d,anthrobot_fidelity,advanced_fidelity,system_maturity}.py` + `~/Workspace/betse-unified`
- [x] Notebook 10 — cancer-as-loss-of-module-identifiability (`notebooks/10_cancer_module_identifiability.ipynb`, stretch, final): the course's metrics turned diagnostic. The TOFT / atavistic register (Soto & Sonnenschein; Trigos et al.; Levin 2021 — alongside, not instead of, the mutational view): cancer ⇒ the regulome's module identifiability falls (the Fiedler-region gap dissolves) + the homeostatic driver set comes loose while the transient/proliferative programs become persistent — the regeneration arrow of Lab 5 reversed ("the wound that doesn't heal"; the atavistic stress/proliferation machinery freed from multicellular constraint). Reuses Lab 4's MII machinery (mii_heuristic, relative_eigengap) on the one real comparison we have (Lab 4's organoid > blueprint > bioprinted trio — small differences, the metric is coarse) + a clearly-labelled schematic of the spectral signature (clear-cliff → soft-cliff → ramp — the relative_eigengap tracks it and ranks the cartoon right, the coarse mii_heuristic doesn't, Lab 4's caveat live) + the dynamical face (the committed kidney-IRI driver split + a toy "cancer flip") + the constructive flip (the anatomical compiler reframing therapy as "normalise, don't (just) kill" — differentiation therapy; the bioelectric instance, repolarisation / the BETSE-JAX optimize_pattern; morphoceuticals — Pio-Lopez & Levin 2023) + the diagnostic flip (the readout that certifies an engineered tissue, run in reverse, flags a diseased one — Davies's "know when you have"). Exercises (run the real primary→organoid→tumour-organoid→cancer-line gradient — the headline open question; which Hanahan–Weinberg hallmark predicts the MII drop; V_mem vs MII; the normalisation control problem — a working two-attractor "differentiated ⇄ tumour" toggle + the dose to revert it, Lab 8's machinery; atavism spectrally — the unreachable subspace vs the unicellular-ancestral gene set). Self-contained; pipeline = `scripts/{benchmark_kidney_modularity,test_nitmb_modularity,benchmark_disease_enrichment,benchmark_tf_disease,benchmark_tang_bioprinting,validate_choose}.py` + `~/Workspace/betse-unified`.  **→ the 10-lab course is now complete (Setup + Labs 1–10 + the colab).**

## Scripts Added (2026-03-15)

- **benchmark_standard.py** — Standard cocitation benchmark (Cora validated)
- **accuracy_ablation.py** — Class-balance ablation study
- **fix_thnn.py** — THNNConv overflow fix (sum-of-logs clamping)
- **extract_cropseq.py** — CROP-seq data extraction from Seurat objects
- **benchmark_comparison.py** — Updated with DHG device mismatch fix

## Technical Debt

- [x] Fix THNNConv NaN stability — root cause: sum-of-logs overflow, not gradient clipping. fix_thnn.py created
- [x] Fix SheafDiffusion OOM — edge_stalk_dim capped
- [ ] Integrate THNNConv fix into hgx core
- [x] Fix Poincare LatentODE dimension handling (fixed in cac6cba)
- [x] Improve perturbation training data — real CROP-seq DE extracted from Seurat objects.
- [x] Implement robust large file downloader with resume support.
- [ ] Add proper train/val/test splits for module detection
- [ ] Add cross-validation to all analyses
- [ ] Profile GPU utilization (currently CPU-heavy for centrality + persistence)
