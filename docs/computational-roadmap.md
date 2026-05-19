# Computational roadmap — directions the project hasn't yet covered

*Survey of computational approaches that are **mature enough to use** but **not yet in this project**, ranked by leverage against the stated goals: **predictably engineer tissue states; close Levin's anatomical compiler loop; bridge the regulome → form gap**. Written 2026-05-13, conditional on the state at commit `7eb180b` (the course at Setup + Labs 1–10 + the [BETSE-JAX](../README.md#bioelectric-layer-companion) bioelectric companion). For the as-built picture, see [`ROADMAP.md`](../ROADMAP.md); for the conceptual map, [`notebooks/README.md`](../notebooks/README.md).*

The labs already cover: static regulome topology ([Lab 4](../notebooks/04_modularity_identifiability.ipynb) — Hodge Laplacians, MII), perturbation fidelity ([Lab 3](../notebooks/03_benchmarking_fidelity.ipynb)), Hypergraph Neural ODEs as the learned plant ([Lab 5](../notebooks/05_hypergraph_neural_odes.ipynb)), linear network control ([Lab 6](../notebooks/06_control_theory.ipynb) / `jaxctrl`), structural identifiability ([Lab 7](../notebooks/07_structural_identifiability.ipynb)), nonlinear optimal control / the anatomical compiler ([Lab 8](../notebooks/08_anatomical_compiler.ipynb)), the wet-lab forward programme ([Lab 9](../notebooks/09_synthetic_morphology_wetlab.ipynb)), and cancer as a diagnostic case ([Lab 10](../notebooks/10_cancer_module_identifiability.ipynb)). Plus the [BETSE-JAX](../README.md#bioelectric-layer-companion) bioelectric layer with inverse design / xenobot motility / Vₘₑₘ→GRN triggering / morphoceutical timelines.

The families below are the major *missing* ones. They aren't a critique of what's there — most are downstream of work already done.

> **Update (2026-05-13) — wet-lab availability re-ranks this list.** Biopunk Lab now has wet-lab capacity to execute experiments from these simulations (see [`docs/wetlab-program.md`](wetlab-program.md)). That moves **§2 (active learning / Bayesian experimental design)** from "useful enrichment" to **urgent** — it's the math needed to decide which experiment to run next under a finite wet-lab budget. **§1 (foundation models + causal inference)** stays highest-leverage on the *prediction* side; the CRISPRi-style wet-lab arm is also natively interventional, ideal for the causal-discovery methods listed there. **§3 (differentiable Cellular Potts)** is **deferred but stubbed** — model-side richer-plant work without an immediate wet-lab validation pathway; revisit once the loop is producing data. Design recorded in [`docs/differentiable-cp.md`](differentiable-cp.md); sister-repo skeleton at `~/Workspace/cpjax/`. **§4 (information theory)** and **§7 (generative perturbation models — CellOT, flow-matching)** become newly relevant because the wet-lab arm needs channel-capacity specs (synNotch) and prior distributions for prediction (perturbation responses).

---

## 1 — Foundation models + causal inference for perturbation prediction *(highest leverage)*

**Why:** Lab 3's fidelity-triple transfer r ≈ 0.13 (and the 60 % raw / 83 %-with-a-trained-predictor direction gap) is the project's most empirically-exposed ceiling. The field has moved a lot here since the project's design, and the gains are realistic.

- **scRNA-seq foundation models.** **scGPT** (Cui et al. 2024, *Nat Methods*), **Geneformer** (Theodoris et al. 2023, *Nature*), **scFoundation** (Hao et al. 2024, *Nat Methods*), **CellPLM** (Wen et al. 2024), and — the most directly relevant — **Prophet** (Theis Lab; already a TODO in `ROADMAP.md` §3D). These are pretrained on 30–100 M cells; zero-shot perturbation prediction routinely beats task-specific models on the kind of benchmarks Lab 3 runs.
- **Causal discovery on interventional CRISPRi data.** The project treats CRISPRi as a *prediction* target; it's natively *interventional* data and a much richer signal. Mature methods: **NOTEARS** (Zheng et al. 2018 — continuous-optimisation DAG learning, naturally JAX-friendly), **DAG-GFlowNets** (Deleu et al. 2022), **dCDI** (Brouillard et al. 2020 — differentiable causal discovery *from interventions*), **JCI** (Mooij et al. 2020). A causally-learned GRN sits next to Pando's associative one, and is plausibly what the *transfer* failures of Lab 3 are diagnostic of: associative regulons that don't generalise under interventions in a new context.

**Where it slots.** Extending Lab 3 with a Prophet / scGPT / NOTEARS comparator track on *real* held-out perturbations (a previous attempt at a dedicated Lab 11 stub-mode demonstration was retired — its synthetic held-out-gene benchmark conflated structure-aware features with the pretraining lift). Effort: ~weeks. Impact: would shift the headline number that gates every downstream lab.

---

## 2 — Active learning, Bayesian optimal experimental design, RL for the closed loop

**Why:** Lab 9 articulates the model-in-the-loop design cycle but doesn't supply the math of *which experiment to do next, given the model's current uncertainty?* Without it, the closed loop with wet-lab partners is prose.

- **Bayesian optimal experimental design** — D-/A-/EIG-optimality; **Foster et al. 2019** "Variational BED". Given the current Hypergraph Neural ODE posterior, pick the perturbation that maximises expected information gain about the parameter of interest (or the actuation policy). **Direct precedent on cell-based simulators:** Ozik et al. 2019 (ref. 78a) — the EMEWS framework running active learning + genetic algorithms over PhysiCell parameter spaces on HPC, for the same kind of tumor-immune dynamics this project's [Lab 10](../notebooks/10_cancer_module_identifiability.ipynb) touches. The wet-lab capstone in [`docs/wetlab-program.md`](wetlab-program.md) is shaped like a Ozik-EMEWS run on a real lab instead of a simulator. **Concrete reference implementation**: [`PhysiBoSS/spheroid-tnf-v2-emews`](https://github.com/PhysiBoSS/spheroid-tnf-v2-emews) — EMEWS over PhysiBoSS for TNF-dosing optimisation on a tumor spheroid (Ponce-de-Leon 2022 *Front Mol Biosci*, doi:10.3389/fmolb.2022.836794), uses GA + CMA-ES, swept parameters are treatment duration / concentration / timing — directly transferable to the wet-lab cycle scheduling.
- **Bayesian optimisation** for actuation cocktails and 4D-bioprinting geometry sweeps (the Gartner-style design map of §4.3(i) is *literally* a continuous-parameter BO problem). GP or TS-based; tools like BoTorch / Trieste.
- **Reinforcement learning** as a generalisation of Lab 8's direct OC: **model-based RL** (PETS, MuZero) on the learned Neural ODE; **offline RL** (CQL, BCQ, IQL) on the existing perturbation database (Pollen + CHOOSE + …); **goal-conditioned RL** with the target tissue state as the goal embedding.
- **Inverse RL** on the developmental pseudotime data: recover the *implicit cost function* / Lyapunov function the developmental program follows. Would give a principled definition of "maturity" beyond the hand-coded marker-gene sums.

**Where it slots.** A "Lab 8.5 — From direct OC to MPC / BO / RL" extension, or as `jaxctrl/examples/` notebooks alongside the existing three. Effort: ~weeks. Impact: makes Lab 9's design cycle actually executable.

---

## 3 — The regulome → form gap: spatial transcriptomics + differentiable Cellular Potts

**Why:** The deepest open problem the project flags. The current Hypergraph Neural ODE steers regulome states; tissues have *shape* too. Two concrete missing pieces:

- **Spatial transcriptomics.** Visium / Stereo-seq / MERFISH / Xenium — the project has *no spatial layer*. Mature methods to integrate: **cell2location** (Kleshchevnikov et al. 2022) for spatial deconvolution, **Tangram** for tissue mapping, **SpaceFlow** / **STAGATE** / **GraphST** for spatially-aware embeddings. The Hypergraph Neural ODE generalises naturally to a **spatial Neural PDE** on a cell-cell graph; `hgx` already supports the cell-graph data structure.
- **A differentiable Cellular Potts simulator** — the sister to [BETSE-JAX](../README.md#bioelectric-layer-companion). BETSE-JAX made the *bioelectric* layer differentiable; nobody has done the same for the *cell-shape / movement / division / adhesion* layer (CC3D/Morpheus territory). **jax-md** (Schoenholz & Cubuk 2020) is the closest precedent — differentiable molecular dynamics, fully GPU. A JAX-native CP with `jax.grad` through the Hamiltonian + Glauber updates would let the anatomical compiler optimise *shape* directly. Hard (stochastic MC isn't naively differentiable — Gumbel-Softmax / score-function estimators / continuous relaxations needed), but the BETSE-JAX architecture (`step_pure` ↔ `lax.scan`) maps cleanly. A real scientific contribution if pulled off. **Full plan in [`docs/differentiable-cp.md`](differentiable-cp.md); namespace-reserved stub at `~/Workspace/cpjax/`** (Phase 0–5 skeletons, `NotImplementedError` bodies, smoke tests).

**Where it slots.** A new `~/Workspace/cpjax` sister repo (stubbed 2026-05-13); a `Lab 12 — Spatial regulomes & differentiable morphology` in this repo consuming from it. Effort: months (CP differentiability is genuinely research-grade). Impact: closes the project's deepest stated gap.

---

## 4 — Information theory of cellular communication

**Why:** Lab 9's synNotch design needs a **channel-capacity** analysis — how many bits/cell-cell-contact does a synNotch carry, how many bits/dose does a morphoceutical schedule carry. The project does no information theory currently, and it's a natural fit given the regulatory-circuit framing.

- **Mutual-information GRN inference.** **ARACNe** (Margolin et al. 2006), **PIDC** (Chan et al. 2017 — partial information decomposition), **MIIC** (Verny et al. 2017). Non-parametric, complementary to Pando's GLM.
- **Information bottleneck** (Tishby–Pereira–Bialek 1999; Alemi et al. 2017 "Deep variational IB") for cell-state representation. The IB-optimal latent *is* a disentangled latent over cell state — a candidate interpretable coordinate system on top of Lab 5's Hypergraph Neural ODE.
- **Channel capacity of synthetic circuits.** Concrete: bits/cell-cell-contact for a synNotch (Morsut/Toda receptors), bits/dose for a morphoceutical schedule (Murugan/Pio-Lopez), bits/Vₘₑₘ-level for a bioelectric prepattern. These are quantitative engineering specs that don't yet exist anywhere.
- **Cheng et al. 2019** "Information flow in biological networks" — relate the Hodge spectrum and the GRN's information capacity directly. A bridge from Lab 4 to the synNotch / bioelectric layers.

**Where it slots.** A small new methods subsection of the paper between fidelity (§2.6) and modularity (§2.3); an information-theoretic readout in the Lab 9 design cycle.

---

## 5 — Probabilistic / Bayesian Neural ODEs

**Why:** Lab 5 fits a *point estimate* of the Neural ODE. Lab 7's structural-identifiability result says this is correct asymptotically ("the parameters are non-identifiable; it's the flow that matters"), but uncertainty *over the flow itself* is missing.

- **Bayesian Neural ODEs**: **Laplace approximation** (Daxberger et al. 2021 "Laplace redux"), **variational ODEs** (Yıldız et al. 2019), Hamiltonian-MCMC over Neural-ODE weights, stochastic-weight-averaging-Gaussian. Probably the lowest-effort way to add calibrated uncertainty.
- **Real stochastic Neural ODEs / SDEs** (Li et al. 2020 "Scalable gradients for SDEs") — Lab 5 has an SDE exercise (a), but the implementation is just a sketch. A real SDE fit gives a *variance* trajectory along pseudotime (cf. the CellRank fate-entropy comparison the exercise prompt mentions).
- **Posterior-predictive cost-to-go for LQR** — when Lab 8's control fails on a target, *is it* because the plant is uncertain there, or because that region is genuinely uncontrollable? Currently impossible to disentangle.
- **Normalizing flows on cell-state densities.** NSF, RealNVP, FFJORD — differentiable, evaluable densities. Lab 8's "target state" becomes a *target distribution*; the SBI inverse becomes well-posed.

**Where it slots.** A Bayesian extension to Lab 5; tightly couples to the SBI inverse Lab 8 mentions. Effort: ~weeks. Impact: closes a real gap in Lab 7/8's argument.

---

## 6 — Combinatorial optimization for actuator-set selection

**Why:** Lab 6 raises the "which subset of TFs to perturb?" question (exercise (e), the control-allocation problem). This is **combinatorial**, not continuous, and the project's `jax.grad`-everywhere posture misses it.

- **Submodular maximisation** (Krause & Golovin 2014): greedy with a 1−1/e approximation guarantee. Maps onto the "marginal controllability gain" framing in Lab 6 exactly.
- **Mixed-integer programming** (Gurobi, SCIP, HiGHS) for exact small-instance solutions as reference points.
- **Matroid-constrained optimisation** for combined budget + diversity constraints ("pick 7 TFs, at least 2 from each fate axis").
- **Greedy + lazy evaluation** for the high-leverage TF ranking from Lab 6 §3.

**Where it slots.** Lab 6 §6(e) — currently an exercise prompt; could be a real solver with one focused PR. Effort: ~days. Impact: makes the high-leverage TF selection from §3 into a *prescription*.

---

## 7 — Generative models for cell states & perturbation responses

The project uses VAEs / flows minimally. Mature alternatives:

- **scVI / scANVI** (Lopez 2018; Xu 2021) — the canonical probabilistic VAE for single-cell. Mentioned in [the memo on three identifiabilities](../README.md) and the BETSE-JAX/ZILLNB discussion; would slot into Lab 8's SBI inverse if/when it's built.
- **Diffusion / flow-matching models for perturbation prediction.** **CellOT** (Bunne et al. 2023), **scDiffEx**, **PerturbDiff** — predict the post-perturbation distribution from the pre-perturbation state via a learned diffusion bridge.
- **Schiebinger 2019 "Waddington-OT" / CellFlow** — optimal-transport-based density estimation that scales to 100 k+ cells; modern revival is in **CellFlow** (Theis Lab; already a project TODO).
- **Generative regulome design**: diffusion / score models *on regulome topology itself* — given a target circuit logic, sample plausible GRN graphs that implement it. A generative complement to the project's analysis-only posture, and a natural post-Lab-10 direction.

---

## 8 — Mechanistic / physics-aware ODEs and PDEs

The project has Hill ODEs ([Lab 1](../notebooks/01_gene_circuit_dynamics.ipynb)) and Neural ODEs ([Lab 5](../notebooks/05_hypergraph_neural_odes.ipynb)) but lacks the in-between:

- **Physics-informed Neural Networks** for the regulome — enforce mass conservation, monotonicity, or Hill-kinetics priors as soft losses on the Neural ODE drift. Cuts down on the structural non-identifiability of Lab 7.
- **Neural PDEs** for reaction-diffusion on a spatial grid (a generalisation of Lab 5 once spatial data is in scope — §3 above).
- **Symplectic / Hamiltonian Neural Networks** (Greydanus et al. 2019) — conservation-respecting dynamics. Probably overkill for GRNs but elegant for the bioelectric layer.
- **Gillespie / tau-leaping simulators** for stochastic gene expression — the discrete-stochastic story (Elowitz & Bois cover it; the project's Lab 1 doesn't). A JAX-native Gillespie with re-parameterised-gradient-estimator surrogates would slot next to BETSE-JAX.

---

## 9 — Topology beyond Hodge Laplacians

[Lab 4](../notebooks/04_modularity_identifiability.ipynb) uses graph Laplacians and Hodge $L_0$/$L_1$. Other relevant topology:

- **Sheaf neural networks** (Bodnar et al. 2022). The repo touched this in Phase 1A (SheafDiffusion OOM, dropped). Worth retrying with the post-BETSE-JAX maturity.
- **Mapper algorithm** (Singh–Memoli–Carlsson 2007) — topological skeleton of cell-state space; complementary to Hodge.
- **Multi-parameter persistent homology** for trajectories — captures how topology changes along pseudotime; tools like **RIVET**, **multipers**.
- **Ollivier–Ricci curvature** on regulome graphs (Sandhu et al. 2015) — quantifies bottleneck structure in a way Fiedler doesn't. Complement to the MII.
- **Group-equivariant neural networks** (Cohen et al. 2016) — if any symmetries exist in the regulome (paralog pairs, evolutionary duplication), an equivariant architecture exploits them.

---

## 10 — LLM / code-agent integration *(orthogonal but real)*

Increasingly mature for biology:

- **Literature mining / hypothesis ranking** — LLM-driven literature reviews; gene-function priors from PubMed; protein-pathway annotation.
- **LLM-as-judge** for biological plausibility of generated regulons or perturbation responses (Liu et al. 2024).
- **LLM-driven experimental design** as a sanity layer over §2's BO/AL.
- **Protein design models** (RFDiffusion, ESM3, AlphaFold-multimer) for the *molecules* the regulatory circuits run on — relevant when Lab 9's synNotch design needs novel receptor scaffolds.

---

## Strategic ranking

Pick three to push first, in this order:

1. **Foundation models + causal inference for perturbation prediction (§1).** Directly attacks the project's most exposed empirical weakness (Lab 3 transfer r ≈ 0.13). Models exist (Prophet/scGPT/Geneformer); data is in place. High effort, **high impact**, well-paved infrastructure. Probably the single highest-leverage move.
2. **Active learning / Bayesian experimental design (§2).** Without it, [Lab 9](../notebooks/09_synthetic_morphology_wetlab.ipynb)'s model-in-the-loop design cycle stays aspirational. Closing the actual loop with wet-lab partners *requires* next-experiment-to-do math. Medium effort, **high impact** on the §4.3 forward programme.
3. **Differentiable Cellular Potts / spatial morphology (§3 second half).** The long-shot but high-payoff one. Closes the regulome→form gap the project flags as deepest. Hard (stochastic MC isn't naively differentiable; needs continuous relaxations), but the BETSE-JAX precedent says it's doable; a real scientific contribution if pulled off. Months of effort, **transformative** if successful.

The others — information theory, Bayesian Neural ODEs, generative models, combinatorial control, advanced topology, LLM agents — are useful enrichments. Each could justify a single notebook or focused script. None are foundational gaps; they're paint, not load-bearing structure.

---

## How to apply

For a future agent picking this up:

- Start with `notebooks/00_setup.ipynb` (the arc) and `ROADMAP.md` (the as-built status). Then this doc.
- When the user revisits a topic in this list, link to its section here rather than re-surveying.
- If the user authorises a push into one of these, expect a Lab-style notebook artefact (cf. [the recipe](../notebooks/README.md)) and/or a new sister repo (cf. BETSE-JAX) as the natural unit. A new Lab is the lower-cost path; a new sister repo is for things (like differentiable Cellular Potts) that are large enough to warrant their own pyproject + uv.lock + CI.
- The [memo on three identifiabilities](../notebooks/00_setup.ipynb) and [the RMT-ablation memory](../README.md) document load-bearing diagnostic priors — *the project's headline numbers are structural, not preprocessing-fragile* — that should constrain how each direction here is framed. None of these methods is a panacea against [Lab 7](../notebooks/07_structural_identifiability.ipynb)'s identifiability geometry or [Lab 6](../notebooks/06_control_theory.ipynb)'s controllability gap; they're complementary leverage points.
