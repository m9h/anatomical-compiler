# Wet-lab programme — from simulation to bench at Biopunk

*Planning document for closing the model-in-the-loop design cycle of [Lab 9](../notebooks/09_synthetic_morphology_wetlab.ipynb) / paper §4.3 at **Biopunk Lab**, the West Coast node of HTGAA 2026a. Written 2026-05-13 against project state `c940a44`. Companion to [`docs/computational-roadmap.md`](computational-roadmap.md) (the forward survey of dry-side work) and [`ROADMAP.md`](../ROADMAP.md) (as-built status).*

The project's simulations now meet a wet lab. This doc proposes what to run first.

## Constraints (typical community-lab posture)

Cost and capability sit on a steep gradient — **plan the first experiments at the qPCR/immunostaining/bulk-RNA-seq tier**, save scRNA-seq for after the model has narrowed the design space:

| readout | cost / sample | resolution | when to use |
|---|---|---|---|
| qPCR (small panel) | $ | 5–20 genes, bulk | first-pass validation of single-TF predictions, marker timecourses |
| Immunostaining / flow | $$ | 5–10 markers per stain | tissue-level patterning checks (Lab 9 (i)–(iii)) |
| Bulk RNA-seq | $$ | whole transcriptome, bulk | maturation programmes, fate-mixture inference |
| scRNA-seq (10x) | $$$$ | single-cell, full | required for Hypergraph Neural ODE fits ([Lab 5](../notebooks/05_hypergraph_neural_odes.ipynb)), MII ([Lab 4](../notebooks/04_modularity_identifiability.ipynb)), the perturbation predictor ([Lab 3](../notebooks/03_benchmarking_fidelity.ipynb)) |
| Optogenetic / ionophore | $$ (reagents) | acute, reversible | bioelectric perturbations ([Lab 9 (v)](../notebooks/09_synthetic_morphology_wetlab.ipynb), BETSE-JAX validation) |

A reasonable budgeting heuristic: **bulk → focused single-cell → broad single-cell**, with the dry side narrowing the design space at each gate.

## The headline questions worth testing first

Each entry below is keyed to a *committed* simulation result; the wet-lab experiment is the validation. Format: *question · simulation prediction · experiment · cost tier · readout*.

### A. Does the Module Identifiability Index ordering hold on a Biopunk-grown sample?

- **Prediction (Lab 4 / `nitmb_modularity_report.json`):** brain organoid (0.381) > fetal-kidney blueprint (0.367) > bioprinted kidney (0.353) — *self-organisation makes sharper modules than cell-by-cell printing*.
- **Experiment:** generate one self-organised organoid and one matched bioprinted construct from the same starting cells; scRNA-seq both at a comparable maturation stage; run `scripts/benchmark_kidney_modularity.py` on each.
- **Cost:** scRNA-seq tier — but only **two samples**, so ~$ minimal compared to a screen.
- **Why first:** **directly tests the project's headline cross-system result** with one of its own readouts; the answer is either "the ordering replicates" (validates the diagnostic) or "it doesn't" (shifts the whole interpretation). Either way, informative.
- **Risk:** scRNA-seq of just two samples is small-N; the MII band is narrow (0.35–0.38) so the effect-of-interest is also small. Power-limited — consider three replicates per arm if budget permits.

### B. Does the homeostatic / stress driver split hold in an organoid injury–recovery time-course?

- **Prediction (Lab 5 / `regenerative_flow_results.json`):** during injury–recovery, the *stable structural drivers* (Lhx1/Cdh1/Pax8/Pax2/Six1/2/Wt1/Foxc2; rollout MSE ≈ 0.05–0.11) fit a smooth Hypergraph Neural ODE; the *transient stress responders* (Fos 4.4 / Jun 1.5 / Cd44 0.93 / Atf3 0.80) don't. The split is the regenerative-flow finding's load-bearing claim.
- **Experiment:** mechanically or chemically wound a Biopunk organoid (mature enough to have differentiated drivers); harvest at t = 0, 1 d, 3 d, 7 d, 14 d; **bulk RNA-seq** the timecourse. Fit a small Neural ODE to the per-TF trajectory (Lab 5 §2 machinery, on bulk-data); compute per-TF rollout MSE; check whether the split appears.
- **Cost:** **bulk-RNA-seq tier** (~ 5 timepoints × few replicates) — much cheaper than scRNA-seq.
- **Why first:** this is the *project's most mechanistically-falsifiable* finding. If a clean stable-vs-transient split shows up in a Biopunk-grown organoid's injury response, the regenerative-flow hypothesis transfers from kidney IRI to *any* tissue. If it doesn't, the finding is tissue-specific.
- **Add-on:** the experiment also generates a *new* benchmark for Lab 5 (an organoid injury time-course, alongside the existing Balzer-2022 kidney IRI).

### C. Does a Lab 8 anatomical-compiler prescription actually rescue an in-silico-fated cell?

- **Prediction (Lab 8 §6(d) demo):** in the toggle-switch toy compiler, a calibrated bioelectric set-point (or TF cocktail) drives a perturbed cell back across the separatrix into the wild-type basin. The simulated kidney-IRI run gives a *73 % reduction* in actuated-TF error / *21 % overall*.
- **Experiment:** at the scale Biopunk can run: pick one strong "landscape tilter" from Lab 5's `perturbation_fates.npy` — **FOXG1** (‖Δfate‖ = 0.84) or **DLX2** (0.82) — and a few candidate rescue TFs the simulation prescribes. Use **CRISPRi** for the KO (a single guide is cheap; lentiviral or RNP); deliver the rescue TFs via mRNA or small-molecule analogues. **qPCR** for a marker panel (10–20 fate-marker genes) on KO-only vs KO+rescue. Optional: bulk RNA-seq on the most informative arm.
- **Cost:** **qPCR tier** for the first pass — the cheapest serious validation possible. **Bulk-RNA-seq tier** for the confirmation arm.
- **Why first:** **the most directly closed-loop experiment** the project can run. Simulation gives a prescription; bench tests whether the prescription works. If yes, you have a wet-lab-validated anatomical-compiler instance. If no, you have a calibration target for the model — which is *also* a result.
- **Risk:** the toggle-switch demo is a *toy* (Lab 8 §6(d)); the real organoid system is higher-dimensional and the prescription may not transfer cleanly. The qPCR-tier first pass de-risks this — if the simulated rescue cocktail moves the marker panel in the predicted direction, scale up; if not, the model needs more before bench.

### D. A focused 4D-bioprinting design-map sweep (Lab 9 (i) at Biopunk scale)

- **Prediction (Gartner 2021 / `gartner_results.json` / paper §sec:results-4d):** higher aspect-ratio print conformations maximise proximal-tubule + podocyte maturity, minimise off-target stroma.
- **Experiment:** if the lab has a [PRINTESS](https://printess.org) (Skylar-Scott $250 open-hardware bioprinter — already a referenced ecosystem tool in this project), run **3–5 print conformations** of the same kidney-progenitor starting material; harvest at a common timepoint; **immunostain** for podocyte (NPHS1/2), proximal-tubule (LRP2/CUBN), stromal (COL1A1) markers; quantify with image analysis.
- **Cost:** **immunostaining tier** — far cheaper than scRNA-seq; if PRINTESS is available, also cheap on the print side.
- **Why first:** the cleanest *physical-actuator* test in the §4.3 menu. No CRISPRi or transfection required; the actuator is *geometry*, which the lab can manipulate directly. Validates the Gartner-style finding in a fresh lab setting.
- **Add-on:** the design-map produced is the *prior* for a Bayesian-optimisation sweep — feeds directly into the active-learning track (`docs/computational-roadmap.md` §2).

### E. SynNotch sender–receiver patterning (Lab 9 (ii) — needs cloning capacity)

- **Prediction:** Morsut 2016 / Toda 2018, 2020 — a sender cell expressing ligand X and a receiver expressing a CD19-synNotch→TF-Y circuit creates an X-controlled Y-on/Y-off spatial pattern. The project's Toda-2020 benchmark (`figures/toda_results.json`, [Lab 9 §2](../notebooks/09_synthetic_morphology_wetlab.ipynb)) maps the ON/OFF toggle into the Neocortex Atlas's growth-vs-arrest axis.
- **Experiment:** **the cheapest closed-loop in synthetic developmental biology**. Get the Morsut/Toda plasmids (Addgene, free for academic use); transfect HEK293T (sender) and a co-cultured reporter line (receiver); image the spatial pattern; **qPCR or bulk RNA-seq** on sorted ON vs OFF receivers; project into the Neocortex Atlas pattern space (`scripts/benchmark_toda_morphogenesis.py`).
- **Cost:** **cloning + transfection + qPCR/bulk tier** — well within community-lab scope if cloning capacity exists.
- **Why first:** validates the *programmed-regulatory-logic* arm of §4.3 directly, on the cheapest cell line; gives Biopunk a synthetic-biology beachhead the rest of the wet-lab programme can build on.
- **Risk:** synNotch circuits are *finicky* — first transfection often gives leaky on-states. Budget two rounds.

### F. Bioelectric perturbation + scRNA-seq (Lab 9 (v) — couples to BETSE-JAX)

- **Prediction (Lab 8 §6(d) two-layer compiler + BETSE-JAX `optimize_pattern`):** a target Vmem set-point drives a downstream regulatory program. The BETSE-JAX inverse-design pipeline gives the required ion currents; a Vmem→GRN coupling (`betse.science.jax.physics.grn_trigger`) predicts the transcriptional response.
- **Experiment:** **acute Vmem perturbation by ionophore** (nigericin, valinomycin, gramicidin — depolarising / hyperpolarising at known thresholds) on a Biopunk cell line for 6 h, 24 h; **scRNA-seq or bulk RNA-seq**; check whether the BETSE-JAX-predicted gene set shifts.
- **Cost:** **bulk RNA-seq tier**, or scRNA-seq if budget allows.
- **Why first:** the BETSE-JAX side is *ready* and the ionophore reagents are cheap. This is the **most direct test of the bioelectric → GRN coupling** the project posits, with the simplest possible actuator (an ionophore is uniform across cells, no spatial structure required).
- **Stretch:** if the bulk-level prediction holds, scale to optogenetic Vmem control + spatial readout — the §4.3(v) full programme.

## Recommended sequencing (model-in-the-loop)

A cycle order that minimises bench-cost and maximises learning per round:

1. **Cycle 1 — calibration** (cheapest tier; one experiment): pick *one* of B (organoid injury bulk-RNA-seq) or D (PRINTESS sweep + immunostaining) — both are bulk/imaging tier, both test load-bearing project claims. Result feeds back into the relevant lab's notebook as a new committed result panel.
2. **Cycle 2 — circuit beachhead** (cloning tier; one experiment): E (synNotch sender–receiver). Builds reusable lab capacity (cloning, transfection, sorting) that every subsequent experiment uses.
3. **Cycle 3 — closed-loop test** (qPCR-tier first; bulk if promising): C (FOXG1-KO rescue prescribed by Lab 8). The first *prescription*-style experiment.
4. **Cycle 4 — bioelectric layer** (bulk/scRNA-seq tier): F (ionophore + RNA-seq). Validates the BETSE-JAX coupling.
5. **Cycle 5 — full single-cell** (scRNA-seq tier): A (organoid vs bioprinted MII ordering). The most expensive validation, run last when the dry-side panel is robust.

This gives the project a path from "we have a wet lab" to "we have a closed loop" without spending the scRNA-seq budget upfront. The active-learning math the [`computational-roadmap.md`](computational-roadmap.md) §2 flagged as urgent applies here: at each cycle, the *next* experiment is the one that most reduces uncertainty about the next prediction; formalise that as a Bayesian-optimal-experimental-design loop once Cycles 1–2 produce calibration data.

## Open questions — to scope further

Before any of the above is concrete:

1. **Equipment** — what cell culture, bioprinting (PRINTESS / commercial?), transfection, imaging, qPCR, sequencing access does Biopunk actually have? Is scRNA-seq local or outsourced?
2. **Budget** — per-experiment ballpark, and how many experiments per quarter.
3. **People & cadence** — who runs the bench; how many parallel experiments at a time; turn-around per cycle.
4. **Existing constructs / cell lines** — anything already in the freezer (organoid protocols, lentiviral CRISPRi lines, synNotch plasmids) that constrains/enables specific experiments.
5. **Partners** — Lim Lab (synNotch), Gartner Lab (DPAC), Levin Lab / Mafe-group (bioelectric), Feinberg / Skylar-Scott (bioprinting) — any of these tied in for reagents / protocols / sequencing?

Once these are answered, this doc becomes specific — concrete protocols, cell-line picks, expected timeline. Right now it's a *prioritised menu*.

## As an HTGAA 2026/2027 final project

Biopunk Lab is the West Coast node of HTGAA 2026a; the course's two Genetic Circuits modules (Week 6: Densmore/Haddock — DNA assembly + the Asimov Kernel; Week 7: Weiss — neuromorphic / intracellular ANNs) are direct upstream context for this whole project, and the dry-side educational track (Setup + [Labs 1–10](../notebooks/README.md)) was designed to *be* the computational companion to those modules. **Closing the loop at the bench is the natural HTGAA capstone.** Concretely:

### Curricular structure (one HTGAA term)

- **Weeks 1–7 (the HTGAA course proper):** students complete Weeks 1–5 of HTGAA, then Weeks 6–7 (genetic circuits — recreate the repressilator, build a synthetic-Notch / synthetic-perceptron toy).
- **Parallel/co-requisite dry side (this project):** students work through **Setup + Labs 1–8** of the notebook track during HTGAA Weeks 1–7. The pacing fits: one lab per week, ~2 hours each, self-contained. By the end of Week 7 a student has the repressilator (Lab 1), the regulome (Lab 2), the fidelity triple (Lab 3), the Module Identifiability Index (Lab 4), the Hypergraph Neural ODE (Lab 5), the control toolkit (Lab 6, optionally Lab 7), and the anatomical compiler (Lab 8) in their head and as runnable code.
- **Weeks 8–12 / capstone:** *the wet-lab cycle.* Pick **one** of the cycles below and execute end-to-end: design (using the Lab-8 compiler), simulate (the Lab-5 plant + the Lab-6 controller), build (Biopunk bench), read out (the Lab-3 fidelity readout or the Lab-4 MII or the Lab-5 driver split), refine (model + plan a Cycle-2). Final deliverable: a written report + a notebook with the wet-lab data committed alongside the existing committed benchmarks.

### The natural HTGAA capstone — **Cycle 2 (synNotch sender–receiver)**

Of the six experiments in this doc, **Cycle 2 (experiment E above)** is the cleanest HTGAA fit:

- **It builds on HTGAA Weeks 6–7 directly.** Students cloned the repressilator in Week 6; the synNotch circuit is the natural next-level circuit (sender ligand → receiver receptor → TF output → reporter). Same molecular biology toolkit, one rung up the complexity ladder.
- **It uses the project's existing benchmark.** The Toda 2020 synNotch dataset is already committed (`figures/toda_results.json`, `scripts/benchmark_toda_morphogenesis.py`, [Lab 9 §2](../notebooks/09_synthetic_morphology_wetlab.ipynb)) — students' own wet-lab Toda-style data slots into the same readout pipeline.
- **It's tractable at community-lab scale in a single term.** Morsut/Toda plasmids from Addgene, HEK293T or similar host line, qPCR or bulk-RNA-seq readout. No bioprinter, no organoid culture, no CRISPRi pipeline required.
- **It tests a load-bearing project claim.** Lab 9 frames every wet-lab modality as "one optimal-control problem on the Hypergraph Neural ODE" with a different actuator B; the synNotch arm is the cleanest test of that framing.
- **The deliverable composes.** Each student's circuit + readout becomes one row of an extended `toda_results.json` — the project accumulates a *community-grown* benchmark across HTGAA cohorts.

A reasonable student final-project deliverable: **a notebook (`notebooks/student_synnotch_<name>.ipynb`) that loads their wet-lab readout, projects it into the Neocortex Atlas pattern space (`projectR`-in-JAX), runs the Lab-3 fidelity triple against the project's predicted regulatory program, and reports the Lab-4 MII of the resulting transcriptional state** — all using infrastructure already in the repo.

### Alternative capstones (if equipment matches better)

- **Cycle 4 / experiment D — PRINTESS bioprinting sweep**, if the lab has a [PRINTESS](https://printess.org) (Skylar-Scott's $250 open-hardware bioprinter; already a referenced ecosystem node in this project). Tractable, no transfection, immunostaining-tier readout, ties to the Gartner-Lab benchmark already committed (`benchmark_gartner_4d.py`).
- **Cycle 4 / experiment F — Bioelectric ionophore + bulk RNA-seq**, if the BETSE-JAX side ([[betse-jax-sister-project]]-equivalent — `~/Workspace/betse-unified`) is the student's interest. Couples the dry-side BETSE-JAX inverse-design pipeline to a wet-lab ionophore perturbation; closes the bioelectric → GRN loop directly.
- **Cycle 1 / experiment B — Organoid injury bulk-RNA-seq**, if the lab has organoid-culture infrastructure. The richest single-experiment result, but heaviest equipment and budget; better as a *cohort* project (multiple students sharing one timecourse) than as an individual capstone.

### What this contributes back to the project

A wet-lab cycle from any HTGAA cohort feeds the dry side concretely:

- **A new committed result JSON** (`figures/biopunk_htgaa_<cohort>_<experiment>_results.json`) — accumulates over cohorts.
- **A new committed h5ad / qPCR table / image set** under `data/biopunk/`, with the cohort's preprocessing pipeline alongside.
- **A new row in the relevant lab's comparison panel** — the project's labs already do "fidelity triple across systems", "MII across systems", "driver stability across systems"; adding a Biopunk row is a one-line change.
- **A wet-lab-data update to the paper's §4.3** — what's currently *proposed* in (i)–(vi) becomes *committed* with each cohort.

The HTGAA-capstone framing also resolves the equipment / budget open question above structurally: **the lab need not have everything; each cohort picks the cycle that matches what it has**, and the dry-side toolchain is rich enough to absorb whatever readout the bench produces.

---

## Dry-side implications (immediate)

The wet-lab availability re-ranks `docs/computational-roadmap.md`:

- **Active learning / experimental design / RL (§2 there)** is no longer "useful enrichment"; it's the urgent next dry-side work. The math of *next experiment to do given current model uncertainty* is the load-bearing infrastructure for everything in this doc. Highest dry-side priority.
- **Foundation models + causal inference (§1 there)** stays high — they make each experiment's prior tighter, and the CRISPRi side (experiment C) is natively interventional, perfect for NOTEARS-style causal discovery.
- **Generative perturbation models (§7 there)** — esp. CellOT / flow-matching for predicting post-perturbation cell-state distributions — become relevant once experiments C and F produce data.
- **Differentiable Cellular Potts (§3 there)** is deferred — useful eventually, but model-side enrichment without immediate validation pathway until tissue-shape readouts come online (experiment D's image-analysis panel is the first hook).

## How to apply

A future agent picking this up should:
- Treat this as a *menu*, not a commitment. The user picks the cycle; the dry side scaffolds the design (`scripts/benchmark_*.py` and the labs already supply the readouts).
- Each picked experiment generates: (i) a new committed h5ad / qPCR table / image set in `data/biopunk/`, (ii) a new committed result JSON in `figures/biopunk_<experiment>_results.json`, (iii) a new comparison panel in the relevant lab notebook + the paper. The repo's existing benchmark/lab/paper structure absorbs new wet-lab data cleanly — no infrastructure changes needed.
- The active-learning code is the missing dry-side piece; build it after Cycle 1 generates the first calibration datum.
- Update [the wetlab-biopunk memory](../../../.claude/projects/-Users-mhough-Workspace-anatomical-compiler/memory/wetlab-biopunk.md) as the scope clarifies.
