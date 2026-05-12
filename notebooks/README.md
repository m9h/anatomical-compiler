# Notebooks — toward a course on computational synthetic morphology

This folder is the seed of an **educational track**: a sequence of notebooks meant to walk a
student (or a class) through the whole arc of the project — from "what is a regulome and why a
hypergraph" to "design an intervention that steers a tissue toward a target state." The idea is
that each notebook is a self-contained lab session, building on the last, ending in a small
exercise; together they cover the methods in `publication/paper.Rnw` and the experimental
programme in its §4.

## What's here now

- **`01_regulomes_and_hypergraphs.ipynb`** — **Lab 1**: what a regulome is; load the Fleck
  incidence matrix; why a regulon is a *hyperedge*, not a clique (the clique-expansion blow-up
  *and* aliasing, worked on a toy example + the full regulome); `hgx` basics (`from_incidence`,
  node/edge degrees, star vs clique expansion, a `UniGCNConv` forward pass); a first structural
  readout (the hypergraph-Laplacian spectrum and its gap — the seed of the Module Identifiability
  Index); exercises (all-pairs Jaccard regulon overlap among the master TFs, heavy-tailedness of
  regulon sizes, graph degree vs hypergraph degree). Self-contained — reads `data/processed/`,
  falls back to a tiny synthetic regulome if absent.
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

## Planned sequence (a 6–8 session course)

1. **Regulomes and hypergraphs.** Gene regulatory networks; why a regulon is a *hyperedge*, not
   a clique. Build the Fleck incidence matrix; basic hypergraph operations in `hgx`. *(Refs: Davidson;
   Fleck et al. 2023; the §1.4 / §2.2 material.)*
2. **Benchmarking fidelity.** Does an organoid regulome predict CRISPRi outcomes in primary cortex?
   Regulon overlap, direction concordance, cross-species conservation. *(Builds on `organoid_hgx_colab.ipynb`;
   Pollen 2026; §3.1–3.3.)*
3. **Modularity and identifiability.** The Hodge Laplacian; the Module Identifiability Index;
   "neurogenic stop-signals." Run it on organoid vs primary vs bioprinted systems. *(Hartwell 1999;
   NITMB framing; §2.3 / §3.x.)*
4. **Dynamics: Hypergraph Neural ODEs.** Fit a latent ODE on a timecourse; separate stable
   structural drivers from transient stress responders; the attractor view of cell identity.
   *(Kauffman; Huang et al. 2009; §2.4 / §3 regenerative-flow.)*
5. **Control theory on cellular dynamics (`jaxctrl`).** Identify a surrogate (SINDy/Koopman) →
   controllability → LQR → driver nodes on a hypergraph. Use the three jaxctrl example notebooks.
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
