# DGX Spark verifier runbook

*Instructions for the verifier agent on DGX Spark: what to run, in what order, what success looks like, and what to commit back. Written 2026-05-14, addressed to the agent (second-person imperative). Live to commit `150bb42`+.*

The local laptop side (the agent that produced this doc) has exhausted what's runnable without (a) the heavier system installs (VCell, OpenCOR), (b) GPU + real foundation-model weights, and (c) real biological datasets. Everything on the laptop side passes at machine precision; everything below is the **next leg** that needs DGX Spark to exercise.

This runbook is **tiered**. Run Tier 0 first; then proceed through Tier 1 → 2 → 3 → 4 as the prerequisites are ready. Each tier has its own success criterion + commit-back artifact. **Don't skip tiers** — Tier 3 depends on real h5ad/edges/promoters/TFs from prior work; Tier 4 depends on the Tier 3 cache.

---

## Tier 0 — Pre-flight

**5 minutes; no GPU needed.** Establishes that the local environment matches the canonical state.

```bash
cd ~/dev/anatomical-compiler        # or wherever the repo is on DGX
git fetch origin
git log --oneline origin/master | head -1
# Expect: 150bb42 (cure: items 4 + 5 ...) or newer

git pull origin master              # in case anything new landed
uv sync                             # rebuild venv if needed
uv run python3 -c "import jax, scanpy, anndata, tellurium, basico, libsbml, libsedml, libcombine; \
  print('jax', jax.__version__); \
  print('scanpy', scanpy.__version__); \
  print('tellurium', tellurium.__version__); \
  print('basico', basico.__version__)"
```

**Success:** every import succeeds and prints a version. If any fails, `uv pip install <missing>` and re-run.

---

## Tier 1 — Baseline self-test

**5 minutes; no GPU; no extra installs.** Reproduces the local laptop's verification results to confirm the DGX environment behaves identically.

```bash
# 1. Regulome-provenance manifest idempotence
uv run python3 scripts/emit_regulome_provenance.py --check

# 2. Three-simulator SBML cross-verification (JAX + Tellurium + COPASI)
uv run python3 scripts/verify_lab1_sbml.py

# 3. The two synthetic ablations
uv run python3 scripts/ablate_edge_priors.py
uv run python3 scripts/ablate_perturb_eig.py
```

**Success criteria** (compare against the committed `figures/`):

| script | expected | from |
|---|---|---|
| `emit_regulome_provenance.py --check` | exits 0, prints `OK` | committed `models/regulome_provenance.json` |
| `verify_lab1_sbml.py` | overall=PASS; tellurium=PASS / copasi=PASS for all 4 circuits at worst rel-L2 ≤ 6e-06; vcell=skipped, opencor=skipped | `figures/lab1_sbml_verification.md` |
| `ablate_edge_priors.py` | best_alpha=0.30, lift_over_pando ≈ +0.109 (seed 0); ±0.03 across seeds | `figures/edge_prior_ablation.md` |
| `ablate_perturb_eig.py` | baseline prior ρ ≈ 0.564; EIG-rank wins over GREEDY by ≈ +0.029 at median budget | `figures/perturb_eig_ablation.md` |

**If everything PASS:** nothing to commit; the DGX environment is sound. **If anything fails:** flag the divergence in a commit message + don't proceed to Tier 2 until resolved.

---

## Tier 2 — Bring the SBML cross-simulator panel from 3 to 5 (VCell + openCOR)

**~30 minutes one-time install; then re-run.** The Lab-1 verification's `vcell` and `opencor` backends are stubs that activate on dependency presence — Tier 2 installs them.

### 2a. VCell via vcell-cli

```bash
# Option A — Docker image (preferred if DGX has Docker)
docker pull ghcr.io/virtualcell/vcell-cli
cat > /usr/local/bin/vcell-cli <<'EOF'
#!/usr/bin/env bash
exec docker run --rm -v "$PWD":/workspace ghcr.io/virtualcell/vcell-cli "$@"
EOF
chmod +x /usr/local/bin/vcell-cli

# Option B — release binary
# Download the latest from https://github.com/virtualcell/vcell-cli/releases
# Extract, then `ln -s /path/to/vcell-cli /usr/local/bin/vcell-cli`

# Verify
vcell-cli --version
```

### 2b. OpenCOR Python module

```bash
# Download from https://opencor.ws (Linux .tar.gz)
cd /opt
sudo curl -O https://opencor.ws/downloads/snapshots/OpenCOR-2024-XX-XX-Linux.tar.gz
sudo tar xzf OpenCOR-2024-XX-XX-Linux.tar.gz
echo 'export PYTHONPATH="/opt/OpenCOR-2024-XX-XX-Linux/python:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
uv run python3 -c "import OpenCOR; print('OpenCOR OK')"
```

### 2c. Re-run the verifier

```bash
uv run python3 scripts/verify_lab1_sbml.py
```

**Success criterion:** all four circuits now show `vcell=PASS(...)` and `opencor=PASS(...)` in addition to the existing `tellurium=PASS` / `copasi=PASS`. **Five-simulator agreement at machine precision** on all 4 Lab-1 circuits is the deliverable.

### 2d. Commit back

If 5-simulator PASS:

```bash
git add figures/lab1_sbml_verification.json figures/lab1_sbml_verification.md
git commit -m "cure-6: five-simulator agreement — JAX + Tellurium + COPASI + VCell + OpenCOR

All four Lab-1 circuits now PASS at machine precision against five
independent integrator families (Dopri5 explicit RK / CVODE BDF / LSODA
stiff-switching / VCell adaptive / OpenCOR adaptive). The CURE-audit
item 6 simulator panel is now fully populated; no further code changes
needed since the VCell and OpenCOR backends in verify_lab1_sbml.py
activate automatically when their binaries / Python modules are
present.

Run on DGX Spark per docs/dgx-verifier-runbook.md Tier 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

git push origin master
```

If any backend FAILs (rel-L2 > 0.05): **don't commit a regression**; open an issue or pause for human review. The integrator-tolerance mismatch is the likely culprit; consider tightening per-backend tolerances in `verify_lab1_sbml.py`.

---

## Tier 3 — Real-mode FM pipeline (the heavy lift)

**~1 hour wall-clock; needs GPU, weights pre-fetched, real biological inputs.** This is the **step 5 of [`docs/foundation-models.md`](foundation-models.md)** — running the FM extractors in `--mode real` against real data.

### 3a. Required inputs

| input | what | how to produce |
|---|---|---|
| `<h5ad>` | single-cell expression matrix | Pollen brain-organoid h5ad (canonical) or Biopunk capstone output |
| `<edges.csv>` | regulome edges with `tf,target` columns | Pando output table — `scripts/02_pando_import.py` produces it |
| `<promoters.fa>` | FASTA of promoter sequences keyed by gene symbol | `scripts/build_promoter_fasta.py` (already on DGX as of commit 3bcc979) on GENCODE v45 + hg38 |
| `<tfs.txt>` | one TF symbol per line | `awk -F, 'NR>1 {print $1}' <edges.csv> \| sort -u > tfs.txt` |

### 3b. Heavy-dep install (one-time)

Per [`docs/dgx-spark-setup.md`](dgx-spark-setup.md) §2 — install via the `Dockerfile.fm` target, or directly:

```bash
uv pip install \
    'torch >= 2.4' 'transformers >= 4.42' 'huggingface-hub >= 0.24' \
    biopython geneformer 'evo-model >= 1.1' borzoi-pytorch
uv pip install 'git+https://github.com/bowang-lab/scGPT'
uv pip install 'git+https://github.com/snap-stanford/UCE'

# JASPAR motif database
mkdir -p /opt/jaspar
curl -L 'https://jaspar.genereg.net/download/data/2024/CORE/JASPAR2024_CORE_non-redundant_pfms_meme.txt' \
     -o /opt/jaspar/JASPAR2024_CORE_vertebrates_non-redundant.meme
export JASPAR_MEME=/opt/jaspar/JASPAR2024_CORE_vertebrates_non-redundant.meme

# Pre-fetch the HuggingFace weights (~50 GB)
huggingface-cli download ctheodoris/Geneformer
huggingface-cli download subercui/scGPT
huggingface-cli download chanzuckerberg/uce
huggingface-cli download togethercomputer/evo-1-131k-base
huggingface-cli download calico/borzoi-v1
```

### 3c. One-command run

```bash
./scripts/run_fm_real_dgx.sh \
    data/pollen.h5ad \
    data/fleck_edges.csv \
    data/promoters_hg38_2kb.fa \
    data/tfs.txt \
    cache/dgx_real_pollen_$(date +%Y%m%d)
```

**Expected per-step wall-clock** (per `docs/dgx-spark-setup.md` §4):
- Geneformer extract: ~2 min / 10 k cells
- scGPT extract: ~1 min
- UCE extract: ~5 min
- motif scan: ~2 min / 100 k edges
- Evo scoring: ~15 min / 100 k edges
- Borzoi scoring: ~30 min / 100 k edges
- scGPT in-silico KD: ~5 min / 1000 TFs
- Two ablations: ~30 s total
- **Total: ~1 hour for a Pollen-scale run.**

### 3d. Success criteria

Each tool writes both a `.npy` and a `manifest.json`. After the run:

```bash
ls cache/dgx_real_pollen_*/
# Expect, at minimum:
#   pollen_uce.npy                    (n_cells, 1280)
#   pollen_geneformer.npy             (n_genes, 512)
#   pollen_scgpt.npy                  (n_cells, 512)
#   fleck_edges_motif.npy             (n_edges,)
#   fleck_edges_evo.npy               (n_edges,)
#   fleck_edges_borzoi.npy            (n_edges,)
#   pollen_scgpt_perturb.npy          (n_tfs, n_genes)
#   *_manifest.json for each
#   ablation_edges.{json,md}          — real-mode edge-prior ablation
#   ablation_perturb_eig.{json,md}    — real-mode BO/EIG ablation
#   logs/*.log
```

Each manifest's `mode` field must be `real` (not `stub` or `auto-fell-back`). **If any manifest shows `stub`, that tool's real backend failed silently** — check `logs/<tool>.log` and don't proceed to Tier 4.

### 3e. Commit back

The full `cache/dgx_real_pollen_*/` directory is too large to commit. Instead:

```bash
# Summarise the cache directory into a single small artifact
uv run python3 - <<'PY'
import json, hashlib, pathlib
cache = sorted(pathlib.Path('cache').glob('dgx_real_pollen_*'))[-1]
summary = {'cache_dir': str(cache), 'artifacts': []}
for f in sorted(cache.rglob('*')):
    if f.is_file():
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        summary['artifacts'].append({
            'path': str(f.relative_to(cache)),
            'bytes': f.stat().st_size,
            'sha256_16': h,
        })
import datetime
out = pathlib.Path(f'figures/dgx_real_run_{datetime.date.today().isoformat()}.json')
out.write_text(json.dumps(summary, indent=2))
print(f'wrote {out}')
PY

git add figures/dgx_real_run_*.json
# Plus the small ablation_*.{json,md} from the cache:
cp cache/dgx_real_pollen_*/ablation_*.{json,md} figures/ 2>/dev/null
git add figures/ablation_*.{json,md}

git commit -m "fm-real: step 5 of docs/foundation-models.md — real-mode runs on <dataset>

Real-mode extraction completed on DGX Spark per
docs/dgx-verifier-runbook.md Tier 3. Full cache directory at
$(ls -d cache/dgx_real_pollen_* | tail -1) (not committed; too large);
SHA-256-fingerprinted summary committed at figures/dgx_real_run_*.json.

Manifests confirm all six extractors ran in mode='real' (not stub).
Real-mode ablation results committed alongside the stub-mode versions
for direct comparison.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

git push origin master
```

---

## Tier 4 — The four headline measurements (the CURE-Validation deliverable)

**~30 minutes once Tier 3 cache exists.** This is what answers the long-pending "do FMs make a difference on the project's *actual* numbers?" The four measurements documented in [`docs/dgx-spark-setup.md`](dgx-spark-setup.md) §7:

| target | what changes | success metric |
|---|---|---|
| [Lab 3](../notebooks/03_benchmarking_fidelity.ipynb) fidelity-triple transfer-r ≈ 0.13 | use Geneformer + scGPT priors on the perturbation predictor | transfer-r on Pollen-test set |
| [Lab 4](../notebooks/04_modularity_identifiability.ipynb) MII gap | rebuild the regulome graph with sequence-edge priors blended in | MII separation between organoid / blueprint / bioprinted |
| [Lab 6](../notebooks/06_control_theory.ipynb) high-leverage TFs | combine scGPT in-silico KD prior with the controllability ranking | agreement on top-10 TFs across methods |
| edge-prior ablation (`scripts/ablate_edge_priors.py`) in real-mode | swap stub for real motif/Evo/Borzoi | F1 lift over Pando alone on the real Fleck edges |

**These don't currently have wrapper scripts** — you'll need to write one per measurement, consuming the Tier 3 cache as numpy arrays per the integration contract in [`docs/foundation-models.md`](foundation-models.md).

Suggested filenames: `scripts/measure_lab3_real.py`, `scripts/measure_lab4_real.py`, `scripts/measure_lab6_real.py`. Each writes `figures/<measure>_real_results.{json,md}` for committable evidence.

If you need a wrapper-script template, base on [`scripts/ablate_perturb_eig.py`](../scripts/ablate_perturb_eig.py) — same shape (CLI + JSON+MD output + multi-seed report).

### Commit back

```bash
git add figures/lab3_real_results.{json,md} \
        figures/lab4_real_results.{json,md} \
        figures/lab6_real_results.{json,md} \
        scripts/measure_lab{3,4,6}_real.py
git commit -m "fm-real: Tier 4 — Lab 3 / 4 / 6 measurements with FM priors

[fill in with the actual transfer-r / MII gap / TF-agreement numbers]"
git push origin master
```

---

## Tier 5 — Post-CURE-6 supplementary tests *(added 2026-05-14 by DGX-side agent)*

These are bite-sized tests the **laptop-side runbook didn't include** but that the audit doc (`docs/cure-audit.md`) implicitly invites. Each is independent — run any subset in any order. They strengthen the credibility claim without needing Tier 3's heavy infra.

### 5a — libsbml/libsedml/libcombine semantic validation

**~2 hours; no GPU; CPU-only.** `verify_lab1_sbml.py` checks *numerical* trajectory agreement but does **not** check whether the emitted SBML/SED-ML is *semantically valid* (units consistent, no dangling id refs, parameters all used). libsbml ships a validator; libsedml does too. Catches issues that don't show up in trajectory comparisons.

```python
# Suggested: scripts/validate_sbml_semantic.py
import libsbml, libsedml, libcombine
from pathlib import Path

for sbml_path in Path("models").glob("lab1_*.sbml"):
    doc = libsbml.SBMLReader().readSBML(str(sbml_path))
    doc.setConsistencyChecks(libsbml.LIBSBML_CAT_MODELING_PRACTICE, True)
    n_err = doc.checkConsistency()
    # report n_err and the .getErrorLog() at severity >= ERROR
```

Same for `.sedml` (libsedml.SedReader) and `.omex` (libcombine.CombineArchive). Output a single `figures/sbml_semantic_validation.{json,md}` with per-file pass/fail + error counts. **Success:** all 4 SBML files pass at LIBSBML_SEV_ERROR (warnings tolerable).

### 5b — Cross-tolerance numerical sensitivity for Lab 1

**~2 hours; no GPU.** The current verifier asserts "JAX vs Tellurium vs COPASI agree at machine precision" but only at the **default tolerances**. Tightening proves the agreement isn't an artifact of loose tolerance.

```bash
# Suggested: scripts/lab1_tolerance_sweep.py
# For each circuit and each rtol in {1e-6, 1e-8, 1e-10, 1e-12}:
#   re-run JAX/Dopri5 and Tellurium/CVODE
#   record worst rel-L2
# Assert rel-L2 decreases monotonically as rtol decreases.
```

Writes `figures/lab1_tolerance_sweep.{json,md}`. **Success:** rel-L2 is monotone non-increasing in tightening rtol for all 4 circuits across both solvers.

### 5c — BioSimulations REST round-trip

**~30 minutes; needs internet.** External third-party verification — the strongest credibility claim available. POST each `models/lab1_*.omex` to https://api.biosimulations.org/runs, poll until done, pull the trajectory CSV, compare to JAX/diffrax reference.

```bash
# Suggested: scripts/biosimulations_roundtrip.py
# For each omex:
#   POST https://api.biosimulations.org/runs with simulator=tellurium (or copasi)
#   poll /runs/<id> until status=SUCCEEDED
#   GET /results/<id>/outputs
#   compare to JAX/diffrax baseline
```

Writes `figures/biosimulations_roundtrip.{json,md}` with the BioSimulations run IDs (citable URLs). **Success:** all 4 circuits agree with the BioSimulations-hosted Tellurium run within tolerance.

### 5d — Hash-pinned reproducibility test in CI

**~half a day; one-time work.** Today `Dockerfile`'s build-time smoke test runs `verify_lab1_sbml.py` + the two ablations but doesn't pin their *outputs*. Add SHA-256 hashes of the committed `figures/lab1_sbml_verification.json` and `figures/{edge_prior,perturb_eig}_ablation.json` to a `tests/test_reproducibility.py`; have the Docker build re-run and assert hash equality. Catches accidental numerical drift from upstream package updates.

Writes `tests/test_reproducibility.py`; modifies `Dockerfile` to invoke it. **Success:** clean Docker builds reproduce the hashes; intentional changes to outputs require updating both the figures and the test.

### 5e — AIC/BIC + parameter-count reporting on the fidelity-triple

**~2 hours; CPU-only.** `docs/cure-audit.md` §C-flags "AIC/BIC reporting" as cheap-but-missing. For Lab 5 (Hypergraph Neural ODE):

```python
# n_params = sum(p.size for p in flatten(model_params))
# AIC = -2 * test_log_likelihood + 2 * n_params
# BIC = -2 * test_log_likelihood + log(n_test) * n_params
```

Add to `scripts/compare_pollen.py`'s output JSON. Same one-line addition for any other scored model. **Success:** `figures/compare_pollen.json` now has `aic` and `bic` fields alongside `transfer_r`.

### 5f — Sauro Table-1 numeric self-scorecard

**~half a day; CPU-only.** The current `docs/cure-audit.md` uses ✅/⚠️/❌ glyphs. A numeric scorecard with explicit coverage percentages is the kind of thing reviewers cite. Iterate Sauro 2025 Table 1's items; for each, compute a coverage number (e.g. *Annotation: X/Y species → ChEBI; X/Y reactions → SBO*; *Provenance: X/Y benchmarks have SHA-fingerprinted manifests*; *Verification: X/Y closed-form models cross-verified against ≥ 2 simulators*).

Writes `figures/cure_scorecard.{json,md}`. **Success:** every Table-1 row has a numeric value, not just a glyph.

---

## Tier 6 — Standing CURE gaps the audit explicitly names *(post-Tier 4)*

These are the items `docs/cure-audit.md` flags as ⚠️ partial that aren't closed by any of Tier 1–5. Bigger lifts, publication-grade returns.

### 6a — Bayesian posterior on Lab 5 (Hypergraph Neural ODE)

**~1–3 days; needs GPU.** The audit's flagged C-Uncertainty gap: structural identifiability (Lab 7) and per-input sensitivity (Lab 6) are covered, but there's no global posterior over the 50k-parameter Neural ODE weights. Laplace approximation at MAP is the cheapest path: compute Hessian (or its diagonal) at the trained weights, sample, propagate to predicted trajectories, report posterior std on transfer-r. Heavier alternative: NUTS via numpyro or HMC via blackjax on a low-rank reparameterisation.

Writes `figures/lab5_posterior.{json,md}` + an updated section in `notebooks/05_hypergraph_neural_odes.ipynb`. **Success:** posterior std on transfer-r reported; the headline 0.13 transfer-r number now has an honest uncertainty interval attached.

### 6b — Annotation completeness audit

**~half a day; CPU-only.** Cure-audit §C-flags "MIRIAM / SBO / KiSAO terms" as ⚠️ partial — items 4 & 5 landed MIRIAM/SBO for Lab 1's 4 closed-form circuits but the regulome substrate is annotated only at the manifest level. A coverage report: % of `hgx.Regulome` entities with Ensembl / NCBI / UniProt / GO annotations; gap analysis on what's missing.

Writes `figures/regulome_annotation_coverage.{json,md}` and extends `models/regulome_provenance.json`. **Success:** coverage percentages are reported, not just "we have Ensembl".

---

## Decision flowchart

```
Pre-flight Tier 0 fails  → debug environment, do not proceed
Tier 1 fails             → flag divergence, do not proceed
Tier 1 PASS              → proceed to Tier 2 (heavier installs) OR skip to Tier 3 (more impactful)
Tier 2 PASS              → 5-simulator panel done; commit; CURE item 6 fully ✅
Tier 3 PASS              → real-mode cache ready; commit summary; proceed to Tier 4
Tier 4 PASS              → headline numbers measured; **this is the real-data CURE-Validation answer**
Tier 5 items             → independent of each other; cherry-pick by leverage/cost
Tier 6 items             → bigger lifts that close the audit's explicitly-named ⚠️ gaps
```

**Order priority if time-limited:** Tier 1 → Tier 3 → Tier 4 → Tier 2 → Tier 5a/5d (cheap audit reinforcement) → Tier 5b/5c/5e (publication-quality numerics) → Tier 6 (paper-worthy gap closure). Tier 3+4 produce the actual scientific result the project's been asking for; Tier 2 is a CURE-completeness move that's lower-impact than the real-data measurements; Tier 5+6 are post-result reinforcement.

---

## What to flag back to the laptop side

Push commits as you go. If any of these conditions hold, **stop and ask** rather than proceeding:

- A backend disagrees materially (rel-L2 > 0.01) with JAX on any Lab-1 circuit. This means a model interpretation differs — investigate before any committal narrative.
- An FM extractor falls back to stub mode silently. Stub-vs-real attribution is the whole point of step 5; a silent fallback corrupts the measurement.
- Lab 3 real-mode transfer-r is *worse* than the 0.13 baseline. Counter to expectations; understand before publishing.
- The real-mode edge-prior ablation gives a *negative* F1 lift. The stub-mode synthetic showed +0.146 ± 0.032; a negative lift on real data would be a publishable null result but worth a sanity check.

These are the "go slow" signals. Everything else: commit, push, move to the next tier.

---

## Cross-references

- [`docs/dgx-spark-setup.md`](dgx-spark-setup.md) — the install / weights / inputs recipe
- [`docs/foundation-models.md`](foundation-models.md) — the integration plan (steps 1–5)
- [`docs/cure-audit.md`](cure-audit.md) — what the verification chain is *for*
- [`scripts/run_fm_real_dgx.sh`](../scripts/run_fm_real_dgx.sh) — the Tier 3 one-command driver
- [`scripts/verify_lab1_sbml.py`](../scripts/verify_lab1_sbml.py) — the Tier 1 + 2 verifier
- [`MODEL_CARD.md`](../MODEL_CARD.md) — the artifact index, for context on what each measurement validates
