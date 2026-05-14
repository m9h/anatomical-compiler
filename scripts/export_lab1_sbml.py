"""export_lab1_sbml — emit Antimony / SBML for the closed-form circuits in Lab 1.

Step 2 of docs/cure-audit.md. Lab 1's four gene circuits (negative auto-
regulation, toggle switch, repressilator, positive autoregulation) are
canonical Hill-rate ODE systems — exactly the kind of model SBML / CellML
are designed to represent. Emitting them as Antimony (Smith et al. 2009,
the COMBINE community's human-readable round-trip format for SBML) closes
the CURE-Understandable + CURE-Reproducible gap on the closed-form half
of the project: the four circuits are now portable to BioModels, Tellurium,
libRoadRunner, COPASI, VCell, openCOR — any COMBINE-compliant toolchain.

This script is pure text generation; no Tellurium / libsbml dependency.
Round-trip *verification* (Antimony → SBML → libRoadRunner sim vs JAX/
diffrax sim, compare trajectories within tolerance) is the companion
script ``scripts/verify_lab1_sbml.py``, which does require Tellurium.

The four models match Lab 1 (`notebooks/01_gene_circuit_dynamics.ipynb`)
1:1 in parameter values, so a round-trip verification establishes
*cross-simulator agreement* on the project's foundational ODE circuits.

Usage
-----
    python scripts/export_lab1_sbml.py [--output-dir models/]

Output: one `lab1_<name>.ant` Antimony file per circuit, plus an index
`models/README.md` documenting the set.

See also
--------
- docs/cure-audit.md §1 (priority item 2)
- notebooks/01_gene_circuit_dynamics.ipynb — the source-of-truth ODEs
- scripts/verify_lab1_sbml.py — the round-trip verifier (Tellurium needed)
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# ----------------------------------------------------------------------------
# The four circuits — kept here as data, with exact Lab 1 parameter values.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Circuit:
    name: str
    title: str            # human-readable
    citation: str         # canonical reference
    pubmed_id: str | None  # for MIRIAM bqmodel:isDescribedBy (audit item 4)
    initial: dict[str, float]
    parameters: dict[str, float]
    rate_rules: dict[str, str]  # species → rate expression in Antimony syntax
    lab1_cell: int        # for cross-reference
    notes: str            # short prose


CIRCUITS = [
    Circuit(
        name="negative_autoregulation",
        title="Negative autoregulation (NAR)",
        citation="Becskei & Serrano 2000, Nature 405:590 — also Lab 1 §2 (Elowitz & Bois).",
        pubmed_id="10864324",  # Becskei & Serrano 2000
        initial={"x": 0.01},
        parameters={"beta": 5.0, "K": 1.0, "n": 2.0, "gam": 1.0},
        rate_rules={
            "x": "beta / (1 + (x/K)^n) - gam * x",
        },
        lab1_cell=5,
        notes=(
            "A single gene representing its own repressor. The NAR circuit reaches "
            "its steady state ~2x faster than an unregulated gene with the same x* "
            "(Lab 1 §2 shows ~2.5x); the fixed point is stable (f'(x*) < 0 by "
            "autodiff in Lab 1)."
        ),
    ),
    Circuit(
        name="toggle_switch",
        title="Toggle switch (Gardner-Cantor-Collins)",
        citation="Gardner, Cantor & Collins 2000, Nature 403:339 — also Lab 1 §3.",
        pubmed_id="10659857",  # Gardner, Cantor & Collins 2000
        initial={"u": 0.2, "v": 1.5},
        parameters={"alpha": 4.0, "n": 3.0},
        rate_rules={
            "u": "alpha / (1 + v^n) - u",
            "v": "alpha / (1 + u^n) - v",
        },
        lab1_cell=7,
        notes=(
            "Two mutually-repressing genes. For α=4, n=3 the system is bistable: "
            "one symmetric saddle at u=v≈1.36 (eigenvalues ±, unstable) plus two "
            "stable attractors near (3.8, 0.1) and (0.1, 3.8). Lab 1 verifies the "
            "phase-portrait via autodiff Jacobians."
        ),
    ),
    Circuit(
        name="repressilator",
        title="Repressilator",
        citation="Elowitz & Leibler 2000, Nature 403:335 — also Lab 1 §4.",
        pubmed_id="10659856",  # Elowitz & Leibler 2000
        initial={"p0": 2.5, "p1": 1.5, "p2": 3.0},
        parameters={"alpha": 10.0, "alpha0": 0.1, "n": 3.0, "gam": 1.0},
        rate_rules={
            # Cyclic repression: gene i is repressed by protein at position
            # (i + 2) mod 3 in Lab 1's PERM = [2, 0, 1].
            "p0": "alpha / (1 + p2^n) + alpha0 - gam * p0",
            "p1": "alpha / (1 + p0^n) + alpha0 - gam * p1",
            "p2": "alpha / (1 + p1^n) + alpha0 - gam * p2",
        },
        lab1_cell=9,
        notes=(
            "Three genes in a cyclic repression loop. For α=10, α₀=0.1, n=3, γ=1 "
            "the symmetric fixed point is unstable (a Hopf pair), producing "
            "sustained oscillations — the substrate for Lab 6's linearise → "
            "controllability → LQR worked example."
        ),
    ),
    Circuit(
        name="positive_autoregulation",
        title="Positive autoregulation (bistability)",
        citation="Becskei, Séraphin & Serrano 2001, EMBO J 20(10):2528-35 — Lab 1 §5 exercise (a).",
        pubmed_id="11350942",  # Becskei et al. 2001
        initial={"x": 0.5},
        # The β parameter is swept in the lab to find a saddle-node bifurcation;
        # we export with β = 4.0, which sits in the bistable regime.
        parameters={"beta0": 0.2, "K": 1.0, "n": 4.0, "gam": 1.0, "beta": 4.0},
        rate_rules={
            "x": "beta0 + beta * (x/K)^n / (1 + (x/K)^n) - gam * x",
        },
        lab1_cell=11,
        notes=(
            "A single gene activating its own production. For β=4 this system has "
            "three fixed points: a low-x stable, a middle unstable, and a high-x "
            "stable — the saddle-node bifurcation that defines bistability. "
            "Sweeping β reveals the hysteresis loop (Lab 1 exercise (a))."
        ),
    ),
]


# ----------------------------------------------------------------------------
# Antimony emission
# ----------------------------------------------------------------------------


def emit_antimony(circuit: Circuit) -> str:
    """Emit an Antimony model string for one circuit.

    The output is reaction-based with split production / degradation
    where the rate has both terms (so it round-trips cleanly through SBML
    L3v2). Single-term rates (toggle's already-net `α/(1+v^n) - u` form,
    written without a separate degradation reaction) are emitted as one
    reaction with a signed rate — Antimony / libRoadRunner accept this,
    matching the Lab 1 RHS exactly.
    """
    lines: list[str] = []

    # Header comment block (CURE-Understandable §3: explicit citation /
    # purpose / scope at the top of every model file).
    header = textwrap.dedent(
        f"""\
        // {circuit.title}
        //
        // Canonical reference: {circuit.citation}
        // Source: anatomical-compiler/notebooks/01_gene_circuit_dynamics.ipynb cell {circuit.lab1_cell}.
        // Auto-generated by scripts/export_lab1_sbml.py — do not edit by hand;
        // re-run the exporter after modifying the circuit definition.
        //
        // {circuit.notes}
        //
        // CURE alignment: Sauro et al. 2025 (arXiv:2502.15597); the round-trip
        // verifier is scripts/verify_lab1_sbml.py.
        """
    )
    lines.append(header.rstrip())
    lines.append("")
    lines.append(f"model *lab1_{circuit.name}()")
    lines.append("")

    # Species declarations + initial concentrations.
    lines.append("    // species")
    for species, init in circuit.initial.items():
        lines.append(f"    species {species} = {init}")
    lines.append("")

    # Parameters.
    lines.append("    // parameters")
    for k, v in circuit.parameters.items():
        lines.append(f"    {k} = {v}")
    lines.append("")

    # Rate equations as single signed-rate reactions.
    # Note: Antimony permits negative-valued rate expressions in `-> sp; rate`
    # reactions, which libRoadRunner handles correctly (it integrates dsp/dt = rate).
    lines.append("    // rate equations")
    for i, (species, rate) in enumerate(circuit.rate_rules.items()):
        lines.append(f"    J{i}: -> {species}; {rate};")
    lines.append("")

    lines.append("end")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# SBO + MIRIAM annotation (CURE-audit item 4)
# ----------------------------------------------------------------------------

# Canonical SBO terms used here.
SBO_POLYPEPTIDE_CHAIN = 252        # Per-species: each Lab-1 species is a protein concentration.
SBO_BIOCHEMICAL_REACTION = 176     # Per-reaction: the combined production-degradation rate
                                   # is a biochemical reaction (generic but accurate for the
                                   # Hill-rate ODE formulation Lab 1 uses).


def annotate_sbml(sbml_string: str, circuit: "Circuit") -> str:
    """Add MIRIAM + SBO annotations to an SBML string via libsbml.

    Model-level: ``bqmodel:isDescribedBy → https://identifiers.org/pubmed/<id>``
    Species-level: ``SBO:0000252`` (polypeptide chain)
    Reaction-level: ``SBO:0000176`` (biochemical reaction)

    The MIRIAM annotation lets BioSimulations, COPASI, VCell, BiVeS and the
    rest of the COMBINE ecosystem resolve the citation automatically; the
    SBO terms give downstream tools semantically-typed access to species
    and reactions.
    """
    import libsbml as ls

    reader = ls.SBMLReader()
    doc = reader.readSBMLFromString(sbml_string)
    if doc.getNumErrors() > 0:
        # Non-fatal warnings are fine; only fail on hard errors.
        for i in range(doc.getNumErrors()):
            err = doc.getError(i)
            if err.getSeverity() >= ls.LIBSBML_SEV_ERROR:
                raise RuntimeError(f"SBML parse error: {err.getMessage()}")
    model = doc.getModel()

    # Model-level MIRIAM: link to the PubMed citation if available.
    if circuit.pubmed_id is not None:
        if not model.isSetMetaId():
            model.setMetaId(f"meta_{model.getId() or 'model'}")
        cv = ls.CVTerm(ls.MODEL_QUALIFIER)
        cv.setModelQualifierType(ls.BQM_IS_DESCRIBED_BY)
        cv.addResource(f"https://identifiers.org/pubmed/{circuit.pubmed_id}")
        model.addCVTerm(cv)

    # Per-species: SBO:0000252 (polypeptide chain). These models are abstract
    # protein concentrations, not specific gene products, so we don't add
    # bqbiol:is → UniProt/Ensembl — that would be a false claim.
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        sp.setSBOTerm(SBO_POLYPEPTIDE_CHAIN)

    # Per-reaction: SBO:0000176 (biochemical reaction).
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        rxn.setSBOTerm(SBO_BIOCHEMICAL_REACTION)

    writer = ls.SBMLWriter()
    return writer.writeSBMLToString(doc)


# ----------------------------------------------------------------------------
# SED-ML simulation descriptor (CURE-audit item 5 prerequisite)
# ----------------------------------------------------------------------------


def emit_sedml(circuit: "Circuit", sbml_filename: str, t_final: float, n_points: int) -> str:
    """Emit a minimal SED-ML L1V4 descriptor for the circuit's simulation.

    The SED-ML records: which model file to load, a uniform-time-course task
    over ``[0, t_final]`` with ``n_points`` outputs, a data-generator per
    species, and a Report. This is what VCell / BioSimulations / Tellurium
    consume to reproduce the Lab 1 simulation experiment in their own engine.
    """
    import libsedml

    doc = libsedml.SedDocument(1, 4)
    # Model: reference the SBML file by name (resolved within the OMEX archive).
    sed_model = doc.createModel()
    sed_model.setId(f"model_{circuit.name}")
    sed_model.setName(circuit.title)
    sed_model.setSource(sbml_filename)
    sed_model.setLanguage("urn:sedml:language:sbml.level-3.version-2")

    # Simulation: uniform time course.
    sim = doc.createUniformTimeCourse()
    sim.setId(f"sim_{circuit.name}")
    sim.setInitialTime(0.0)
    sim.setOutputStartTime(0.0)
    sim.setOutputEndTime(t_final)
    sim.setNumberOfPoints(n_points - 1)  # SED-ML counts intervals
    alg = sim.createAlgorithm()
    alg.setKisaoID("KISAO:0000019")  # CVODE — what libRoadRunner uses by default

    # Task: link model + sim.
    task = doc.createTask()
    task.setId(f"task_{circuit.name}")
    task.setModelReference(sed_model.getId())
    task.setSimulationReference(sim.getId())

    # Time data generator.
    dg_time = doc.createDataGenerator()
    dg_time.setId("dg_time")
    var_t = dg_time.createVariable()
    var_t.setId("var_time")
    var_t.setTaskReference(task.getId())
    var_t.setSymbol("urn:sedml:symbol:time")
    dg_time.setMath(libsedml.parseFormula("var_time"))

    # One data generator per species.
    report = doc.createReport()
    report.setId(f"report_{circuit.name}")
    set_time = report.createDataSet()
    set_time.setId("ds_time")
    set_time.setLabel("time")
    set_time.setDataReference(dg_time.getId())

    for sp_name in circuit.rate_rules.keys():
        dg = doc.createDataGenerator()
        dg.setId(f"dg_{sp_name}")
        var = dg.createVariable()
        var.setId(f"var_{sp_name}")
        var.setTaskReference(task.getId())
        var.setTarget(
            f"/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='{sp_name}']"
        )
        dg.setMath(libsedml.parseFormula(f"var_{sp_name}"))
        ds = report.createDataSet()
        ds.setId(f"ds_{sp_name}")
        ds.setLabel(sp_name)
        ds.setDataReference(dg.getId())

    writer = libsedml.SedWriter()
    return writer.writeSedMLToString(doc)


# ----------------------------------------------------------------------------
# OMEX / COMBINE archive bundling (CURE-audit item 5)
# ----------------------------------------------------------------------------


def write_omex_archive(
    omex_path: Path,
    sbml_path: Path,
    sedml_path: Path,
    description: str,
) -> None:
    """Bundle an SBML model + SED-ML descriptor into a COMBINE/OMEX archive.

    The .omex file is a zip with a manifest.xml listing every component
    and its format URI. libcombine handles the manifest; we just register
    the entries and call writeToFile. BioSimulations, VCell, COPASI, and
    Tellurium all consume this format directly.
    """
    import libcombine as lc

    if omex_path.exists():
        omex_path.unlink()
    archive = lc.CombineArchive()
    # Add the SBML model as the primary entry (master=True).
    archive.addFile(
        str(sbml_path),
        sbml_path.name,
        lc.KnownFormats.lookupFormat("sbml"),
        False,
    )
    # Add the SED-ML descriptor as the master simulation experiment.
    archive.addFile(
        str(sedml_path),
        sedml_path.name,
        lc.KnownFormats.lookupFormat("sed-ml"),
        True,  # master — this is what VCell / BioSimulations executes
    )
    # Top-level metadata (description) on the archive itself.
    meta = lc.OmexDescription()
    meta.setAbout(".")
    meta.setDescription(description)
    meta.setCreated(lc.OmexDescription.getCurrentDateAndTime())
    archive.addMetadata(".", meta)

    archive.writeToFile(str(omex_path))


# ----------------------------------------------------------------------------
# Index README
# ----------------------------------------------------------------------------


def emit_readme(circuits: list[Circuit]) -> str:
    lines: list[str] = []
    lines.append("# `models/` — Antimony / SBML exports of the project's closed-form circuits")
    lines.append("")
    lines.append(
        "*Auto-generated by [`scripts/export_lab1_sbml.py`](../scripts/export_lab1_sbml.py). "
        "Round-trip-verified by [`scripts/verify_lab1_sbml.py`](../scripts/verify_lab1_sbml.py). "
        "Part of the CURE-audit priority list ([`docs/cure-audit.md`](../docs/cure-audit.md) §2).*"
    )
    lines.append("")
    lines.append(
        "Each circuit ships in four formats per the CURE-Reproducible + Understandable + "
        "Credible-annotation pillars (Sauro et al. 2025, ref. 99a):"
    )
    lines.append("")
    lines.append(
        "- **`.ant`** — [Antimony](https://tellurium.readthedocs.io/en/latest/antimony.html) "
        "source-of-truth; auto-generated by the exporter from the `Circuit` dataclass."
    )
    lines.append(
        "- **`.sbml`** — SBML L3v2 round-tripped via Tellurium + annotated via libsbml with "
        "MIRIAM (`bqmodel:isDescribedBy` → PubMed) + SBO terms (species: 0000252 polypeptide "
        "chain; reactions: 0000176 biochemical reaction)."
    )
    lines.append(
        "- **`.sedml`** — SED-ML L1V4 simulation experiment descriptor (uniform time course, "
        "CVODE algorithm KISAO:0000019, one data generator per species)."
    )
    lines.append(
        "- **`.omex`** — COMBINE/OMEX archive bundling SBML + SED-ML + metadata.rdf. "
        "Master entry is the SED-ML; this is what VCell, BioSimulations, COPASI, and "
        "Tellurium each execute directly."
    )
    lines.append("")
    lines.append(
        "Plus **`regulome_provenance.json`** — the MIRIAM-style manifest for the project's "
        "Pando-on-CHOOSE regulome substrate (different model class, not SBML-shaped). "
        "Re-emit with `python scripts/emit_regulome_provenance.py`."
    )
    lines.append("")
    lines.append("## What's here")
    lines.append("")
    lines.append("| circuit | Antimony | SBML | SED-ML | OMEX | Lab 1 cell | canonical reference |")
    lines.append("|---|---|---|---|---|---:|---|")
    for c in circuits:
        lines.append(
            f"| {c.title} | "
            f"[`.ant`](lab1_{c.name}.ant) | "
            f"[`.sbml`](lab1_{c.name}.sbml) | "
            f"[`.sedml`](lab1_{c.name}.sedml) | "
            f"[`.omex`](lab1_{c.name}.omex) | "
            f"[cell {c.lab1_cell}](../notebooks/01_gene_circuit_dynamics.ipynb) | "
            f"{c.citation} |"
        )
    lines.append("")
    lines.append("## How to consume")
    lines.append("")
    lines.append("**Submit an OMEX archive to BioSimulations (any of their bundled simulators):**")
    lines.append("")
    lines.append("```bash")
    lines.append("curl -X POST https://api.biosimulations.org/runs \\")
    lines.append("     -F 'simulator=tellurium' \\")
    lines.append("     -F 'file=@models/lab1_repressilator.omex'")
    lines.append("```")
    lines.append("")
    lines.append("**Submit to VCell directly via vcell-cli:**")
    lines.append("")
    lines.append("```bash")
    lines.append("vcell-cli execute --archive models/lab1_repressilator.omex --out-dir results/")
    lines.append("```")
    lines.append("")
    lines.append("**Verify any model round-trips through Tellurium + COPASI + matches Lab 1's JAX simulation:**")
    lines.append("")
    lines.append("```bash")
    lines.append("uv pip install tellurium     # ~150 MB; pulls libsbml + libRoadRunner + scipy")
    lines.append("python scripts/verify_lab1_sbml.py")
    lines.append("```")
    lines.append("")
    lines.append("**Load a model into Tellurium:**")
    lines.append("")
    lines.append("```python")
    lines.append("import tellurium as te")
    lines.append("with open('models/lab1_repressilator.ant') as f:")
    lines.append("    model = te.loada(f.read())")
    lines.append("result = model.simulate(0, 40, 401)")
    lines.append("model.plot(result)")
    lines.append("```")
    lines.append("")
    lines.append("**Convert to SBML L3v2:**")
    lines.append("")
    lines.append("```python")
    lines.append("import tellurium as te")
    lines.append("sbml = te.antimonyToSBML(open('models/lab1_repressilator.ant').read())")
    lines.append("open('models/lab1_repressilator.sbml', 'w').write(sbml)")
    lines.append("```")
    lines.append("")
    lines.append("## Why Antimony, not SBML directly")
    lines.append("")
    lines.append(
        "Antimony is human-readable; SBML L3v2 XML is not. The committed artifact is the "
        "Antimony source-of-truth; the SBML is one `te.antimonyToSBML` call away. "
        "This matches the CURE-Understandable §1 recommendation: \"explicit representation "
        "not intertwined with implementation\" — Antimony is the explicit form, libSBML "
        "is the lossless XML form."
    )
    lines.append("")
    lines.append("## What's *not* here, honestly")
    lines.append("")
    lines.append(
        "The Hypergraph Neural ODE ([Lab 5](../notebooks/05_hypergraph_neural_odes.ipynb)) "
        "and the cell-state SBI inverse ([Lab 8](../notebooks/08_anatomical_compiler.ipynb)) "
        "are *learned* parameterised models — no closed-form SBML representation exists, and "
        "trying to force one would defeat the point. The CURE-aligned answer for those is in "
        "[`docs/cure-audit.md`](../docs/cure-audit.md) §\"structural non-compliance\" — emit the "
        "anatomical compiler's *output* in a standard format (PhysiCell grammar, per Johnson "
        "et al. 2025 — ref. 77c), not the *model* itself."
    )
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--output-dir", default="models", help="output directory for .ant files (default: models/)"
    )
    p.add_argument(
        "--no-readme", action="store_true", help="skip emitting the models/README.md index"
    )
    p.add_argument(
        "--no-sbml",
        action="store_true",
        help="skip the Tellurium-driven SBML round-trip (only emit .ant)",
    )
    p.add_argument(
        "--no-omex",
        action="store_true",
        help="skip SED-ML + OMEX bundling (item 5); only emit .ant + .sbml",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"export_lab1_sbml: writing to {out_dir}/")
    # The pipeline: Antimony (text) → SBML (via Tellurium) → MIRIAM/SBO-
    # annotated SBML (via libsbml) → SED-ML descriptor (via libsedml) →
    # OMEX archive (via libcombine). Tellurium drags in libsbml + libsedml
    # + libcombine, so the full chain runs from a single uv pip install.
    try:
        import tellurium as te  # noqa: F401
        have_tellurium = True
    except ImportError:
        have_tellurium = False

    # SED-ML / OMEX use libsedml + libcombine which ship with tellurium.
    have_omex = have_tellurium
    if have_omex:
        try:
            import libsedml  # noqa: F401
            import libcombine  # noqa: F401
        except ImportError:
            have_omex = False

    # Default simulation horizon for the SED-ML descriptor.
    sedml_t_final = 10.0
    sedml_n_points = 501

    for c in CIRCUITS:
        ant_path = out_dir / f"lab1_{c.name}.ant"
        text = emit_antimony(c)
        ant_path.write_text(text, encoding="utf-8")
        print(f"  wrote {ant_path}  ({len(text)} bytes)")

        if have_tellurium and not args.no_sbml:
            sbml_path = out_dir / f"lab1_{c.name}.sbml"
            sbml_text = te.antimonyToSBML(text)  # type: ignore[name-defined]
            # Add MIRIAM (PubMed) + SBO (polypeptide chain / biochemical reaction)
            # annotations via libsbml — CURE-audit item 4.
            sbml_text = annotate_sbml(sbml_text, c)
            sbml_path.write_text(sbml_text, encoding="utf-8")
            print(f"  wrote {sbml_path}  ({len(sbml_text)} bytes, MIRIAM+SBO annotated)")

            if have_omex and not args.no_omex:
                # Per-circuit SED-ML descriptor — what BioSimulations / VCell run.
                t_final = 6.0 if c.name == "repressilator" else sedml_t_final
                sedml_path = out_dir / f"lab1_{c.name}.sedml"
                sedml_text = emit_sedml(c, sbml_path.name, t_final, sedml_n_points)
                sedml_path.write_text(sedml_text, encoding="utf-8")
                print(f"  wrote {sedml_path}  ({len(sedml_text)} bytes)")

                # Bundle SBML + SED-ML into a COMBINE archive — CURE-audit item 5.
                omex_path = out_dir / f"lab1_{c.name}.omex"
                description = (
                    f"{c.title}. {c.citation} "
                    f"Source: anatomical-compiler/notebooks/01_gene_circuit_dynamics.ipynb "
                    f"cell {c.lab1_cell}. "
                    f"OMEX archive auto-generated by scripts/export_lab1_sbml.py."
                )
                write_omex_archive(omex_path, sbml_path, sedml_path, description)
                print(f"  wrote {omex_path}  ({omex_path.stat().st_size} bytes)")

    if not args.no_readme:
        readme_path = out_dir / "README.md"
        readme_path.write_text(emit_readme(CIRCUITS), encoding="utf-8")
        print(f"  wrote {readme_path}")

    if have_omex and not args.no_omex:
        formats = "(Antimony + MIRIAM/SBO-annotated SBML + SED-ML + OMEX)"
    elif have_tellurium and not args.no_sbml:
        formats = "(Antimony + SBML; install libsedml+libcombine for OMEX bundling)"
    else:
        formats = "(Antimony only — install tellurium for SBML/SED-ML/OMEX)"
    print(f"done — {len(CIRCUITS)} circuits exported {formats}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
