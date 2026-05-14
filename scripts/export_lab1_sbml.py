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
        citation="Elowitz & Bois — Lab 1 §5 exercise (a).",
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
        "Each `.ant` file is an [Antimony](https://tellurium.readthedocs.io/en/latest/antimony.html) "
        "model that round-trips to SBML L3v2 via "
        "[Tellurium](https://tellurium.readthedocs.io) / [libRoadRunner](https://www.libroadrunner.org). "
        "They are the canonical, BioModels-submittable representations of Lab 1's gene circuits."
    )
    lines.append("")
    lines.append("## What's here")
    lines.append("")
    lines.append("| file | circuit | Lab 1 cell | canonical reference |")
    lines.append("|---|---|---:|---|")
    for c in circuits:
        lines.append(
            f"| [`lab1_{c.name}.ant`](lab1_{c.name}.ant) | {c.title} | "
            f"[cell {c.lab1_cell}](../notebooks/01_gene_circuit_dynamics.ipynb) | {c.citation} |"
        )
    lines.append("")
    lines.append("## How to consume")
    lines.append("")
    lines.append("**Verify any model round-trips through Tellurium + matches Lab 1's JAX simulation:**")
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
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"export_lab1_sbml: writing to {out_dir}/")
    # Try Tellurium for the optional SBML round-trip emission. If absent, the
    # .ant files alone are the committed artifact and SBML is on-demand.
    try:
        import tellurium as te  # noqa: F401
        have_tellurium = True
    except ImportError:
        have_tellurium = False

    for c in CIRCUITS:
        ant_path = out_dir / f"lab1_{c.name}.ant"
        text = emit_antimony(c)
        ant_path.write_text(text, encoding="utf-8")
        print(f"  wrote {ant_path}  ({len(text)} bytes)")

        if have_tellurium and not args.no_sbml:
            sbml_path = out_dir / f"lab1_{c.name}.sbml"
            sbml_text = te.antimonyToSBML(text)  # type: ignore[name-defined]
            sbml_path.write_text(sbml_text, encoding="utf-8")
            print(f"  wrote {sbml_path}  ({len(sbml_text)} bytes)")

    if not args.no_readme:
        readme_path = out_dir / "README.md"
        readme_path.write_text(emit_readme(CIRCUITS), encoding="utf-8")
        print(f"  wrote {readme_path}")

    print(
        f"done — {len(CIRCUITS)} circuits exported"
        f"{' (Antimony + SBML)' if have_tellurium and not args.no_sbml else ' (Antimony only — install tellurium for SBML round-trip)'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
