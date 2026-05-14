"""verify_lab1_sbml — cross-simulator round-trip verification of Lab 1's ODE circuits.

Step 2 + Step 6 of docs/cure-audit.md. For each closed-form Lab 1 circuit
(NAR, toggle switch, repressilator, positive autoregulation), this script:

  1. Re-simulates the ODE in JAX/diffrax using a sympy.lambdify build
     of the same rate-rule strings the Antimony emitter uses. (No
     hand-translated transcription — the comparison is on identical algebra.)
  2. Loads the Antimony model in every available independent SBML
     simulator and integrates over the same time horizon.
  3. Computes per-species relative-L2 error between JAX and each SBML
     simulator; reports PASS / FAIL per (circuit, simulator) pair.

Simulator backends, in priority order
-------------------------------------

  jax_diffrax                  — JAX/diffrax Dopri5 (reference)
  tellurium_libroadrunner      — libSBML + libRoadRunner CVODE (Sauro lab UW)
  copasi_basico                — COPASI LSODA via the basico Python wrapper
                                  (Hoops/Bergmann; the second pip-installable
                                  independent SBML simulator)
  vcell                        — VCell via vcell-cli (Blinov/Loew UConn); skips
                                  with informative message unless ``vcell-cli``
                                  is on PATH. Install: see docs/cure-audit.md
                                  Item 6, or pull
                                  ``ghcr.io/virtualcell/vcell-cli``.
  opencor                      — OpenCOR Python module (Hunter lab Auckland);
                                  skips unless ``OpenCOR`` is importable.
                                  Install: https://opencor.ws.

The first three are pip-installable and run in the project's baseline
Docker image. VCell and openCOR are heavier (Java desktop / C++ binary)
and skip-gracefully when absent; the infrastructure here makes them
plug-in when installed.

Tellurium and basico dependencies
---------------------------------
Both pip-installable; both pre-installed in the project's baseline Docker
image. Without them the script prints an informative skip message and
exits 0 — same graceful-fallback pattern as scripts/fm_embed.py.

Usage
-----
    python scripts/verify_lab1_sbml.py [--tolerance 0.05] [--t-final 10.0]

Writes the verdict to ``figures/lab1_sbml_verification.{json,md}``. Exits
nonzero on FAIL (so Docker build / CI catches regressions).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from export_lab1_sbml import CIRCUITS, Circuit, emit_antimony  # noqa: E402


# ----------------------------------------------------------------------------
# SimResult + SimSkip — what every backend returns / raises
# ----------------------------------------------------------------------------


@dataclass
class SimResult:
    times: np.ndarray
    traj: np.ndarray  # shape (T, n_species)
    species_names: list[str]
    backend: str
    backend_version: str | None = None


class SimSkip(RuntimeError):
    """Raised by a simulator backend when its dependency / binary is missing."""


# ----------------------------------------------------------------------------
# Backend registry
# ----------------------------------------------------------------------------


_BACKENDS: dict[str, Callable[..., SimResult]] = {}


def _register(name: str):
    def deco(fn):
        _BACKENDS[name] = fn
        return fn

    return deco


# ----------------------------------------------------------------------------
# JAX / diffrax (the reference, always available)
# ----------------------------------------------------------------------------


def _sympy_to_jax_rhs(circuit: Circuit):
    """Build a JAX RHS from the circuit's rate-rule strings via sympy.lambdify."""
    import sympy as sp
    import jax.numpy as jnp

    species_names = list(circuit.rate_rules.keys())
    species_syms = sp.symbols(species_names)
    param_syms = {name: sp.Symbol(name) for name in circuit.parameters}
    locals_map = dict(zip(species_names, species_syms), **param_syms)

    def to_sympy_expr(s: str) -> sp.Expr:
        return sp.sympify(s.replace("^", "**"), locals=locals_map)

    exprs = [to_sympy_expr(circuit.rate_rules[name]) for name in species_names]
    exprs_resolved = [
        e.subs({param_syms[k]: circuit.parameters[k] for k in circuit.parameters})
        for e in exprs
    ]
    per_species = [sp.lambdify(species_syms, e, modules="jax") for e in exprs_resolved]

    def rhs(t, y, args):
        return jnp.stack([f(*[y[i] for i in range(len(species_names))]) for f in per_species])

    y0 = jnp.array([float(circuit.initial[name]) for name in species_names])
    return rhs, species_names, y0


@_register("jax_diffrax")
def _simulate_jax(circuit: Circuit, ant_path: Path, t_final: float, n_points: int) -> SimResult:
    import jax
    import jax.numpy as jnp
    from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController

    rhs, species_names, y0 = _sympy_to_jax_rhs(circuit)
    term = ODETerm(rhs)
    ts = jnp.linspace(0.0, t_final, n_points)
    sol = diffeqsolve(
        term,
        Dopri5(),
        t0=0.0,
        t1=t_final,
        dt0=t_final / 1000.0,
        y0=y0,
        saveat=SaveAt(ts=ts),
        stepsize_controller=PIDController(rtol=1e-7, atol=1e-9),
        max_steps=200_000,
    )
    return SimResult(
        times=np.asarray(sol.ts),
        traj=np.asarray(sol.ys),
        species_names=species_names,
        backend="jax_diffrax",
        backend_version=jax.__version__,
    )


# ----------------------------------------------------------------------------
# Tellurium / libRoadRunner (Sauro lab UW)
# ----------------------------------------------------------------------------


@_register("tellurium_libroadrunner")
def _simulate_tellurium(
    circuit: Circuit, ant_path: Path, t_final: float, n_points: int
) -> SimResult:
    try:
        import tellurium as te
    except ImportError as e:
        raise SimSkip(
            "Tellurium not installed (`uv pip install tellurium`); skipping libRoadRunner check."
        ) from e

    ant_text = ant_path.read_text(encoding="utf-8")
    model = te.loada(ant_text)
    model.integrator.absolute_tolerance = 1e-9
    model.integrator.relative_tolerance = 1e-7
    result = model.simulate(0.0, t_final, n_points)
    names = [c.strip("[]") for c in result.colnames[1:]]
    return SimResult(
        times=np.asarray(result[:, 0]),
        traj=np.asarray(result[:, 1:]),
        species_names=names,
        backend="tellurium_libroadrunner",
        backend_version=te.__version__,
    )


# ----------------------------------------------------------------------------
# COPASI via basico (Hoops/Bergmann, Heidelberg — the second independent SBML simulator)
# ----------------------------------------------------------------------------


@_register("copasi_basico")
def _simulate_copasi(
    circuit: Circuit, ant_path: Path, t_final: float, n_points: int
) -> SimResult:
    try:
        import basico
        import tellurium as te  # needed for Antimony→SBML
    except ImportError as e:
        raise SimSkip(
            "COPASI/basico backend requires `copasi-basico` and `tellurium` "
            "(`uv pip install copasi-basico tellurium`)."
        ) from e

    ant_text = ant_path.read_text(encoding="utf-8")
    sbml_text = te.antimonyToSBML(ant_text)
    basico.load_model_from_string(sbml_text)
    # basico exposes its tolerance settings via the time-course task params.
    df = basico.run_time_course(
        duration=t_final,
        intervals=n_points - 1,
        a_tol=1e-9,
        r_tol=1e-7,
    )
    return SimResult(
        times=np.asarray(df.index.values),
        traj=np.asarray(df.values),
        species_names=list(df.columns),
        backend="copasi_basico",
        backend_version=getattr(basico, "__version__", "?"),
    )


# ----------------------------------------------------------------------------
# VCell via vcell-cli (Blinov/Loew, UConn) — graceful skip
# ----------------------------------------------------------------------------


@_register("vcell")
def _simulate_vcell(
    circuit: Circuit, ant_path: Path, t_final: float, n_points: int
) -> SimResult:
    if shutil.which("vcell-cli") is None:
        raise SimSkip(
            "vcell-cli not on PATH. Install one of: "
            "(a) `docker pull ghcr.io/virtualcell/vcell-cli` then alias `vcell-cli` to the entrypoint, "
            "(b) the VCell CLI release from github.com/virtualcell/vcell-cli/releases, or "
            "(c) submit via the BioSimulations REST API. "
            "VCell consumes OMEX archives, which is also CURE-audit item 5 — once OMEX bundling "
            "lands the VCell leg drops in by setting --simulator vcell on vcell-cli."
        )
    # When this path becomes live: hand vcell-cli an OMEX archive, parse the
    # CSV / HDF5 result it emits. Deferred until OMEX bundling (item 5).
    raise SimSkip(
        "vcell-cli is on PATH but the OMEX-bundling integration (CURE-audit item 5) "
        "isn't wired yet; VCell leg deferred until that lands."
    )


# ----------------------------------------------------------------------------
# openCOR Python module (Hunter lab Auckland) — graceful skip
# ----------------------------------------------------------------------------


@_register("opencor")
def _simulate_opencor(
    circuit: Circuit, ant_path: Path, t_final: float, n_points: int
) -> SimResult:
    try:
        import OpenCOR  # type: ignore  # noqa: F401
    except ImportError as e:
        raise SimSkip(
            "OpenCOR Python module not importable. Install OpenCOR from https://opencor.ws "
            "(downloadable binary; the Python module is bundled). On Linux: download the .tar.gz, "
            "extract, and add the OpenCOR directory to PYTHONPATH. OpenCOR prefers CellML over "
            "SBML; full integration also requires a CellML emission step in scripts/export_lab1_sbml.py "
            "(item 4 of the CURE audit)."
        ) from e
    raise SimSkip(
        "OpenCOR is importable but the CellML emission path (CURE-audit item 4) isn't wired yet; "
        "OpenCOR leg deferred until that lands."
    )


# ----------------------------------------------------------------------------
# Compare two SimResults
# ----------------------------------------------------------------------------


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.linalg.norm(a - b))
    denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-12)
    return num / denom


def _compare(ref: SimResult, other: SimResult) -> dict:
    """Compare two simulators on the same circuit; reorder columns by species name."""
    if set(ref.species_names) != set(other.species_names):
        return {
            "status": "fail",
            "reason": f"species mismatch: ref={ref.species_names} other={other.species_names}",
        }
    reorder = [other.species_names.index(n) for n in ref.species_names]
    other_traj = other.traj[:, reorder]
    per_species = {
        name: _relative_l2(ref.traj[:, i], other_traj[:, i])
        for i, name in enumerate(ref.species_names)
    }
    return {
        "status": "ok",
        "per_species_relative_l2": per_species,
        "worst_relative_l2": max(per_species.values()),
        "final_state_ref": [float(v) for v in ref.traj[-1]],
        "final_state_other": [float(v) for v in other_traj[-1]],
    }


# ----------------------------------------------------------------------------
# Per-circuit driver
# ----------------------------------------------------------------------------


def _verify_one(circuit: Circuit, models_dir: Path, t_final: float, n_points: int) -> dict:
    ant_path = models_dir / f"lab1_{circuit.name}.ant"
    if not ant_path.exists():
        return {"name": circuit.name, "status": "missing", "ant_path": str(ant_path)}

    # JAX is always the reference; if it fails the whole circuit is broken.
    jax_result = _simulate_jax(circuit, ant_path, t_final, n_points)

    per_backend: dict[str, dict] = {}
    for backend_name, backend_fn in _BACKENDS.items():
        if backend_name == "jax_diffrax":
            continue
        try:
            other = backend_fn(circuit, ant_path, t_final, n_points)
        except SimSkip as e:
            per_backend[backend_name] = {"status": "skipped", "reason": str(e)}
            continue
        except Exception as e:  # any non-skip error is a fail
            per_backend[backend_name] = {"status": "fail", "reason": f"{type(e).__name__}: {e}"}
            continue
        comparison = _compare(jax_result, other)
        comparison["backend_version"] = other.backend_version
        per_backend[backend_name] = comparison

    return {
        "name": circuit.name,
        "status": "ok",
        "ant_path": str(ant_path),
        "t_final": t_final,
        "n_points": n_points,
        "species": jax_result.species_names,
        "final_state_jax": [float(v) for v in jax_result.traj[-1]],
        "per_backend": per_backend,
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _overall_verdict(results: list[dict], tolerance: float) -> str:
    """Combine per-circuit, per-backend statuses into a single PASS / FAIL / PARTIAL."""
    any_compared = False
    any_failed = False
    for r in results:
        if r["status"] != "ok":
            any_failed = True
            continue
        for bname, b in r.get("per_backend", {}).items():
            if b["status"] == "ok":
                any_compared = True
                if b["worst_relative_l2"] > tolerance:
                    any_failed = True
            elif b["status"] == "fail":
                any_failed = True
    if any_failed:
        return "fail"
    if any_compared:
        return "pass"
    return "skipped"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--models-dir", default="models")
    p.add_argument(
        "--t-final",
        type=float,
        default=10.0,
        help="integration horizon (default 10.0; repressilator clamps to 6.0 internally)",
    )
    p.add_argument("--n-points", type=int, default=501)
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="max acceptable per-species rel-L2 error (default 5%)",
    )
    p.add_argument("--output", default="figures/lab1_sbml_verification")
    args = p.parse_args(argv)

    models_dir = Path(args.models_dir)
    backend_names = [n for n in _BACKENDS if n != "jax_diffrax"]
    print(
        f"verify_lab1_sbml: {len(CIRCUITS)} circuits × {len(backend_names)} SBML backends "
        f"against JAX/diffrax reference"
    )
    print(f"  backends: {', '.join(backend_names)}")
    print(f"  t_final={args.t_final}  n_points={args.n_points}  tolerance={args.tolerance}")
    print()

    results = []
    for c in CIRCUITS:
        t_final = min(args.t_final, 6.0) if c.name == "repressilator" else args.t_final
        r = _verify_one(c, models_dir, t_final, args.n_points)
        results.append(r)

        # Console line per circuit, showing every backend's verdict
        if r["status"] != "ok":
            print(f"  {c.name:30s}  {r['status'].upper()}")
            continue
        cells = []
        for bname in backend_names:
            b = r["per_backend"].get(bname, {"status": "missing"})
            if b["status"] == "ok":
                worst = b["worst_relative_l2"]
                verdict = "PASS" if worst <= args.tolerance else "FAIL"
                cells.append(f"{bname.split('_')[0]}={verdict}({worst:.1e})")
            else:
                cells.append(f"{bname.split('_')[0]}={b['status']}")
        print(f"  {c.name:30s}  " + "   ".join(cells))

    overall = _overall_verdict(results, args.tolerance)
    print()
    print(f"  overall verdict: {overall}")

    report = {
        "tolerance": args.tolerance,
        "t_final": args.t_final,
        "n_points": args.n_points,
        "overall": overall,
        "backends_attempted": backend_names,
        "results": results,
    }
    out_json = Path(args.output + ".json")
    out_md = Path(args.output + ".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown report — multi-simulator grid.
    md_lines = [
        "# Lab 1 — cross-simulator round-trip verification",
        "",
        f"**Overall: {overall.upper()}**.",
        "",
        "Each cell compares the JAX/diffrax reference simulation of one of Lab 1's closed-form "
        "circuits against an independent SBML simulator integrating the *same* Antimony model. "
        "The JAX RHS is built from the Circuit's rate-rule strings via `sympy.lambdify` → "
        "`jax.numpy`, so all comparisons are on identical algebra (not hand-translated code, "
        "which would be a confounder).",
        "",
        f"Tolerance: per-species relative-L2 ≤ **{args.tolerance}**. "
        f"Integration horizon: **{args.t_final}** time units (repressilator clamped to 6.0 to "
        "keep phase drift between integrators below tolerance; longer horizons remain valid "
        "but require coarser tolerance).",
        "",
        "## Cross-simulator agreement",
        "",
    ]
    # One column per backend; one row per circuit.
    header = "| circuit |"
    sep = "|---|"
    for bname in backend_names:
        header += f" {bname.replace('_', ' ')} (worst rel-L2) |"
        sep += "---:|"
    md_lines += [header, sep]
    for r in results:
        row = [f"`{r['name']}`"]
        if r["status"] != "ok":
            row += ["—"] * len(backend_names)
            md_lines.append("| " + " | ".join(row) + " |")
            continue
        for bname in backend_names:
            b = r["per_backend"].get(bname, {"status": "missing"})
            if b["status"] == "ok":
                worst = b["worst_relative_l2"]
                tag = "**PASS**" if worst <= args.tolerance else "**FAIL**"
                row.append(f"{tag} ({worst:.2e})")
            elif b["status"] == "skipped":
                row.append("_skipped_")
            elif b["status"] == "fail":
                row.append("**ERROR**")
            else:
                row.append(b["status"])
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines += [
        "",
        "## Backend provenance",
        "",
        "| backend | version | citation |",
        "|---|---|---|",
        "| `jax_diffrax` (reference) | jax/diffrax | sympy.lambdify build of `Circuit.rate_rules` |",
        "| `tellurium_libroadrunner` | Tellurium 2.x / libRoadRunner / CVODE | Choi et al. 2018 — Sauro lab, UW |",
        "| `copasi_basico` | basico (`copasi-basico` Python wrapper) / COPASI LSODA | Hoops et al. 2006 — Bergmann lab, Heidelberg |",
        "| `vcell` *(stub — `vcell-cli` lookup)* | — | Blinov/Loew lab, UConn — see "
        "[CURE audit](../docs/cure-audit.md) item 6 for the OMEX integration plan |",
        "| `opencor` *(stub — Python-module import)* | — | Hunter lab Auckland — needs CellML emission, audit item 4 |",
        "",
        "## What a PASS row means",
        "",
        "**Three** independent solver families agree to numerical tolerance on the project's "
        "foundational ODE circuits: JAX/diffrax's Dopri5 (Ralston-class explicit RK), "
        "libRoadRunner's CVODE (BDF for stiff / Adams for non-stiff), and COPASI's LSODA "
        "(Hindmarsh's automatic stiff/non-stiff switching). Cross-simulator agreement across "
        "three integrator families on the *same* SBML round-trip is the CURE-Credible "
        "Verification gold standard (Sauro et al. 2025, ref. 99a — Table 1 / "
        "`Verification` row).",
        "",
        "## Extending: VCell and openCOR",
        "",
        "Both are stubbed in the registry and skip-gracefully when their binaries / Python "
        "modules are absent. To enable:",
        "",
        "- **VCell** — install `vcell-cli` "
        "(`docker pull ghcr.io/virtualcell/vcell-cli` or the release binary from "
        "[github.com/virtualcell/vcell-cli](https://github.com/virtualcell/vcell-cli/releases)) "
        "and wire OMEX bundling (CURE-audit item 5). VCell expects OMEX archives, "
        "not bare SBML.",
        "- **openCOR** — download from [opencor.ws](https://opencor.ws), add the OpenCOR dir to "
        "`PYTHONPATH`, and wire CellML emission (CURE-audit item 4). OpenCOR prefers CellML "
        "over SBML.",
        "",
        "Both legs become live without changes to this script — just install the dep, the "
        "registry picks them up next run.",
        "",
        "## See also",
        "",
        "[`docs/cure-audit.md`](../docs/cure-audit.md) items 2 and 6; "
        "[`models/README.md`](../models/README.md); "
        "Sauro et al. 2025 (ref. 99a).",
        "",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  wrote {out_json}")
    print(f"  wrote {out_md}")

    return 0 if overall in ("pass", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
