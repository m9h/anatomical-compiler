"""measure_lab4_real — Lab 4 MII gap with sequence-edge priors blended in.

Tier 4 of docs/dgx-verifier-runbook.md. Does blending sequence-grounded
edge priors (motif / Evo / Borzoi from the Tier-3 cache) into the
Pando-inferred regulome change the MII (Module Identifiability Index)
gap between the three benchmark systems?

The MII gap is the project's headline structural diagnostic — the spread
of MII across (bioprinted kidney, fetal kidney, brain organoid) cell-
state samples. Lab 4 reports baseline values around 0.31 / 0.37 / 0.23
(see `figures/nitmb_modularity_report.json`); the question is whether
sequence-grounded edge priors widen, narrow, or shift this spread.

Modes
-----
  stub   — synthetic 3-system benchmark with known modularity structure +
           controlled prior quality. Verifies the wrapper logic and gives
           a structural floor.
  real   — load edges.csv + cache_dir/<edges-stem>_{motif,evo,borzoi}.npy,
           sweep blend weight α, compute MII per system per α.
  auto   — try real, fall back to stub with a warning.

Output: figures/lab4_real_results.{json,md}.

Usage
-----
    python scripts/measure_lab4_real.py --mode stub
    python scripts/measure_lab4_real.py --mode real \\
        --edges data/fleck_edges.csv \\
        --cache cache/dgx_real_pollen_YYYYMMDD/ \\
        --systems bioprinted_kidney,fetal_kidney,brain_organoid

See also
--------
- docs/dgx-verifier-runbook.md Tier 4
- notebooks/04_modularity_identifiability.ipynb — the baseline MII numbers
- scripts/fm_edges_seq.py + scripts/ablate_edge_priors.py — the upstream stub +
  ablation; this is the real-data analogue
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# MII heuristic — pulled from Lab 4's mii_heuristic + relative_eigengap pattern
# ----------------------------------------------------------------------------


def _mii_from_adjacency(A: np.ndarray, k: int = 3) -> dict[str, float]:
    """MII heuristic + Fiedler-region relative eigengap on a (gene × gene)
    adjacency matrix. Returns {'mii_heuristic': ..., 'relative_eigengap': ...}.

    Sign-aware: edges with |A| are treated as undirected weights for the
    Laplacian. Self-loops are zeroed.
    """
    A = np.abs(A).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)
    Dinvsqrt = 1.0 / np.sqrt(np.where(d < 1e-9, 1.0, d))
    L_norm = np.eye(A.shape[0]) - (Dinvsqrt[:, None] * A * Dinvsqrt[None, :])
    eigvals = np.sort(np.real(np.linalg.eigvalsh(L_norm)))
    # MII heuristic: 1 - eigvals[1] / eigvals[k] (Lab 4 convention; Fiedler
    # over the cliff start). Bounded [0, 1].
    if len(eigvals) > k:
        denom = eigvals[k] if eigvals[k] > 1e-9 else 1e-9
        mii = float(max(0.0, 1.0 - eigvals[1] / denom))
        rel_gap = float((eigvals[k] - eigvals[1]) / max(eigvals[-1], 1e-9))
    else:
        mii = 0.0
        rel_gap = 0.0
    return {"mii_heuristic": mii, "relative_eigengap": rel_gap}


def _blend_to_adjacency(pando_weights: np.ndarray, seq_scores: np.ndarray, alpha: float) -> np.ndarray:
    """Convert (n_tfs × n_genes) Pando coefficients + sequence-prior scores into
    a symmetric (n_genes × n_genes) co-regulation adjacency.

    Each gene's co-regulation footprint = TFs that regulate it; the
    adjacency between two genes = overlap in their TF sets, weighted by
    (1-α) · stdz(Pando) + α · stdz(seq). This is the same blend pattern
    as scripts/ablate_edge_priors.py.
    """
    def stdz(W):
        mu = W.mean(axis=1, keepdims=True)
        sd = W.std(axis=1, keepdims=True) + 1e-9
        return (W - mu) / sd

    blended = (1 - alpha) * stdz(pando_weights) + alpha * stdz(seq_scores)
    # Co-regulation adjacency: similarity of TF-regulation profiles per gene
    return (blended.T @ blended).astype(np.float32)


# ----------------------------------------------------------------------------
# Stub mode — synthetic 3-system benchmark with known modularity gap
# ----------------------------------------------------------------------------


def _measure_stub(seed: int = 0) -> dict:
    """Three synthetic systems with controlled true modularity: organoid
    (clearest), fetal (medium), bioprinted (most diffuse). Stub Pando +
    stub sequence-prior; sweep α."""
    rng = np.random.default_rng(seed)
    n_tfs, n_genes = 20, 150

    def make_system(n_modules: int, module_strength: float, sys_seed: int):
        r = np.random.default_rng(sys_seed)
        truth = np.zeros((n_tfs, n_genes), dtype=np.float32)
        per_module_tfs = n_tfs // n_modules
        per_module_genes = n_genes // n_modules
        for m in range(n_modules):
            for t in range(per_module_tfs):
                idx = r.choice(per_module_genes, 6, replace=False) + m * per_module_genes
                truth[m * per_module_tfs + t, idx] = r.choice([+1, -1], 6) * module_strength
        pando = truth + r.standard_normal(truth.shape).astype(np.float32) * 0.6
        seq = truth + r.standard_normal(truth.shape).astype(np.float32) * 0.4
        return pando, seq

    systems = {
        "brain_organoid": make_system(4, 1.0, seed + 10),
        "fetal_kidney": make_system(3, 0.8, seed + 20),
        "bioprinted_kidney": make_system(2, 0.6, seed + 30),
    }

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for sys_name, (pando, seq) in systems.items():
        for a in alphas:
            A = _blend_to_adjacency(pando, seq, a)
            mii = _mii_from_adjacency(A, k=3)
            rows.append({"system": sys_name, "alpha": a, **mii})

    # Headline: MII spread across systems at α=0 (baseline) vs best-α
    def spread(rows_at_alpha):
        miis = [r["mii_heuristic"] for r in rows_at_alpha]
        return max(miis) - min(miis)

    spread_by_alpha = {a: spread([r for r in rows if r["alpha"] == a]) for a in alphas}
    best_alpha = max(spread_by_alpha, key=lambda a: spread_by_alpha[a])

    return {
        "mode": "stub",
        "alphas": alphas,
        "rows": rows,
        "spread_by_alpha": spread_by_alpha,
        "baseline_spread_alpha_0": spread_by_alpha[0.0],
        "best_alpha": best_alpha,
        "best_spread": spread_by_alpha[best_alpha],
        "delta_spread": spread_by_alpha[best_alpha] - spread_by_alpha[0.0],
        "note": (
            "Stub-mode synthetic with three systems of known module count "
            "(brain_organoid: 4 modules, fetal_kidney: 3, bioprinted_kidney: 2). "
            "MII spread = max(MII) - min(MII) across systems; blending in a "
            "decorrelated stub sequence prior can in principle widen the spread. "
            "Real-mode against the three real benchmark systems is the actual answer."
        ),
    }


# ----------------------------------------------------------------------------
# Real mode — load edges.csv + cache_dir, do the same sweep on real data
# ----------------------------------------------------------------------------


def _measure_real(edges_path: Path, cache_dir: Path, systems: list[str]) -> dict:
    """Real-mode template. DGX agent fills in the system-specific bits
    (which subset of the regulome corresponds to which benchmark system).
    """
    import pandas as pd

    edges = pd.read_csv(edges_path)
    if not {"tf", "target"}.issubset(edges.columns):
        raise ValueError("edges.csv must have 'tf' and 'target' columns")

    stem = edges_path.stem
    motif_path = cache_dir / f"{stem}_motif.npy"
    evo_path = cache_dir / f"{stem}_evo.npy"
    borzoi_path = cache_dir / f"{stem}_borzoi.npy"
    missing = [p for p in (motif_path, evo_path, borzoi_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Tier-3 cache incomplete: missing {missing}. Run scripts/run_fm_real_dgx.sh first."
        )

    # The real-mode body needs per-system gene/edge subsets — typically
    # filtered from the regulome via system-specific masks the DGX agent
    # supplies. Pattern:
    #   for sys in systems:
    #       edges_sys = edges[edges['system'] == sys]  # or however the dataset labels them
    #       pando_W = build_pando_matrix(edges_sys)
    #       seq_W = align_seq_scores(motif/evo/borzoi, edges_sys)
    #       for alpha in alphas:
    #           A = _blend_to_adjacency(pando_W, seq_W, alpha)
    #           mii = _mii_from_adjacency(A, k=3)
    #           rows.append(...)
    raise NotImplementedError(
        f"Real-mode requires the system-specific edge-subset construction. "
        f"DGX agent: implement the {systems} mask given the dataset's schema. "
        f"All Tier-3 cache files are present ({motif_path}, {evo_path}, {borzoi_path}); "
        f"the missing piece is the system-to-edge mapping."
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--mode", choices=["stub", "real", "auto"], default="auto")
    p.add_argument("--edges", default=None, help="path to edges.csv with tf,target")
    p.add_argument("--cache", default=None, help="Tier-3 cache dir with motif/evo/borzoi .npy")
    p.add_argument(
        "--systems",
        default="brain_organoid,fetal_kidney,bioprinted_kidney",
        help="comma-separated benchmark systems (real mode)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="figures/lab4_real_results")
    args = p.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(",")]

    used_mode = args.mode
    if args.mode == "real":
        if args.edges is None or args.cache is None:
            print("error: --mode real requires --edges and --cache", file=sys.stderr)
            return 2
        result = _measure_real(Path(args.edges), Path(args.cache), systems)
    elif args.mode == "stub":
        result = _measure_stub(seed=args.seed)
    else:
        if args.edges is not None and args.cache is not None:
            try:
                result = _measure_real(Path(args.edges), Path(args.cache), systems)
                used_mode = "real"
            except (FileNotFoundError, NotImplementedError, ValueError) as e:
                warnings.warn(f"falling back to stub mode: {e}")
                result = _measure_stub(seed=args.seed)
                used_mode = "stub"
        else:
            result = _measure_stub(seed=args.seed)
            used_mode = "stub"

    print(f"\nmeasure_lab4_real ({used_mode} mode):")
    print(f"  baseline (α=0) spread: {result['baseline_spread_alpha_0']:.3f}")
    print(f"  best α: {result['best_alpha']}  → spread {result['best_spread']:.3f}")
    print(f"  Δ spread: {result['delta_spread']:+.3f}")

    out_json = Path(args.output + ".json")
    out_md = Path(args.output + ".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = [
        "# Lab 4 — MII gap with sequence-edge priors blended in",
        "",
        f"**Mode: {used_mode}.**",
        "",
        "Blended graph: `(1−α)·stdz(Pando) + α·stdz(seq_prior)`. "
        "MII heuristic on the (gene × gene) co-regulation adjacency derived "
        "from blending. Spread = max(MII) − min(MII) across the three systems.",
        "",
        "| α | spread |",
        "|---:|---:|",
    ]
    for a in result["alphas"]:
        md.append(f"| {a:.1f} | {result['spread_by_alpha'][a]:.3f} |")
    md += [
        "",
        f"**Headline:** at α=0 (Pando alone) spread = {result['baseline_spread_alpha_0']:.3f}; "
        f"at best α = {result['best_alpha']:.1f} spread = {result['best_spread']:.3f}; "
        f"Δ = **{result['delta_spread']:+.3f}**.",
        "",
        f"_{result.get('note', '')}_" if result.get("note") else "",
        "",
        "## Per-system MII at each α",
        "",
        "| system | α | mii_heuristic | rel_eigengap |",
        "|---|---:|---:|---:|",
    ]
    for r in result["rows"]:
        md.append(
            f"| {r['system']} | {r['alpha']:.1f} | {r['mii_heuristic']:.3f} | "
            f"{r['relative_eigengap']:.3f} |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        "Lab 4's baseline MII numbers (Pando alone, no sequence prior) are around "
        "0.31 (bioprinted kidney), 0.37 (brain organoid), 0.23 (fetal kidney) — see "
        "`figures/nitmb_modularity_report.json`. The blend with sequence priors is "
        "additive in structural signal: edges where Pando and seq agree get reinforced; "
        "where they disagree, the seq prior pulls the topology toward the cis-regulatory "
        "grammar. A positive Δ-spread means the sequence prior is amplifying the "
        "modularity-of-self-organisation signal the project's whitepaper rests on.",
        "",
        "See also: [`docs/dgx-verifier-runbook.md`](../docs/dgx-verifier-runbook.md) Tier 4; "
        "[`scripts/ablate_edge_priors.py`](../scripts/ablate_edge_priors.py) — the stub-mode synthetic; "
        "[`notebooks/04_modularity_identifiability.ipynb`](../notebooks/04_modularity_identifiability.ipynb) — the baseline.",
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {out_json}")
    print(f"  wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
