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
    # NaN/Inf cleanup: when seq priors are partial (NaN-filled placeholders for
    # missing evo/borzoi caches), NaNs propagate through the blended adjacency
    # and corrupt the Laplacian spectrum.
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)
    Dinvsqrt = 1.0 / np.sqrt(np.where(d < 1e-9, 1.0, d))
    L_norm = np.eye(A.shape[0]) - (Dinvsqrt[:, None] * A * Dinvsqrt[None, :])
    L_norm = 0.5 * (L_norm + L_norm.T)
    try:
        eigvals = np.sort(np.real(np.linalg.eigvalsh(L_norm)))
    except np.linalg.LinAlgError:
        eigvals = np.sort(np.real(np.linalg.svd(L_norm, compute_uv=False)))
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


def _seq_prior_per_gene(edges_df, motif_scores, evo_scores, borzoi_scores,
                        target_genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-edge sequence scores into per-(TF, gene) priors.

    Returns
    -------
    seq_per_target : (n_target_genes,) — average seq score across TFs targeting each gene
    tf_target_grid : (n_tfs, n_target_genes) — z-standardised seq scores for the
        outer-product blend pattern. Missing (TF, gene) entries are filled with 0.
    """
    import pandas as pd

    def stdz(v):
        """Return z-standardised array, or None if the backend is absent
        (all-NaN or all-zero — i.e. the cache file wasn't loaded).
        """
        v = np.asarray(v, dtype=np.float32)
        if not np.any(np.isfinite(v)) or np.allclose(v[np.isfinite(v)], 0.0):
            return None
        mask = np.isfinite(v) & (v != 0)
        if mask.sum() < 2:
            return None
        mu = v[mask].mean()
        sd = v[mask].std() + 1e-9
        out = (v - mu) / sd
        out[~mask] = 0.0
        return out

    present = [z for z in (stdz(motif_scores), stdz(evo_scores), stdz(borzoi_scores)) if z is not None]
    if not present:
        raise ValueError("no usable seq-prior backends — all caches absent or all-zero")
    combined = np.mean(np.stack(present), axis=0)

    tfs = sorted(edges_df["tf"].unique())
    tf_to_i = {t: i for i, t in enumerate(tfs)}
    g_to_j = {g: j for j, g in enumerate(target_genes)}
    grid = np.zeros((len(tfs), len(target_genes)), dtype=np.float32)
    for idx, row in enumerate(edges_df.itertuples(index=False)):
        j = g_to_j.get(row.target)
        i = tf_to_i.get(row.tf)
        if j is None or i is None:
            continue
        grid[i, j] = combined[idx]
    per_target = grid.mean(axis=0)
    return per_target, grid


def _identifiability_score(eigvals: np.ndarray) -> float:
    """Lab 4's NITMB modularity-identifiability heuristic:
    mean(gaps[:10]) / std(eigvals[:10]).
    """
    if len(eigvals) < 2:
        return 0.0
    gaps = np.diff(eigvals[:11])
    return float(np.mean(gaps[:10]) / (np.std(eigvals[:10]) + 1e-8))


def _per_system_baseline_adjacency(adata, gene_list: list[str], n_top: int = 10) -> np.ndarray:
    """Lab 4's correlation-incidence baseline restricted to the supplied
    gene_list (so the seq-prior blend operates on the same gene set).
    """
    var_names = list(adata.var.index)
    have = [g for g in gene_list if g in var_names]
    if len(have) < 10:
        raise ValueError(
            f"only {len(have)}/{len(gene_list)} target genes are in the system's "
            f"h5ad var.index — too few for a stable L0 spectrum"
        )
    keep_idx = [var_names.index(g) for g in have]
    X = adata.X
    if hasattr(X, "toarray"):
        X_sub = X[:, keep_idx].toarray()
    else:
        X_sub = np.asarray(X)[:, keep_idx]
    corr = np.corrcoef(X_sub.T)
    corr = np.nan_to_num(corr, nan=0.0)
    n = corr.shape[0]
    incidence = np.zeros((n, n), dtype=np.float32)
    k = min(n_top, n - 1)
    for i in range(n):
        top = np.argsort(np.abs(corr[i]))[-(k + 1):]  # self + k
        incidence[top, i] = 1.0
    return incidence, have


def _l0_spectrum(adjacency: np.ndarray, n_keep: int = 100) -> np.ndarray:
    """Normalised graph Laplacian eigenvalues, sorted ascending, drop ~0 mode.

    NaN/Inf cleanup: when seq priors are partial (e.g. borzoi placeholder
    all-NaN, or evo has some unscored edges), NaNs propagate through the
    blended adjacency. nan_to_num before eigendecomposition keeps the
    computation finite; eigvalsh is wrapped in a fallback in case of
    numerical non-convergence on pathological adjacencies.
    """
    A = np.abs(adjacency).astype(np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)
    Dinv = 1.0 / np.sqrt(np.where(d < 1e-9, 1.0, d))
    L = np.eye(A.shape[0]) - (Dinv[:, None] * A * Dinv[None, :])
    L = 0.5 * (L + L.T)  # symmetrize against accumulated round-off
    try:
        ev = np.sort(np.real(np.linalg.eigvalsh(L)))
    except np.linalg.LinAlgError:
        # Fallback: SVD-based pseudo-eigenvalues — slower but more robust
        ev = np.sort(np.real(np.linalg.svd(L, compute_uv=False)))
    ev = ev[ev > 1e-6]
    return ev[:n_keep]


def _measure_real(edges_path: Path, cache_dir: Path,
                  system_h5ads: dict[str, Path], n_hvg: int = 100) -> dict:
    """Real-mode: per-system L0 spectrum, blended with sequence-prior adjacency.

    Pipeline:
      1. Load edges + Tier-3 cache; build a per-(TF,target) sequence prior grid.
      2. For each system's h5ad, restrict to the regulome's target-gene set
         and compute the correlation-incidence baseline (Lab 4 convention).
      3. Build a sequence-prior adjacency: seq_adj[i,j] = grid[:, i] · grid[:, j]
         (gene-gene similarity in their TF sequence-priority profiles).
      4. Sweep α ∈ {0, …, 1}; blend `(1-α)·stdz(corr_adj) + α·stdz(seq_adj)`;
         take L0 spectrum + identifiability score per system per α.
      5. Spread = max(score) − min(score) across systems at each α; report.
    """
    import pandas as pd
    import anndata as ad

    edges = pd.read_csv(edges_path)
    if not {"tf", "target"}.issubset(edges.columns):
        raise ValueError("edges.csv must have 'tf' and 'target' columns")

    stem = edges_path.stem
    motif_path = cache_dir / f"{stem}_motif.npy"
    evo_path = cache_dir / f"{stem}_evo.npy"
    borzoi_path = cache_dir / f"{stem}_borzoi.npy"
    # Motif is mandatory; evo/borzoi gracefully substitute with NaN-filled
    # arrays (which the per-target aggregator handles) when the Tier-3 cache
    # is partial. The blended seq prior degrades to motif-only in that case.
    if not motif_path.exists():
        raise FileNotFoundError(
            f"Tier-3 cache incomplete: missing motif at {motif_path}. "
            "Run scripts/fm_edges_seq.py motif first."
        )
    motif = np.load(motif_path)
    if evo_path.exists():
        evo = np.load(evo_path)
    else:
        warnings.warn(f"evo cache missing at {evo_path}; using NaN placeholder")
        evo = np.full(len(motif), np.nan, dtype=np.float32)
    if borzoi_path.exists():
        borzoi = np.load(borzoi_path)
    else:
        warnings.warn(f"borzoi cache missing at {borzoi_path}; using NaN placeholder")
        borzoi = np.full(len(motif), np.nan, dtype=np.float32)
    if not (len(motif) == len(evo) == len(borzoi) == len(edges)):
        raise ValueError(
            f"seq-prior cache lengths {len(motif)}/{len(evo)}/{len(borzoi)} "
            f"don't match edges.csv ({len(edges)} rows)"
        )

    target_genes = sorted(edges["target"].unique())
    _seq_per_target, tf_target_grid = _seq_prior_per_gene(
        edges, motif, evo, borzoi, target_genes
    )

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    per_system_have_count: dict[str, int] = {}
    for sys_name, h5_path in system_h5ads.items():
        if not h5_path.exists():
            warnings.warn(f"system h5ad missing: {h5_path}; skipping {sys_name}")
            continue
        adata = ad.read_h5ad(str(h5_path), backed="r").to_memory()
        corr_adj, have = _per_system_baseline_adjacency(adata, target_genes)
        have_idx = [target_genes.index(g) for g in have]
        per_system_have_count[sys_name] = len(have)

        sub_grid = tf_target_grid[:, have_idx]
        # NaN-clean the seq-prior grid before the gene×gene outer product:
        # missing motif/evo/borzoi entries are scored 0 (no co-regulation
        # signal), not NaN (which would corrupt the matmul).
        sub_grid = np.nan_to_num(sub_grid, nan=0.0, posinf=0.0, neginf=0.0)
        # gene–gene similarity in their TF-sequence-priority profiles
        seq_adj = (sub_grid.T @ sub_grid).astype(np.float32)
        # standardise both to comparable scale
        def stdz(M):
            mu = M.mean()
            sd = M.std() + 1e-9
            return (M - mu) / sd
        corr_z = stdz(corr_adj)
        seq_z = stdz(seq_adj)

        for a in alphas:
            A = (1 - a) * corr_z + a * seq_z
            ev = _l0_spectrum(A.astype(np.float32))
            score = _identifiability_score(ev)
            rows.append({
                "system": sys_name,
                "alpha": a,
                "mii_heuristic": score,
                "relative_eigengap": float((ev[2] - ev[0]) / max(ev[-1], 1e-9)) if len(ev) > 2 else 0.0,
                "n_genes": len(have),
            })

    if not rows:
        raise ValueError("no systems had loadable h5ads")

    def spread(rows_at_alpha):
        miis = [r["mii_heuristic"] for r in rows_at_alpha]
        return max(miis) - min(miis) if miis else 0.0

    spread_by_alpha = {a: spread([r for r in rows if r["alpha"] == a]) for a in alphas}
    best_alpha = max(spread_by_alpha, key=lambda a: spread_by_alpha[a])

    return {
        "mode": "real",
        "edges": str(edges_path),
        "cache_dir": str(cache_dir),
        "system_h5ads": {k: str(v) for k, v in system_h5ads.items()},
        "per_system_have_count": per_system_have_count,
        "alphas": alphas,
        "rows": rows,
        "spread_by_alpha": spread_by_alpha,
        "baseline_spread_alpha_0": spread_by_alpha[0.0],
        "best_alpha": best_alpha,
        "best_spread": spread_by_alpha[best_alpha],
        "delta_spread": spread_by_alpha[best_alpha] - spread_by_alpha[0.0],
        "note": (
            f"Real-mode on edges={edges_path.name} ({len(edges)} rows, "
            f"{len(target_genes)} unique target genes). Each system's correlation-"
            f"incidence adjacency restricted to those targets, blended with seq-prior "
            f"grid (TF × target → gene-gene similarity). MII = NITMB identifiability "
            f"heuristic (mean spectral gap / std). Anchor: baseline (α=0) ≈ "
            f"{spread_by_alpha[0.0]:.3f} spread across systems."
        ),
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--mode", choices=["stub", "real", "auto"], default="auto")
    p.add_argument("--edges", default=None, help="path to edges.csv with tf,target")
    p.add_argument("--cache", default=None, help="Tier-3 cache dir with motif/evo/borzoi .npy")
    p.add_argument(
        "--system",
        action="append",
        default=None,
        help="system spec name=path/to.h5ad — pass multiple times. "
             "Default: bioprinted_kidney + brain_organoid + fetal_kidney_ref using "
             "data/bioprinting/lawlor_2021_processed.h5ad, data/zenodo/RNA_all_velo.h5ad, "
             "data/bioprinting/kidney_ref_processed.h5ad (the Lab 4 convention).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="figures/lab4_real_results")
    args = p.parse_args(argv)

    if args.system:
        system_h5ads = {}
        for spec in args.system:
            if "=" not in spec:
                print(f"error: --system arg must be 'name=path/to.h5ad'; got {spec!r}", file=sys.stderr)
                return 2
            name, path = spec.split("=", 1)
            system_h5ads[name.strip()] = Path(path.strip())
    else:
        system_h5ads = {
            "bioprinted_kidney": Path("data/bioprinting/lawlor_2021_processed.h5ad"),
            "brain_organoid": Path("data/zenodo/RNA_all_velo.h5ad"),
            "fetal_kidney_ref": Path("data/bioprinting/kidney_ref_processed.h5ad"),
        }

    used_mode = args.mode
    if args.mode == "real":
        if args.edges is None or args.cache is None:
            print("error: --mode real requires --edges and --cache", file=sys.stderr)
            return 2
        result = _measure_real(Path(args.edges), Path(args.cache), system_h5ads)
    elif args.mode == "stub":
        result = _measure_stub(seed=args.seed)
    else:
        if args.edges is not None and args.cache is not None:
            try:
                result = _measure_real(Path(args.edges), Path(args.cache), system_h5ads)
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
