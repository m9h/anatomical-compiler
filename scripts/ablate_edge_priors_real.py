"""ablate_edge_priors_real — real-mode counterpart to ablate_edge_priors.py.

The synthetic ablation answers "does a stub sequence-prior help if we knew
the ground truth?" — useful as a ceiling. This script answers the harder
question: **on the actual Pollen regulome, does a real motif PWM scan add
F1 over a Pando-style expression-correlation baseline?**

Truth set: the binary `data/pollen/processed/incidence.npy` regulome
(44 TFs × 2739 target genes; 32k positives, 88k true negatives).
Pando baseline: per-(TF, target) |Pearson(TF_expr, target_expr)| across the
Pollen slice expression matrix.
Sequence prior: cached motif PSSM scores from
`cache/real_run/pollen_edges_full_motif.npy` (full 120k candidate set).
F1 score: top-K threshold where K = number of positives.

Output: figures/edge_prior_ablation_real.{json,md}

Tier 4 of docs/dgx-verifier-runbook.md — the real-data answer to the
edge-prior ablation row, complementing the synthetic ablation in
figures/edge_prior_ablation.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np


def _f1_at_k(scores: np.ndarray, truth: np.ndarray, k: int) -> tuple[float, float, float]:
    """Top-K F1. Returns (precision, recall, f1)."""
    # NaN scores rank last (treated as -inf)
    s = np.where(np.isfinite(scores), scores, -np.inf)
    top_k_idx = np.argpartition(-s, k - 1)[:k]
    pred = np.zeros_like(truth, dtype=bool)
    pred[top_k_idx] = True
    tp = int(((pred) & (truth.astype(bool))).sum())
    fp = int(((pred) & (~truth.astype(bool))).sum())
    fn = int(((~pred) & (truth.astype(bool))).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1


def _stdz(x: np.ndarray) -> np.ndarray:
    """Standardize ignoring NaNs."""
    m = np.nanmean(x)
    s = np.nanstd(x) + 1e-9
    z = (x - m) / s
    return np.where(np.isfinite(z), z, 0.0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--edges", default="data/pollen_edges_full.csv",
                   help="CSV with tf,target,truth columns (full candidate set)")
    p.add_argument("--motif-cache", default="cache/real_run/pollen_edges_full_motif.npy",
                   help="motif scores for each row in --edges (same order)")
    p.add_argument("--h5ad", default="data/pollen_slice_geneformer.h5ad",
                   help="expression matrix for the Pando baseline")
    p.add_argument("--output", default="figures/edge_prior_ablation_real")
    args = p.parse_args(argv)

    import pandas as pd

    print("loading edges + truth…", file=sys.stderr)
    edges = pd.read_csv(args.edges)
    if not {"tf", "target", "truth"}.issubset(edges.columns):
        print("error: edges CSV must have tf, target, truth columns", file=sys.stderr)
        return 2
    truth = edges["truth"].to_numpy().astype(np.int8)
    n_pos = int(truth.sum())
    print(f"  {len(edges)} candidates; {n_pos} positives, {len(edges) - n_pos} negatives",
          file=sys.stderr)

    print("loading motif scores…", file=sys.stderr)
    motif = np.load(args.motif_cache)
    if len(motif) != len(edges):
        raise ValueError(
            f"motif cache length {len(motif)} != edges {len(edges)}"
        )
    n_motif_finite = int(np.isfinite(motif).sum())
    print(f"  {n_motif_finite}/{len(motif)} motif-scored", file=sys.stderr)

    print("loading expression for Pando baseline…", file=sys.stderr)
    import anndata as ad
    adata = ad.read_h5ad(args.h5ad)
    var_index = {name: i for i, name in enumerate(adata.var_names)}
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float32)

    print(f"  pollen slice: {adata.shape}", file=sys.stderr)

    # Compute Pando baseline: |Pearson(TF_expr, target_expr)| per candidate
    print("computing Pando baseline (per-pair |Pearson|)…", file=sys.stderr)
    pando = np.full(len(edges), np.nan, dtype=np.float32)
    n_pando_finite = 0
    # Vectorize per TF: standardize TF expression once, then for all targets
    tf_unique = edges["tf"].unique()
    tf_z_cache: dict[str, np.ndarray] = {}
    for tf in tf_unique:
        if tf not in var_index:
            continue
        col = X[:, var_index[tf]].astype(np.float64)
        std = col.std()
        if std < 1e-8:
            continue
        tf_z_cache[tf] = (col - col.mean()) / std
    # For each row, look up TF and target expressions
    rows_by_tf = edges.groupby("tf")
    for tf, group in rows_by_tf:
        if tf not in tf_z_cache:
            continue
        tf_z = tf_z_cache[tf]
        for idx, target in zip(group.index, group["target"]):
            if target not in var_index:
                continue
            col = X[:, var_index[target]].astype(np.float64)
            std = col.std()
            if std < 1e-8:
                continue
            tg_z = (col - col.mean()) / std
            r = float(np.dot(tf_z, tg_z) / len(tf_z))
            pando[idx] = abs(r)
            n_pando_finite += 1
    print(f"  Pando-scored {n_pando_finite}/{len(edges)}", file=sys.stderr)

    # F1 at top-K with K = n_positives
    p_pando, r_pando, f1_pando = _f1_at_k(pando, truth, n_pos)
    p_motif, r_motif, f1_motif = _f1_at_k(motif, truth, n_pos)

    # Blended ranking — sweep α
    pando_z = _stdz(pando)
    motif_z = _stdz(motif)
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    f1_combined = []
    for a in alphas:
        combined = (1.0 - a) * pando_z + a * motif_z
        _, _, f1 = _f1_at_k(combined, truth, n_pos)
        f1_combined.append(f1)
    best_idx = int(np.argmax(f1_combined))
    best_alpha = alphas[best_idx]
    best_f1 = f1_combined[best_idx]
    delta = best_f1 - f1_pando

    if delta > 0.005:
        verdict = "KEEP — sequence prior (motif) adds F1 over Pando alone"
    elif delta < -0.005:
        verdict = "DROP — motif underperforms Pando alone (sign of negative lift)"
    else:
        verdict = "NEUTRAL — no meaningful gain or loss from blending motif"

    result = {
        "n_candidates": len(edges),
        "n_positives": n_pos,
        "n_negatives": len(edges) - n_pos,
        "n_motif_finite": n_motif_finite,
        "n_pando_finite": n_pando_finite,
        "f1_pando_only": f1_pando,
        "p_pando": p_pando,
        "r_pando": r_pando,
        "f1_motif_only": f1_motif,
        "p_motif": p_motif,
        "r_motif": r_motif,
        "alpha_grid": alphas,
        "f1_combined": f1_combined,
        "best_alpha": best_alpha,
        "best_f1_combined": best_f1,
        "delta_vs_pando": delta,
        "verdict": verdict,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_json = out.with_suffix(".json")
    out_md = out.with_suffix(".md")
    out_json.write_text(json.dumps(result, indent=2))

    md = [
        "# Edge-prior ablation (real-mode)",
        "",
        f"- Truth: {n_pos:,} positive edges / {len(edges) - n_pos:,} negatives (44 TFs × 2,739 targets)",
        f"- Motif scored: {n_motif_finite:,}/{len(edges):,}",
        f"- Pando scored: {n_pando_finite:,}/{len(edges):,}",
        "",
        "## F1 at top-K (K = n_positives)",
        "",
        f"- F1 (Pando alone)  = {f1_pando:.4f}  (P={p_pando:.3f}, R={r_pando:.3f})",
        f"- F1 (motif alone)  = {f1_motif:.4f}  (P={p_motif:.3f}, R={r_motif:.3f})",
        f"- F1 (best blend)   = {best_f1:.4f}  at α = {best_alpha}",
        f"- Δ over Pando      = {delta:+.4f}",
        "",
        f"**Verdict: {verdict}**",
        "",
        "## α sweep",
        "",
        "| α | F1 |",
        "|---|---|",
    ]
    for a, f in zip(alphas, f1_combined):
        md.append(f"| {a:.1f} | {f:.4f} |")
    out_md.write_text("\n".join(md))

    print()
    print(f"ablate_edge_priors_real:")
    print(f"  F1 (Pando alone)  = {f1_pando:.4f}")
    print(f"  F1 (motif alone)  = {f1_motif:.4f}")
    print(f"  best α            = {best_alpha}")
    print(f"  best F1 (blend)   = {best_f1:.4f}")
    print(f"  Δ over Pando      = {delta:+.4f}")
    print(f"  verdict           = {verdict}")
    print(f"  wrote {out_json}")
    print(f"  wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
