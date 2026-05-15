"""measure_lab6_real — Lab 6 high-leverage TF ranking vs scGPT in-silico KD.

Tier 4 of docs/dgx-verifier-runbook.md. Does scGPT's zero-shot in-silico
KD prediction agree with Lab 6's controllability-derived high-leverage
TF ranking? Spearman ρ + top-10 set overlap.

The disagreement is the EIG signal for the wet-lab BO loop (step 4 of
docs/foundation-models.md): TFs where the two methods *disagree* are the
ones where a wet-lab measurement carries the most information per query.

Modes
-----
  stub   — synthetic regulome with two independent rankings of known
           ground truth; checks the wrapper math.
  real   — load h5ad + cache_dir/<stem>_scgpt_perturb.npy, compute
           Jacobian-based controllability ranking, compare.
  auto   — try real, fall back to stub.

Output: figures/lab6_real_results.{json,md}.

Usage
-----
    python scripts/measure_lab6_real.py --mode stub
    python scripts/measure_lab6_real.py --mode real \\
        --h5ad data/pollen.h5ad \\
        --cache cache/dgx_real_pollen_YYYYMMDD/ \\
        --tfs data/tfs.txt

See also
--------
- docs/dgx-verifier-runbook.md Tier 4
- scripts/ablate_perturb_eig.py — the stub-mode BO/EIG ablation (this is real-mode)
- scripts/fm_perturb_scgpt.py — the upstream scGPT KD extractor
- notebooks/06_control_theory.ipynb — the Jacobian/controllability baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# Ranking-agreement metrics
# ----------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D arrays. Plain numpy."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    n = len(a)
    return float(1.0 - 6.0 * np.sum((ra - rb) ** 2) / (n * (n * n - 1)))


def _top_k_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Jaccard overlap between top-k indices of |a| and |b|."""
    top_a = set(np.argsort(-np.abs(a))[:k])
    top_b = set(np.argsort(-np.abs(b))[:k])
    if not (top_a or top_b):
        return 0.0
    return len(top_a & top_b) / len(top_a | top_b)


def _eig_rank_acquisition(jac_imp: np.ndarray, scgpt_imp: np.ndarray, budget: int) -> list[int]:
    """The same EIG-rank acquisition function from ablate_perturb_eig.py:
    return the top-budget TFs by rank disagreement between the two predictors."""
    jac_rank = np.argsort(np.argsort(jac_imp))
    scgpt_rank = np.argsort(np.argsort(scgpt_imp))
    rank_disagreement = np.abs(jac_rank - scgpt_rank)
    return [int(i) for i in np.argsort(-rank_disagreement)[:budget]]


# ----------------------------------------------------------------------------
# Stub mode — synthetic two-ranking comparison with known truth
# ----------------------------------------------------------------------------


def _measure_stub(seed: int = 0) -> dict:
    """Build a synthetic with known TF importance, plus two noisy predictors
    (Jacobian-like + scGPT-like). Realistic-regime tuning matches
    ablate_perturb_eig.py: baseline prior ρ ≈ 0.56."""
    rng = np.random.default_rng(seed)
    n_tfs = 30

    true_importance = rng.uniform(0.5, 4.0, n_tfs).astype(np.float32)

    # Jacobian-style noisy estimator: standardised + Gaussian noise.
    jac_imp = true_importance + rng.standard_normal(n_tfs).astype(np.float32) * 1.2

    # scGPT-style: correlated with truth but with different error profile
    # (Laplace tails vs Jacobian's Gaussian), modelling decorrelated errors.
    scgpt_imp = true_importance + rng.laplace(scale=0.8, size=n_tfs).astype(np.float32)

    rho = _spearman(jac_imp, scgpt_imp)
    overlap_10 = _top_k_overlap(jac_imp, scgpt_imp, k=10)
    overlap_20 = _top_k_overlap(jac_imp, scgpt_imp, k=20)
    rho_jac_truth = _spearman(jac_imp, true_importance)
    rho_scgpt_truth = _spearman(scgpt_imp, true_importance)
    eig_top_5 = _eig_rank_acquisition(jac_imp, scgpt_imp, budget=5)

    return {
        "mode": "stub",
        "n_tfs": n_tfs,
        "spearman_jac_vs_scgpt": rho,
        "top10_overlap_jaccard": overlap_10,
        "top20_overlap_jaccard": overlap_20,
        "spearman_jac_vs_truth": rho_jac_truth,
        "spearman_scgpt_vs_truth": rho_scgpt_truth,
        "eig_top_5_indices": eig_top_5,
        "note": (
            "Stub-mode synthetic, ablate_perturb_eig.py realistic-regime parity (baseline "
            "prior ρ ≈ 0.56). Spearman(jac, scgpt) measures cross-method agreement; the "
            "EIG-rank top-5 are the wet-lab queries with highest expected information gain. "
            "Real-mode against Pollen + scgpt_perturb cache is the actual deliverable."
        ),
    }


# ----------------------------------------------------------------------------
# Real mode — load h5ad + cache, compute Jacobian rank, compare with scGPT KD
# ----------------------------------------------------------------------------


def _jacobian_importance_from_h5ad(adata, tfs: list[str]) -> tuple[np.ndarray, list[str]]:
    """Estimate per-TF controllability importance from observational
    expression. Returns ridge-regression L2 norms aligned to *the subset of
    tfs actually present in adata*.

    Lab 6's full method: ridge regression of every gene on all TF
    expression vectors, then ‖B‖₂ per TF as the controllability index.
    Adds the *aligned* TF list as the second return so callers can match
    the cache to a possibly-smaller TF subset.
    """
    if "tf_activity" in adata.obsm:
        A = np.asarray(adata.obsm["tf_activity"], dtype=np.float32)
        # obsm['tf_activity'] columns are assumed to match `tfs` order; if
        # not, the caller is responsible for aligning.
        tfs_kept = list(tfs)
    else:
        var_names = list(adata.var.index)
        tfs_kept = [t for t in tfs if t in var_names]
        if not tfs_kept:
            raise ValueError(
                f"None of the {len(tfs)} cache TFs are in adata.var.index. "
                f"First 5 sought: {tfs[:5]}. First 5 var.index: {var_names[:5]}. "
                "Either supply adata.obsm['tf_activity'] (n_cells × n_tfs aligned to tfs.txt) "
                "or re-extract the cache against this h5ad's gene set."
            )
        tf_idx = [var_names.index(t) for t in tfs_kept]
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        A = X[:, tf_idx].astype(np.float32)
    Y = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    Y = Y.astype(np.float32)
    n_tfs_kept = A.shape[1]
    AtA = A.T @ A + 1.0 * np.eye(n_tfs_kept, dtype=np.float32)
    B = np.linalg.solve(AtA, A.T @ Y)
    return np.linalg.norm(B, axis=1), tfs_kept


def _measure_real(h5ad_path: Path, cache_dir: Path, tfs_path: Path) -> dict:
    """Real-mode: load h5ad + scgpt_perturb cache, compute Jacobian importance,
    compare rankings.

    The two rankings come from independent constructions:
      - jac_imp[i] = ‖B[i, :]‖₂ where B = ridge(Y ~ TF_expr) on adata
      - scgpt_imp[i] = ‖scgpt_perturb[i, :]‖₂ — scGPT zero-shot KD magnitude

    Mismatches in TF presence: the cache may list 44 TFs but only 38 are
    expressed in adata.var.index. We intersect and compare on the
    intersection (reported as `n_tfs_compared`).
    """
    import anndata as ad

    adata = ad.read_h5ad(str(h5ad_path))

    with open(tfs_path, encoding="utf-8") as f:
        tfs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    stem = h5ad_path.stem
    scgpt_path = cache_dir / f"{stem}_scgpt_perturb.npy"
    if not scgpt_path.exists():
        raise FileNotFoundError(
            f"Tier-3 cache incomplete: missing {scgpt_path}. Run scripts/run_fm_real_dgx.sh first."
        )
    scgpt_perturb = np.load(scgpt_path)  # shape (n_tfs_cache, n_genes_cache)
    if scgpt_perturb.shape[0] != len(tfs):
        raise ValueError(
            f"scgpt_perturb shape {scgpt_perturb.shape} doesn't match tfs.txt length "
            f"{len(tfs)}. Were they extracted with the same TF list?"
        )
    scgpt_imp_full = np.linalg.norm(scgpt_perturb, axis=1)

    jac_imp_kept, tfs_kept = _jacobian_importance_from_h5ad(adata, tfs)
    keep_idx = np.array([tfs.index(t) for t in tfs_kept])
    scgpt_imp = scgpt_imp_full[keep_idx]
    assert len(jac_imp_kept) == len(scgpt_imp) == len(tfs_kept)

    rho = _spearman(jac_imp_kept, scgpt_imp)
    overlap_10 = _top_k_overlap(jac_imp_kept, scgpt_imp, k=10)
    overlap_20 = _top_k_overlap(jac_imp_kept, scgpt_imp, k=20)
    eig_top_5 = _eig_rank_acquisition(jac_imp_kept, scgpt_imp, budget=5)
    eig_top_10 = _eig_rank_acquisition(jac_imp_kept, scgpt_imp, budget=min(10, len(tfs_kept)))

    return {
        "mode": "real",
        "h5ad": str(h5ad_path),
        "cache_dir": str(cache_dir),
        "tfs": str(tfs_path),
        "n_tfs_in_cache": len(tfs),
        "n_tfs_compared": len(tfs_kept),
        "tfs_compared": tfs_kept,
        "spearman_jac_vs_scgpt": rho,
        "top10_overlap_jaccard": overlap_10,
        "top20_overlap_jaccard": overlap_20,
        "eig_top_5_tfs": [tfs_kept[i] for i in eig_top_5],
        "eig_top_10_tfs": [tfs_kept[i] for i in eig_top_10],
        "jac_importance_by_tf": {tfs_kept[i]: float(jac_imp_kept[i]) for i in range(len(tfs_kept))},
        "scgpt_importance_by_tf": {tfs_kept[i]: float(scgpt_imp[i]) for i in range(len(tfs_kept))},
        "note": (
            f"Real-mode on {h5ad_path.name} ({len(tfs_kept)}/{len(tfs)} TFs survived "
            f"intersection with adata.var.index). The EIG-rank top-5/10 are the wet-lab "
            f"BO loop's next-query candidates (per docs/wetlab-program.md). "
            f"Anchor: stub-mode realistic-regime ρ ≈ 0.56 (see ablate_perturb_eig.py)."
        ),
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--mode", choices=["stub", "real", "auto"], default="auto")
    p.add_argument("--h5ad", default=None)
    p.add_argument("--cache", default=None)
    p.add_argument("--tfs", default=None, help="tfs.txt — one TF symbol per line")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="figures/lab6_real_results")
    args = p.parse_args(argv)

    used_mode = args.mode
    if args.mode == "real":
        if args.h5ad is None or args.cache is None or args.tfs is None:
            print("error: --mode real requires --h5ad, --cache, --tfs", file=sys.stderr)
            return 2
        result = _measure_real(Path(args.h5ad), Path(args.cache), Path(args.tfs))
    elif args.mode == "stub":
        result = _measure_stub(seed=args.seed)
    else:
        if all(x is not None for x in (args.h5ad, args.cache, args.tfs)):
            try:
                result = _measure_real(Path(args.h5ad), Path(args.cache), Path(args.tfs))
                used_mode = "real"
            except (FileNotFoundError, NotImplementedError, ValueError) as e:
                warnings.warn(f"falling back to stub mode: {e}")
                result = _measure_stub(seed=args.seed)
                used_mode = "stub"
        else:
            result = _measure_stub(seed=args.seed)
            used_mode = "stub"

    print(f"\nmeasure_lab6_real ({used_mode} mode):")
    print(f"  Spearman(jac, scgpt) = {result['spearman_jac_vs_scgpt']:+.3f}")
    print(f"  top-10 Jaccard       = {result['top10_overlap_jaccard']:.3f}")
    print(f"  top-20 Jaccard       = {result['top20_overlap_jaccard']:.3f}")
    if used_mode == "real":
        print(f"  EIG top-5 TFs        = {result['eig_top_5_tfs']}")
    else:
        print(f"  EIG top-5 indices    = {result['eig_top_5_indices']}")

    out_json = Path(args.output + ".json")
    out_md = Path(args.output + ".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = [
        "# Lab 6 — high-leverage TF ranking vs scGPT in-silico KD",
        "",
        f"**Mode: {used_mode}.**",
        "",
        "Two independent rankings of the same TF set, scored by Spearman ρ and "
        "top-K Jaccard overlap:",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| Spearman(jac, scgpt) | **{result['spearman_jac_vs_scgpt']:+.3f}** |",
        f"| top-10 Jaccard overlap | {result['top10_overlap_jaccard']:.3f} |",
        f"| top-20 Jaccard overlap | {result['top20_overlap_jaccard']:.3f} |",
        "",
        "## EIG-rank wet-lab queries",
        "",
        "The top-K TFs by *disagreement* between the two predictors. These are the "
        "highest-expected-information-gain wet-lab queries — running them resolves the "
        "ambiguity that the disagreement encodes.",
        "",
    ]
    if used_mode == "real":
        md += ["**Top 5 (real):**", ""]
        for tf in result["eig_top_5_tfs"]:
            md.append(f"- `{tf}`")
        md += ["", "**Top 10 (real):**", ""]
        for tf in result["eig_top_10_tfs"]:
            md.append(f"- `{tf}`")
    else:
        md.append(f"_Stub mode — TF indices only:_ {result['eig_top_5_indices']}")
    md += [
        "",
        f"_{result.get('note', '')}_" if result.get("note") else "",
        "",
        "## Interpretation",
        "",
        "Lab 6's controllability ranking is *linear* (Jacobian of the LTI surrogate "
        "around an attractor; ‖B‖₂ per-TF). scGPT's in-silico KD is *nonlinear* (a "
        "30 M-cell-pretrained transformer predicting post-perturbation state). Where they "
        "agree (high Spearman), confidence is high. Where they disagree, the wet lab "
        "resolves the inconsistency — that's the BO acquisition signal "
        "`scripts/ablate_perturb_eig.py` quantified in stub-mode (EIG-rank beats GREEDY "
        "by +0.029 Spearman in the realistic regime). This script is the real-mode answer.",
        "",
        "See also: [`docs/dgx-verifier-runbook.md`](../docs/dgx-verifier-runbook.md) Tier 4; "
        "[`docs/wetlab-program.md`](../docs/wetlab-program.md) — the cycles these EIG queries inform; "
        "[`scripts/ablate_perturb_eig.py`](../scripts/ablate_perturb_eig.py) — the stub.",
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {out_json}")
    print(f"  wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
