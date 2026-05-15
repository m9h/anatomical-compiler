"""measure_lab3_real — Lab 3 fidelity-triple transfer-r with FM priors swapped in.

Tier 4 of docs/dgx-verifier-runbook.md. The headline measurement the
project has been waiting on: does swapping FM (Geneformer + scGPT + UCE)
embeddings into Lab 3's perturbation-response predictor lift the
cross-context transfer-r above the 0.13 baseline?

Two arms:

  baseline   — predictor uses one-hot gene + raw cell-mean features.
  fm         — predictor uses Geneformer (gene features) + UCE (cell
               encoding), loaded from the Tier-3 cache.

Both train on context A and test on held-out context B. The headline
number is the difference: ``r_fm - r_baseline``.

Modes
-----
  stub   — synthetic 2-context regulome with stub FM features. Verifies
           the wrapper logic + reproduces a structure-aware floor.
  real   — load h5ad with .obs['context'] + .obs['perturbed_tf'] labels,
           load cache_dir/<stem>_{geneformer,uce}.npy, run the comparison.
  auto   — try real, fall back to stub with a warning.

Output: figures/lab3_real_results.{json,md}.

Usage
-----
    # Local stub-mode self-test:
    python scripts/measure_lab3_real.py --mode stub

    # DGX real-mode run (Tier-3 cache must exist):
    python scripts/measure_lab3_real.py --mode real \\
        --h5ad data/pollen.h5ad \\
        --cache cache/dgx_real_pollen_YYYYMMDD/

See also
--------
- docs/dgx-verifier-runbook.md Tier 4 — the runbook this implements
- docs/foundation-models.md step 5 — the original integration plan
- notebooks/03_benchmarking_fidelity.ipynb — the baseline 0.13 transfer-r number
- scripts/fm_embed.py — the upstream cache producer
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


# ----------------------------------------------------------------------------
# Stub mode — synthetic 2-context regulome with FM-stub features
# ----------------------------------------------------------------------------


def _make_stub_data(seed: int = 0):
    """Two contexts sharing a latent regulome, with controlled batch effects.

    The setup matches Lab 11's held-out-gene framing minus the rigging:
    here both arms can see every gene, but the predictor must generalise
    across the *context* axis (different cell-state distribution + batch
    noise). One-hot gene features still carry signal (per-gene memorisation
    of the response shape), but the FM-stub embedding's lower-rank
    representation should generalise more cleanly.
    """
    rng = np.random.default_rng(seed)
    n_genes, n_cells, n_factors = 200, 300, 4
    n_tfs = 20

    # Shared latent regulome: factor loadings (gene level)
    H = rng.standard_normal((n_factors, n_genes)).astype(np.float32)

    # Two contexts differ in cell-factor-score distribution + noise level
    def make_context(noise: float, baseline_shift: float, ctx_seed: int):
        r = np.random.default_rng(ctx_seed)
        W = r.standard_normal((n_cells, n_factors)).astype(np.float32)
        X = W @ H + baseline_shift + r.standard_normal((n_cells, n_genes)).astype(np.float32) * noise
        return X, W

    X_A, W_A = make_context(noise=0.4, baseline_shift=0.0, ctx_seed=seed + 1)
    X_B, W_B = make_context(noise=0.6, baseline_shift=0.3, ctx_seed=seed + 2)

    # True perturbation response per gene: -alpha * gene-gene Jacobian
    R_true = (-0.5 * (H.T @ H)).astype(np.float32)

    # FM-stub features (matches scripts/fm_embed.py stub backends):
    # Geneformer-stub: SVD of context-A expression -> 512-d per gene.
    U, s, Vt = np.linalg.svd(X_A - X_A.mean(axis=0, keepdims=True), full_matrices=False)
    geneformer_stub = (Vt.T * s).astype(np.float32)
    # UCE-stub: PCA + random projection on cells, 1280-d. Not directly used
    # for the gene-feature predictor below but written to cache for completeness.
    uce_stub = (U * s).astype(np.float32)

    return {
        "X_A": X_A, "X_B": X_B, "R_true": R_true,
        "geneformer_stub": geneformer_stub, "uce_stub": uce_stub,
        "n_genes": n_genes,
    }


# ----------------------------------------------------------------------------
# Predictor — ridge regression on gene features, scaled by per-context expression
# ----------------------------------------------------------------------------


def _fit_score(features: np.ndarray, X_train: np.ndarray, X_test: np.ndarray,
               R_true: np.ndarray, ridge: float = 1.0) -> float:
    """Train: features → standardised response; test against R_true on the
    flat (gene × perturbation) panel. Returns Pearson r.

    Returns 0.0 (with a flag) when the prediction is degenerate (one-hot
    features on held-out genes — same handling as Lab 11).
    """
    cell_mean_train = X_train.mean(axis=0)
    cell_mean_test = X_test.mean(axis=0)
    target = R_true / np.where(np.abs(cell_mean_train) < 1e-3, 1e-3, cell_mean_train)[None, :]

    A = features
    AtA = A.T @ A + ridge * np.eye(A.shape[1], dtype=np.float32)
    B = np.linalg.solve(AtA, A.T @ target)
    pred = (features @ B) * cell_mean_test[None, :]
    if pred.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(pred.ravel(), R_true.ravel())[0, 1])


def _measure_stub(seed: int = 0) -> dict:
    data = _make_stub_data(seed)
    one_hot = np.eye(data["n_genes"], dtype=np.float32)
    r_baseline = _fit_score(one_hot, data["X_A"], data["X_B"], data["R_true"])
    r_fm = _fit_score(data["geneformer_stub"], data["X_A"], data["X_B"], data["R_true"])
    return {
        "mode": "stub",
        "transfer_r_baseline": r_baseline,
        "transfer_r_fm": r_fm,
        "delta": r_fm - r_baseline,
        "note": (
            "Stub-mode floor demonstration. Real-mode numbers on the Pollen brain-organoid "
            "h5ad against the published 0.13 baseline are the answer the project's actually "
            "been waiting on; this stub just exercises the wrapper logic."
        ),
    }


# ----------------------------------------------------------------------------
# Real mode — load from h5ad + cache, do the same comparison on real data
# ----------------------------------------------------------------------------


def _build_response_panel(X_ctx: np.ndarray,
                          tf_labels: np.ndarray,
                          control_mask_ctx: np.ndarray,
                          tfs_use: list[str],
                          min_cells: int = 10) -> tuple[np.ndarray, list[str]]:
    """Per-gene mean KD response in a context.

    Returns
    -------
    panel : (n_genes, len(tfs_kept)) float32
        panel[g, j] = mean(X over cells with tf_labels == tfs_kept[j], gene g)
                    - mean(X over control cells, gene g)
    tfs_kept : list[str]
        TFs that had ≥ min_cells perturbed-cells in this context.
    """
    mu_ctrl = X_ctx[control_mask_ctx].mean(axis=0)
    panel_cols = []
    tfs_kept = []
    for tf in tfs_use:
        m = tf_labels == tf
        if m.sum() < min_cells:
            continue
        panel_cols.append(X_ctx[m].mean(axis=0) - mu_ctrl)
        tfs_kept.append(tf)
    if not tfs_kept:
        raise ValueError("no TFs had enough cells in this context for the response panel")
    return np.stack(panel_cols, axis=1).astype(np.float32), tfs_kept


def _fit_predict_score(features: np.ndarray,
                       R_train: np.ndarray,
                       R_test: np.ndarray,
                       ridge: float = 1.0,
                       _onehot: bool = False) -> float:
    """Ridge regression: features (n_genes × d) → R_train (n_genes × n_tfs);
    score predicted vs held-out R_test by flattened Pearson r.

    When `_onehot=True`, skip the explicit features @ B step: with A = I,
    pred = R_train / (1 + ridge), so the correlation collapses to
    corr(R_train, R_test) directly. Avoids O(n_genes²) memory allocation
    for the identity baseline at Pollen scale (~6 GB for 38 k genes).
    """
    if _onehot:
        if R_train.std() < 1e-8 or R_test.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(R_train.ravel(), R_test.ravel())[0, 1])
    A = features
    AtA = A.T @ A + ridge * np.eye(A.shape[1], dtype=np.float32)
    B = np.linalg.solve(AtA.astype(np.float32), A.T @ R_train.astype(np.float32))
    pred = features @ B
    if pred.std() < 1e-8 or R_test.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(pred.ravel(), R_test.ravel())[0, 1])


def _fm_to_gene_features(fm: np.ndarray, X: np.ndarray, n_genes: int, n_cells: int) -> np.ndarray:
    """Project an FM embedding to per-gene features.

    Supports two cache shapes:
      - (n_genes, dim) — already per-gene (Geneformer V1 token embeddings,
        stub geneformer); returned unchanged.
      - (n_cells, dim) — per-cell (Geneformer V2 cls, scGPT, UCE); projected
        to per-gene via cell-expression-weighted averaging:
          gene_features[g, :] = sum_c (X[c, g] * cell_emb[c, :]) / sum_c X[c, g]
        which is the centroid of the FM embedding over cells expressing gene g.

    Raises a clear error otherwise.
    """
    if fm.shape[0] == n_genes:
        return fm.astype(np.float32)
    if fm.shape[0] == n_cells:
        norm = X.sum(axis=0) + 1e-6                       # (n_genes,)
        gene_feat = (X.T @ fm.astype(np.float32))         # (n_genes, dim)
        return (gene_feat / norm[:, None]).astype(np.float32)
    raise ValueError(
        f"FM cache shape {fm.shape} matches neither n_genes={n_genes} "
        f"nor n_cells={n_cells} along axis 0. Re-extract or check the h5ad stem."
    )


def _measure_real(h5ad_path: Path, cache_dir: Path,
                  context_col: str = "timepoint",
                  perturb_col: str = "Gene_target_single",
                  perturb_status_col: str | None = "perturbation",
                  control_label: str | None = "NT",
                  na_label: str = "NA") -> dict:
    """Real-mode: Pollen-style schema.

    Default columns match `data/pollen/screen_annotated.h5ad`:
      - obs['timepoint']           — cross-context split (D0 vs D7)
      - obs['Gene_target_single']  — per-cell TF KD label ('NA' = no target)
      - obs['perturbation']        — 'WT' / 'NT' (non-targeting) / 'Perturbed'

    Cache shape handling:
      - geneformer/scgpt/uce caches can be per-cell (V2 cls / per-cell scGPT /
        UCE — the layout DGX actually produces) or per-gene (V1 token mode).
        `_fm_to_gene_features` projects per-cell embeddings to per-gene via
        expression-weighted averaging.

    Override via CLI flags for other datasets.
    """
    import anndata as ad

    adata = ad.read_h5ad(str(h5ad_path))
    stem = h5ad_path.stem
    geneformer_path = cache_dir / f"{stem}_geneformer.npy"
    uce_path = cache_dir / f"{stem}_uce.npy"
    scgpt_path = cache_dir / f"{stem}_scgpt.npy"
    if not geneformer_path.exists():
        raise FileNotFoundError(
            f"Tier-3 cache incomplete: need {geneformer_path}. "
            "Run scripts/run_fm_real_dgx.sh first."
        )
    geneformer = np.load(geneformer_path)
    n_genes = adata.shape[1]
    n_cells = adata.shape[0]
    if geneformer.shape[0] not in (n_genes, n_cells):
        raise ValueError(
            f"Geneformer cache shape {geneformer.shape} matches neither n_genes={n_genes} "
            f"nor n_cells={n_cells} along axis 0."
        )
    has_uce = uce_path.exists()
    has_scgpt = scgpt_path.exists()
    uce = np.load(uce_path) if has_uce else None
    scgpt = np.load(scgpt_path) if has_scgpt else None

    for col in (context_col, perturb_col):
        if col not in adata.obs.columns:
            raise ValueError(
                f"h5ad.obs is missing required column '{col}'. "
                f"Available: {list(adata.obs.columns)}"
            )

    contexts = list(adata.obs[context_col].unique())
    if len(contexts) < 2:
        raise ValueError(f"need ≥2 unique values in obs['{context_col}']; got {contexts}")
    ctx_a, ctx_b = contexts[0], contexts[1]
    mask_a = (adata.obs[context_col] == ctx_a).to_numpy()
    mask_b = (adata.obs[context_col] == ctx_b).to_numpy()

    # Control cell mask: prefer 'NT' rows in perturb_status_col if present,
    # else fall back to perturb_col == na_label (e.g., 'NA' or 'WT').
    if perturb_status_col is not None and perturb_status_col in adata.obs.columns and control_label is not None:
        control_mask = (adata.obs[perturb_status_col] == control_label).to_numpy()
    else:
        control_mask = (adata.obs[perturb_col] == na_label).to_numpy()
    if control_mask.sum() < 20:
        raise ValueError(
            f"<20 control cells found (looked for {perturb_status_col}=={control_label} "
            f"or {perturb_col}=={na_label}). Total controls: {int(control_mask.sum())}."
        )

    tfs_use = sorted(
        t for t in adata.obs[perturb_col].unique()
        if isinstance(t, str) and t not in (na_label, "WT") and "," not in t
    )

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X_A = X[mask_a].astype(np.float32)
    X_B = X[mask_b].astype(np.float32)
    R_A, tfs_kept_A = _build_response_panel(
        X_A, adata.obs[perturb_col].to_numpy()[mask_a],
        control_mask[mask_a], tfs_use,
    )
    R_B, tfs_kept_B = _build_response_panel(
        X_B, adata.obs[perturb_col].to_numpy()[mask_b],
        control_mask[mask_b], tfs_use,
    )
    tfs_common = [t for t in tfs_kept_A if t in tfs_kept_B]
    if not tfs_common:
        raise ValueError("no TFs had ≥10 perturbed cells in both contexts")
    idx_A = np.array([tfs_kept_A.index(t) for t in tfs_common])
    idx_B = np.array([tfs_kept_B.index(t) for t in tfs_common])
    R_A = R_A[:, idx_A]
    R_B = R_B[:, idx_B]

    # One-hot baseline: closed-form pred = R_train / (1+ridge); we never
    # materialise the n_genes × n_genes identity (would be ~6 GB on Pollen).
    r_baseline = _fit_predict_score(np.empty((0, 0)), R_A, R_B, _onehot=True)
    gene_geneformer = _fm_to_gene_features(geneformer, X, n_genes, n_cells)
    r_fm = _fit_predict_score(gene_geneformer, R_A, R_B)

    extras: dict[str, Any] = {
        "geneformer_cache_shape": list(geneformer.shape),
        "geneformer_was_per_cell": bool(geneformer.shape[0] == n_cells),
    }
    if has_uce:
        gene_uce = _fm_to_gene_features(uce, X, n_genes, n_cells)
        feat_combined = np.concatenate([gene_geneformer, gene_uce], axis=1)
        extras["transfer_r_fm_geneformer_plus_uce"] = _fit_predict_score(
            feat_combined, R_A, R_B
        )
        extras["uce_cache_shape"] = list(uce.shape)
    if has_scgpt:
        gene_scgpt = _fm_to_gene_features(scgpt, X, n_genes, n_cells)
        extras["transfer_r_fm_scgpt"] = _fit_predict_score(gene_scgpt, R_A, R_B)
        extras["scgpt_cache_shape"] = list(scgpt.shape)
        # Combined Geneformer + scGPT + UCE if all present
        if has_uce:
            feat_all = np.concatenate(
                [gene_geneformer, gene_scgpt, _fm_to_gene_features(uce, X, n_genes, n_cells)],
                axis=1,
            )
            extras["transfer_r_fm_all"] = _fit_predict_score(feat_all, R_A, R_B)

    return {
        "mode": "real",
        "h5ad": str(h5ad_path),
        "cache_dir": str(cache_dir),
        "context_col": context_col,
        "context_a": str(ctx_a),
        "context_b": str(ctx_b),
        "n_genes": int(n_genes),
        "n_cells_a": int(mask_a.sum()),
        "n_cells_b": int(mask_b.sum()),
        "n_controls": int(control_mask.sum()),
        "tfs_evaluated": tfs_common,
        "n_tfs_evaluated": len(tfs_common),
        "transfer_r_baseline": r_baseline,
        "transfer_r_fm": r_fm,
        "delta": r_fm - r_baseline,
        **extras,
        "note": (
            f"Real-mode on {h5ad_path.name} ({n_genes} genes, "
            f"{mask_a.sum()}+{mask_b.sum()} cells across {ctx_a}/{ctx_b}). "
            f"R_train computed on context {ctx_a}, R_test on {ctx_b}. "
            f"Lab 3 baseline anchor: transfer-r ≈ 0.13 — delta should be read against that."
        ),
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--mode", choices=["stub", "real", "auto"], default="auto")
    p.add_argument("--h5ad", default=None, help="path to .h5ad with context + perturbation labels")
    p.add_argument("--cache", default=None, help="Tier-3 cache directory with FM .npy")
    p.add_argument("--context-col", default="timepoint", help="obs column for cross-context split")
    p.add_argument("--perturb-col", default="Gene_target_single", help="obs column for per-cell TF KD label")
    p.add_argument("--perturb-status-col", default="perturbation", help="obs column for control flag (set empty to skip)")
    p.add_argument("--control-label", default="NT", help="value in --perturb-status-col indicating controls")
    p.add_argument("--na-label", default="NA", help="value in --perturb-col indicating no target (fallback control)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="figures/lab3_real_results")
    args = p.parse_args(argv)

    real_kwargs = dict(
        context_col=args.context_col,
        perturb_col=args.perturb_col,
        perturb_status_col=args.perturb_status_col or None,
        control_label=args.control_label or None,
        na_label=args.na_label,
    )

    used_mode = args.mode
    if args.mode == "real":
        if args.h5ad is None or args.cache is None:
            print("error: --mode real requires --h5ad and --cache", file=sys.stderr)
            return 2
        result = _measure_real(Path(args.h5ad), Path(args.cache), **real_kwargs)
    elif args.mode == "stub":
        result = _measure_stub(seed=args.seed)
    else:  # auto
        if args.h5ad is not None and args.cache is not None:
            try:
                result = _measure_real(Path(args.h5ad), Path(args.cache), **real_kwargs)
                used_mode = "real"
            except (FileNotFoundError, NotImplementedError, ValueError) as e:
                warnings.warn(f"falling back to stub mode: {e}")
                result = _measure_stub(seed=args.seed)
                used_mode = "stub"
        else:
            result = _measure_stub(seed=args.seed)
            used_mode = "stub"

    print(f"\nmeasure_lab3_real ({used_mode} mode):")
    print(f"  transfer_r_baseline = {result['transfer_r_baseline']:+.3f}")
    print(f"  transfer_r_fm       = {result['transfer_r_fm']:+.3f}")
    print(f"  Δ                   = {result['delta']:+.3f}")
    print(f"  (Lab 3's published baseline = 0.13; FM lift = r_fm - 0.13)")

    out_json = Path(args.output + ".json")
    out_md = Path(args.output + ".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = [
        "# Lab 3 — fidelity-triple transfer-r with FM priors",
        "",
        f"**Mode: {used_mode}.**",
        "",
        "| arm | transfer-r |",
        "|---|---:|",
        f"| baseline (one-hot / learned-from-scratch) | {result['transfer_r_baseline']:+.3f} |",
        f"| FM-augmented (Geneformer features) | {result['transfer_r_fm']:+.3f} |",
        f"| **Δ** | **{result['delta']:+.3f}** |",
        "",
        f"_{result.get('note', '')}_" if result.get("note") else "",
        "",
        "## Interpretation",
        "",
        "The published Lab 3 baseline is **transfer-r ≈ 0.13** "
        "(`notebooks/03_benchmarking_fidelity.ipynb`). The FM-augmented arm using "
        "Geneformer gene embeddings (and optionally UCE cell embeddings) should lift this "
        "if the 30 M-cell pretraining carries co-regulation structure that the one-hot "
        "baseline can't access. A positive Δ in real-mode is the empirical answer to "
        "_'do FMs make a difference on the project's actual numbers'_.",
        "",
        "See also: [`docs/dgx-verifier-runbook.md`](../docs/dgx-verifier-runbook.md) Tier 4; "
        "[`docs/foundation-models.md`](../docs/foundation-models.md) step 5; "
        "[`notebooks/11_foundation_model_pipeline.ipynb`](../notebooks/11_foundation_model_pipeline.ipynb).",
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {out_json}")
    print(f"  wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
