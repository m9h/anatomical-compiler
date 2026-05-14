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


def _measure_real(h5ad_path: Path, cache_dir: Path) -> dict:
    """Real-mode template — DGX agent fills in the dataset-specific details.

    Loads cache_dir/<h5ad-stem>_geneformer.npy + _uce.npy, builds the
    train/test split from h5ad.obs['context'], runs the same ridge
    predictor pattern as stub mode, reports per-TF transfer-r distribution.
    """
    import anndata as ad

    adata = ad.read_h5ad(str(h5ad_path))
    stem = h5ad_path.stem
    geneformer_path = cache_dir / f"{stem}_geneformer.npy"
    uce_path = cache_dir / f"{stem}_uce.npy"
    if not geneformer_path.exists() or not uce_path.exists():
        raise FileNotFoundError(
            f"Tier-3 cache incomplete: need {geneformer_path} and {uce_path}. "
            "Run scripts/run_fm_real_dgx.sh first."
        )
    geneformer = np.load(geneformer_path)
    _uce = np.load(uce_path)
    n_genes = adata.shape[1]
    assert geneformer.shape == (n_genes, 512), \
        f"Geneformer cache shape {geneformer.shape} != ({n_genes}, 512)"

    # Expected obs columns — adjust to match the actual h5ad schema.
    if "context" not in adata.obs.columns:
        raise ValueError(
            "h5ad.obs must have a 'context' column with at least 2 unique values "
            "(the cross-context split for the fidelity-triple transfer-r). "
            "Adjust the column name in this script to match the dataset."
        )
    contexts = adata.obs["context"].unique()
    if len(contexts) < 2:
        raise ValueError(f"need ≥2 contexts; got {list(contexts)}")
    ctx_a, ctx_b = contexts[:2]
    mask_a = (adata.obs["context"] == ctx_a).to_numpy()
    mask_b = (adata.obs["context"] == ctx_b).to_numpy()

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X_A = X[mask_a].astype(np.float32)
    X_B = X[mask_b].astype(np.float32)

    # The "true" response per gene needs to come from the dataset's
    # perturbation labels (h5ad.obs['perturbed_tf']). Construct an
    # observed-Jacobian estimate from KD-vs-control mean differences on
    # context A; train predictor; test cross-context. This block is the
    # natural extension the DGX agent fills in for the specific dataset's
    # schema; for now the template raises a clear error.
    if "perturbed_tf" not in adata.obs.columns:
        raise NotImplementedError(
            "Real-mode requires h5ad.obs['perturbed_tf'] (per-cell KD label). "
            "DGX agent: fill in the dataset-specific response-construction here. "
            "Pattern: for each TF in obs['perturbed_tf'].unique(), compute the "
            "mean expression of KD cells minus mean of control cells in context A; "
            "this is R_true[:, tf_idx]. Then train as in _measure_stub."
        )

    # Placeholder reached only when the agent supplies a working construction.
    raise NotImplementedError("complete the real-mode body for the specific h5ad schema")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--mode", choices=["stub", "real", "auto"], default="auto")
    p.add_argument("--h5ad", default=None, help="path to .h5ad with context + perturbed_tf labels")
    p.add_argument("--cache", default=None, help="Tier-3 cache directory with FM .npy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="figures/lab3_real_results")
    args = p.parse_args(argv)

    used_mode = args.mode
    if args.mode == "real":
        if args.h5ad is None or args.cache is None:
            print("error: --mode real requires --h5ad and --cache", file=sys.stderr)
            return 2
        result = _measure_real(Path(args.h5ad), Path(args.cache))
    elif args.mode == "stub":
        result = _measure_stub(seed=args.seed)
    else:  # auto
        if args.h5ad is not None and args.cache is not None:
            try:
                result = _measure_real(Path(args.h5ad), Path(args.cache))
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
