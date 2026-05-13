"""fm_perturb_scgpt — in-silico knockdown via scGPT, cached for BO / Lab 6 / Lab 8 use.

Step 4 of docs/foundation-models.md. Companion to:

* scripts/fm_embed.py        — node features (cell / gene embeddings).
* scripts/fm_edges_seq.py    — edge priors (sequence-grounded).
* this file                  — perturbation-response priors (in-silico KD).

What this gives the project
---------------------------
For each candidate TF, scGPT predicts the average response of every other
gene to that TF being knocked down. The output is an (n_tfs × n_genes)
matrix that's directly comparable to:

* Lab 6's Jacobian-derived controllability matrix B (the linearised
  plant; what `jaxctrl` operates on).
* Lab 8's `hgx` forward-predicted perturbation responses.
* The §4.3 wet-lab forward programme's KO-rescue + synNotch cycles.

The active-learning move: **wherever scGPT and the project's Jacobian
disagree most, the wet lab should run an experiment** — that's the
highest-EIG perturbation under the current model. scripts/ablate_perturb_eig.py
quantifies this on a synthetic-truth benchmark.

Real / stub / auto contract
---------------------------
Same as fm_embed.py and fm_edges_seq.py:

* `real`  — load the actual scGPT checkpoint; run the per-TF KD API.
            Lazy-imports `scgpt`; raises informatively if missing.
* `stub`  — deterministic structured response based on the supplied
            "true" regulome (or random if none provided). Models a
            scGPT-like prediction with controlled false-positive and
            false-negative rates. The right attribution control:
            isolates "any structure-aware perturbation prior helps"
            from "scGPT's 33 M-cell pretraining helps".
* `auto`  — try real, fall back to stub with warning.

Usage
-----
    python scripts/fm_perturb_scgpt.py \\
        --input data/pollen.h5ad \\
        --tfs data/candidate_tfs.txt \\
        --output cache/ \\
        --mode stub

Writes `<stem>_scgpt_perturb.npy` of shape (n_tfs, n_genes) and a
matching manifest JSON.

See also
--------
- docs/foundation-models.md §4 — the intervention this implements.
- scripts/ablate_perturb_eig.py — the BO / EIG-disagreement ablation.
- docs/wetlab-program.md — the closed-loop context the EIG signal informs.
- docs/dgx-spark-setup.md — real-mode execution on the 128 GB GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

MODELS = {
    "scgpt": {
        "citation": "Cui et al. 2024, Nat Methods (Wang lab, Toronto/Vector)",
        "checkpoint": "subercui/scGPT",
        "dgx_gb": 4,  # tiny by FM standards
    },
}


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------


def _load_h5ad(path: Path):
    import anndata as ad

    return ad.read_h5ad(str(path))


def _load_tfs(path: Path) -> list[str]:
    """One TF symbol per line, optional header line starting with '#'."""
    tfs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tfs.append(line.split()[0])
    if not tfs:
        raise ValueError(f"no TFs parsed from {path}")
    return tfs


def _tf_seed(tf: str, salt: int) -> int:
    h = hashlib.sha256(f"perturb|{tf}|{salt}".encode()).digest()
    return int.from_bytes(h[:4], "little")


# ----------------------------------------------------------------------------
# Stub backend
# ----------------------------------------------------------------------------


def _stub_response(
    adata,
    tfs: list[str],
    seed: int,
    truth: np.ndarray | None = None,
    fpr: float = 0.05,
    fnr: float = 0.40,
) -> np.ndarray:
    """A (n_tfs × n_genes) predicted-KD-response matrix.

    If a "truth" matrix is provided (synthetic-benchmark use), the stub
    returns a controlled-noise version of it — recovers true downstream
    effects with probability (1 - fnr) and adds false positives at rate
    fpr. Without truth, the stub returns a deterministic structured
    response from each TF's identity hash. Both paths are reproducible.
    """
    n_genes = adata.shape[1]
    n_tfs = len(tfs)
    rng_master = np.random.default_rng(seed)

    if truth is not None:
        assert truth.shape == (n_tfs, n_genes), \
            f"truth shape {truth.shape} != ({n_tfs}, {n_genes})"
        response = truth.astype(np.float32, copy=True)
        # Flip false positives in (was-zero) entries:
        flip_fp = rng_master.random(truth.shape) < fpr
        new_signs = rng_master.choice([+1.0, -1.0], size=truth.shape).astype(np.float32)
        response = np.where((response == 0) & flip_fp, new_signs * 0.6, response)
        # Drop false negatives in (was-nonzero) entries:
        flip_fn = rng_master.random(truth.shape) < fnr
        response = np.where((np.abs(response) > 0) & flip_fn, 0.0, response)
        # Add measurement noise:
        response += (rng_master.standard_normal(truth.shape).astype(np.float32) * 0.2)
        return response

    # Truth-free fallback: deterministic per-TF Gaussian response, weighted by
    # the *mean expression* of each gene in `adata` (so it has structure even
    # without a regulome). Useful for the smoke-test path.
    expr_mean = (
        np.asarray(adata.X.mean(axis=0)).ravel()
        if hasattr(adata.X, "mean")
        else adata.X.mean(axis=0)
    )
    expr_weight = (expr_mean - expr_mean.mean()) / (expr_mean.std() + 1e-6)
    response = np.zeros((n_tfs, n_genes), dtype=np.float32)
    for i, tf in enumerate(tfs):
        rng_tf = np.random.default_rng(_tf_seed(tf, seed))
        per_tf = rng_tf.standard_normal(n_genes).astype(np.float32)
        response[i] = per_tf * 0.6 + expr_weight * rng_tf.normal(scale=0.4)
    return response


# ----------------------------------------------------------------------------
# Real backend (lazy)
# ----------------------------------------------------------------------------


def _real_response(adata, tfs: list[str], seed: int, **kwargs) -> np.ndarray:
    """scGPT in-silico KD. Lazy-imports `scgpt`; raises if missing."""
    try:
        import scgpt  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "fm_perturb_scgpt real-mode requires the `scgpt` package "
            "(https://github.com/bowang-lab/scGPT). Fall back to --mode stub for "
            "tutorial use; install scgpt + checkpoint on DGX Spark per "
            "docs/dgx-spark-setup.md for real-mode."
        ) from e
    n_genes = adata.shape[1]
    response = np.zeros((len(tfs), n_genes), dtype=np.float32)
    for i, tf in enumerate(tfs):
        # The real scGPT KD API returns per-cell deltas; we average to per-TF.
        # Concrete call shape is pinned to the model card at run time.
        delta = scgpt.in_silico_kd(adata, tf)  # type: ignore[attr-defined]
        response[i] = np.asarray(delta).mean(axis=0).astype(np.float32)
    return response


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def predict_kd_responses(
    adata,
    tfs: list[str],
    mode: str = "auto",
    seed: int = 0,
    truth: np.ndarray | None = None,
    fpr: float = 0.05,
    fnr: float = 0.40,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict (n_tfs × n_genes) in-silico KD responses.

    The `truth` / `fpr` / `fnr` args only affect stub mode — they let the
    synthetic-benchmark ablation control prediction quality. Real mode
    ignores them.
    """
    used_mode = mode
    if mode == "real":
        response = _real_response(adata, tfs, seed)
    elif mode == "stub":
        response = _stub_response(adata, tfs, seed, truth=truth, fpr=fpr, fnr=fnr)
    elif mode == "auto":
        try:
            response = _real_response(adata, tfs, seed)
            used_mode = "real"
        except RuntimeError as e:
            warnings.warn(f"fm_perturb_scgpt: falling back to stub mode: {e}")
            response = _stub_response(adata, tfs, seed, truth=truth, fpr=fpr, fnr=fnr)
            used_mode = "stub"
    else:
        raise ValueError(f"unknown mode {mode!r}; choose real/stub/auto")

    manifest = {
        "model": "scgpt",
        "mode": used_mode,
        "n_tfs": len(tfs),
        "n_genes": int(adata.shape[1]),
        "citation": MODELS["scgpt"]["citation"],
        "checkpoint": MODELS["scgpt"]["checkpoint"],
        "dgx_estimated_gb": MODELS["scgpt"]["dgx_gb"],
        "seed": seed,
        "stub_fpr": fpr if used_mode == "stub" else None,
        "stub_fnr": fnr if used_mode == "stub" else None,
        "fm_perturb_scgpt_version": "0.1.0",
    }
    return response, manifest


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _cmd_extract(args) -> int:
    input_path = Path(args.input)
    tfs_path = Path(args.tfs)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2
    if not tfs_path.exists():
        print(f"error: tfs file not found: {tfs_path}", file=sys.stderr)
        return 2

    print(f"fm_perturb_scgpt: loading {input_path}")
    adata = _load_h5ad(input_path)
    print(f"fm_perturb_scgpt: adata = {adata}")

    tfs = _load_tfs(tfs_path)
    print(f"fm_perturb_scgpt: {len(tfs)} TFs loaded")

    response, manifest = predict_kd_responses(
        adata, tfs, mode=args.mode, seed=args.seed
    )
    manifest["input_path"] = str(input_path)
    manifest["tfs_path"] = str(tfs_path)

    stem = input_path.stem
    out_npy = output_dir / f"{stem}_scgpt_perturb.npy"
    out_manifest = output_dir / f"{stem}_scgpt_perturb_manifest.json"
    np.save(out_npy, response)
    out_manifest.write_text(json.dumps(manifest, indent=2))

    print(
        f"fm_perturb_scgpt: wrote {out_npy}  shape={response.shape}  "
        f"|mean|={np.abs(response).mean():.3f}"
    )
    print(f"fm_perturb_scgpt: wrote {out_manifest}  mode={manifest['mode']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--input", required=True, help="path to input .h5ad")
    p.add_argument("--tfs", required=True, help="text file, one TF per line")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--mode", choices=["real", "stub", "auto"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    return _cmd_extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
