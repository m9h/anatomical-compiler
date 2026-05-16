"""fm_perturb_measured — empirical-CRISPRi KD response panel from a Perturb-seq h5ad.

A drop-in replacement for ``scripts/fm_perturb_scgpt.py``'s output cache
contract, but the response panel comes from **measured perturbation
data in the h5ad itself** rather than from an in-silico predictor.

Why this matters here
---------------------
``scripts/fm_perturb_scgpt.py``'s real-mode is structurally blocked:
the installed scgpt-human checkpoint is the embedder, not a perturbation
predictor. Per `docs/dgx-verifier-runbook.md`'s Tier 4 §Lab 6, that
blocks the controllability-vs-FM ranking comparison.

For Pollen-style Perturb-seq h5ads, every cell labelled with a single-TF
sgRNA *is* a measured KD readout — the very signal a zero-shot in-silico
KD predictor would try to approximate. Using the measured data directly:

* Cleaner science — no zero-shot generalisation gap. The ground truth
  is the comparison reference.
* Unblocks Lab 6 with no new dependencies and no FM fine-tune.
* The follow-up Lab 6 question (controllability ≈ scGPT-zero-shot)
  becomes (controllability ≈ measured-KD), which is arguably the
  *primary* validation question — does the linear-control surrogate
  agree with what biology actually does when you ablate a TF?

The standard scGPT path is still useful when (a) the project moves to a
dataset *without* dense Perturb-seq coverage, or (b) a proper
scGPT-perturbation fine-tune (scgpt-norman / scgpt-replogle) gets
wired in. This script is the path that's runnable *today*.

Cache contract
--------------
Same as ``fm_perturb_scgpt.py``: writes ``<stem>_<model>_perturb.npy``
of shape ``(n_tfs, n_genes)`` and a matching manifest JSON. ``<model>``
defaults to ``measured`` (override via ``--cache-stem`` to write as
``scgpt_perturb`` if you want a literal drop-in).

The TF order matches the supplied ``--tfs`` file. Rows for TFs absent
from ``adata.obs[--perturb-col]`` are zero-filled and flagged in the
manifest's ``tfs_missing`` field.

Response formula
----------------
For each TF *t* with at least ``--min-cells`` perturbed cells:

    response[t, g] = mean(X[cells where obs[perturb_col] == t, g])
                   - mean(X[control_cells, g])

where ``control_cells`` is identified by ``obs[control_col] == control_label``
(default: ``obs['perturbation'] == 'NT'``). Falls back to
``obs[perturb_col] == na_label`` if the status column is absent.

Usage
-----
    # Pollen full screen (137 k cells, 44 single-TF perturbations):
    python scripts/fm_perturb_measured.py \\
        --input data/pollen/screen_annotated.h5ad \\
        --tfs   data/pollen/processed/tf_names.json \\
        --output cache/measured/ \\
        --cache-stem scgpt_perturb   # writes <input-stem>_scgpt_perturb.npy
                                     # so measure_lab6_real.py picks it up
                                     # under the default convention

    # Or keep them separate so it's visually clear the source isn't scGPT:
    python scripts/fm_perturb_measured.py \\
        --input data/pollen/screen_annotated.h5ad \\
        --tfs   data/pollen/processed/tf_names.json \\
        --output cache/measured/
    # → screen_annotated_measured_perturb.npy

See also
--------
- scripts/fm_perturb_scgpt.py — the in-silico counterpart (currently blocked).
- scripts/measure_lab6_real.py — the downstream consumer.
- docs/dgx-verifier-runbook.md Tier 4 — Lab 6 unblock paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_h5ad(path: Path):
    import anndata as ad

    return ad.read_h5ad(str(path))


def _load_tfs(path: Path) -> list[str]:
    """Accept either one-symbol-per-line or a JSON list."""
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: JSON must be a list of TF symbols")
        return [str(t).strip() for t in data if str(t).strip()]
    return [ln.split()[0] for ln in text.splitlines()
            if ln.strip() and not ln.startswith("#")]


def _measured_response(
    adata,
    tfs: list[str],
    perturb_col: str,
    control_col: str | None,
    control_label: str | None,
    na_label: str,
    min_cells: int,
) -> tuple[np.ndarray, list[str], int, int]:
    """Build the (n_tfs × n_genes) measured KD response panel.

    Returns
    -------
    response : (n_tfs, n_genes) float32 — zero-rows for missing TFs.
    tfs_missing : list[str] — TFs absent from obs[perturb_col] or too sparse.
    n_controls : int — number of control cells found.
    n_cells_per_tf : dict — for diagnostic logging.
    """
    if perturb_col not in adata.obs.columns:
        raise ValueError(
            f"h5ad.obs missing required column {perturb_col!r}. "
            f"Available: {list(adata.obs.columns)}"
        )
    perturb = adata.obs[perturb_col].astype(str).to_numpy()

    if control_col is not None and control_col in adata.obs.columns and control_label is not None:
        control_mask = (adata.obs[control_col].astype(str) == control_label).to_numpy()
    else:
        control_mask = (perturb == na_label)
    n_controls = int(control_mask.sum())
    if n_controls < 20:
        raise ValueError(
            f"<20 control cells found (looked for {control_col}=={control_label} or "
            f"{perturb_col}=={na_label}); only {n_controls}."
        )

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float32)
    mu_ctrl = X[control_mask].mean(axis=0)

    n_genes = adata.shape[1]
    response = np.zeros((len(tfs), n_genes), dtype=np.float32)
    tfs_missing: list[str] = []
    cells_per_tf: dict[str, int] = {}
    for i, tf in enumerate(tfs):
        kd_mask = (perturb == tf)
        n_kd = int(kd_mask.sum())
        cells_per_tf[tf] = n_kd
        if n_kd < min_cells:
            tfs_missing.append(tf)
            continue
        response[i] = X[kd_mask].mean(axis=0) - mu_ctrl
    return response, tfs_missing, n_controls, cells_per_tf


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

    print(f"fm_perturb_measured: loading {input_path}")
    adata = _load_h5ad(input_path)
    print(f"fm_perturb_measured: adata = {adata}")

    tfs = _load_tfs(tfs_path)
    print(f"fm_perturb_measured: {len(tfs)} TFs loaded")

    response, tfs_missing, n_controls, cells_per_tf = _measured_response(
        adata,
        tfs,
        perturb_col=args.perturb_col,
        control_col=args.control_col or None,
        control_label=args.control_label or None,
        na_label=args.na_label,
        min_cells=args.min_cells,
    )

    stem = input_path.stem
    cache_stem = args.cache_stem
    out_npy = output_dir / f"{stem}_{cache_stem}.npy"
    out_manifest = output_dir / f"{stem}_{cache_stem}_manifest.json"
    np.save(out_npy, response)
    manifest: dict[str, Any] = {
        "model": "measured",
        "mode": "real",
        "source": "empirical CRISPRi mean(perturbed) − mean(control) per TF",
        "input_path": str(input_path),
        "tfs_path": str(tfs_path),
        "n_tfs": len(tfs),
        "n_tfs_present": len(tfs) - len(tfs_missing),
        "n_tfs_missing": len(tfs_missing),
        "tfs_missing": tfs_missing,
        "n_genes": int(adata.shape[1]),
        "n_controls": n_controls,
        "min_cells": args.min_cells,
        "perturb_col": args.perturb_col,
        "control_col": args.control_col or None,
        "control_label": args.control_label or None,
        "na_label": args.na_label,
        "cells_per_tf": cells_per_tf,
        "fm_perturb_measured_version": "0.1.0",
    }
    out_manifest.write_text(json.dumps(manifest, indent=2))

    n_present = len(tfs) - len(tfs_missing)
    print(
        f"fm_perturb_measured: wrote {out_npy}  shape={response.shape}  "
        f"|mean|={np.abs(response).mean():.4f}  "
        f"({n_present}/{len(tfs)} TFs with ≥{args.min_cells} cells)"
    )
    print(f"fm_perturb_measured: wrote {out_manifest}")
    if tfs_missing:
        print(f"  missing TFs (zero-rowed): {tfs_missing}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--input", required=True, help="path to input .h5ad")
    p.add_argument("--tfs", required=True, help="text file (one symbol per line) or JSON list")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--perturb-col", default="Gene_target_single",
                   help="obs column with per-cell TF KD label (default for Pollen)")
    p.add_argument("--control-col", default="perturbation",
                   help="obs column with control flag (set empty to skip)")
    p.add_argument("--control-label", default="NT",
                   help="value in --control-col indicating controls")
    p.add_argument("--na-label", default="NA",
                   help="value in --perturb-col indicating no target (fallback control)")
    p.add_argument("--min-cells", type=int, default=10,
                   help="minimum perturbed cells per TF (zero-row otherwise)")
    p.add_argument("--cache-stem", default="measured_perturb",
                   help="filename stem after <h5ad-stem>_ ; set to scgpt_perturb "
                        "to be a literal drop-in for the scGPT cache")
    args = p.parse_args(argv)
    return _cmd_extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
