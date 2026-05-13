"""fm_embed — extract foundation-model embeddings from an h5ad, cache as .npy.

The integration contract from ``docs/foundation-models.md``: extract once,
cache as .npy, then run the JAX pipeline on the cached arrays. No
fine-tuning, no recurring inference cost.

Subcommands
-----------
uce         Universal Cell Embedding (Stanford + CZI; Rosen, Quake, Leskovec
            et al. 2024). Returns cells × 1280.
geneformer  Geneformer (Theodoris et al. 2023, Nature; Gladstone / UCSF).
            Returns genes × 512.
scgpt       scGPT (Cui et al. 2024, Nat Methods; Wang lab, Toronto / Vector).
            Returns cells × 512 (cell-level embedding).

Modes
-----
real   Load the actual checkpoint via `transformers` / model-specific loader.
       Requires GPU + downloaded weights + the appropriate optional
       dependency installed. Raises informatively if any are missing.
stub   Deterministic synthetic embedding of the correct dimensions. Useful
       for: (a) tutorial / smoke-test runs that don't need real biology,
       (b) wiring up the downstream pipeline before allocating GPU budget,
       (c) ablation controls (compare real-FM-prior vs structure-matched
       random-projection prior to attribute gains).
auto   Try real, fall back to stub with a warning. Default.

The stubs are intentionally *structure-preserving*: a real FM encodes
co-expression structure, so the stubs project the input expression matrix
through a (deterministic, seeded) low-rank decomposition. This means
downstream comparisons against one-hot baselines are still informative —
the stub shows what a "co-expression-aware embedding of the same input"
can do, separately from "what 30 M extra cells of pretraining buys you".

Usage
-----
    python scripts/fm_embed.py uce        --input data/pollen.h5ad --output cache/
    python scripts/fm_embed.py geneformer --input data/pollen.h5ad --output cache/ --mode stub
    python scripts/fm_embed.py scgpt      --input data/pollen.h5ad --output cache/

Each call writes ``<stem>_<model>.npy`` and ``<stem>_<model>_manifest.json``
into the output directory.

See also
--------
- docs/foundation-models.md — design rationale, model catalogue, pipeline
- notebooks/11_foundation_model_pipeline.ipynb — the consumer of these caches
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ----------------------------------------------------------------------------
# Configuration — model dimensions and provenance pins
# ----------------------------------------------------------------------------

MODELS = {
    "uce": {
        "dim": 1280,
        "axis": "cells",
        "citation": "Rosen et al. 2024 (Stanford+CZI)",
        "checkpoint": "chanzuckerberg/uce",  # HF handle
    },
    "geneformer": {
        "dim": 512,
        "axis": "genes",
        "citation": "Theodoris et al. 2023, Nature (Gladstone/UCSF)",
        "checkpoint": "ctheodoris/Geneformer",
    },
    "scgpt": {
        "dim": 512,
        "axis": "cells",
        "citation": "Cui et al. 2024, Nat Methods (Wang lab, Toronto/Vector)",
        "checkpoint": "subercui/scGPT",
    },
}


@dataclass
class Manifest:
    model: str
    mode: str  # "real" or "stub"
    dim: int
    axis: str  # "cells" or "genes"
    n: int  # number of items along the axis
    input_path: str
    input_sha256_8: str  # first 8 hex chars
    citation: str
    checkpoint: str
    seed: int
    fm_embed_version: str = "0.1.0"


# ----------------------------------------------------------------------------
# Input loading
# ----------------------------------------------------------------------------


def _load_h5ad(path: Path):
    """Load an h5ad. Lazy-imports anndata so the CLI works without it for `--help`."""
    import anndata as ad

    return ad.read_h5ad(str(path))


def _input_fingerprint(path: Path) -> str:
    """SHA-256 of the file, first 8 hex chars. Cheap, sufficient for cache invalidation."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


# ----------------------------------------------------------------------------
# Stub backends — deterministic, structure-preserving embeddings
# ----------------------------------------------------------------------------


def _log1p_normalised(X: np.ndarray) -> np.ndarray:
    """Library-size normalise + log1p. Matches the standard scanpy preamble."""
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    target = float(np.median(row_sums))
    X_norm = X / row_sums * target
    return np.log1p(X_norm).astype(np.float32)


def _stub_uce(X: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """Cells × dim. Cell-level PCA padded with a seeded Gaussian projection.

    Captures the dominant cell-state structure (top PCs) in the first
    `min(n_cells, 50)` dimensions, then a deterministic random projection
    fills the remaining dims with co-expression-aware features.
    """
    n_cells, n_genes = X.shape
    X_log = _log1p_normalised(X)
    X_centered = X_log - X_log.mean(axis=0, keepdims=True)
    k_pca = min(50, n_cells - 1, n_genes)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pca = (U[:, :k_pca] * s[:k_pca]).astype(np.float32)
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((n_genes, dim - k_pca), dtype=np.float32) / np.sqrt(n_genes)
    extras = X_log @ proj
    emb = np.concatenate([pca, extras], axis=1)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def _stub_geneformer(X: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """Genes × dim. Gene-level co-expression embedding via SVD of the normalised
    expression matrix; encodes the kind of gene–gene structure Geneformer's
    rank-value pretraining captures, without the 30 M-cell prior.
    """
    n_cells, n_genes = X.shape
    X_log = _log1p_normalised(X)
    X_centered = X_log - X_log.mean(axis=0, keepdims=True)
    # SVD of (cells × genes); right singular vectors are gene loadings.
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    k_real = min(dim, Vt.shape[0])
    gene_emb = (Vt[:k_real].T * s[:k_real]).astype(np.float32)
    if k_real < dim:
        rng = np.random.default_rng(seed)
        pad = rng.standard_normal((n_genes, dim - k_real), dtype=np.float32) * 0.01
        gene_emb = np.concatenate([gene_emb, pad], axis=1)
    return gene_emb


def _stub_scgpt(X: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """Cells × dim. Same family as UCE-stub but at scGPT's narrower 512-d width."""
    return _stub_uce(X, dim, seed)


_STUBS = {
    "uce": _stub_uce,
    "geneformer": _stub_geneformer,
    "scgpt": _stub_scgpt,
}


# ----------------------------------------------------------------------------
# Real backends — guarded; raise informatively if deps / weights missing
# ----------------------------------------------------------------------------


def _real_uce(adata, dim: int, seed: int) -> np.ndarray:
    """Universal Cell Embedding (Stanford + CZI). Lazy-imports the official package."""
    try:
        # The CZI release ships under `uce` or `cellxgene_census` depending on
        # version; we keep the import permissive and surface a clear error.
        import uce  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "UCE real-mode requires the `uce` package: pip install uce-models. "
            "Fall back to --mode stub for tutorial/smoke-test use."
        ) from e
    # Implementation note: the official UCE API is `uce.embed(adata)` returning
    # an (n_cells, 1280) array. We do not exercise this in the smoke-tested path.
    return uce.embed(adata)  # type: ignore[attr-defined]


def _real_geneformer(adata, dim: int, seed: int) -> np.ndarray:
    """Geneformer (UCSF/Gladstone). Lazy-imports `geneformer`."""
    try:
        from geneformer import EmbExtractor  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Geneformer real-mode requires the `geneformer` package "
            "(https://huggingface.co/ctheodoris/Geneformer). Fall back to "
            "--mode stub for tutorial/smoke-test use."
        ) from e
    extractor = EmbExtractor(model_type="Pretrained", emb_mode="gene")
    return extractor.extract_embs(adata)  # implementation pinned to model card


def _real_scgpt(adata, dim: int, seed: int) -> np.ndarray:
    """scGPT (Toronto/Vector). Lazy-imports `scgpt`."""
    try:
        import scgpt  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "scGPT real-mode requires the `scgpt` package "
            "(https://github.com/bowang-lab/scGPT). Fall back to --mode stub."
        ) from e
    return scgpt.embed_cells(adata)  # type: ignore[attr-defined]


_REAL = {
    "uce": _real_uce,
    "geneformer": _real_geneformer,
    "scgpt": _real_scgpt,
}


# ----------------------------------------------------------------------------
# Public API — what the notebook imports
# ----------------------------------------------------------------------------


def extract(
    adata,
    model: str,
    mode: str = "auto",
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract an embedding from an in-memory AnnData; return (array, manifest_dict).

    This is the function ``notebooks/11_foundation_model_pipeline.ipynb`` calls.

    Parameters
    ----------
    adata
        AnnData with ``.X`` as the count matrix.
    model
        One of "uce", "geneformer", "scgpt".
    mode
        "real" | "stub" | "auto". Default "auto": try real, fall back to stub
        with a warning.
    seed
        Seed for the stub's random projection. Ignored in real mode.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {list(MODELS)}")
    spec = MODELS[model]
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float32, copy=False)

    used_mode = mode
    if mode == "real":
        emb = _REAL[model](adata, spec["dim"], seed)
    elif mode == "stub":
        emb = _STUBS[model](X, spec["dim"], seed)
    elif mode == "auto":
        try:
            emb = _REAL[model](adata, spec["dim"], seed)
            used_mode = "real"
        except RuntimeError as e:
            warnings.warn(f"fm_embed: falling back to stub mode ({model}): {e}")
            emb = _STUBS[model](X, spec["dim"], seed)
            used_mode = "stub"
    else:
        raise ValueError(f"unknown mode {mode!r}; choose from real/stub/auto")

    manifest = {
        "model": model,
        "mode": used_mode,
        "dim": spec["dim"],
        "axis": spec["axis"],
        "n": emb.shape[0],
        "citation": spec["citation"],
        "checkpoint": spec["checkpoint"],
        "seed": seed,
        "fm_embed_version": "0.1.0",
    }
    return emb, manifest


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _cmd_extract(args) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"fm_embed: loading {input_path}")
    adata = _load_h5ad(input_path)
    print(f"fm_embed: adata = {adata}")

    emb, manifest = extract(adata, model=args.model, mode=args.mode, seed=args.seed)
    manifest["input_path"] = str(input_path)
    manifest["input_sha256_8"] = _input_fingerprint(input_path)

    stem = input_path.stem
    emb_path = output_dir / f"{stem}_{args.model}.npy"
    manifest_path = output_dir / f"{stem}_{args.model}_manifest.json"
    np.save(emb_path, emb)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"fm_embed: wrote {emb_path}  shape={emb.shape}  dtype={emb.dtype}")
    print(f"fm_embed: wrote {manifest_path}  mode={manifest['mode']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="model", required=True)
    for name in MODELS:
        sp = sub.add_parser(name, help=f"{name} — {MODELS[name]['citation']}")
        sp.add_argument("--input", required=True, help="path to input .h5ad")
        sp.add_argument("--output", required=True, help="output directory for .npy + manifest")
        sp.add_argument("--mode", choices=["real", "stub", "auto"], default="auto")
        sp.add_argument("--seed", type=int, default=0)
        sp.set_defaults(func=_cmd_extract)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
