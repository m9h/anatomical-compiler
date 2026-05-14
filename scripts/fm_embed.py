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
    """Universal Cell Embedding (Stanford + CZI).

    UCE has no clean in-process Python API: the upstream entry is
    `eval_single_anndata.py`, which reads an h5ad, runs the encoder, and
    writes an h5ad with embeddings in `.obsm["X_uce"]`. We invoke that
    script as a subprocess and read the result back.

    Required env:
        UCE_REPO       path to the cloned snap-stanford/UCE repository
                       (containing eval_single_anndata.py).
        UCE_MODEL_LOC  path to the model `.torch` checkpoint
                       (e.g. .../4layer_model.torch). Defaults to the path
                       the script auto-downloads if unset.
        UCE_SPECIES    species string for the dataset (default "human").
    """
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    import anndata as ad

    uce_repo = os.environ.get("UCE_REPO")
    if not uce_repo:
        raise RuntimeError(
            "UCE real-mode needs UCE_REPO pointing to the snap-stanford/UCE repo "
            "(eval_single_anndata.py lives there). Inside the FM container the repo "
            "is cloned to /opt/UCE; set UCE_REPO=/opt/UCE."
        )
    script = Path(uce_repo) / "eval_single_anndata.py"
    if not script.exists():
        raise RuntimeError(f"UCE script not found: {script}")

    species = os.environ.get("UCE_SPECIES", "human")
    model_loc = os.environ.get("UCE_MODEL_LOC")
    batch_size = int(os.environ.get("UCE_BATCH_SIZE", "100"))

    with tempfile.TemporaryDirectory(prefix="uce_") as work:
        work = Path(work)
        in_path = work / "input.h5ad"
        adata.write_h5ad(in_path)

        cmd = [
            sys.executable, str(script),
            "--adata_path", str(in_path),
            "--dir", str(work),
            "--species", species,
            "--batch_size", str(batch_size),
        ]
        if model_loc:
            cmd += ["--model_loc", model_loc]
        subprocess.run(cmd, check=True, cwd=uce_repo)

        # UCE writes <stem>_uce_adata.h5ad with X_uce in .obsm
        out = next(work.glob("*_uce_adata.h5ad"), None)
        if out is None:
            raise RuntimeError(f"UCE produced no output h5ad in {work}")
        ad_out = ad.read_h5ad(out)
        if "X_uce" not in ad_out.obsm:
            raise RuntimeError("UCE output h5ad has no obsm['X_uce']")
        return np.asarray(ad_out.obsm["X_uce"], dtype=np.float32)


def _real_geneformer(adata, dim: int, seed: int) -> np.ndarray:
    """Geneformer (UCSF/Gladstone).

    The actual API is two-stage: `TranscriptomeTokenizer.tokenize_data()`
    writes a HuggingFace Dataset of rank-value-encoded sequences to disk,
    and `EmbExtractor.extract_embs()` consumes that dataset and writes
    embeddings out. We orchestrate both in a temp dir and return numpy.

    Requirements:
        - adata.var must contain "ensembl_id" (we synthesise it if var_names
          are already Ensembl IDs).
        - adata.obs must contain "n_counts" (we compute it if missing).
    """
    try:
        from geneformer import EmbExtractor, TranscriptomeTokenizer  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Geneformer real-mode requires `pip install geneformer`. "
            "Available in the FM container (Dockerfile.fm)."
        ) from e

    import os
    import tempfile
    from pathlib import Path

    adata = adata.copy()
    # Ensure ensembl_id column — Geneformer's tokenizer keys on it.
    if "ensembl_id" not in adata.var.columns:
        names = list(adata.var_names[:10])
        if all(n.startswith(("ENSG", "ENSMUSG", "ENSRNOG")) for n in names):
            adata.var["ensembl_id"] = adata.var_names
        else:
            raise RuntimeError(
                "Geneformer requires Ensembl IDs in adata.var['ensembl_id']. "
                "Map gene symbols to Ensembl IDs first (e.g. via mygene or biomart)."
            )
    if "n_counts" not in adata.obs.columns:
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        adata.obs["n_counts"] = X.sum(axis=1).astype(np.float32)

    model_dir = os.environ.get("GENEFORMER_MODEL", "ctheodoris/Geneformer")
    forward_batch_size = int(os.environ.get("GENEFORMER_BATCH", "64"))
    emb_mode = os.environ.get("GENEFORMER_EMB_MODE", "cls")  # cls = cell

    with tempfile.TemporaryDirectory(prefix="geneformer_") as work:
        work = Path(work)
        h5_dir = work / "h5ad"
        h5_dir.mkdir()
        adata.write_h5ad(h5_dir / "input.h5ad")

        tok_dir = work / "tok"
        tok_dir.mkdir()
        tokenizer = TranscriptomeTokenizer()
        tokenizer.tokenize_data(
            data_directory=str(h5_dir),
            output_directory=str(tok_dir),
            output_prefix="input",
            file_format="h5ad",
        )

        emb_dir = work / "emb"
        emb_dir.mkdir()
        extractor = EmbExtractor(
            model_type="Pretrained",
            num_classes=0,
            emb_mode=emb_mode,
            max_ncells=None,
            emb_layer=-1,
            forward_batch_size=forward_batch_size,
        )
        result = extractor.extract_embs(
            model_directory=model_dir,
            input_data_file=str(tok_dir / "input.dataset"),
            output_directory=str(emb_dir),
            output_prefix="emb",
            output_torch_embs=True,
        )
        # (embs_df, torch_embs, avg_attentions) when output_torch_embs=True
        torch_embs = result[1] if isinstance(result, tuple) and len(result) >= 2 else result
        return torch_embs.detach().cpu().numpy().astype(np.float32)


def _real_scgpt(adata, dim: int, seed: int) -> np.ndarray:
    """scGPT (Toronto/Vector).

    Uses `scgpt.tasks.embed_data` (the modern public entry point). Returns
    cell-level embeddings; reads them back from `obsm['X_scGPT']` of the
    returned AnnData.

    Required env:
        SCGPT_MODEL_DIR  directory holding the scGPT checkpoint
                         (args.json, vocab.json, best_model.pt).
        SCGPT_GENE_COL   column in adata.var holding gene identifiers
                         (default "feature_name"; fall back to var_names).
    """
    try:
        from scgpt.tasks import embed_data  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "scGPT real-mode requires `pip install git+https://github.com/bowang-lab/scGPT`. "
            "Available in the FM container (Dockerfile.fm)."
        ) from e

    import os
    from pathlib import Path

    model_dir = os.environ.get("SCGPT_MODEL_DIR")
    if not model_dir or not Path(model_dir).exists():
        raise RuntimeError(
            "scGPT real-mode needs SCGPT_MODEL_DIR pointing to the checkpoint "
            "(must contain args.json, vocab.json, best_model.pt). "
            "Pre-fetch via `huggingface-cli download subercui/scGPT_human`."
        )

    gene_col = os.environ.get("SCGPT_GENE_COL", "feature_name")
    if gene_col not in adata.var.columns:
        # Fall back to var_names — embed_data requires the column to exist.
        adata = adata.copy()
        adata.var[gene_col] = adata.var_names

    batch_size = int(os.environ.get("SCGPT_BATCH", "64"))
    device = os.environ.get("SCGPT_DEVICE", "cuda")

    out = embed_data(
        adata,
        model_dir=model_dir,
        gene_col=gene_col,
        batch_size=batch_size,
        device=device,
        return_new_adata=True,
    )
    if "X_scGPT" not in out.obsm:
        raise RuntimeError("scGPT output AnnData has no obsm['X_scGPT']")
    return np.asarray(out.obsm["X_scGPT"], dtype=np.float32)


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
