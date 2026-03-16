"""
data_loader.py — Shared data loading module for organoid-hgx-benchmark analyses.

Loads the Fleck et al. 2023 cerebral organoid data from Zenodo processed files.
All analysis scripts should import from this module rather than loading data directly.

Data sources:
    - Pando GRN coefficients (grn_modules.tsv)
    - RNA expression (RNA_data.h5ad, 49,718 cells x 33,538 genes)
    - Count matrices + rich metadata from data_matrices/ (34,089 cells)
    - RNA velocity (RNA_all_velo.h5ad, optional)

Usage:
    from data_loader import load_pando_grn, load_expression, load_metadata
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEY_TFS = ["GLI3", "FOXG1", "TBR1", "DLX1", "DLX2", "EMX1", "EOMES", "NEUROD6"]

FATE_NAMES = ["Telencephalon", "Early", "Neural Tube"]

FATE_COLORS = {
    "Telencephalon": "#e41a1c",
    "Early": "#377eb8",
    "Neural Tube": "#4daf4a",
}

STAGE_ORDER = ["iPSC", "nect_nepi", "npc", "neuron"]

# Lineage labels as they appear in meta.tsv.gz (lowercase)
_LINEAGE_MAP = {
    "telencephalon": 0,
    "early": 1,
    "nt": 2,
}

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_cache: dict = {}


def _clear_cache() -> None:
    """Clear all cached data (useful for testing or freeing memory)."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Data directory detection
# ---------------------------------------------------------------------------

def find_data_dir() -> Path:
    """Auto-detect data directory.

    Checks the following locations in order:
        1. ``/workspace/benchmark/data`` (inside Docker on DGX Spark)
        2. ``../data`` relative to this script's location
        3. ``data/`` relative to the current working directory

    Returns
    -------
    Path
        Resolved path to the data directory.

    Raises
    ------
    FileNotFoundError
        If no data directory can be found.
    """
    candidates = [
        Path("/workspace/benchmark/data"),
        Path(__file__).resolve().parent.parent / "data",
        Path.cwd() / "data",
    ]
    for p in candidates:
        if p.is_dir():
            logger.info("Using data directory: %s", p)
            return p
    raise FileNotFoundError(
        "Could not find data directory. Looked in:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


# ---------------------------------------------------------------------------
# Pando GRN
# ---------------------------------------------------------------------------

def load_pando_grn(
    data_dir: Optional[Path] = None,
    padj_threshold: float = 0.05,
) -> tuple:
    """Load Pando GRN into an hgx Hypergraph.

    Parameters
    ----------
    data_dir : Path, optional
        Root data directory. Auto-detected if *None*.
    padj_threshold : float
        Adjusted p-value threshold for filtering edges (default 0.05).

    Returns
    -------
    tuple of (hg, coefs_df, gene_names, tf_names)
        hg : hgx.Hypergraph
            Hypergraph with ~2,792 nodes and ~720 hyperedges (one per TF regulon).
        coefs_df : pandas.DataFrame
            Raw coefficient table (all rows, unfiltered).
        gene_names : list[str]
            Gene names ordered by node index in the hypergraph.
        tf_names : list[str]
            TF names ordered by hyperedge index in the hypergraph.
    """
    cache_key = ("pando_grn", padj_threshold)
    if cache_key in _cache:
        logger.info("Returning cached Pando GRN (padj < %s)", padj_threshold)
        return _cache[cache_key]

    import pandas as pd
    import hgx
    import devograph

    if data_dir is None:
        data_dir = find_data_dir()
    data_dir = Path(data_dir)

    # Locate the coefficients file
    coef_path = data_dir / "pando" / "grn_modules.tsv"
    if not coef_path.exists():
        coef_path = data_dir / "pando" / "coefs.tsv"
    if not coef_path.exists():
        raise FileNotFoundError(
            f"Cannot find Pando coefficients at {data_dir / 'pando'}. "
            "Expected grn_modules.tsv or coefs.tsv."
        )

    print(f"[data_loader] Loading Pando GRN from {coef_path} ...")
    coefs_df = pd.read_csv(coef_path, sep="\t")
    print(
        f"[data_loader]   Raw coefficients: {len(coefs_df):,} rows, "
        f"{coefs_df['tf'].nunique()} TFs, {coefs_df['target'].nunique()} targets"
    )

    # Build hypergraph via hgx
    hg = devograph.load_pando_modules(
        coef_csv=str(coef_path),
        modules_csv=None,
        padj_threshold=padj_threshold,
        tf_col="tf",
        target_col="target",
        estimate_col="estimate",
        padj_col="padj",
    )

    # Extract ordered gene and TF names from the hypergraph
    # hgx stores node/edge labels; we pull them out in index order.
    n_nodes = hg.incidence.shape[0]
    n_edges = hg.incidence.shape[1]

    # Gene names: if hg exposes .node_labels or similar, use that;
    # otherwise reconstruct from the filtered coefficient table.
    filtered = coefs_df[coefs_df["padj"] <= padj_threshold].copy()
    all_genes_ordered = sorted(
        set(filtered["tf"].unique()) | set(filtered["target"].unique())
    )
    tf_names_ordered = sorted(filtered["tf"].unique())

    # Prefer the hypergraph's own ordering if available
    if hasattr(hg, "node_labels") and hg.node_labels is not None:
        gene_names = list(hg.node_labels)
    else:
        gene_names = all_genes_ordered

    if hasattr(hg, "edge_labels") and hg.edge_labels is not None:
        tf_names = list(hg.edge_labels)
    else:
        tf_names = tf_names_ordered

    print(
        f"[data_loader]   Hypergraph: {n_nodes:,} nodes, {n_edges:,} hyperedges "
        f"(padj < {padj_threshold})"
    )

    result = (hg, coefs_df, gene_names, tf_names)
    _cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Expression (h5ad)
# ---------------------------------------------------------------------------

def load_expression(
    data_dir: Optional[Path] = None,
    subset_genes: Optional[list[str]] = None,
) -> "anndata.AnnData":
    """Load RNA expression from ``RNA_data.h5ad``.

    Parameters
    ----------
    data_dir : Path, optional
        Root data directory. Auto-detected if *None*.
    subset_genes : list[str], optional
        If provided, subset the AnnData to only these gene names.
        Genes not found in the data are silently skipped.

    Returns
    -------
    anndata.AnnData
        Expression matrix with 49,718 cells x N genes.
    """
    import anndata as ad

    if data_dir is None:
        data_dir = find_data_dir()
    data_dir = Path(data_dir)

    h5ad_path = data_dir / "expression" / "RNA_data.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Expression file not found: {h5ad_path}")

    # Cache the full AnnData; subsetting is cheap
    if "expression_h5ad" not in _cache:
        print(f"[data_loader] Loading expression from {h5ad_path} ...")
        adata = ad.read_h5ad(h5ad_path)
        print(
            f"[data_loader]   Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes"
        )
        _cache["expression_h5ad"] = adata
    else:
        adata = _cache["expression_h5ad"]
        logger.info("Returning cached expression AnnData")

    if subset_genes is not None:
        available = [g for g in subset_genes if g in adata.var_names]
        missing = set(subset_genes) - set(available)
        if missing:
            logger.warning(
                "%d/%d requested genes not found in expression data (e.g. %s)",
                len(missing),
                len(subset_genes),
                list(missing)[:5],
            )
        adata = adata[:, available].copy()
        print(
            f"[data_loader]   Subsetted to {adata.n_vars:,} genes "
            f"({len(missing)} not found)"
        )

    return adata


# ---------------------------------------------------------------------------
# Rich metadata from data_matrices/
# ---------------------------------------------------------------------------

def load_metadata(data_dir: Optional[Path] = None) -> "pandas.DataFrame":
    """Load rich cell metadata from ``data_matrices/meta.tsv.gz``.

    This metadata comes from the integrated RNA+ATAC dataset and contains
    34,089 cells with columns including ``lineage``, ``velocity_pseudotime``,
    ``stage_manual``, ``pt_bin``, and more.

    Parameters
    ----------
    data_dir : Path, optional
        Root data directory. Auto-detected if *None*.

    Returns
    -------
    pandas.DataFrame
        Metadata table indexed by ``cellID``.
    """
    import pandas as pd

    if "metadata" in _cache:
        logger.info("Returning cached metadata")
        return _cache["metadata"]

    if data_dir is None:
        data_dir = find_data_dir()
    data_dir = Path(data_dir)

    meta_path = data_dir / "zenodo" / "data_matrices" / "meta.tsv.gz"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}. "
            "Ensure data_matrices/ has been downloaded."
        )

    print(f"[data_loader] Loading metadata from {meta_path} ...")
    meta = pd.read_csv(meta_path, sep="\t", compression="gzip")
    if "cellID" in meta.columns:
        meta = meta.set_index("cellID")
    print(
        f"[data_loader]   Loaded: {len(meta):,} cells, "
        f"{len(meta.columns)} columns"
    )
    print(
        f"[data_loader]   Lineage distribution: "
        + ", ".join(
            f"{k}={v:,}"
            for k, v in meta["lineage"].value_counts().items()
        )
    )

    _cache["metadata"] = meta
    return meta


# ---------------------------------------------------------------------------
# Expression from count matrices + metadata
# ---------------------------------------------------------------------------

def load_expression_with_metadata(
    data_dir: Optional[Path] = None,
    subset_genes: Optional[list[str]] = None,
) -> "anndata.AnnData":
    """Load expression from sparse count matrix and attach rich metadata.

    Reads ``counts.mtx.gz``, ``features.tsv.gz``, ``barcodes.tsv.gz``, and
    ``meta.tsv.gz`` from the ``data_matrices/`` directory. The resulting
    AnnData has all metadata columns in ``.obs`` (including pseudotime,
    lineage, stage).

    Parameters
    ----------
    data_dir : Path, optional
        Root data directory. Auto-detected if *None*.
    subset_genes : list[str], optional
        If provided, subset to only these gene names.

    Returns
    -------
    anndata.AnnData
        Expression matrix with metadata in ``.obs``.

    Raises
    ------
    FileNotFoundError
        If the data_matrices/ directory or required files are missing.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy.io import mmread
    import gzip

    cache_key = "expression_with_metadata"
    if cache_key in _cache and subset_genes is None:
        logger.info("Returning cached expression-with-metadata AnnData")
        return _cache[cache_key]

    if data_dir is None:
        data_dir = find_data_dir()
    data_dir = Path(data_dir)

    mat_dir = data_dir / "zenodo" / "data_matrices"
    if not mat_dir.is_dir():
        raise FileNotFoundError(
            f"data_matrices directory not found: {mat_dir}. "
            "Fall back to load_expression() for the h5ad file."
        )

    # --- Load sparse count matrix ---
    counts_path = mat_dir / "counts.mtx.gz"
    print(f"[data_loader] Loading count matrix from {counts_path} ...")
    counts = mmread(str(counts_path)).tocsc()
    print(f"[data_loader]   Count matrix shape: {counts.shape}")

    # --- Load features (gene names) ---
    features_path = mat_dir / "features.tsv.gz"
    with gzip.open(features_path, "rt") as f:
        gene_names = [line.strip() for line in f]
    print(f"[data_loader]   Features: {len(gene_names):,} genes")

    # --- Load barcodes ---
    barcodes_path = mat_dir / "barcodes.tsv.gz"
    with gzip.open(barcodes_path, "rt") as f:
        barcodes = [line.strip() for line in f]
    print(f"[data_loader]   Barcodes: {len(barcodes):,} cells")

    # --- Load metadata ---
    meta = load_metadata(data_dir)

    # The count matrix is genes x cells (MatrixMarket convention for scRNA-seq),
    # but AnnData expects cells x genes. Check and transpose if needed.
    if counts.shape[0] == len(gene_names) and counts.shape[1] == len(barcodes):
        # genes x cells -> transpose to cells x genes
        counts = counts.T
    elif counts.shape[0] == len(barcodes) and counts.shape[1] == len(gene_names):
        pass  # already cells x genes
    else:
        raise ValueError(
            f"Count matrix shape {counts.shape} does not match "
            f"{len(barcodes)} barcodes x {len(gene_names)} features"
        )

    counts = counts.tocsr()

    # Build AnnData
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=gene_names),
    )

    # Merge metadata into .obs for cells that exist in both
    common_cells = adata.obs.index.intersection(meta.index)
    if len(common_cells) > 0:
        adata = adata[common_cells].copy()
        adata.obs = adata.obs.join(meta, how="left")
        print(
            f"[data_loader]   Merged metadata for {len(common_cells):,} cells "
            f"(out of {len(barcodes):,} barcodes and {len(meta):,} meta rows)"
        )
    else:
        # Try matching without the barcode suffix or using the meta index directly
        print(
            "[data_loader]   WARNING: No barcode overlap with metadata index. "
            "Attaching metadata by positional order if sizes match."
        )
        if len(barcodes) == len(meta):
            adata.obs = meta.reset_index(drop=True)
            adata.obs.index = barcodes

    # Subset genes if requested
    if subset_genes is not None:
        available = [g for g in subset_genes if g in adata.var_names]
        adata = adata[:, available].copy()
        print(f"[data_loader]   Subsetted to {adata.n_vars:,} genes")

    print(
        f"[data_loader]   Final AnnData: {adata.n_obs:,} cells x {adata.n_vars:,} genes"
    )

    if subset_genes is None:
        _cache[cache_key] = adata

    return adata


# ---------------------------------------------------------------------------
# Temporal (pseudotime-binned) expression
# ---------------------------------------------------------------------------

def load_temporal_expression(
    data_dir: Optional[Path] = None,
    grn_genes: Optional[list[str]] = None,
    num_bins: int = 10,
) -> tuple:
    """Build pseudotime-binned expression matrix for temporal dynamics.

    Steps:
        1. Load metadata (velocity_pseudotime from ``meta.tsv.gz``)
        2. Load expression for GRN genes
        3. Bin cells into *num_bins* equal-width pseudotime bins
        4. Compute mean expression per bin
        5. Compute lineage composition per bin

    Parameters
    ----------
    data_dir : Path, optional
        Root data directory.
    grn_genes : list[str], optional
        Genes to include. If *None*, uses all GRN genes from ``load_pando_grn``.
    num_bins : int
        Number of pseudotime bins (default 10).

    Returns
    -------
    tuple of (expr_matrix, pseudotime_centers, gene_names, lineage_fractions)
        expr_matrix : np.ndarray, shape (num_bins, n_genes, 1)
            Mean expression per pseudotime bin, with trailing dim for
            compatibility with hgx signal processing.
        pseudotime_centers : np.ndarray, shape (num_bins,)
            Center pseudotime value for each bin.
        gene_names : list[str]
            Gene names matching axis 1 of expr_matrix.
        lineage_fractions : np.ndarray, shape (num_bins, 3)
            Fraction of cells in each bin belonging to
            [telencephalon, early, nt] lineages.
    """
    import numpy as np
    import pandas as pd

    if data_dir is None:
        data_dir = find_data_dir()
    data_dir = Path(data_dir)

    # --- Determine gene list ---
    if grn_genes is None:
        _, _, grn_genes, _ = load_pando_grn(data_dir)
        print(
            f"[data_loader] Using {len(grn_genes):,} GRN genes for temporal expression"
        )

    # --- Try loading from data_matrices (preferred: has pseudotime) ---
    mat_dir = data_dir / "zenodo" / "data_matrices"
    if mat_dir.is_dir():
        adata = load_expression_with_metadata(data_dir, subset_genes=grn_genes)
    else:
        # Fall back to h5ad — but we need pseudotime from metadata
        print(
            "[data_loader]   WARNING: data_matrices/ not found. "
            "Falling back to RNA_data.h5ad with 'age' as a pseudotime proxy."
        )
        adata = load_expression(data_dir, subset_genes=grn_genes)
        # Use 'age' (organoid days) as a rough pseudotime proxy
        if "age" in adata.obs.columns:
            ages = adata.obs["age"].values.astype(float)
            adata.obs["velocity_pseudotime"] = (ages - ages.min()) / (
                ages.max() - ages.min()
            )
        else:
            raise ValueError(
                "No pseudotime available: data_matrices/ missing and "
                "RNA_data.h5ad has no 'age' column."
            )

    # --- Filter to cells with valid pseudotime ---
    if "velocity_pseudotime" not in adata.obs.columns:
        raise ValueError(
            "velocity_pseudotime column not found in metadata. "
            "Ensure data_matrices/meta.tsv.gz is available."
        )

    pt = adata.obs["velocity_pseudotime"].values.astype(float)
    valid_mask = np.isfinite(pt)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(
            f"[data_loader]   Dropping {n_invalid:,} cells with NaN pseudotime"
        )
    adata = adata[valid_mask].copy()
    pt = pt[valid_mask]

    print(
        f"[data_loader] Building temporal expression: {adata.n_obs:,} cells, "
        f"{adata.n_vars:,} genes, {num_bins} bins"
    )

    # --- Bin cells by pseudotime ---
    bin_edges = np.linspace(pt.min(), pt.max(), num_bins + 1)
    bin_assignments = np.digitize(pt, bin_edges[1:-1])  # 0-indexed bins

    # Pseudotime bin centers
    pseudotime_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # --- Dense expression matrix for averaging ---
    # For large datasets, convert to dense per-bin to avoid memory issues
    import scipy.sparse as sp

    X = adata.X
    gene_names_out = list(adata.var_names)
    n_genes = len(gene_names_out)

    expr_matrix = np.zeros((num_bins, n_genes, 1), dtype=np.float32)
    lineage_fractions = np.zeros((num_bins, 3), dtype=np.float32)

    has_lineage = "lineage" in adata.obs.columns

    for b in range(num_bins):
        mask = bin_assignments == b
        n_cells_bin = mask.sum()
        if n_cells_bin == 0:
            continue

        # Mean expression
        X_bin = X[mask]
        if sp.issparse(X_bin):
            X_bin = X_bin.toarray()
        expr_matrix[b, :, 0] = np.mean(X_bin, axis=0).ravel()

        # Lineage fractions
        if has_lineage:
            lineages_bin = adata.obs["lineage"].values[mask]
            for lineage_name, lineage_idx in _LINEAGE_MAP.items():
                lineage_fractions[b, lineage_idx] = (
                    np.sum(lineages_bin == lineage_name) / n_cells_bin
                )

        if (b + 1) % max(1, num_bins // 5) == 0:
            print(
                f"[data_loader]   Bin {b + 1}/{num_bins}: "
                f"{n_cells_bin:,} cells, pt=[{bin_edges[b]:.3f}, {bin_edges[b+1]:.3f}]"
            )

    print(
        f"[data_loader]   Temporal expression matrix: {expr_matrix.shape} "
        f"(bins x genes x 1)"
    )

    return expr_matrix, pseudotime_centers, gene_names_out, lineage_fractions


# ---------------------------------------------------------------------------
# RNA velocity (optional)
# ---------------------------------------------------------------------------

def load_velocity_h5ad(
    data_dir: Optional[Path] = None,
) -> "anndata.AnnData | None":
    """Load RNA velocity AnnData if available.

    Parameters
    ----------
    data_dir : Path, optional
        Root data directory.

    Returns
    -------
    anndata.AnnData or None
        The velocity AnnData with spliced/unspliced layers, or *None*
        if the file has not been downloaded yet.
    """
    import anndata as ad

    if "velocity_h5ad" in _cache:
        return _cache["velocity_h5ad"]

    if data_dir is None:
        data_dir = find_data_dir()
    data_dir = Path(data_dir)

    velo_path = data_dir / "zenodo" / "RNA_all_velo.h5ad"
    if not velo_path.exists():
        print(
            f"[data_loader] Velocity file not found: {velo_path}. "
            "Returning None (file may not be downloaded yet)."
        )
        return None

    print(f"[data_loader] Loading velocity AnnData from {velo_path} ...")
    print("[data_loader]   (This is ~6 GB and may take a while.)")
    adata = ad.read_h5ad(velo_path)
    print(
        f"[data_loader]   Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes"
    )
    if adata.layers:
        print(f"[data_loader]   Layers: {list(adata.layers.keys())}")

    _cache["velocity_h5ad"] = adata
    return adata


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_gene_index(gene_names: list[str], gene: str) -> int | None:
    """Find the index of a gene by name.

    Parameters
    ----------
    gene_names : list[str]
        Ordered list of gene names.
    gene : str
        Gene name to look up.

    Returns
    -------
    int or None
        Index in *gene_names*, or *None* if not found.
    """
    try:
        return gene_names.index(gene)
    except ValueError:
        return None


def get_key_tf_indices(gene_names: list[str]) -> dict[str, int]:
    """Return a dict mapping key TF names to their indices in *gene_names*.

    Key TFs are biologically important transcription factors for cerebral
    organoid development: GLI3, FOXG1, TBR1, DLX1, DLX2, EMX1, EOMES,
    NEUROD6.

    Parameters
    ----------
    gene_names : list[str]
        Ordered list of gene names (e.g., from ``load_pando_grn``).

    Returns
    -------
    dict[str, int]
        Mapping from TF name to index. TFs not found in *gene_names*
        are omitted with a warning.
    """
    result = {}
    for tf in KEY_TFS:
        idx = get_gene_index(gene_names, tf)
        if idx is not None:
            result[tf] = idx
        else:
            logger.warning("Key TF '%s' not found in gene_names", tf)
    found = len(result)
    total = len(KEY_TFS)
    if found < total:
        print(
            f"[data_loader] WARNING: Found {found}/{total} key TFs in gene list. "
            f"Missing: {[tf for tf in KEY_TFS if tf not in result]}"
        )
    return result


# ---------------------------------------------------------------------------
# Quick summary (for interactive use / testing)
# ---------------------------------------------------------------------------

def summarize_available_data(data_dir: Optional[Path] = None) -> None:
    """Print a summary of which data files are available."""
    if data_dir is None:
        data_dir = find_data_dir()
    data_dir = Path(data_dir)

    files = {
        "Pando GRN (grn_modules.tsv)": data_dir / "pando" / "grn_modules.tsv",
        "Pando GRN (coefs.tsv)": data_dir / "pando" / "coefs.tsv",
        "RNA expression (RNA_data.h5ad)": data_dir / "expression" / "RNA_data.h5ad",
        "Count matrix (counts.mtx.gz)": data_dir / "zenodo" / "data_matrices" / "counts.mtx.gz",
        "Features (features.tsv.gz)": data_dir / "zenodo" / "data_matrices" / "features.tsv.gz",
        "Barcodes (barcodes.tsv.gz)": data_dir / "zenodo" / "data_matrices" / "barcodes.tsv.gz",
        "Metadata (meta.tsv.gz)": data_dir / "zenodo" / "data_matrices" / "meta.tsv.gz",
        "RNA velocity (RNA_all_velo.h5ad)": data_dir / "zenodo" / "RNA_all_velo.h5ad",
    }

    print(f"\n{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"{'='*60}")
    for label, path in files.items():
        exists = path.exists()
        size = ""
        if exists:
            size_bytes = path.stat().st_size
            if size_bytes > 1e9:
                size = f" ({size_bytes / 1e9:.1f} GB)"
            elif size_bytes > 1e6:
                size = f" ({size_bytes / 1e6:.1f} MB)"
            elif size_bytes > 1e3:
                size = f" ({size_bytes / 1e3:.1f} KB)"
        status = "FOUND" + size if exists else "MISSING"
        print(f"  [{status:>20s}]  {label}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summarize_available_data()
