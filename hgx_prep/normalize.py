"""Expression normalization: library-size, log1p, z-score."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

def normalize_expression(
    counts: sp.spmatrix | np.ndarray,
    *,
    target_sum: float = 1e4,
    pre_normalized: bool = False,
) -> np.ndarray:
    """Normalize a (genes x cells) count matrix.

    Steps: library-size normalization -> log1p -> z-score per gene.
    If pre_normalized=True, skip library-size norm (input already log-transformed).
    Returns dense float32 array (genes x cells).
    """
    if pre_normalized:
        if sp.issparse(counts):
            log_expr = counts.toarray().astype(np.float64)
        else:
            log_expr = np.asarray(counts, dtype=np.float64)
    else:
        if sp.issparse(counts):
            counts = sp.csc_matrix(counts, dtype=np.float64)
        else:
            counts = np.asarray(counts, dtype=np.float64)

        # Library size normalization (per cell = per column)
        lib_sizes = np.asarray(counts.sum(axis=0)).ravel()
        lib_sizes[lib_sizes == 0] = 1.0

        if sp.issparse(counts):
            inv_lib = sp.diags(target_sum / lib_sizes)
            normalized = counts @ inv_lib
            log_expr = np.log1p(normalized.toarray())
        else:
            normalized = counts * (target_sum / lib_sizes[np.newaxis, :])
            log_expr = np.log1p(normalized)

    # Z-score per gene (row-wise)
    means = log_expr.mean(axis=1, keepdims=True)
    stds = log_expr.std(axis=1, keepdims=True)
    stds[stds < 1e-8] = 1.0
    scaled = ((log_expr - means) / stds).astype(np.float32)
    return scaled
