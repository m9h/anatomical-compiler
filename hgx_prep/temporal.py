"""Pseudotime binning for temporal expression profiles."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class TemporalResult:
    expression: np.ndarray       # (n_bins, n_genes) float32
    pseudotime_centers: np.ndarray  # (n_bins,) float32
    lineage_fractions: np.ndarray | None  # (n_bins, n_lineages) or None
    lineage_names: list[str] | None

def bin_temporal(
    expression: np.ndarray,
    pseudotime: np.ndarray,
    *,
    num_bins: int = 10,
    lineage: np.ndarray | None = None,
    lineage_names: list[str] | None = None,
) -> TemporalResult:
    """Bin expression by pseudotime.

    Args:
        expression: (n_genes, n_cells) scaled expression matrix
        pseudotime: (n_cells,) pseudotime values in [0, 1]
        num_bins: number of equal-width bins
        lineage: (n_cells,) string lineage labels (optional)
        lineage_names: ordered list of lineage names (optional)
    """
    n_genes = expression.shape[0]
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    temporal_expression = np.zeros((num_bins, n_genes), dtype=np.float32)
    pseudotime_centers = np.zeros(num_bins, dtype=np.float32)

    n_lineages = len(lineage_names) if lineage_names else 0
    lineage_fractions = (
        np.zeros((num_bins, n_lineages), dtype=np.float32) if n_lineages > 0 else None
    )

    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b < num_bins - 1:
            mask = (pseudotime >= lo) & (pseudotime < hi)
        else:
            mask = (pseudotime >= lo) & (pseudotime <= hi)

        if mask.sum() == 0:
            continue

        temporal_expression[b] = expression[:, mask].mean(axis=1)
        pseudotime_centers[b] = pseudotime[mask].mean()

        if lineage is not None and lineage_names and lineage_fractions is not None:
            bin_lineage = lineage[mask]
            n_bin = mask.sum()
            for li, lname in enumerate(lineage_names):
                lineage_fractions[b, li] = (bin_lineage == lname).sum() / n_bin

    return TemporalResult(
        expression=temporal_expression,
        pseudotime_centers=pseudotime_centers,
        lineage_fractions=lineage_fractions,
        lineage_names=lineage_names,
    )
