"""GRN loading and incidence matrix construction."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class GRNResult:
    incidence: np.ndarray          # (n_genes, n_edges) float32
    gene_names: list[str]          # sorted, length n_genes
    tf_names: list[str]            # sorted, length n_edges
    module_labels: np.ndarray      # (n_genes,) int32, argmax assignment (-1 if unassigned)
    tf_target_map: dict[str, list[int]]  # tf -> list of target gene indices
    method: str                    # "pando", "de", "supplied"

def load_pando_grn(
    coefs_path: Path,
    gene_names: list[str],
    *,
    padj_threshold: float = 0.05,
) -> GRNResult:
    """Build incidence matrix from Pando GRN coefficients.

    Extracted from 00_preprocess.py steps 1 + 8.

    Args:
        coefs_path: Path to Pando coefs.tsv (columns: tf, target, estimate, padj)
        gene_names: sorted list of gene names in the expression matrix
        padj_threshold: significance threshold for edges
    """
    coefs = pd.read_csv(coefs_path, sep="\t")
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    sig_coefs = coefs[coefs["padj"] < padj_threshold].copy()
    tf_names: list[str] = []
    tf_target_map: dict[str, list[int]] = {}

    for tf in sorted(sig_coefs["tf"].unique()):
        if tf not in gene_name_to_idx:
            continue
        targets = sig_coefs.loc[sig_coefs["tf"] == tf, "target"].unique()
        target_indices = [gene_name_to_idx[t] for t in targets if t in gene_name_to_idx]
        if len(target_indices) == 0:
            continue
        tf_names.append(tf)
        tf_target_map[tf] = target_indices

    n_edges = len(tf_names)
    incidence = np.zeros((n_genes, n_edges), dtype=np.float32)
    for ei, tf in enumerate(tf_names):
        tf_idx = gene_name_to_idx[tf]
        incidence[tf_idx, ei] = 1.0
        for ti in tf_target_map[tf]:
            incidence[ti, ei] = 1.0

    module_labels = _compute_module_labels(incidence, n_genes)

    return GRNResult(
        incidence=incidence,
        gene_names=gene_names,
        tf_names=tf_names,
        module_labels=module_labels,
        tf_target_map=tf_target_map,
        method="pando",
    )


def build_incidence_from_de(
    de_results: dict[str, dict],
    gene_names: list[str],
    *,
    log2fc_threshold: float = 0.25,
    pval_threshold: float = 0.05,
    gene_idx_in_source: dict[str, int] | None = None,
) -> GRNResult:
    """Build incidence matrix from per-TF differential expression.

    Extracted from preprocess_pollen.py step 7.

    Args:
        de_results: {tf_name: {"log2fc": array, "pval": array, ...}}
        gene_names: sorted list of shared gene names
        log2fc_threshold: minimum |log2FC| for significance
        pval_threshold: p-value threshold
        gene_idx_in_source: {gene_name: index_in_source_arrays} mapping
    """
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)
    tf_list = sorted(de_results.keys())
    K = len(tf_list)

    incidence = np.zeros((n_genes, K), dtype=np.float32)
    tf_target_map: dict[str, list[int]] = {}

    for ki, tf in enumerate(tf_list):
        targets = []
        if tf in gene_name_to_idx:
            incidence[gene_name_to_idx[tf], ki] = 1.0

        log2fc = de_results[tf]["log2fc"]
        pval = de_results[tf]["pval"]

        for gi, gene in enumerate(gene_names):
            if gene_idx_in_source is not None:
                src_idx = gene_idx_in_source.get(gene)
                if src_idx is None:
                    continue
            else:
                src_idx = gi

            if abs(log2fc[src_idx]) > log2fc_threshold and pval[src_idx] < pval_threshold:
                incidence[gi, ki] = 1.0
                targets.append(gi)

        tf_target_map[tf] = targets

    module_labels = _compute_module_labels(incidence, n_genes)

    return GRNResult(
        incidence=incidence,
        gene_names=gene_names,
        tf_names=tf_list,
        module_labels=module_labels,
        tf_target_map=tf_target_map,
        method="de",
    )


def load_supplied_grn(
    grn_path: Path,
    gene_names: list[str],
    *,
    padj_threshold: float = 0.05,
) -> GRNResult:
    """Load a user-supplied GRN edge list as a Pando-format TSV.

    Expects columns: tf, target, estimate (optional), padj (optional).
    Falls back to treating all edges as significant if padj is missing.
    """
    df = pd.read_csv(grn_path, sep="\t")

    if "padj" in df.columns:
        df = df[df["padj"] < padj_threshold]

    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    tf_names: list[str] = []
    tf_target_map: dict[str, list[int]] = {}

    for tf in sorted(df["tf"].unique()):
        if tf not in gene_name_to_idx:
            continue
        targets = df.loc[df["tf"] == tf, "target"].unique()
        target_indices = [gene_name_to_idx[t] for t in targets if t in gene_name_to_idx]
        if len(target_indices) == 0:
            continue
        tf_names.append(tf)
        tf_target_map[tf] = target_indices

    n_edges = len(tf_names)
    incidence = np.zeros((n_genes, n_edges), dtype=np.float32)
    for ei, tf in enumerate(tf_names):
        incidence[gene_name_to_idx[tf], ei] = 1.0
        for ti in tf_target_map[tf]:
            incidence[ti, ei] = 1.0

    module_labels = _compute_module_labels(incidence, n_genes)

    return GRNResult(
        incidence=incidence,
        gene_names=gene_names,
        tf_names=tf_names,
        module_labels=module_labels,
        tf_target_map=tf_target_map,
        method="supplied",
    )


def _compute_module_labels(incidence: np.ndarray, n_genes: int) -> np.ndarray:
    """Assign each gene to its primary module (argmax of incidence row)."""
    row_sums = incidence.sum(axis=1)
    module_labels = np.full(n_genes, -1, dtype=np.int32)
    assigned_mask = row_sums > 0
    module_labels[assigned_mask] = incidence[assigned_mask].argmax(axis=1).astype(np.int32)
    return module_labels
