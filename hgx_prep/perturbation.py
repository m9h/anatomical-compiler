"""Perturbation effect computation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import scipy.sparse as sp

@dataclass
class PerturbationResult:
    masks: np.ndarray          # (K, n_genes) bool
    effects: np.ndarray        # (K, n_genes) float32
    tf_list: list[str]         # length K
    method: str                # "observed_de" or "grn_propagation"


def compute_de_perturbations(
    expression: sp.spmatrix | np.ndarray,
    guide_labels: np.ndarray,
    controls: list[str],
    gene_names: list[str],
    *,
    gene_idx_in_source: dict[str, int] | None = None,
    min_cells: int = 10,
) -> PerturbationResult:
    """Compute perturbation effects from observed Perturb-seq DE.

    Extracted from preprocess_pollen.py steps 3+5.

    Args:
        expression: (cells x genes) expression matrix (AnnData convention)
        guide_labels: (n_cells,) string labels for each cell's perturbation
        controls: list of control label strings
        gene_names: sorted list of output gene names
        gene_idx_in_source: {gene_name: column_index_in_expression}
        min_cells: minimum cells per TF to compute DE
    """
    from scipy.stats import norm

    n_genes = len(gene_names)
    ctrl_mask = np.isin(guide_labels, controls)

    # Control statistics
    if sp.issparse(expression):
        ctrl_mean = np.asarray(expression[ctrl_mask].mean(axis=0)).ravel()
        ctrl_var = np.asarray(
            expression[ctrl_mask].power(2).mean(axis=0)
        ).ravel() - ctrl_mean ** 2
    else:
        X = np.asarray(expression, dtype=np.float32)
        ctrl_mean = X[ctrl_mask].mean(axis=0)
        ctrl_var = X[ctrl_mask].var(axis=0)

    n_ctrl = ctrl_mask.sum()

    # Per-TF DE
    unique_tfs = sorted(set(guide_labels) - set(controls))
    tf_list = []
    masks_list = []
    effects_list = []

    for tf in unique_tfs:
        tf_mask = guide_labels == tf
        n_tf = tf_mask.sum()
        if n_tf < min_cells:
            continue

        if sp.issparse(expression):
            tf_mean = np.asarray(expression[tf_mask].mean(axis=0)).ravel()
        else:
            tf_mean = X[tf_mask].mean(axis=0)

        # log2FC
        pseudo = 0.1
        log2fc = np.log2(tf_mean + pseudo) - np.log2(ctrl_mean + pseudo)

        # Build output arrays for shared genes
        mask = np.zeros(n_genes, dtype=bool)
        effect = np.zeros(n_genes, dtype=np.float32)

        if tf in (gene_idx_in_source or {}):
            # Mark the TF itself in the shared gene space
            for gi, gene in enumerate(gene_names):
                if gene == tf:
                    mask[gi] = True
                    break
        elif gene_idx_in_source is None:
            # Direct mapping
            for gi, gene in enumerate(gene_names):
                if gene == tf:
                    mask[gi] = True
                    break

        for gi, gene in enumerate(gene_names):
            if gene_idx_in_source is not None:
                src_idx = gene_idx_in_source.get(gene)
                if src_idx is None:
                    continue
            else:
                src_idx = gi
            effect[gi] = log2fc[src_idx]

        # Normalize to unit variance
        std = effect.std()
        if std > 1e-8:
            effect = effect / std

        tf_list.append(tf)
        masks_list.append(mask)
        effects_list.append(effect)

    return PerturbationResult(
        masks=np.array(masks_list, dtype=bool),
        effects=np.array(effects_list, dtype=np.float32),
        tf_list=tf_list,
        method="observed_de",
    )


def compute_grn_perturbations(
    coefs: pd.DataFrame,
    gene_name_to_idx: dict[str, int],
    key_tfs: list[str],
    n_genes: int,
    *,
    decay_2hop: float = 0.4,
) -> PerturbationResult:
    """Compute perturbation effects via GRN coefficient propagation.

    Extracted from 00_preprocess.py step 9.
    Uses 1-hop direct GRN coefficients + 2-hop propagation through
    intermediate TFs with a decay factor.

    Args:
        coefs: Pando coefficients DataFrame (tf, target, estimate, padj)
        gene_name_to_idx: {gene_name: index} mapping
        key_tfs: list of TFs to simulate knockouts for
        n_genes: total number of genes
        decay_2hop: decay factor for 2-hop propagation
    """
    sig_coefs = coefs[coefs["padj"] < 0.05]

    # Build per-TF target map with coefficients
    tf_to_targets: dict[str, dict[str, float]] = {}
    for tf in sorted(sig_coefs["tf"].unique()):
        tf_rows = sig_coefs[sig_coefs["tf"] == tf]
        tf_to_targets[tf] = {
            row["target"]: float(row["estimate"])
            for _, row in tf_rows.iterrows()
            if row["target"] in gene_name_to_idx
        }

    available_tfs = [tf for tf in key_tfs if tf in gene_name_to_idx]
    K = len(available_tfs)

    masks = np.zeros((K, n_genes), dtype=bool)
    effects = np.zeros((K, n_genes), dtype=np.float32)

    for ki, tf in enumerate(available_tfs):
        tf_idx = gene_name_to_idx[tf]
        masks[ki, tf_idx] = True

        # 1-hop: direct GRN coefficients
        direct_targets = tf_to_targets.get(tf, {})
        for target, estimate in direct_targets.items():
            tidx = gene_name_to_idx[target]
            effects[ki, tidx] = -estimate  # KO = reverse estimated effect

        # 2-hop: propagate through intermediate TFs
        for intermediate, coef_tf_inter in direct_targets.items():
            if intermediate not in tf_to_targets:
                continue
            for gene2, coef_inter_gene2 in tf_to_targets[intermediate].items():
                gidx = gene_name_to_idx[gene2]
                if effects[ki, gidx] == 0.0:  # don't override 1-hop
                    effects[ki, gidx] = -coef_tf_inter * coef_inter_gene2 * decay_2hop

        # Normalize to unit variance
        std = effects[ki].std()
        if std > 1e-8:
            effects[ki] /= std

    return PerturbationResult(
        masks=masks,
        effects=effects,
        tf_list=available_tfs,
        method="grn_propagation",
    )
