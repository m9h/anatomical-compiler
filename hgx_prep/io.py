"""Standardized output writer and validator."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


# Required files in a valid processed directory
REQUIRED_FILES = [
    "summary.json",
    "gene_names.json",
    "tf_names.json",
    "incidence.npy",
    "node_features_pca.npy",
    "perturbation_masks.npy",
    "perturbation_effects.npy",
]


def write_processed(
    out_dir: Path,
    *,
    gene_names: list[str],
    tf_names: list[str],
    incidence: np.ndarray,
    node_features_pca: np.ndarray,
    perturbation_masks: np.ndarray,
    perturbation_effects: np.ndarray,
    eigenvalues: np.ndarray | None = None,
    module_labels: np.ndarray | None = None,
    temporal_expression: np.ndarray | None = None,
    pseudotime_centers: np.ndarray | None = None,
    lineage_fractions: np.ndarray | None = None,
    fate_probabilities: np.ndarray | None = None,
    perturbation_fates: np.ndarray | None = None,
    cell_type_fractions: np.ndarray | None = None,
    source_doi: str | None = None,
    source_geo: str | None = None,
    grn_method: str = "unknown",
    pca_details: dict | None = None,
    extra_metadata: dict | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write all outputs to a standardized processed/ directory.

    Validates shapes and writes .npy + .json files.
    Returns the output directory path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_genes = len(gene_names)
    n_edges = len(tf_names)

    # Validate shapes
    assert incidence.shape == (n_genes, n_edges), (
        f"incidence shape {incidence.shape} != ({n_genes}, {n_edges})"
    )
    assert node_features_pca.shape[0] == n_genes, (
        f"features rows {node_features_pca.shape[0]} != {n_genes}"
    )
    assert perturbation_masks.shape[1] == n_genes, (
        f"perturbation_masks cols {perturbation_masks.shape[1]} != {n_genes}"
    )
    assert perturbation_effects.shape == perturbation_masks.shape, (
        f"effects shape {perturbation_effects.shape} != masks {perturbation_masks.shape}"
    )

    # Write arrays
    np.save(out_dir / "incidence.npy", incidence)
    np.save(out_dir / "node_features_pca.npy", node_features_pca)
    np.save(out_dir / "perturbation_masks.npy", perturbation_masks)
    np.save(out_dir / "perturbation_effects.npy", perturbation_effects)

    if eigenvalues is not None:
        np.save(out_dir / "eigenvalues.npy", eigenvalues)
    if module_labels is not None:
        np.save(out_dir / "module_labels.npy", module_labels)
    if temporal_expression is not None:
        np.save(out_dir / "temporal_expression.npy", temporal_expression)
    if pseudotime_centers is not None:
        np.save(out_dir / "pseudotime_centers.npy", pseudotime_centers)
    if lineage_fractions is not None:
        np.save(out_dir / "lineage_fractions.npy", lineage_fractions)
    if fate_probabilities is not None:
        np.save(out_dir / "fate_probabilities.npy", fate_probabilities)
    if perturbation_fates is not None:
        np.save(out_dir / "perturbation_fates.npy", perturbation_fates)
    if cell_type_fractions is not None:
        np.save(out_dir / "cell_type_fractions.npy", cell_type_fractions)
    if extra_arrays:
        for name, arr in extra_arrays.items():
            np.save(out_dir / f"{name}.npy", arr)

    # Write JSON files
    with open(out_dir / "gene_names.json", "w") as f:
        json.dump(gene_names, f)
    with open(out_dir / "tf_names.json", "w") as f:
        json.dump(tf_names, f)

    # Build summary
    summary = {
        "n_genes": n_genes,
        "n_edges": n_edges,
        "n_perturbations": int(perturbation_masks.shape[0]),
        "feature_dim": int(node_features_pca.shape[1]),
        "source_doi": source_doi,
        "source_geo": source_geo,
        "grn_method": grn_method,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "hgx-prep",
    }
    if pca_details:
        summary["pca"] = pca_details
    if extra_metadata:
        summary.update(extra_metadata)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Wrote {len(list(out_dir.glob('*.npy')))} .npy files + "
          f"{len(list(out_dir.glob('*.json')))} .json files to {out_dir}")
    return out_dir


def validate_processed(processed_dir: Path) -> tuple[bool, list[str]]:
    """Check that a processed directory has the required minimum files.

    Returns (is_valid, list_of_missing_files).
    """
    missing = []
    for fname in REQUIRED_FILES:
        if not (processed_dir / fname).exists():
            missing.append(fname)
    return len(missing) == 0, missing
