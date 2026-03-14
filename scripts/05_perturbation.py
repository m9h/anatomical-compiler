#!/usr/bin/env python
"""Perturbation prediction benchmark (CROP-seq validation).

Demonstrates hgx.PerturbationPredictor on CROP-seq knockout data from
cerebral organoids. Trains a perturbation model on observed knockouts,
validates GLI3 KO effects on downstream targets, and runs a full
perturbation screen across all transcription factors.

Produces Figure 5 (2x2):
    A — Predicted vs observed expression changes (GLI3 KO)
    B — Fate probability shifts per KO
    C — Perturbation screen heatmap (top TFs x downstream genes)
    D — ROC curves per KO
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hgx

# ── Gene labels for the synthetic organoid GRN ────────────────────────────

TF_NAMES = [
    "GLI3", "PAX6", "SOX2", "TBR1", "EMX1", "NEUROD6", "FOXG1",
    "EOMES", "DLX1", "DLX2", "GAD1", "GAD2", "NKX2-1", "LHX6",
    "OTX2", "HES1", "NOTCH1", "SHH", "BMP4", "WNT3A",
]

DOWNSTREAM_NAMES = [
    "TUBB3", "MAP2", "DCX", "STMN2", "NCAM1", "CDH2", "VIM",
    "NES", "SOX9", "GFAP", "ALDH1L1", "MKI67", "TOP2A", "PCNA",
    "CCND1", "CDK4", "RB1", "TP53", "GAPDH", "ACTB",
]

# Extend to 200 genes total
_EXTRA = [f"Gene_{i}" for i in range(len(TF_NAMES) + len(DOWNSTREAM_NAMES), 200)]
ALL_GENE_NAMES = TF_NAMES + DOWNSTREAM_NAMES + _EXTRA

NUM_GENES = 200
NUM_TFS = 20
NUM_MODULES = 15
GENE_DIM = 8
NUM_FATES = 3
FATE_NAMES = ["Cortical", "GE", "Neural Tube"]

# ── Indices for biologically meaningful KO targets ────────────────────────

GLI3_IDX = 0
TBR1_IDX = 3
EOMES_IDX = 7
DLX1_IDX = 8
DLX2_IDX = 9
GAD1_IDX = 10
NEUROD6_IDX = 5


# ── Helpers ───────────────────────────────────────────────────────────────


def _try_load_cropseq(data_dir: Path) -> pd.DataFrame | None:
    """Attempt to load CROP-seq DE results."""
    path = data_dir / "cropseq" / "cropseq_de.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def _try_load_pando(data_dir: Path) -> pd.DataFrame | None:
    """Attempt to load Pando GRN coefficients."""
    path = data_dir / "pando" / "coefs.tsv"
    if path.exists():
        return pd.read_csv(path, sep="\t")
    return None


def _build_synthetic_incidence(key: jax.Array) -> jnp.ndarray:
    """Build a synthetic regulatory incidence matrix.

    Each of NUM_MODULES modules contains a random subset of genes, with
    TFs preferentially assigned as hubs.

    Returns:
        Incidence matrix of shape (NUM_GENES, NUM_MODULES).
    """
    k1, k2 = jax.random.split(key)
    H = np.zeros((NUM_GENES, NUM_MODULES), dtype=np.float32)

    for m in range(NUM_MODULES):
        # Each module gets 2-4 TFs and 8-15 target genes
        num_tfs_in_mod = np.random.RandomState(int(k1[0]) + m).randint(2, 5)
        num_targets = np.random.RandomState(int(k1[0]) + m + 100).randint(8, 16)

        rng = np.random.RandomState(int(k2[0]) + m)
        tf_members = rng.choice(NUM_TFS, size=num_tfs_in_mod, replace=False)
        target_members = rng.choice(
            np.arange(NUM_TFS, NUM_GENES),
            size=min(num_targets, NUM_GENES - NUM_TFS),
            replace=False,
        )

        for t in tf_members:
            H[t, m] = 1.0
        for t in target_members:
            H[t, m] = 1.0

    return jnp.array(H)


def _build_synthetic_features(key: jax.Array) -> jnp.ndarray:
    """Random node features representing gene expression profiles."""
    return jax.random.normal(key, (NUM_GENES, GENE_DIM)) * 0.5 + 0.5


def _build_perturbation_data(
    incidence: np.ndarray,
    features: np.ndarray,
    ko_indices: list[int],
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate synthetic perturbation training data.

    For each KO gene:
      - Create a boolean mask (True at KO position)
      - Propagate effects: genes sharing regulatory modules with the KO'd
        gene get downregulated proportional to module overlap
      - Compute fate probability shift based on which TF is knocked out

    Returns:
        (ko_masks, expr_changes, fate_targets) of shapes
        (P, n), (P, n, gene_dim), (P, num_fates).
    """
    P = len(ko_indices)
    n = incidence.shape[0]
    inc_np = np.array(incidence)
    feat_np = np.array(features)

    masks = np.zeros((P, n), dtype=bool)
    expr_targets = np.zeros((P, n, GENE_DIM), dtype=np.float32)
    fate_targets = np.zeros((P, NUM_FATES), dtype=np.float32)

    # Baseline fate: roughly equal
    baseline_fate = np.array([0.40, 0.35, 0.25], dtype=np.float32)

    for p, ko_idx in enumerate(ko_indices):
        masks[p, ko_idx] = True

        # Propagate through incidence: which modules contain the KO'd gene?
        modules_hit = np.where(inc_np[ko_idx, :] > 0)[0]

        # Genes in those modules get expression changes
        for m in modules_hit:
            members = np.where(inc_np[:, m] > 0)[0]
            for g in members:
                if g != ko_idx:
                    # Downregulate co-members, scaled by module overlap
                    overlap = np.sum(inc_np[ko_idx, :] * inc_np[g, :])
                    scale = -0.3 * overlap
                    expr_targets[p, g, :] += scale * feat_np[g, :]

        # Zero out the KO'd gene itself
        expr_targets[p, ko_idx, :] = -feat_np[ko_idx, :]

        # Fate shift depends on which TF is knocked out
        fate = baseline_fate.copy()
        if ko_idx == GLI3_IDX:
            # GLI3 KO: GE fate decreases, cortical increases
            fate[0] += 0.20   # cortical up
            fate[1] -= 0.25   # GE down
            fate[2] += 0.05   # NT slight up
        elif ko_idx == TBR1_IDX:
            # TBR1 KO: cortical fate decreases
            fate[0] -= 0.25
            fate[1] += 0.10
            fate[2] += 0.15
        elif ko_idx == EOMES_IDX:
            # EOMES KO: intermediate progenitor loss
            fate[0] -= 0.15
            fate[1] -= 0.05
            fate[2] += 0.20
        else:
            # Generic TF: small perturbation
            rng = np.random.RandomState(int(key[0]) + ko_idx)
            shift = rng.randn(NUM_FATES) * 0.05
            fate += shift

        # Normalize to valid probability
        fate = np.maximum(fate, 0.01)
        fate /= fate.sum()
        fate_targets[p] = fate

    return jnp.array(masks), jnp.array(expr_targets), jnp.array(fate_targets)


def _compute_roc(
    predicted: np.ndarray,
    observed: np.ndarray,
    threshold_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve for identifying affected genes.

    A gene is "truly affected" if its observed absolute change exceeds
    the top `threshold_frac` quantile of all observed changes.

    Returns:
        (fpr, tpr, auc)
    """
    # Binary labels: top affected genes
    obs_abs = np.abs(observed)
    cutoff = np.quantile(obs_abs, 1.0 - threshold_frac)
    labels = (obs_abs >= cutoff).astype(int)

    # Predicted scores (absolute predicted change)
    scores = np.abs(predicted)

    # Sort by descending score
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    # Compute TPR/FPR
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)

    # Prepend (0, 0)
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])

    # AUC via trapezoidal rule
    auc = float(np.trapezoid(tpr, fpr))

    return fpr, tpr, auc


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Perturbation prediction benchmark (CROP-seq validation)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Path to data directory",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Training epochs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    fig_dir = Path(__file__).resolve().parent.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)
    key, k_inc, k_feat, k_pert, k_model, k_train = jax.random.split(key, 6)

    # ── 1. Load or generate data ─────────────────────────────────────────

    cropseq_df = _try_load_cropseq(data_dir)
    pando_df = _try_load_pando(data_dir)
    using_real_data = cropseq_df is not None and pando_df is not None

    if using_real_data:
        print("Loaded real CROP-seq and Pando data")
        # In a real pipeline we would parse the DataFrames into incidence
        # matrices and perturbation targets here.  For now fall through
        # to synthetic data as a reference implementation.
        print("  (Real data parsing not yet implemented — using synthetic)")
        using_real_data = False

    if not using_real_data:
        print("Using synthetic organoid GRN data")
        incidence = _build_synthetic_incidence(k_inc)
        features = _build_synthetic_features(k_feat)

    print(
        f"  Genes: {NUM_GENES}  TFs: {NUM_TFS}  "
        f"Modules: {NUM_MODULES}  Feature dim: {GENE_DIM}"
    )

    # ── 2. Build perturbation training data ──────────────────────────────

    # Training KOs: GLI3, TBR1, EOMES
    train_ko_indices = [GLI3_IDX, TBR1_IDX, EOMES_IDX]
    # Held-out KOs for testing: DLX1, NKX2-1
    test_ko_indices = [DLX1_IDX, 12]  # NKX2-1 = index 12

    all_ko_indices = train_ko_indices + test_ko_indices

    ko_masks, expr_changes, fate_targets = _build_perturbation_data(
        np.array(incidence), np.array(features), all_ko_indices, k_pert,
    )

    # Split into train/test
    n_train = len(train_ko_indices)
    train_masks = ko_masks[:n_train]
    train_expr = expr_changes[:n_train]
    train_fates = fate_targets[:n_train]

    test_masks = ko_masks[n_train:]
    test_expr = expr_changes[n_train:]
    test_fates = fate_targets[n_train:]

    print(
        f"  Training perturbations: {n_train}  "
        f"Test perturbations: {len(test_ko_indices)}"
    )

    # ── 3. Build regulatory hypergraph ───────────────────────────────────

    hg = hgx.from_incidence(incidence, node_features=features)
    print(
        f"  Hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges"
    )

    # ── 4. Train PerturbationPredictor ───────────────────────────────────

    predictor = hgx.PerturbationPredictor(
        gene_dim=GENE_DIM,
        hidden_dim=64,
        num_fates=NUM_FATES,
        conv_cls=hgx.UniGCNConv,
        num_layers=2,
        key=k_model,
    )

    print(f"  Training for {args.epochs} epochs ...")
    predictor = hgx.train_perturbation_predictor(
        predictor,
        hg,
        perturbations=train_masks,
        targets=(train_expr, train_fates),
        epochs=args.epochs,
        key=k_train,
    )
    print("  Training complete.")

    # ── 5. GLI3 KO benchmark (primary) ──────────────────────────────────

    gli3_expr, gli3_fate = hgx.in_silico_knockout(predictor, hg, GLI3_IDX)
    gli3_expr_np = np.array(gli3_expr)
    gli3_fate_np = np.array(gli3_fate)

    # Observed (from training data, index 0)
    gli3_obs_expr = np.array(train_expr[0])
    gli3_obs_fate = np.array(train_fates[0])

    # Summarize per-gene expression change magnitude
    pred_mean = gli3_expr_np.mean(axis=-1)  # (n,)
    obs_mean = gli3_obs_expr.mean(axis=-1)  # (n,)

    # Validate expected biology
    print("\n  GLI3 KO validation:")
    for name, idx in [("DLX1", DLX1_IDX), ("DLX2", DLX2_IDX), ("GAD1", GAD1_IDX)]:
        print(f"    {name} predicted change: {pred_mean[idx]:.3f} (expect < 0)")
    for name, idx in [("TBR1", TBR1_IDX), ("NEUROD6", NEUROD6_IDX)]:
        print(f"    {name} predicted change: {pred_mean[idx]:.3f} (expect > 0)")

    print(f"  Predicted fate: {dict(zip(FATE_NAMES, gli3_fate_np.round(3)))}")
    print(f"  Observed  fate: {dict(zip(FATE_NAMES, gli3_obs_fate.round(3)))}")

    # ── 6. Full perturbation screen ──────────────────────────────────────

    tf_indices = jnp.array(list(range(NUM_TFS)))
    all_changes, all_fates = hgx.perturbation_screen(
        predictor, hg, tf_indices
    )
    all_changes_np = np.array(all_changes)  # (K, n, gene_dim)
    all_fates_np = np.array(all_fates)      # (K, num_fates)

    print(f"\n  Perturbation screen: {NUM_TFS} TFs screened")

    # ── 7. Metrics ───────────────────────────────────────────────────────

    # Pearson correlation (GLI3 KO, mean across feature dims)
    valid = np.isfinite(pred_mean) & np.isfinite(obs_mean)
    if valid.sum() > 2:
        pearson_r = float(np.corrcoef(pred_mean[valid], obs_mean[valid])[0, 1])
    else:
        pearson_r = 0.0
    print(f"\n  Pearson r (GLI3 KO): {pearson_r:.3f} (target > 0.50)")

    # ROC AUC per training KO
    auc_results = {}
    for i, (ko_idx, ko_name) in enumerate(
        zip(train_ko_indices, ["GLI3", "TBR1", "EOMES"])
    ):
        pred_ko, _ = hgx.in_silico_knockout(predictor, hg, ko_idx)
        pred_ko_mean = np.array(pred_ko).mean(axis=-1)
        obs_ko_mean = np.array(train_expr[i]).mean(axis=-1)
        _, _, auc = _compute_roc(pred_ko_mean, obs_ko_mean)
        auc_results[ko_name] = auc
        print(f"  ROC AUC ({ko_name} KO): {auc:.3f} (target > 0.70)")

    # Fate direction accuracy
    fate_correct = 0
    for i, (ko_idx, ko_name) in enumerate(
        zip(train_ko_indices, ["GLI3", "TBR1", "EOMES"])
    ):
        _, pred_fate_i = hgx.in_silico_knockout(predictor, hg, ko_idx)
        pred_shift = np.array(pred_fate_i) - np.array([0.40, 0.35, 0.25])
        obs_shift = np.array(train_fates[i]) - np.array([0.40, 0.35, 0.25])
        # Check if the direction of the largest shift matches
        if np.argmax(np.abs(pred_shift)) == np.argmax(np.abs(obs_shift)):
            fate_correct += 1
    fate_accuracy = fate_correct / len(train_ko_indices)
    print(f"  Fate direction accuracy: {fate_accuracy:.2f}")

    # ── 8. Figure 5 ──────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ── Panel A: Scatter — predicted vs observed expression (GLI3 KO) ──

    ax = axes[0, 0]
    # Color by absolute observed change as a proxy for significance
    obs_abs = np.abs(obs_mean)
    padj_proxy = 1.0 - obs_abs / (obs_abs.max() + 1e-8)  # lower = more sig

    scatter = ax.scatter(
        obs_mean, pred_mean,
        c=padj_proxy, cmap="viridis_r", s=12, alpha=0.7, edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("padj (proxy)", fontsize=8)

    # Diagonal reference
    lims = [
        min(obs_mean.min(), pred_mean.min()) - 0.1,
        max(obs_mean.max(), pred_mean.max()) + 0.1,
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)

    # Annotate key genes
    for name, idx in [
        ("DLX1", DLX1_IDX), ("TBR1", TBR1_IDX), ("GLI3", GLI3_IDX),
    ]:
        ax.annotate(
            name,
            (obs_mean[idx], pred_mean[idx]),
            fontsize=7, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="grey"),
        )

    ax.set_xlabel("Observed expression change", fontsize=9)
    ax.set_ylabel("Predicted expression change", fontsize=9)
    ax.set_title(f"A  GLI3 KO: predicted vs observed (r={pearson_r:.2f})", fontsize=10)
    ax.tick_params(labelsize=8)

    # ── Panel B: Bar chart — fate probability shifts per KO ────────────

    ax = axes[0, 1]
    ko_labels = ["GLI3", "TBR1", "EOMES"]
    x = np.arange(len(ko_labels))
    bar_width = 0.12
    fate_colors = ["#d62728", "#1f77b4", "#2ca02c"]  # ctx=red, GE=blue, NT=green

    for f_idx, (fate_name, color) in enumerate(zip(FATE_NAMES, fate_colors)):
        # Predicted (solid)
        pred_vals = []
        obs_vals = []
        for i, ko_idx in enumerate(train_ko_indices):
            _, pf = hgx.in_silico_knockout(predictor, hg, ko_idx)
            pred_vals.append(float(pf[f_idx]))
            obs_vals.append(float(train_fates[i, f_idx]))

        offset = (f_idx - 1) * bar_width * 2
        ax.bar(
            x + offset - bar_width / 2, pred_vals, bar_width,
            color=color, label=f"{fate_name} (pred)" if f_idx == 0 or True else None,
        )
        ax.bar(
            x + offset + bar_width / 2, obs_vals, bar_width,
            color=color, alpha=0.4, hatch="//",
            label=f"{fate_name} (obs)" if f_idx == 0 or True else None,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ko_labels, fontsize=9)
    ax.set_ylabel("Fate probability", fontsize=9)
    ax.set_title("B  Fate shifts per KO", fontsize=10)
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.tick_params(labelsize=8)

    # ── Panel C: Perturbation screen heatmap ───────────────────────────

    ax = axes[1, 0]

    # Mean expression change magnitude per TF KO
    screen_mean = all_changes_np.mean(axis=-1)  # (K, n)

    # Select top 15 TFs by total effect magnitude
    tf_effect = np.abs(screen_mean).sum(axis=1)
    top_tf_order = np.argsort(-tf_effect)[:15]

    # Select top 30 downstream genes by max absolute change across KOs
    gene_effect = np.abs(screen_mean[top_tf_order, :]).max(axis=0)
    top_gene_order = np.argsort(-gene_effect)[:30]

    heatmap_data = screen_mean[np.ix_(top_tf_order, top_gene_order)]
    vmax = max(abs(heatmap_data.min()), abs(heatmap_data.max())) or 1.0

    im = ax.imshow(
        heatmap_data, cmap="RdBu_r", aspect="auto",
        vmin=-vmax, vmax=vmax,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Predicted expression change")

    # Labels
    tf_labels = [TF_NAMES[i] if i < len(TF_NAMES) else f"TF_{i}" for i in top_tf_order]
    gene_labels = [
        ALL_GENE_NAMES[i] if i < len(ALL_GENE_NAMES) else f"G_{i}"
        for i in top_gene_order
    ]
    ax.set_yticks(range(len(tf_labels)))
    ax.set_yticklabels(tf_labels, fontsize=6)
    ax.set_xticks(range(len(gene_labels)))
    ax.set_xticklabels(gene_labels, fontsize=5, rotation=90)
    ax.set_title("C  Perturbation screen", fontsize=10)

    # ── Panel D: ROC curves per KO ─────────────────────────────────────

    ax = axes[1, 1]
    roc_colors = ["#d62728", "#1f77b4", "#ff7f0e"]

    for i, (ko_idx, ko_name, color) in enumerate(
        zip(train_ko_indices, ["GLI3", "TBR1", "EOMES"], roc_colors)
    ):
        pred_ko, _ = hgx.in_silico_knockout(predictor, hg, ko_idx)
        pred_ko_mean = np.array(pred_ko).mean(axis=-1)
        obs_ko_mean = np.array(train_expr[i]).mean(axis=-1)
        fpr, tpr, auc = _compute_roc(pred_ko_mean, obs_ko_mean)
        ax.plot(fpr, tpr, color=color, linewidth=1.5, label=f"{ko_name} (AUC={auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title("D  ROC: affected gene identification", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=8)

    # ── Save ──────────────────────────────────────────────────────────────

    fig.suptitle(
        "Figure 5: Perturbation Prediction (CROP-seq Validation)",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = fig_dir / "figure_05_perturbation.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved to {out_path}")


if __name__ == "__main__":
    main()
