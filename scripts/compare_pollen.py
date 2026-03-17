#!/usr/bin/env python3
"""Cross-dataset comparison: Fleck et al. organoids vs Pollen/Ding et al. primary tissue.

Evaluates whether a perturbation predictor trained on organoid GRN data
(Fleck et al. 2023) generalizes to primary cortical CRISPRi perturbations
(Pollen/Ding et al. 2026). This is the key benchmark for Phase 3.

Comparisons:
  1. GRN overlap — How many Fleck regulon TFs are also perturbed in Pollen?
  2. Direction concordance — Do shared TF perturbations show same downstream
     gene direction in both datasets?
  3. Transfer prediction — Train PerturbationPredictor on Fleck, test on Pollen
  4. Within-dataset baseline — Train+test within Pollen (upper bound)

Produces: figures/pollen_comparison.png (4-panel figure)

Usage:
    uv run python scripts/compare_pollen.py
    uv run python scripts/compare_pollen.py --fleck-dir data/processed --pollen-dir data/pollen/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import hgx
    import jax
    import jax.numpy as jnp
    HAS_HGX = True
except ImportError:
    HAS_HGX = False

try:
    import devograph
    HAS_DEVOGRAPH = True
except ImportError:
    HAS_DEVOGRAPH = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: Path, name: str) -> dict | None:
    """Load preprocessed dataset arrays."""
    d = {"name": name, "dir": data_dir}

    for fname in ["gene_names", "tf_names"]:
        path = data_dir / f"{fname}.json"
        if path.exists():
            with open(path) as f:
                d[fname] = json.load(f)
        else:
            print(f"  WARNING: {path} not found")
            return None

    for fname in ["perturbation_masks", "perturbation_effects",
                   "incidence", "node_features_pca"]:
        path = data_dir / f"{fname}.npy"
        if path.exists():
            d[fname] = np.load(path)
        else:
            print(f"  WARNING: {path} not found")

    summary_path = data_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            d["summary"] = json.load(f)

    return d


# ---------------------------------------------------------------------------
# 1. GRN Overlap Analysis
# ---------------------------------------------------------------------------

def analyze_grn_overlap(fleck: dict, pollen: dict) -> dict:
    """Analyze overlap between Fleck GRN TFs and Pollen CRISPRi targets."""
    print("\n" + "=" * 64)
    print("  Comparison 1: GRN / Perturbation Overlap")
    print("=" * 64)

    fleck_tfs = set(fleck["tf_names"])
    pollen_tfs = set(pollen["tf_names"])
    fleck_genes = set(fleck["gene_names"])
    pollen_genes = set(pollen["gene_names"])

    shared_tfs = fleck_tfs & pollen_tfs
    shared_genes = fleck_genes & pollen_genes

    print(f"  Fleck GRN TFs:       {len(fleck_tfs)}")
    print(f"  Pollen CRISPRi TFs:  {len(pollen_tfs)}")
    print(f"  Shared TFs:          {len(shared_tfs)}")
    if shared_tfs:
        print(f"  Shared TFs:          {sorted(shared_tfs)}")
    print(f"  Fleck genes:         {len(fleck_genes)}")
    print(f"  Pollen genes:        {len(pollen_genes)}")
    print(f"  Shared genes:        {len(shared_genes)}")

    # Hypergeometric test: is the TF overlap more than expected by chance?
    # Population = all human TFs (~1600), drawn = fleck_tfs, successes = pollen_tfs
    N_human_tfs = 1600  # approximate
    if shared_tfs:
        pval = float(stats.hypergeom.sf(
            len(shared_tfs) - 1,
            N_human_tfs,
            len(pollen_tfs),
            len(fleck_tfs),
        ))
        print(f"  Enrichment p-value:  {pval:.4g}")
    else:
        pval = 1.0
        print("  No shared TFs — cannot compute enrichment")

    return {
        "fleck_tfs": sorted(fleck_tfs),
        "pollen_tfs": sorted(pollen_tfs),
        "shared_tfs": sorted(shared_tfs),
        "n_shared_genes": len(shared_genes),
        "enrichment_pval": pval,
    }


# ---------------------------------------------------------------------------
# 2. Direction Concordance
# ---------------------------------------------------------------------------

def analyze_direction_concordance(fleck: dict, pollen: dict) -> dict:
    """Compare DE direction for shared TFs across shared genes."""
    print("\n" + "=" * 64)
    print("  Comparison 2: Direction Concordance (Shared TF Perturbations)")
    print("=" * 64)

    fleck_genes = fleck["gene_names"]
    pollen_genes = pollen["gene_names"]
    fleck_tfs = fleck["tf_names"]
    pollen_tfs = pollen["tf_names"]

    fleck_gene_idx = {g: i for i, g in enumerate(fleck_genes)}
    pollen_gene_idx = {g: i for i, g in enumerate(pollen_genes)}

    shared_tfs = set(fleck_tfs) & set(pollen_tfs)
    shared_genes = sorted(set(fleck_genes) & set(pollen_genes))

    if not shared_tfs:
        print("  No shared TFs — skipping concordance analysis")
        return {"concordance": None, "reason": "no_shared_tfs"}

    fleck_eff = fleck.get("perturbation_effects")
    pollen_eff = pollen.get("perturbation_effects")

    if fleck_eff is None or pollen_eff is None:
        print("  Missing perturbation effects arrays")
        return {"concordance": None, "reason": "missing_effects"}

    results = {}
    all_concordances = []
    all_correlations = []

    print(f"\n  {'TF':<12} {'Shared':>7} {'Concord':>8} {'Pearson_r':>10} {'p-val':>10}")
    print("  " + "-" * 50)

    for tf in sorted(shared_tfs):
        # Find TF index in each dataset
        fi = fleck_tfs.index(tf) if tf in fleck_tfs else None
        pi = pollen_tfs.index(tf) if tf in pollen_tfs else None
        if fi is None or pi is None:
            continue

        # Extract effects for shared genes
        f_vals = []
        p_vals = []
        for gene in shared_genes:
            if gene in fleck_gene_idx and gene in pollen_gene_idx:
                f_vals.append(float(fleck_eff[fi, fleck_gene_idx[gene]]))
                p_vals.append(float(pollen_eff[pi, pollen_gene_idx[gene]]))

        f_arr = np.array(f_vals)
        p_arr = np.array(p_vals)

        # Direction concordance (fraction of genes with same sign)
        both_nonzero = (f_arr != 0) & (p_arr != 0)
        if both_nonzero.sum() > 5:
            same_sign = (np.sign(f_arr[both_nonzero]) == np.sign(p_arr[both_nonzero]))
            concordance = float(same_sign.mean())
        else:
            concordance = float("nan")

        # Pearson correlation
        if np.std(f_arr) > 1e-8 and np.std(p_arr) > 1e-8:
            r, pval = stats.pearsonr(f_arr, p_arr)
        else:
            r, pval = 0.0, 1.0

        results[tf] = {
            "n_shared_genes": len(shared_genes),
            "n_nonzero_both": int(both_nonzero.sum()),
            "concordance": concordance,
            "pearson_r": float(r),
            "pearson_pval": float(pval),
        }
        all_concordances.append(concordance)
        all_correlations.append(float(r))

        print(f"  {tf:<12} {len(shared_genes):>7} {concordance:>8.2%} "
              f"{r:>10.4f} {pval:>10.4g}")

    mean_conc = float(np.nanmean(all_concordances)) if all_concordances else 0
    mean_r = float(np.nanmean(all_correlations)) if all_correlations else 0
    print(f"\n  Mean concordance: {mean_conc:.2%}")
    print(f"  Mean Pearson r:   {mean_r:.4f}")

    return {
        "per_tf": results,
        "mean_concordance": mean_conc,
        "mean_pearson_r": mean_r,
        "shared_genes": shared_genes,
    }


# ---------------------------------------------------------------------------
# 3. Transfer Prediction (train on Fleck, test on Pollen)
# ---------------------------------------------------------------------------

def transfer_prediction(fleck: dict, pollen: dict) -> dict:
    """Train PerturbationPredictor on Fleck, evaluate on Pollen."""
    print("\n" + "=" * 64)
    print("  Comparison 3: Transfer Prediction (Fleck -> Pollen)")
    print("=" * 64)

    if not HAS_HGX or not HAS_DEVOGRAPH:
        print("  hgx/devograph not available — skipping transfer prediction")
        return {"skipped": True, "reason": "missing_deps"}

    fleck_inc = fleck.get("incidence")
    fleck_feat = fleck.get("node_features_pca")
    fleck_masks = fleck.get("perturbation_masks")
    fleck_eff = fleck.get("perturbation_effects")

    if any(x is None for x in [fleck_inc, fleck_feat, fleck_masks, fleck_eff]):
        print("  Missing Fleck arrays for training")
        return {"skipped": True, "reason": "missing_fleck_data"}

    pollen_eff = pollen.get("perturbation_effects")
    if pollen_eff is None:
        print("  Missing Pollen perturbation effects for evaluation")
        return {"skipped": True, "reason": "missing_pollen_data"}

    # Build Fleck hypergraph
    hg = hgx.from_incidence(
        jnp.array(fleck_inc),
        node_features=jnp.array(fleck_feat),
    )

    # Dummy fate targets (3 fates)
    n_perts = fleck_masks.shape[0]
    fleck_fates = np.zeros((n_perts, 3), dtype=np.float32)

    # Reshape effects if needed
    feat_dim = fleck_feat.shape[1]
    if fleck_eff.ndim == 1:
        fleck_eff_3d = np.expand_dims(fleck_eff, -1)
    elif fleck_eff.ndim == 2:
        fleck_eff_3d = np.expand_dims(fleck_eff, -1)
    else:
        fleck_eff_3d = fleck_eff

    # Train
    key = jax.random.PRNGKey(42)
    k_model, k_train = jax.random.split(key)

    try:
        predictor = devograph.PerturbationPredictor(
            gene_dim=feat_dim, hidden_dim=64, num_fates=3,
            conv_cls=hgx.UniGCNConv, num_layers=2, key=k_model,
        )
        predictor = devograph.train_perturbation_predictor(
            predictor, hg,
            perturbations=jnp.array(fleck_masks),
            targets=(jnp.array(fleck_eff_3d), jnp.array(fleck_fates)),
            epochs=200, key=k_train,
        )
        print("  Predictor trained on Fleck data")
    except Exception as e:
        print(f"  Training failed: {e}")
        return {"skipped": True, "reason": f"training_error: {e}"}

    # Evaluate on shared TFs
    fleck_tfs = fleck["tf_names"]
    pollen_tfs = pollen["tf_names"]
    fleck_genes = fleck["gene_names"]
    pollen_genes = pollen["gene_names"]
    fleck_gene_idx = {g: i for i, g in enumerate(fleck_genes)}
    pollen_gene_idx = {g: i for i, g in enumerate(pollen_genes)}

    shared_tfs = set(fleck_tfs) & set(pollen_tfs)
    shared_genes = sorted(set(fleck_genes) & set(pollen_genes))

    results = {}
    for tf in sorted(shared_tfs):
        if tf not in fleck_gene_idx:
            continue
        gene_idx = fleck_gene_idx[tf]

        try:
            pred_ko, pred_fate = devograph.in_silico_knockout(
                predictor, hg, gene_idx
            )
            pred_arr = np.array(pred_ko)
            if pred_arr.ndim > 1:
                pred_arr = pred_arr.mean(axis=-1)
        except Exception as e:
            print(f"  Prediction failed for {tf}: {e}")
            continue

        # Get Pollen observed effects for shared genes
        pi = pollen_tfs.index(tf)
        pred_shared = []
        obs_shared = []
        for gene in shared_genes:
            if gene in fleck_gene_idx and gene in pollen_gene_idx:
                pred_shared.append(float(pred_arr[fleck_gene_idx[gene]]))
                obs_shared.append(float(pollen_eff[pi, pollen_gene_idx[gene]]))

        pred_arr_s = np.array(pred_shared)
        obs_arr_s = np.array(obs_shared)

        if np.std(pred_arr_s) > 1e-8 and np.std(obs_arr_s) > 1e-8:
            r, pval = stats.pearsonr(pred_arr_s, obs_arr_s)
        else:
            r, pval = 0.0, 1.0

        both_nz = (pred_arr_s != 0) & (obs_arr_s != 0)
        if both_nz.sum() > 5:
            dir_acc = float((np.sign(pred_arr_s[both_nz]) == np.sign(obs_arr_s[both_nz])).mean())
        else:
            dir_acc = float("nan")

        results[tf] = {
            "pearson_r": float(r),
            "pearson_pval": float(pval),
            "direction_accuracy": dir_acc,
        }
        print(f"  {tf}: r={r:.4f} (p={pval:.4g}), direction={dir_acc:.2%}")

    mean_r = float(np.mean([v["pearson_r"] for v in results.values()])) if results else 0
    mean_dir = float(np.nanmean([v["direction_accuracy"] for v in results.values()])) if results else 0
    print(f"\n  Transfer mean r: {mean_r:.4f}, mean direction: {mean_dir:.2%}")

    return {
        "per_tf": results,
        "mean_pearson_r": mean_r,
        "mean_direction_accuracy": mean_dir,
    }


# ---------------------------------------------------------------------------
# 4. Within-Pollen baseline
# ---------------------------------------------------------------------------

def within_pollen_baseline(pollen: dict) -> dict:
    """Leave-one-TF-out prediction within Pollen (upper bound)."""
    print("\n" + "=" * 64)
    print("  Comparison 4: Within-Pollen Baseline (LOO)")
    print("=" * 64)

    if not HAS_HGX or not HAS_DEVOGRAPH:
        print("  hgx/devograph not available — skipping")
        return {"skipped": True}

    inc = pollen.get("incidence")
    feat = pollen.get("node_features_pca")
    masks = pollen.get("perturbation_masks")
    eff = pollen.get("perturbation_effects")

    if any(x is None for x in [inc, feat, masks, eff]):
        print("  Missing Pollen arrays")
        return {"skipped": True, "reason": "missing_data"}

    hg = hgx.from_incidence(jnp.array(inc), node_features=jnp.array(feat))
    K = masks.shape[0]
    feat_dim = feat.shape[1]
    fates = np.zeros((K, 3), dtype=np.float32)

    if eff.ndim == 2:
        eff_3d = np.expand_dims(eff, -1)
    else:
        eff_3d = eff

    pollen_tfs = pollen["tf_names"]
    pollen_genes = pollen["gene_names"]
    pollen_gene_idx = {g: i for i, g in enumerate(pollen_genes)}

    results = {}
    # LOO: for each TF, train on others, predict held-out
    for ki in range(min(K, 10)):  # cap at 10 to avoid very long runs
        tf = pollen_tfs[ki]
        train_idx = [i for i in range(K) if i != ki]

        train_masks = masks[train_idx]
        train_eff = eff_3d[train_idx]
        train_fates = fates[train_idx]

        key = jax.random.PRNGKey(42 + ki)
        k_model, k_train = jax.random.split(key)

        try:
            predictor = devograph.PerturbationPredictor(
                gene_dim=feat_dim, hidden_dim=64, num_fates=3,
                conv_cls=hgx.UniGCNConv, num_layers=2, key=k_model,
            )
            predictor = devograph.train_perturbation_predictor(
                predictor, hg,
                perturbations=jnp.array(train_masks),
                targets=(jnp.array(train_eff), jnp.array(train_fates)),
                epochs=100, key=k_train,
            )

            if tf in pollen_gene_idx:
                pred_ko, _ = devograph.in_silico_knockout(
                    predictor, hg, pollen_gene_idx[tf]
                )
                pred = np.array(pred_ko)
                if pred.ndim > 1:
                    pred = pred.mean(axis=-1)

                obs = eff[ki]
                if np.std(pred) > 1e-8 and np.std(obs) > 1e-8:
                    r, pval = stats.pearsonr(pred, obs)
                else:
                    r, pval = 0.0, 1.0

                results[tf] = {"pearson_r": float(r), "pearson_pval": float(pval)}
                print(f"  LOO {tf}: r={r:.4f} (p={pval:.4g})")

        except Exception as e:
            print(f"  LOO {tf} failed: {e}")

    mean_r = float(np.mean([v["pearson_r"] for v in results.values()])) if results else 0
    print(f"\n  Within-Pollen LOO mean r: {mean_r:.4f}")

    return {"per_tf": results, "mean_pearson_r": mean_r}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def generate_figure(
    overlap: dict,
    concordance: dict,
    transfer: dict,
    baseline: dict,
    fig_dir: Path,
) -> Path:
    """Generate 4-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── Panel A: TF Overlap Venn-like bar chart ──
    ax = axes[0, 0]
    fleck_only = len(set(overlap["fleck_tfs"]) - set(overlap["shared_tfs"]))
    pollen_only = len(set(overlap["pollen_tfs"]) - set(overlap["shared_tfs"]))
    shared = len(overlap["shared_tfs"])

    bars = ax.bar(
        ["Fleck only", "Shared", "Pollen only"],
        [fleck_only, shared, pollen_only],
        color=["#e41a1c", "#984ea3", "#377eb8"],
        alpha=0.8,
    )
    ax.set_ylabel("Number of TFs")
    ax.set_title("A. TF Overlap (Fleck GRN vs Pollen CRISPRi)",
                  fontweight="bold")
    ax.bar_label(bars, fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    if shared > 0:
        ax.text(0.95, 0.95, f"p={overlap['enrichment_pval']:.2g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                style="italic")

    # ── Panel B: Direction Concordance ──
    ax = axes[0, 1]
    conc_data = concordance.get("per_tf", {})
    if conc_data:
        tfs = sorted(conc_data.keys())
        concs = [conc_data[tf]["concordance"] for tf in tfs]
        rs = [conc_data[tf]["pearson_r"] for tf in tfs]

        x = np.arange(len(tfs))
        width = 0.35
        ax.bar(x - width / 2, concs, width, label="Direction concordance",
               color="#e41a1c", alpha=0.8)
        ax.bar(x + width / 2, rs, width, label="Pearson r",
               color="#377eb8", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(tfs, rotation=45, ha="right", fontsize=8)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No shared TFs", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_title("B. Direction Concordance (Shared TFs)", fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel C: Transfer vs Within-dataset ──
    ax = axes[1, 0]
    transfer_data = transfer.get("per_tf", {})
    baseline_data = baseline.get("per_tf", {})

    if transfer_data or baseline_data:
        categories = ["Transfer\n(Fleck->Pollen)", "Within-Pollen\n(LOO)"]
        means = [
            transfer.get("mean_pearson_r", 0),
            baseline.get("mean_pearson_r", 0),
        ]
        colors = ["#e41a1c", "#377eb8"]
        bars = ax.bar(categories, means, color=colors, alpha=0.8)
        ax.bar_label(bars, fmt="%.3f", fontsize=10)
        ax.set_ylabel("Mean Pearson r")
        ax.axhline(y=0, color="black", linewidth=0.5)
    else:
        ax.text(0.5, 0.5, "No prediction data\n(requires hgx + devograph)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
    ax.set_title("C. Transfer vs Within-Dataset Prediction", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel D: Summary Table ──
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("D. Comparison Summary", fontweight="bold")

    lines = [
        "Fleck et al. 2023 vs Pollen/Ding et al. 2026",
        "=" * 48,
        "",
        f"Fleck: {len(overlap['fleck_tfs'])} GRN TFs (organoids, Pando)",
        f"Pollen: {len(overlap['pollen_tfs'])} CRISPRi TFs (primary cortex)",
        f"Shared TFs: {len(overlap['shared_tfs'])}",
        f"Shared genes: {overlap['n_shared_genes']}",
        "",
        f"Direction concordance: {concordance.get('mean_concordance', 0):.1%}",
        f"Effect correlation: {concordance.get('mean_pearson_r', 0):.4f}",
        "",
        f"Transfer prediction r: {transfer.get('mean_pearson_r', 0):.4f}",
        f"Within-dataset LOO r: {baseline.get('mean_pearson_r', 0):.4f}",
    ]

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    fig.suptitle(
        "Cross-Dataset Comparison: Organoid GRN vs Primary Cortex CRISPRi",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "pollen_comparison.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved to {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset comparison: Fleck organoids vs Pollen primary cortex"
    )
    parser.add_argument("--fleck-dir", type=str, default=None,
                        help="Fleck processed data dir (auto-detected)")
    parser.add_argument("--pollen-dir", type=str, default=None,
                        help="Pollen processed data dir (auto-detected)")
    parser.add_argument("--skip-prediction", action="store_true",
                        help="Skip prediction comparisons (3 & 4)")
    args = parser.parse_args()

    # Resolve directories
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    fig_dir = script_dir.parent / "figures"

    if args.fleck_dir:
        fleck_dir = Path(args.fleck_dir)
    else:
        candidates = [
            Path("/workspace/benchmark/data/processed"),
            data_dir / "processed",
        ]
        fleck_dir = next((p for p in candidates if p.is_dir()), None)

    if args.pollen_dir:
        pollen_dir = Path(args.pollen_dir)
    else:
        candidates = [
            Path("/workspace/benchmark/data/pollen/processed"),
            data_dir / "pollen" / "processed",
        ]
        pollen_dir = next((p for p in candidates if p.is_dir()), None)

    print("=" * 64)
    print("  Cross-Dataset Comparison: Fleck vs Pollen")
    print("=" * 64)
    print(f"  Fleck dir:  {fleck_dir}")
    print(f"  Pollen dir: {pollen_dir}")
    print(f"  hgx:        {'available' if HAS_HGX else 'NOT available'}")
    print(f"  devograph:  {'available' if HAS_DEVOGRAPH else 'NOT available'}")

    if fleck_dir is None:
        sys.exit("ERROR: Cannot find Fleck processed data. Pass --fleck-dir.")
    if pollen_dir is None:
        sys.exit("ERROR: Cannot find Pollen processed data. "
                 "Run preprocess_pollen.py first, or pass --pollen-dir.")

    # Load datasets
    print("\nLoading datasets...")
    fleck = load_dataset(fleck_dir, "Fleck et al. 2023")
    pollen = load_dataset(pollen_dir, "Pollen/Ding et al. 2026")

    if fleck is None or pollen is None:
        sys.exit("ERROR: Failed to load one or both datasets")

    print(f"  Fleck:  {len(fleck['gene_names'])} genes, {len(fleck['tf_names'])} TFs")
    print(f"  Pollen: {len(pollen['gene_names'])} genes, {len(pollen['tf_names'])} TFs")

    # Run comparisons
    overlap = analyze_grn_overlap(fleck, pollen)
    concordance = analyze_direction_concordance(fleck, pollen)

    if args.skip_prediction:
        transfer = {"skipped": True, "mean_pearson_r": 0}
        baseline = {"skipped": True, "mean_pearson_r": 0}
    else:
        transfer = transfer_prediction(fleck, pollen)
        baseline = within_pollen_baseline(pollen)

    # Generate figure
    fig_path = generate_figure(overlap, concordance, transfer, baseline, fig_dir)

    # Summary
    print("\n" + "=" * 64)
    print("  COMPARISON SUMMARY")
    print("=" * 64)
    print(f"  Shared TFs:            {len(overlap['shared_tfs'])}")
    print(f"  Direction concordance: {concordance.get('mean_concordance', 0):.1%}")
    print(f"  Effect correlation:    {concordance.get('mean_pearson_r', 0):.4f}")
    print(f"  Transfer r:            {transfer.get('mean_pearson_r', 0):.4f}")
    print(f"  Within-dataset r:      {baseline.get('mean_pearson_r', 0):.4f}")
    print(f"  Figure: {fig_path}")
    print("=" * 64)

    # Save results JSON
    results_path = fig_dir / "pollen_comparison_results.json"
    results = {
        "overlap": {k: v for k, v in overlap.items()},
        "concordance": {
            "mean_concordance": concordance.get("mean_concordance"),
            "mean_pearson_r": concordance.get("mean_pearson_r"),
        },
        "transfer": {
            "mean_pearson_r": transfer.get("mean_pearson_r", 0),
            "mean_direction_accuracy": transfer.get("mean_direction_accuracy", 0),
        },
        "baseline": {
            "mean_pearson_r": baseline.get("mean_pearson_r", 0),
        },
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results JSON: {results_path}")


if __name__ == "__main__":
    main()
