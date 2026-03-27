#!/usr/bin/env python3
"""Filtered transfer analysis: evaluate prediction on top DE genes only.

The all-genes Pearson r (0.127) is diluted by thousands of near-zero effects.
This script evaluates transfer on the top-N most differentially expressed genes
per TF in the Pollen CRISPRi data, giving a cleaner signal-to-noise picture.

Also computes:
- Stratified analysis by effect size quantile
- Per-TF breakdown at multiple N thresholds
- Comparison to GEARS-style metrics (top-20 DE genes)

Usage:
    uv run python scripts/filtered_transfer_analysis.py
"""

from __future__ import annotations

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
    import devograph
    HAS_HGX = True
except ImportError:
    HAS_HGX = False


def load_dataset(data_dir: Path, name: str) -> dict | None:
    d = {"name": name, "dir": data_dir}
    for fname in ["gene_names", "tf_names"]:
        path = data_dir / f"{fname}.json"
        if path.exists():
            with open(path) as f:
                d[fname] = json.load(f)
        else:
            return None
    for fname in ["perturbation_masks", "perturbation_effects",
                   "incidence", "node_features_pca"]:
        path = data_dir / f"{fname}.npy"
        if path.exists():
            d[fname] = np.load(path)
    # Perturbed TF subset
    key_tf_path = data_dir / "key_tf_indices.json"
    eff = d.get("perturbation_effects")
    if key_tf_path.exists() and eff is not None:
        with open(key_tf_path) as f:
            d["perturbed_tf_names"] = list(json.load(f).keys())
    elif eff is not None:
        d["perturbed_tf_names"] = d["tf_names"][:eff.shape[0]]
    return d


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    fig_dir = script_dir.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    fleck_dir = data_dir / "processed"
    pollen_dir = data_dir / "pollen" / "processed"

    if not HAS_HGX:
        sys.exit("ERROR: hgx/devograph not available")

    print("Loading datasets...")
    fleck = load_dataset(fleck_dir, "Fleck")
    pollen = load_dataset(pollen_dir, "Pollen")
    if fleck is None or pollen is None:
        sys.exit("ERROR: Missing dataset")

    # Load Pollen DE for ranking genes by effect size
    de_path = pollen_dir / "pollen_de.csv"
    if not de_path.exists():
        sys.exit("ERROR: pollen_de.csv not found — run preprocess_pollen.py first")
    de_df = pd.read_csv(de_path)

    fleck_genes = fleck["gene_names"]
    pollen_genes = pollen["gene_names"]
    pollen_pert_tfs = pollen.get("perturbed_tf_names", pollen["tf_names"])
    fleck_gene_idx = {g: i for i, g in enumerate(fleck_genes)}
    pollen_gene_idx = {g: i for i, g in enumerate(pollen_genes)}
    shared_genes = sorted(set(fleck_genes) & set(pollen_genes))

    # -------------------------------------------------------------------
    # Train predictor on Fleck
    # -------------------------------------------------------------------
    print("Training predictor on Fleck data...")
    fleck_inc = fleck["incidence"]
    fleck_feat = fleck["node_features_pca"]
    fleck_masks = fleck["perturbation_masks"]
    fleck_eff = fleck["perturbation_effects"]

    hg = hgx.from_incidence(
        jnp.array(fleck_inc), node_features=jnp.array(fleck_feat),
    )

    n_perts = fleck_masks.shape[0]
    feat_dim = fleck_feat.shape[1]
    fleck_fates = np.zeros((n_perts, 3), dtype=np.float32)
    fleck_eff_3d = np.expand_dims(fleck_eff, -1) if fleck_eff.ndim == 2 else fleck_eff

    key = jax.random.PRNGKey(42)
    k_model, k_train = jax.random.split(key)

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
    print("  Done.\n")

    # -------------------------------------------------------------------
    # Predict knockouts and evaluate at multiple thresholds
    # -------------------------------------------------------------------
    shared_tfs = sorted(set(fleck["tf_names"]) & set(pollen_pert_tfs))
    pollen_eff = pollen["perturbation_effects"]

    N_values = [20, 50, 100, 200, 500, 1000, 2739]
    # Per-TF results at each N
    results_by_n = {n: [] for n in N_values}
    # Per-TF full vectors for stratified analysis
    per_tf_data = {}

    print(f"{'TF':<12}", end="")
    for n in N_values:
        print(f" {'top'+str(n):>8}", end="")
    print()
    print("-" * (12 + 9 * len(N_values)))

    for tf in shared_tfs:
        if tf not in fleck_gene_idx:
            continue
        gene_idx = fleck_gene_idx[tf]

        try:
            pred_ko, _ = devograph.in_silico_knockout(predictor, hg, gene_idx)
            pred_arr = np.array(pred_ko)
            if pred_arr.ndim > 1:
                pred_arr = pred_arr.mean(axis=-1)
        except Exception as e:
            print(f"  {tf}: prediction failed ({e})")
            continue

        pi = pollen_pert_tfs.index(tf)

        # Get all predicted/observed for shared genes
        pred_shared = []
        obs_shared = []
        gene_list = []
        for gene in shared_genes:
            if gene in fleck_gene_idx and gene in pollen_gene_idx:
                pred_shared.append(float(pred_arr[fleck_gene_idx[gene]]))
                obs_shared.append(float(pollen_eff[pi, pollen_gene_idx[gene]]))
                gene_list.append(gene)

        pred_arr_s = np.array(pred_shared)
        obs_arr_s = np.array(obs_shared)

        # Rank genes by absolute observed effect (Pollen ground truth)
        abs_obs = np.abs(obs_arr_s)
        rank_idx = np.argsort(-abs_obs)  # descending

        per_tf_data[tf] = {
            "pred": pred_arr_s,
            "obs": obs_arr_s,
            "genes": gene_list,
            "rank": rank_idx,
        }

        print(f"{tf:<12}", end="")
        for n in N_values:
            top_idx = rank_idx[:n]
            p = pred_arr_s[top_idx]
            o = obs_arr_s[top_idx]
            if np.std(p) > 1e-8 and np.std(o) > 1e-8:
                r, _ = stats.pearsonr(p, o)
            else:
                r = 0.0
            results_by_n[n].append({"tf": tf, "r": r})
            print(f" {r:>8.3f}", end="")
        print()

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  FILTERED TRANSFER ANALYSIS SUMMARY")
    print("=" * 64)

    summary_rows = []
    print(f"\n  {'Top-N':>8} {'Mean r':>8} {'Median r':>9} {'#TFs>0.1':>9} {'#TFs>0.2':>9} {'Direction':>10}")
    print("  " + "-" * 56)
    for n in N_values:
        rs = [d["r"] for d in results_by_n[n]]
        mean_r = np.mean(rs)
        median_r = np.median(rs)
        n_above_01 = sum(1 for r in rs if r > 0.1)
        n_above_02 = sum(1 for r in rs if r > 0.2)

        # Direction accuracy at this N
        dir_accs = []
        for tf_data in per_tf_data.values():
            top_idx = tf_data["rank"][:n]
            p = tf_data["pred"][top_idx]
            o = tf_data["obs"][top_idx]
            nz = (p != 0) & (o != 0)
            if nz.sum() > 3:
                dir_accs.append(float((np.sign(p[nz]) == np.sign(o[nz])).mean()))
        mean_dir = np.mean(dir_accs) if dir_accs else 0

        summary_rows.append({
            "top_n": n, "mean_r": mean_r, "median_r": median_r,
            "n_above_01": n_above_01, "n_above_02": n_above_02,
            "mean_direction": mean_dir,
        })
        print(f"  {n:>8} {mean_r:>8.4f} {median_r:>9.4f} {n_above_01:>9} {n_above_02:>9} {mean_dir:>9.1%}")

    # -------------------------------------------------------------------
    # Stratified analysis: effect size quintiles
    # -------------------------------------------------------------------
    print("\n  Stratified by observed effect size quintile (pooled across TFs):")
    all_pred = np.concatenate([d["pred"] for d in per_tf_data.values()])
    all_obs = np.concatenate([d["obs"] for d in per_tf_data.values()])
    abs_all_obs = np.abs(all_obs)
    quintile_edges = np.percentile(abs_all_obs, [0, 20, 40, 60, 80, 100])

    print(f"  {'Quintile':>10} {'|effect| range':>20} {'N':>8} {'Pearson r':>10} {'Direction':>10}")
    print("  " + "-" * 62)
    for q in range(5):
        lo, hi = quintile_edges[q], quintile_edges[q + 1]
        mask = (abs_all_obs >= lo) & (abs_all_obs <= hi) if q == 4 else \
               (abs_all_obs >= lo) & (abs_all_obs < hi)
        if mask.sum() < 10:
            continue
        p_q = all_pred[mask]
        o_q = all_obs[mask]
        if np.std(p_q) > 1e-8 and np.std(o_q) > 1e-8:
            r_q, _ = stats.pearsonr(p_q, o_q)
        else:
            r_q = 0.0
        nz = (p_q != 0) & (o_q != 0)
        dir_q = float((np.sign(p_q[nz]) == np.sign(o_q[nz])).mean()) if nz.sum() > 3 else 0
        label = f"Q{q+1} ({'largest' if q == 4 else 'smallest' if q == 0 else ''})"
        print(f"  {label:>10} {lo:>8.3f} – {hi:>8.3f} {mask.sum():>8} {r_q:>10.4f} {dir_q:>9.1%}")

    # -------------------------------------------------------------------
    # GEARS-style metric: top-20 DE genes
    # -------------------------------------------------------------------
    print("\n  GEARS-style top-20 DE comparison:")
    print(f"  {'TF':<12} {'r (top20)':>10} {'dir (top20)':>12} {'r (all)':>10}")
    print("  " + "-" * 46)
    gears_rs = []
    for tf in sorted(per_tf_data.keys()):
        d = per_tf_data[tf]
        top20 = d["rank"][:20]
        p20 = d["pred"][top20]
        o20 = d["obs"][top20]
        if np.std(p20) > 1e-8 and np.std(o20) > 1e-8:
            r20, _ = stats.pearsonr(p20, o20)
        else:
            r20 = 0.0
        nz20 = (p20 != 0) & (o20 != 0)
        dir20 = float((np.sign(p20[nz20]) == np.sign(o20[nz20])).mean()) if nz20.sum() > 3 else 0

        r_all = [d2["r"] for d2 in results_by_n[2739] if d2["tf"] == tf][0]
        gears_rs.append(r20)
        print(f"  {tf:<12} {r20:>10.4f} {dir20:>11.1%} {r_all:>10.4f}")

    print(f"\n  GEARS top-20 mean r: {np.mean(gears_rs):.4f}")
    print(f"  GEARS top-20 median r: {np.median(gears_rs):.4f}")

    # -------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: r vs top-N threshold
    ax = axes[0, 0]
    mean_rs = [np.mean([d["r"] for d in results_by_n[n]]) for n in N_values]
    median_rs = [np.median([d["r"] for d in results_by_n[n]]) for n in N_values]
    ax.plot(N_values, mean_rs, "o-", color="#e41a1c", label="Mean r", linewidth=2)
    ax.plot(N_values, median_rs, "s--", color="#377eb8", label="Median r", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Top-N DE genes per TF")
    ax.set_ylabel("Pearson r (transfer)")
    ax.set_title("A. Transfer correlation vs gene filter", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linewidth=0.5)

    # Panel B: per-TF r at top-20 vs top-all
    ax = axes[0, 1]
    tfs_sorted = sorted(per_tf_data.keys())
    r20s = []
    r_alls = []
    for tf in tfs_sorted:
        r20s.append([d["r"] for d in results_by_n[20] if d["tf"] == tf][0])
        r_alls.append([d["r"] for d in results_by_n[2739] if d["tf"] == tf][0])
    ax.scatter(r_alls, r20s, alpha=0.7, s=40, c="#984ea3")
    for i, tf in enumerate(tfs_sorted):
        if abs(r20s[i]) > 0.2 or abs(r_alls[i]) > 0.15:
            ax.annotate(tf, (r_alls[i], r20s[i]), fontsize=7, alpha=0.8)
    ax.plot([-0.1, 0.3], [-0.1, 0.3], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("Pearson r (all 2,739 genes)")
    ax.set_ylabel("Pearson r (top-20 DE genes)")
    ax.set_title("B. Per-TF: filtered vs unfiltered", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: stratified quintile bars
    ax = axes[1, 0]
    quintile_rs = []
    quintile_dirs = []
    for q in range(5):
        lo, hi = quintile_edges[q], quintile_edges[q + 1]
        mask = (abs_all_obs >= lo) & (abs_all_obs <= hi) if q == 4 else \
               (abs_all_obs >= lo) & (abs_all_obs < hi)
        if mask.sum() < 10:
            quintile_rs.append(0)
            quintile_dirs.append(0.5)
            continue
        p_q = all_pred[mask]
        o_q = all_obs[mask]
        if np.std(p_q) > 1e-8 and np.std(o_q) > 1e-8:
            r_q, _ = stats.pearsonr(p_q, o_q)
        else:
            r_q = 0.0
        nz = (p_q != 0) & (o_q != 0)
        dir_q = float((np.sign(p_q[nz]) == np.sign(o_q[nz])).mean()) if nz.sum() > 3 else 0.5
        quintile_rs.append(r_q)
        quintile_dirs.append(dir_q)

    x = np.arange(5)
    width = 0.35
    ax.bar(x - width/2, quintile_rs, width, label="Pearson r", color="#e41a1c", alpha=0.8)
    ax.bar(x + width/2, quintile_dirs, width, label="Direction acc", color="#377eb8", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\n(smallest)", "Q2", "Q3", "Q4", "Q5\n(largest)"])
    ax.set_ylabel("Score")
    ax.set_title("C. Performance by observed effect size quintile", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    # Panel D: summary text
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("D. Summary", fontweight="bold")
    lines = [
        "Fleck -> Pollen Transfer Prediction",
        "=" * 40,
        "",
        f"TFs evaluated:     {len(per_tf_data)}",
        f"Shared genes:      {len(shared_genes)}",
        "",
        "       All genes    Top-100    Top-20",
        f"Mean r:  {np.mean([d['r'] for d in results_by_n[2739]]):.4f}"
        f"       {np.mean([d['r'] for d in results_by_n[100]]):.4f}"
        f"      {np.mean([d['r'] for d in results_by_n[20]]):.4f}",
        f"Dir:     {summary_rows[-1]['mean_direction']:.1%}"
        f"       {summary_rows[2]['mean_direction']:.1%}"
        f"      {summary_rows[0]['mean_direction']:.1%}",
        "",
        "Top transfer TFs (top-20 DE r):",
    ]
    # Top 5 TFs by top-20 r
    tf_r20 = [(tf, [d["r"] for d in results_by_n[20] if d["tf"] == tf][0])
              for tf in per_tf_data.keys()]
    tf_r20.sort(key=lambda x: -x[1])
    for tf, r in tf_r20[:5]:
        lines.append(f"  {tf:<12} r={r:.3f}")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    fig.suptitle("Filtered Transfer Analysis: Fleck Organoid -> Pollen CRISPRi",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig_path = fig_dir / "pollen_filtered_transfer.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved to {fig_path}")

    # Save results
    results_path = fig_dir / "pollen_filtered_results.json"
    results = {
        "summary_by_n": summary_rows,
        "gears_top20_mean_r": float(np.mean(gears_rs)),
        "gears_top20_median_r": float(np.median(gears_rs)),
        "per_tf_top20": {tf: float([d["r"] for d in results_by_n[20] if d["tf"] == tf][0])
                         for tf in per_tf_data.keys()},
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
