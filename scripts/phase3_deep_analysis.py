#!/usr/bin/env python3
"""Phase 3 deep analysis: does the Fleck GRN topology capture real biology?

Three analyses that test whether organoid-derived GRN structure transfers
to primary cortex CRISPRi perturbations:

  A. GRN topology test — LOO on Pollen using Fleck / Pollen / random incidence
  B. Direct regulon comparison — Jaccard + direction concordance for shared TFs
  C. Domain adaptation — does augmenting with Fleck data help Pollen LOO?

Usage:
    uv run python scripts/phase3_deep_analysis.py
"""

from __future__ import annotations

import json
import time
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

if not HAS_HGX:
    sys.exit("ERROR: hgx/devograph required")


# ===================================================================
# Helpers
# ===================================================================

def load_dataset(data_dir: Path) -> dict | None:
    d = {"dir": data_dir}
    for fname in ["gene_names", "tf_names"]:
        path = data_dir / f"{fname}.json"
        if not path.exists():
            return None
        with open(path) as f:
            d[fname] = json.load(f)
    for fname in ["perturbation_masks", "perturbation_effects",
                   "incidence", "node_features_pca"]:
        path = data_dir / f"{fname}.npy"
        if path.exists():
            d[fname] = np.load(path)
    key_tf_path = data_dir / "key_tf_indices.json"
    eff = d.get("perturbation_effects")
    if key_tf_path.exists() and eff is not None:
        with open(key_tf_path) as f:
            d["perturbed_tf_names"] = list(json.load(f).keys())
    elif eff is not None:
        d["perturbed_tf_names"] = d["tf_names"][:eff.shape[0]]
    return d


def train_and_predict_loo(incidence, features, masks, effects,
                          tf_names, gene_idx_map, n_tfs_eval=10,
                          epochs=150):
    """LOO prediction: train on N-1 TFs, predict held-out. Returns per-TF r."""
    K = masks.shape[0]
    feat_dim = features.shape[1]
    eff_3d = np.expand_dims(effects, -1) if effects.ndim == 2 else effects
    fates = np.zeros((K, 3), dtype=np.float32)

    hg = hgx.from_incidence(jnp.array(incidence), node_features=jnp.array(features))
    results = {}

    for ki in range(min(K, n_tfs_eval)):
        tf = tf_names[ki]
        train_idx = [i for i in range(K) if i != ki]

        key = jax.random.PRNGKey(42 + ki)
        k_model, k_train = jax.random.split(key)

        try:
            predictor = devograph.PerturbationPredictor(
                gene_dim=feat_dim, hidden_dim=64, num_fates=3,
                conv_cls=hgx.UniGCNConv, num_layers=2, key=k_model,
            )
            predictor = devograph.train_perturbation_predictor(
                predictor, hg,
                perturbations=jnp.array(masks[train_idx]),
                targets=(jnp.array(eff_3d[train_idx]),
                         jnp.array(fates[train_idx])),
                epochs=epochs, key=k_train,
            )

            if tf in gene_idx_map:
                pred_ko, _ = devograph.in_silico_knockout(
                    predictor, hg, gene_idx_map[tf],
                )
                pred = np.array(pred_ko)
                if pred.ndim > 1:
                    pred = pred.mean(axis=-1)

                obs = effects[ki]

                # All-gene r
                r_all, p_all = stats.pearsonr(pred, obs) if np.std(pred) > 1e-8 and np.std(obs) > 1e-8 else (0.0, 1.0)

                # Top-100 DE r
                top100 = np.argsort(-np.abs(obs))[:100]
                p100, o100 = pred[top100], obs[top100]
                r_100, _ = stats.pearsonr(p100, o100) if np.std(p100) > 1e-8 and np.std(o100) > 1e-8 else (0.0, 1.0)

                # Top-20 DE r
                top20 = np.argsort(-np.abs(obs))[:20]
                p20, o20 = pred[top20], obs[top20]
                r_20, _ = stats.pearsonr(p20, o20) if np.std(p20) > 1e-8 and np.std(o20) > 1e-8 else (0.0, 1.0)

                results[tf] = {"r_all": float(r_all), "r_100": float(r_100),
                               "r_20": float(r_20)}
        except Exception as e:
            print(f"    {tf} failed: {e}")

    return results


# ===================================================================
# Analysis A: GRN topology test
# ===================================================================

def analysis_a_topology(fleck, pollen, shared_genes, n_eval=10):
    """Compare LOO on Pollen using different incidence matrices."""
    print("\n" + "=" * 64)
    print("  A. GRN TOPOLOGY TEST")
    print("  Does Fleck's Pando GRN structure help predict Pollen KOs?")
    print("=" * 64)

    pollen_genes = pollen["gene_names"]
    pollen_gene_idx = {g: i for i, g in enumerate(pollen_genes)}
    fleck_genes = fleck["gene_names"]
    fleck_gene_idx = {g: i for i, g in enumerate(fleck_genes)}

    pollen_tfs = pollen.get("perturbed_tf_names", pollen["tf_names"])
    pollen_eff = pollen["perturbation_effects"]
    pollen_masks = pollen["perturbation_masks"]
    pollen_feat = pollen["node_features_pca"]
    pollen_inc = pollen["incidence"]
    n_genes = len(pollen_genes)
    n_tfs = len(pollen_tfs)

    # --- Condition 1: Pollen's own DE-derived incidence ---
    print("\n  [1] Pollen DE-derived incidence (own GRN)")
    t0 = time.perf_counter()
    res_pollen = train_and_predict_loo(
        pollen_inc, pollen_feat, pollen_masks, pollen_eff,
        pollen_tfs, pollen_gene_idx, n_tfs_eval=n_eval,
    )
    dt1 = time.perf_counter() - t0
    print(f"      {len(res_pollen)} TFs evaluated in {dt1:.0f}s")

    # --- Condition 2: Fleck Pando incidence projected to shared genes ---
    print("\n  [2] Fleck Pando incidence (organoid GRN)")
    fleck_inc = fleck["incidence"]  # (n_fleck_genes, n_fleck_edges)
    # Project to Pollen gene space: keep rows for shared genes
    shared_idx_in_fleck = [fleck_gene_idx[g] for g in pollen_genes if g in fleck_gene_idx]
    shared_idx_in_pollen = [pollen_gene_idx[g] for g in pollen_genes if g in fleck_gene_idx]
    fleck_inc_proj = np.zeros((n_genes, fleck_inc.shape[1]), dtype=np.float32)
    for pi, fi in zip(shared_idx_in_pollen, shared_idx_in_fleck):
        fleck_inc_proj[pi] = fleck_inc[fi]
    # Drop empty edges
    active = fleck_inc_proj.sum(axis=0) >= 2
    fleck_inc_proj = fleck_inc_proj[:, active]
    print(f"      Projected: {fleck_inc_proj.shape} ({active.sum()} active edges)")

    t0 = time.perf_counter()
    res_fleck = train_and_predict_loo(
        fleck_inc_proj, pollen_feat, pollen_masks, pollen_eff,
        pollen_tfs, pollen_gene_idx, n_tfs_eval=n_eval,
    )
    dt2 = time.perf_counter() - t0
    print(f"      {len(res_fleck)} TFs evaluated in {dt2:.0f}s")

    # --- Condition 3: Random incidence (same density as Pollen) ---
    print("\n  [3] Random incidence (null baseline)")
    rng = np.random.default_rng(42)
    density = pollen_inc.mean()
    rand_inc = (rng.random((n_genes, pollen_inc.shape[1])) < density).astype(np.float32)

    t0 = time.perf_counter()
    res_random = train_and_predict_loo(
        rand_inc, pollen_feat, pollen_masks, pollen_eff,
        pollen_tfs, pollen_gene_idx, n_tfs_eval=n_eval,
    )
    dt3 = time.perf_counter() - t0
    print(f"      {len(res_random)} TFs evaluated in {dt3:.0f}s")

    # --- Condition 4: No incidence (identity = GCN on complete graph) ---
    print("\n  [4] Identity incidence (no GRN, complete graph)")
    # Use identity-like incidence: each gene is its own singleton edge
    # This reduces UniGCNConv to a simple MLP (no message passing benefit)
    id_inc = np.eye(n_genes, dtype=np.float32)

    t0 = time.perf_counter()
    res_identity = train_and_predict_loo(
        id_inc, pollen_feat, pollen_masks, pollen_eff,
        pollen_tfs, pollen_gene_idx, n_tfs_eval=n_eval,
    )
    dt4 = time.perf_counter() - t0
    print(f"      {len(res_identity)} TFs evaluated in {dt4:.0f}s")

    # --- Summary ---
    print("\n  " + "-" * 60)
    print(f"  {'Incidence':<25} {'r_all':>8} {'r_100':>8} {'r_20':>8}")
    print("  " + "-" * 60)

    conditions = [
        ("Pollen DE (own GRN)", res_pollen),
        ("Fleck Pando (organoid)", res_fleck),
        ("Random (null)", res_random),
        ("Identity (no GRN)", res_identity),
    ]
    summary_a = {}
    for name, res in conditions:
        if not res:
            print(f"  {name:<25} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue
        mean_all = np.mean([v["r_all"] for v in res.values()])
        mean_100 = np.mean([v["r_100"] for v in res.values()])
        mean_20 = np.mean([v["r_20"] for v in res.values()])
        print(f"  {name:<25} {mean_all:>8.4f} {mean_100:>8.4f} {mean_20:>8.4f}")
        summary_a[name] = {"r_all": mean_all, "r_100": mean_100, "r_20": mean_20,
                           "per_tf": res}

    return summary_a


# ===================================================================
# Analysis B: Direct regulon comparison
# ===================================================================

def analysis_b_regulon_comparison(fleck, pollen, shared_genes):
    """Compare GRN edges: Fleck Pando membership vs Pollen DE significance."""
    print("\n" + "=" * 64)
    print("  B. DIRECT REGULON COMPARISON")
    print("  Fleck Pando edges vs Pollen CRISPRi DE edges")
    print("=" * 64)

    fleck_genes = fleck["gene_names"]
    pollen_genes = pollen["gene_names"]
    fleck_gene_idx = {g: i for i, g in enumerate(fleck_genes)}
    pollen_gene_idx = {g: i for i, g in enumerate(pollen_genes)}

    fleck_inc = fleck["incidence"]  # (n_fleck_genes, n_fleck_tfs)
    pollen_inc = pollen["incidence"]  # (n_pollen_genes, n_pollen_tfs)
    fleck_tfs = fleck["tf_names"]  # 720 TFs
    pollen_tfs = pollen.get("perturbed_tf_names", pollen["tf_names"])

    # TFs present in both
    shared_tfs = sorted(set(fleck_tfs) & set(pollen_tfs))
    print(f"\n  Shared TFs with incidence data: {len(shared_tfs)}")

    if not shared_tfs:
        print("  No shared TFs")
        return {}

    # For each shared TF, compare regulon membership
    results = {}
    print(f"\n  {'TF':<12} {'Fleck':>6} {'Pollen':>7} {'Inter':>6} {'Union':>6} "
          f"{'Jaccard':>8} {'Fisher_p':>10} {'Dir_conc':>9}")
    print("  " + "-" * 70)

    all_jaccards = []
    all_fisher_ps = []
    all_dir_concs = []

    for tf in shared_tfs:
        fi = fleck_tfs.index(tf)
        pi = pollen_tfs.index(tf)

        # Regulon members in shared gene space
        fleck_members = set()
        pollen_members = set()

        for gene in shared_genes:
            if gene in fleck_gene_idx:
                gi_f = fleck_gene_idx[gene]
                if fleck_inc[gi_f, fi] > 0:
                    fleck_members.add(gene)
            if gene in pollen_gene_idx:
                gi_p = pollen_gene_idx[gene]
                if pollen_inc[gi_p, pi] > 0:
                    pollen_members.add(gene)

        intersection = fleck_members & pollen_members
        union = fleck_members | pollen_members
        jaccard = len(intersection) / len(union) if union else 0

        # Fisher's exact test for overlap enrichment
        n = len(shared_genes)
        a = len(intersection)
        b = len(fleck_members) - a
        c = len(pollen_members) - a
        d = n - a - b - c
        _, fisher_p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")

        # Direction concordance for intersection genes
        dir_conc = float("nan")
        if len(intersection) > 5:
            pollen_eff = pollen["perturbation_effects"]
            # Get Pollen effects for intersection genes
            signs = []
            for gene in intersection:
                gi_p = pollen_gene_idx[gene]
                eff_val = pollen_eff[pi, gi_p]
                if abs(eff_val) > 0.1:
                    signs.append(np.sign(eff_val))
            # For Fleck, incidence is binary, so direction = whether
            # gene is up/down in Pollen for genes that Fleck says are targets
            # We look at whether the sign is consistent within the regulon
            if len(signs) > 3:
                most_common = 1.0 if sum(s > 0 for s in signs) > len(signs) / 2 else -1.0
                dir_conc = sum(1 for s in signs if s == most_common) / len(signs)

        results[tf] = {
            "fleck_size": len(fleck_members),
            "pollen_size": len(pollen_members),
            "intersection": len(intersection),
            "union": len(union),
            "jaccard": jaccard,
            "fisher_p": fisher_p,
            "direction_concordance": dir_conc,
        }
        all_jaccards.append(jaccard)
        if fisher_p < 1:
            all_fisher_ps.append(fisher_p)
        if not np.isnan(dir_conc):
            all_dir_concs.append(dir_conc)

        sig = "*" if fisher_p < 0.05 else " "
        print(f"  {tf:<12} {len(fleck_members):>6} {len(pollen_members):>7} "
              f"{len(intersection):>6} {len(union):>6} {jaccard:>8.3f} "
              f"{fisher_p:>10.2g}{sig} {dir_conc:>8.1%}" if not np.isnan(dir_conc) else
              f"  {tf:<12} {len(fleck_members):>6} {len(pollen_members):>7} "
              f"{len(intersection):>6} {len(union):>6} {jaccard:>8.3f} "
              f"{fisher_p:>10.2g}{sig} {'N/A':>9}")

    n_sig = sum(1 for r in results.values() if r["fisher_p"] < 0.05)
    n_sig_bonf = sum(1 for r in results.values()
                     if r["fisher_p"] < 0.05 / len(shared_tfs))

    print(f"\n  Mean Jaccard:          {np.mean(all_jaccards):.4f}")
    print(f"  Median Jaccard:        {np.median(all_jaccards):.4f}")
    print(f"  TFs with sig overlap:  {n_sig}/{len(shared_tfs)} (p<0.05)")
    print(f"  TFs sig after Bonf:    {n_sig_bonf}/{len(shared_tfs)}")
    if all_dir_concs:
        print(f"  Mean direction conc:   {np.mean(all_dir_concs):.1%}")

    return {
        "per_tf": results,
        "mean_jaccard": float(np.mean(all_jaccards)),
        "n_sig_fisher": n_sig,
        "n_sig_bonferroni": n_sig_bonf,
        "mean_direction_concordance": float(np.mean(all_dir_concs)) if all_dir_concs else None,
    }


# ===================================================================
# Analysis C: Domain adaptation
# ===================================================================

def analysis_c_domain_adaptation(fleck, pollen, shared_genes, n_eval=10):
    """Does augmenting Pollen LOO training with Fleck data help?"""
    print("\n" + "=" * 64)
    print("  C. DOMAIN ADAPTATION")
    print("  Does adding Fleck perturbation data help Pollen LOO?")
    print("=" * 64)

    pollen_genes = pollen["gene_names"]
    pollen_gene_idx = {g: i for i, g in enumerate(pollen_genes)}
    pollen_tfs = pollen.get("perturbed_tf_names", pollen["tf_names"])
    pollen_eff = pollen["perturbation_effects"]
    pollen_masks = pollen["perturbation_masks"]
    pollen_feat = pollen["node_features_pca"]
    pollen_inc = pollen["incidence"]
    n_genes = len(pollen_genes)

    fleck_genes = fleck["gene_names"]
    fleck_gene_idx = {g: i for i, g in enumerate(fleck_genes)}
    fleck_pert_tfs = fleck.get("perturbed_tf_names", fleck["tf_names"])
    fleck_eff = fleck["perturbation_effects"]
    fleck_masks = fleck["perturbation_masks"]

    # Map Fleck perturbation data to Pollen gene space
    n_fleck_perts = fleck_eff.shape[0]
    fleck_eff_proj = np.zeros((n_fleck_perts, n_genes), dtype=np.float32)
    fleck_masks_proj = np.zeros((n_fleck_perts, n_genes), dtype=bool)

    for gi_p, gene in enumerate(pollen_genes):
        if gene in fleck_gene_idx:
            gi_f = fleck_gene_idx[gene]
            fleck_eff_proj[:, gi_p] = fleck_eff[:, gi_f]
            fleck_masks_proj[:, gi_p] = fleck_masks[:, gi_f]

    K = pollen_masks.shape[0]
    feat_dim = pollen_feat.shape[1]
    hg = hgx.from_incidence(jnp.array(pollen_inc), node_features=jnp.array(pollen_feat))

    print(f"\n  Pollen TFs: {K}, Fleck augmentation: {n_fleck_perts} TFs")
    print(f"  Evaluating {min(K, n_eval)} TFs LOO...\n")

    res_pollen_only = {}
    res_augmented = {}

    print(f"  {'TF':<12} {'Pollen_r':>9} {'Pollen_r100':>12} "
          f"{'Aug_r':>9} {'Aug_r100':>12} {'Delta_r':>9}")
    print("  " + "-" * 65)

    for ki in range(min(K, n_eval)):
        tf = pollen_tfs[ki]
        train_idx = [i for i in range(K) if i != ki]

        # -- Pollen only --
        key = jax.random.PRNGKey(42 + ki)
        k_model, k_train = jax.random.split(key)
        try:
            pred_po = devograph.PerturbationPredictor(
                gene_dim=feat_dim, hidden_dim=64, num_fates=3,
                conv_cls=hgx.UniGCNConv, num_layers=2, key=k_model,
            )
            fates_train = np.zeros((len(train_idx), 3), dtype=np.float32)
            eff_train = np.expand_dims(pollen_eff[train_idx], -1)
            pred_po = devograph.train_perturbation_predictor(
                pred_po, hg,
                perturbations=jnp.array(pollen_masks[train_idx]),
                targets=(jnp.array(eff_train), jnp.array(fates_train)),
                epochs=150, key=k_train,
            )
            if tf in pollen_gene_idx:
                ko_pred, _ = devograph.in_silico_knockout(pred_po, hg, pollen_gene_idx[tf])
                pred_arr = np.array(ko_pred)
                if pred_arr.ndim > 1:
                    pred_arr = pred_arr.mean(axis=-1)
                obs = pollen_eff[ki]
                r_po, _ = stats.pearsonr(pred_arr, obs) if np.std(pred_arr) > 1e-8 else (0.0, 1.0)
                top100 = np.argsort(-np.abs(obs))[:100]
                r_po_100, _ = stats.pearsonr(pred_arr[top100], obs[top100]) if np.std(pred_arr[top100]) > 1e-8 else (0.0, 1.0)
                res_pollen_only[tf] = {"r_all": float(r_po), "r_100": float(r_po_100)}
        except Exception as e:
            print(f"    {tf} Pollen-only failed: {e}")
            continue

        # -- Augmented with Fleck --
        key = jax.random.PRNGKey(42 + ki)
        k_model, k_train = jax.random.split(key)
        try:
            aug_masks = np.concatenate([pollen_masks[train_idx], fleck_masks_proj])
            aug_eff = np.concatenate([pollen_eff[train_idx], fleck_eff_proj])
            aug_eff_3d = np.expand_dims(aug_eff, -1)
            aug_fates = np.zeros((aug_masks.shape[0], 3), dtype=np.float32)

            pred_aug = devograph.PerturbationPredictor(
                gene_dim=feat_dim, hidden_dim=64, num_fates=3,
                conv_cls=hgx.UniGCNConv, num_layers=2, key=k_model,
            )
            pred_aug = devograph.train_perturbation_predictor(
                pred_aug, hg,
                perturbations=jnp.array(aug_masks),
                targets=(jnp.array(aug_eff_3d), jnp.array(aug_fates)),
                epochs=150, key=k_train,
            )
            if tf in pollen_gene_idx:
                ko_pred, _ = devograph.in_silico_knockout(pred_aug, hg, pollen_gene_idx[tf])
                pred_arr = np.array(ko_pred)
                if pred_arr.ndim > 1:
                    pred_arr = pred_arr.mean(axis=-1)
                obs = pollen_eff[ki]
                r_aug, _ = stats.pearsonr(pred_arr, obs) if np.std(pred_arr) > 1e-8 else (0.0, 1.0)
                top100 = np.argsort(-np.abs(obs))[:100]
                r_aug_100, _ = stats.pearsonr(pred_arr[top100], obs[top100]) if np.std(pred_arr[top100]) > 1e-8 else (0.0, 1.0)
                res_augmented[tf] = {"r_all": float(r_aug), "r_100": float(r_aug_100)}

                delta = r_aug - r_po if tf in res_pollen_only else 0
                print(f"  {tf:<12} {r_po:>9.4f} {r_po_100:>12.4f} "
                      f"{r_aug:>9.4f} {r_aug_100:>12.4f} {delta:>+9.4f}")
        except Exception as e:
            print(f"    {tf} augmented failed: {e}")

    # Summary
    if res_pollen_only and res_augmented:
        shared = set(res_pollen_only.keys()) & set(res_augmented.keys())
        mean_po = np.mean([res_pollen_only[t]["r_all"] for t in shared])
        mean_aug = np.mean([res_augmented[t]["r_all"] for t in shared])
        mean_po_100 = np.mean([res_pollen_only[t]["r_100"] for t in shared])
        mean_aug_100 = np.mean([res_augmented[t]["r_100"] for t in shared])

        print(f"\n  Mean Pollen-only r:    {mean_po:.4f}  (top-100: {mean_po_100:.4f})")
        print(f"  Mean Augmented r:      {mean_aug:.4f}  (top-100: {mean_aug_100:.4f})")
        print(f"  Delta (Aug - Pollen):  {mean_aug - mean_po:+.4f}  (top-100: {mean_aug_100 - mean_po_100:+.4f})")

        n_improved = sum(1 for t in shared
                        if res_augmented[t]["r_all"] > res_pollen_only[t]["r_all"])
        print(f"  TFs improved:          {n_improved}/{len(shared)}")

    return {
        "pollen_only": res_pollen_only,
        "augmented": res_augmented,
    }


# ===================================================================
# Figure
# ===================================================================

def generate_figure(summary_a, summary_b, summary_c, fig_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: topology test bar chart
    ax = axes[0, 0]
    if summary_a:
        names = list(summary_a.keys())
        r_all = [summary_a[n]["r_all"] for n in names]
        r_100 = [summary_a[n]["r_100"] for n in names]
        r_20 = [summary_a[n]["r_20"] for n in names]
        x = np.arange(len(names))
        w = 0.25
        ax.bar(x - w, r_all, w, label="All genes", color="#e41a1c", alpha=0.8)
        ax.bar(x, r_100, w, label="Top-100 DE", color="#377eb8", alpha=0.8)
        ax.bar(x + w, r_20, w, label="Top-20 DE", color="#4daf4a", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([n.split("(")[0].strip() for n in names],
                          rotation=20, ha="right", fontsize=9)
        ax.legend(fontsize=8)
        ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("A. GRN Topology Test (LOO prediction)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Jaccard distribution
    ax = axes[0, 1]
    if summary_b and "per_tf" in summary_b:
        jaccards = [v["jaccard"] for v in summary_b["per_tf"].values()]
        fishers = [-np.log10(max(v["fisher_p"], 1e-20)) for v in summary_b["per_tf"].values()]
        tf_names_b = list(summary_b["per_tf"].keys())
        scatter = ax.scatter(jaccards, fishers, alpha=0.7, s=40, c="#984ea3")
        ax.axhline(y=-np.log10(0.05), color="red", linestyle="--", alpha=0.5,
                   label="p=0.05")
        ax.axhline(y=-np.log10(0.05 / len(tf_names_b)), color="red",
                   linestyle=":", alpha=0.5, label="Bonferroni")
        for i, tf in enumerate(tf_names_b):
            if fishers[i] > 3 or jaccards[i] > 0.15:
                ax.annotate(tf, (jaccards[i], fishers[i]), fontsize=7, alpha=0.8)
        ax.legend(fontsize=8)
    ax.set_xlabel("Jaccard overlap")
    ax.set_ylabel("-log10(Fisher p)")
    ax.set_title("B. Regulon Overlap: Fleck vs Pollen", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel C: domain adaptation
    ax = axes[1, 0]
    if summary_c.get("pollen_only") and summary_c.get("augmented"):
        po = summary_c["pollen_only"]
        aug = summary_c["augmented"]
        shared = sorted(set(po.keys()) & set(aug.keys()))
        r_po = [po[t]["r_all"] for t in shared]
        r_aug = [aug[t]["r_all"] for t in shared]
        ax.scatter(r_po, r_aug, alpha=0.7, s=50, c="#ff7f00")
        for i, tf in enumerate(shared):
            ax.annotate(tf, (r_po[i], r_aug[i]), fontsize=7, alpha=0.8)
        lim = [min(min(r_po), min(r_aug)) - 0.05, max(max(r_po), max(r_aug)) + 0.05]
        ax.plot(lim, lim, "k--", alpha=0.3, label="y=x")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.legend()
    ax.set_xlabel("Pollen-only LOO r")
    ax.set_ylabel("Augmented (+ Fleck) LOO r")
    ax.set_title("C. Domain Adaptation: + Fleck data", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel D: summary
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("D. Phase 3 Verdict", fontweight="bold")
    lines = ["Phase 3 Deep Analysis Results", "=" * 40, ""]
    if summary_a:
        lines.append("A. GRN Topology (LOO, top-100 DE):")
        for name, data in summary_a.items():
            lines.append(f"   {name:<22} r={data['r_100']:.4f}")
        lines.append("")
    if summary_b:
        lines.append(f"B. Regulon overlap: {summary_b.get('n_sig_fisher', 0)}/{len(summary_b.get('per_tf', {}))} sig")
        lines.append(f"   Mean Jaccard: {summary_b.get('mean_jaccard', 0):.4f}")
        lines.append("")
    if summary_c.get("pollen_only") and summary_c.get("augmented"):
        po = summary_c["pollen_only"]
        aug = summary_c["augmented"]
        shared = set(po.keys()) & set(aug.keys())
        if shared:
            m_po = np.mean([po[t]["r_all"] for t in shared])
            m_aug = np.mean([aug[t]["r_all"] for t in shared])
            lines.append(f"C. Domain adaptation delta: {m_aug - m_po:+.4f}")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    fig.suptitle("Phase 3: Does Organoid GRN Transfer to Primary Cortex?",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = fig_dir / "phase3_deep_analysis.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# ===================================================================
# Main
# ===================================================================

def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    fig_dir = script_dir.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("Loading datasets...")
    fleck = load_dataset(data_dir / "processed")
    pollen = load_dataset(data_dir / "pollen" / "processed")
    if fleck is None or pollen is None:
        sys.exit("ERROR: Missing dataset")

    shared_genes = sorted(set(fleck["gene_names"]) & set(pollen["gene_names"]))
    print(f"  Fleck: {len(fleck['gene_names'])} genes, {len(fleck['tf_names'])} TFs")
    print(f"  Pollen: {len(pollen['gene_names'])} genes, "
          f"{len(pollen.get('perturbed_tf_names', pollen['tf_names']))} TFs")
    print(f"  Shared genes: {len(shared_genes)}")

    # Run all three analyses
    summary_a = analysis_a_topology(fleck, pollen, shared_genes, n_eval=10)
    summary_b = analysis_b_regulon_comparison(fleck, pollen, shared_genes)
    summary_c = analysis_c_domain_adaptation(fleck, pollen, shared_genes, n_eval=10)

    # Generate figure
    fig_path = generate_figure(summary_a, summary_b, summary_c, fig_dir)
    print(f"\n  Figure: {fig_path}")

    # Save results
    results = {
        "topology_test": {k: {kk: vv for kk, vv in v.items() if kk != "per_tf"}
                          for k, v in summary_a.items()} if summary_a else None,
        "regulon_comparison": {k: v for k, v in summary_b.items()
                               if k != "per_tf"} if summary_b else None,
        "domain_adaptation": {
            "pollen_only_mean_r": float(np.mean([v["r_all"] for v in summary_c["pollen_only"].values()])) if summary_c.get("pollen_only") else None,
            "augmented_mean_r": float(np.mean([v["r_all"] for v in summary_c["augmented"].values()])) if summary_c.get("augmented") else None,
        } if summary_c else None,
    }
    results_path = fig_dir / "phase3_deep_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: {results_path}")

    # Final verdict
    print("\n" + "=" * 64)
    print("  PHASE 3 VERDICT")
    print("=" * 64)
    if summary_a:
        fleck_r = summary_a.get("Fleck Pando (organoid)", {}).get("r_100", 0)
        random_r = summary_a.get("Random (null)", {}).get("r_100", 0)
        pollen_r = summary_a.get("Pollen DE (own GRN)", {}).get("r_100", 0)
        print(f"  Topology: Fleck r_100={fleck_r:.4f} vs Random r_100={random_r:.4f}")
        if fleck_r > random_r:
            print(f"  -> Fleck GRN DOES help (+{fleck_r - random_r:.4f} over random)")
        else:
            print(f"  -> Fleck GRN does NOT help ({fleck_r - random_r:+.4f} vs random)")
    if summary_b:
        print(f"  Regulon overlap: {summary_b.get('n_sig_bonferroni', 0)} TFs sig after Bonferroni")
    if summary_c.get("pollen_only") and summary_c.get("augmented"):
        po = summary_c["pollen_only"]
        aug = summary_c["augmented"]
        shared = set(po.keys()) & set(aug.keys())
        if shared:
            delta = np.mean([aug[t]["r_all"] for t in shared]) - np.mean([po[t]["r_all"] for t in shared])
            print(f"  Domain adaptation: {delta:+.4f} (augmented vs Pollen-only)")
    print("=" * 64)


if __name__ == "__main__":
    main()
