"""Activity-induced "Learning Regulome" — Hrvatin 2018 GSE102827.

Solé Open Problem: Bio-Computing & Synthetic Agency.
Original plan called for GSE207577 (Kagan DishBrain), which is private until
2026-12-31 and is MEA-only. This benchmark uses the canonical scRNA-seq
analog: Hrvatin et al. (2018) light-stimulation paradigm in mouse V1.

We score immediate-early gene (IEG) and late-response regulons across
0h / 1h / 4h conditions, then compare TF-activity dynamics with the
Fleck Pando regulon prior to identify "Plasticity-associated regulons" —
TFs whose targets shift coordinately with activity, the transcriptomic
signature of the "embodied learning" state.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats


# Immediate-early genes (rapid: ~30 min - 1 h after stimulation)
IEGS = ["FOS", "FOSB", "EGR1", "EGR2", "ARC", "JUN", "JUNB", "NPAS4", "NR4A1", "NR4A2"]

# Late-response genes (peak 2-6 h)
LATE = ["BDNF", "HOMER1", "DUSP1", "GADD45B", "RGS2", "CREM"]

# Mouse symbols are typically Title-case (Fos, Egr1) — handle both
def _resolve_symbols(adata, symbols):
    found = []
    for s in symbols:
        if s in adata.var_names:
            found.append(s)
        elif s.title() in adata.var_names:
            found.append(s.title())
        elif s.lower() in adata.var_names:
            found.append(s.lower())
    return found


def load_pando_regulons(grn_path):
    if not grn_path.exists():
        return None
    df = pd.read_csv(grn_path, sep="\t")
    if "tf" not in df.columns or "target" not in df.columns:
        return None
    return df.groupby("tf")["target"].apply(list).to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/agency/hrvatin_2018_processed.h5ad")
    parser.add_argument("--grn", default="data/zenodo/grn_modules.tsv")
    parser.add_argument("--out-fig", default="figures/learning_regulome.png")
    parser.add_argument("--out-json", default="figures/learning_regulome_results.json")
    args = parser.parse_args()

    adata_path = Path(args.input)
    if not adata_path.exists():
        print(f"  Error: {adata_path} not found.")
        print("  Run: python scripts/download_agency.py --hrvatin")
        print("       python scripts/preprocess_agency.py")
        return

    print(f"Loading Hrvatin V1 data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    iegs = _resolve_symbols(adata, IEGS)
    lates = _resolve_symbols(adata, LATE)
    print(f"  IEGs found:  {iegs}")
    print(f"  Late genes:  {lates}")
    if iegs:
        sc.tl.score_genes(adata, iegs, score_name="ieg_score")
    if lates:
        sc.tl.score_genes(adata, lates, score_name="late_score")

    # Per-stim group means
    stim_col = "stim_duration" if "stim_duration" in adata.obs.columns else None
    timecourse = {}
    if stim_col is not None:
        for stim, sub in adata.obs.groupby(stim_col):
            timecourse[str(stim)] = {
                "n_cells": int(len(sub)),
                "ieg_mean": float(sub.get("ieg_score", pd.Series([np.nan])).mean()),
                "late_mean": float(sub.get("late_score", pd.Series([np.nan])).mean()),
            }

    # Plasticity-associated regulons via Pando GRN scoring
    regulons = load_pando_regulons(Path(args.grn))
    plasticity_tfs = ["FOS", "JUN", "EGR1", "NPAS4", "CREB1", "MEF2C", "NR4A1", "BDNF", "ARC"]
    regulon_dynamics = {}
    if regulons is not None and stim_col is not None:
        for tf in plasticity_tfs:
            tf_key = tf if tf in regulons else (tf.title() if tf.title() in regulons else None)
            if tf_key is None:
                continue
            targets = _resolve_symbols(adata, regulons[tf_key])
            if len(targets) < 5:
                continue
            score_col = f"reg_{tf}"
            sc.tl.score_genes(adata, targets, score_name=score_col)
            stim_means = {
                str(s): float(sub[score_col].mean())
                for s, sub in adata.obs.groupby(stim_col)
            }
            # Stimulation effect: max(1h,4h) − 0h
            base = stim_means.get("0h", min(stim_means.values()))
            peak = max(v for k, v in stim_means.items() if k != "0h") if len(stim_means) > 1 else base
            regulon_dynamics[tf] = {
                "by_stim": stim_means,
                "induction": peak - base,
                "n_targets": len(targets),
            }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if stim_col is not None and "ieg_score" in adata.obs.columns:
        order = ["0h", "1h", "4h"]
        order = [o for o in order if o in adata.obs[stim_col].unique()]
        means_ieg = [adata.obs.loc[adata.obs[stim_col] == o, "ieg_score"].mean() for o in order]
        means_late = [adata.obs.loc[adata.obs[stim_col] == o, "late_score"].mean() for o in order if "late_score" in adata.obs.columns]
        axes[0].plot(order, means_ieg, "o-", label="IEG", color="firebrick")
        if "late_score" in adata.obs.columns:
            axes[0].plot(order[:len(means_late)], means_late, "s-", label="Late", color="steelblue")
        axes[0].set_xlabel("Stimulation duration")
        axes[0].set_ylabel("Mean signature score")
        axes[0].set_title("Activity-dependent gene programs")
        axes[0].legend()

    if regulon_dynamics:
        tfs = list(regulon_dynamics.keys())
        inds = [regulon_dynamics[t]["induction"] for t in tfs]
        order_idx = np.argsort(inds)
        tfs = [tfs[i] for i in order_idx]
        inds = [inds[i] for i in order_idx]
        axes[1].barh(tfs, inds, color="firebrick")
        axes[1].axvline(0, color="black", lw=0.5)
        axes[1].set_xlabel("Δ regulon score (peak − baseline)")
        axes[1].set_title("Plasticity-associated regulons")

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    print(f"  Figure saved to {out_fig}")

    results = {
        "n_cells": int(adata.shape[0]),
        "iegs_found": iegs,
        "late_found": lates,
        "timecourse": timecourse,
        "regulon_dynamics": regulon_dynamics,
        "regulon_source": "Pando" if regulons else "missing",
        "note_dishbrain": "GSE207577 (Kagan DishBrain) is private until 2026-12-31; "
                          "Hrvatin GSE102827 used as scRNA-seq proxy for activity-induced regulome",
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {out_json}")


if __name__ == "__main__":
    main()
