"""Vascular-Neural Cross-talk regulons in vOrganoids (Shi 2020, GSE131094).

Solé Open Problem: Vascularization & Metabolic Integration ("metabolic wall").
Quantifies regulons whose activity depends on the presence of vascular cells —
i.e. cross-talk hyperedges that exist in vOrganoids but are absent in
non-vascularized cortical organoids. Uses the Fleck Pando GRN as the regulon
prior so we share a common module library across all benchmarks.
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


# Endothelial / vascular markers — used both to confirm vascular cells are
# present and to score "cross-talk" regulons in neural cells.
VASCULAR_MARKERS = ["PECAM1", "CDH5", "KDR", "CLDN5", "VWF", "FLT1", "TIE1"]

# Neural / cortical markers — used to gate the "neural-side" cells.
NEURAL_MARKERS = ["SOX2", "PAX6", "FOXG1", "NEUROD2", "TBR1", "SATB2", "BCL11B"]

# Candidate cross-talk TFs (vascular niche signaling -> neural progenitors).
# These appear in Pando regulons that previous papers flagged as
# vasculature-responsive (Notch/HIF/VEGF axes).
CROSSTALK_TFS = ["HES1", "HES5", "RBPJ", "EPAS1", "HIF1A", "ETS1", "SOX17", "GATA2"]


def load_pando_regulons(grn_path):
    if not grn_path.exists():
        return None
    df = pd.read_csv(grn_path, sep="\t")
    # grn_modules.tsv: 'tf', 'target', plus weight cols
    if "tf" not in df.columns or "target" not in df.columns:
        return None
    return df.groupby("tf")["target"].apply(list).to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/bioprinting/vorganoid_2020_processed.h5ad")
    parser.add_argument("--grn", default="data/zenodo/grn_modules.tsv",
                        help="Pando GRN (regulon prior) for cross-talk scoring")
    parser.add_argument("--out-fig", default="figures/vorganoid_crosstalk.png")
    parser.add_argument("--out-json", default="figures/vorganoid_crosstalk_results.json")
    args = parser.parse_args()

    adata_path = Path(args.input)
    if not adata_path.exists():
        print(f"  Error: {adata_path} not found.")
        print("  Run scripts/download_bioprinting.py --vorganoid then preprocess_bioprinting.py")
        return

    print(f"Loading vOrganoid data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    # GSE131094 ships pre-normalized log-expression (values include negatives),
    # so don't re-normalize. Only filter cells with truly empty rows.
    x_min = float(adata.X.min())
    if x_min < 0:
        print(f"  Detected pre-normalized values (min={x_min:.3f}); skipping normalize/log1p")
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Confirm conditions are parseable
    if "condition" not in adata.obs.columns or adata.obs["condition"].nunique() < 2:
        print("  Warning: no condition split detected. Check label parsing in preprocess.")
        print(f"  Found columns: {list(adata.obs.columns)}")
        print(f"  Conditions: {adata.obs.get('condition', pd.Series()).value_counts().to_dict()}")

    # 1. Vascular & neural identity scores
    vmarkers = [g for g in VASCULAR_MARKERS if g in adata.var_names]
    nmarkers = [g for g in NEURAL_MARKERS if g in adata.var_names]
    print(f"  Vascular markers found: {vmarkers}")
    print(f"  Neural markers found:   {nmarkers}")
    if vmarkers:
        sc.tl.score_genes(adata, vmarkers, score_name="vascular_score")
    if nmarkers:
        sc.tl.score_genes(adata, nmarkers, score_name="neural_score")

    # 2. Cross-talk regulon scoring against Pando GRN, falling back to TF expression
    regulons = load_pando_regulons(Path(args.grn))
    crosstalk_scores = {}
    for tf in CROSSTALK_TFS:
        if regulons and tf in regulons:
            targets = [g for g in regulons[tf] if g in adata.var_names]
        else:
            targets = []
        if len(targets) >= 5:
            score_col = f"reg_{tf}"
            sc.tl.score_genes(adata, targets, score_name=score_col)
        elif tf in adata.var_names:
            score_col = f"expr_{tf}"
            x = adata[:, tf].X
            if hasattr(x, "toarray"):
                x = x.toarray()
            adata.obs[score_col] = np.asarray(x).flatten()
        else:
            continue
        crosstalk_scores[tf] = score_col

    # 3. Per-condition cross-talk delta (vOrganoid vs Organoid)
    deltas = {}
    if "condition" in adata.obs.columns and adata.obs["condition"].nunique() >= 2:
        is_vasc = adata.obs.get("is_vascularized", adata.obs["condition"].str.contains("v", case=False))
        for tf, col in crosstalk_scores.items():
            v_vals = adata.obs.loc[is_vasc, col].dropna().values
            o_vals = adata.obs.loc[~is_vasc, col].dropna().values
            if len(v_vals) < 10 or len(o_vals) < 10:
                continue
            t, p = stats.mannwhitneyu(v_vals, o_vals, alternative="two-sided")
            deltas[tf] = {
                "vorganoid_mean": float(np.mean(v_vals)),
                "organoid_mean": float(np.mean(o_vals)),
                "delta": float(np.mean(v_vals) - np.mean(o_vals)),
                "u_stat": float(t),
                "p_value": float(p),
            }

    # 4. Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if "vascular_score" in adata.obs.columns and "condition" in adata.obs.columns:
        for cond, sub in adata.obs.groupby("condition"):
            axes[0].hist(sub["vascular_score"], bins=40, alpha=0.5, label=str(cond))
        axes[0].set_xlabel("Vascular identity score")
        axes[0].set_ylabel("Cells")
        axes[0].set_title("PECAM1/CDH5/KDR signature")
        axes[0].legend()

    if deltas:
        tfs = list(deltas.keys())
        ds = [deltas[t]["delta"] for t in tfs]
        ps = [deltas[t]["p_value"] for t in tfs]
        colors = ["firebrick" if p < 0.01 else "gray" for p in ps]
        axes[1].barh(tfs, ds, color=colors)
        axes[1].axvline(0, color="black", lw=0.5)
        axes[1].set_xlabel("Δ regulon score (vOrganoid − Organoid)")
        axes[1].set_title("Vascular-Neural Cross-talk regulons (red: p<0.01)")

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    print(f"  Figure saved to {out_fig}")

    results = {
        "n_cells": int(adata.shape[0]),
        "n_genes": int(adata.shape[1]),
        "conditions": adata.obs.get("condition", pd.Series()).value_counts().to_dict(),
        "timepoints": adata.obs.get("timepoint", pd.Series()).value_counts().to_dict(),
        "vascular_markers_found": vmarkers,
        "neural_markers_found": nmarkers,
        "crosstalk_regulons": deltas,
        "regulon_source": "Pando" if regulons else "TF-expression-fallback",
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {out_json}")


if __name__ == "__main__":
    main()
