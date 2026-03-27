#!/usr/bin/env python3
"""Phase 3 multi-dataset validation: regulon overlap across datasets."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy import stats
import anndata as ad, scipy.sparse as sp

NT_PATTERNS = ["non-targeting", "NT", "CTRL", "scramble", "NTC", "safe"]
NT_EXACT = {"NA", "nan", "NaN", "none", "None"}

def is_control(label):
    s = str(label)
    return s in NT_EXACT or any(p.lower() in s.lower() for p in NT_PATTERNS)

def load_fleck(data_dir):
    d = {}
    for name in ["gene_names", "tf_names"]:
        with open(data_dir / f"{name}.json") as f: d[name] = json.load(f)
    d["incidence"] = np.load(data_dir / "incidence.npy")
    return d

def preprocess_h5ad(path, fleck_genes, guide_col=None):
    print(f"\n  Loading {path.name}...")
    adata = ad.read_h5ad(path)
    print(f"    {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"    obs: {list(adata.obs.columns)[:12]}")
    if guide_col is None:
        for col in ["Gene_target_single", "Gene_target", "perturbation"]:
            if col in adata.obs.columns and 5 <= adata.obs[col].dropna().nunique() <= 200:
                guide_col = col; break
    if guide_col is None:
        print("    No guide column found"); return None
    guides = adata.obs[guide_col].astype(str)
    controls = [g for g in sorted(guides.unique()) if is_control(g)]
    tfs = [g for g in sorted(guides.unique()) if not is_control(g)]
    print(f"    '{guide_col}': {len(tfs)} TFs, {len(controls)} controls")
    if len(tfs) < 3: print("    Too few TFs"); return None
    ctrl_mask = guides.isin(controls) if controls else guides.isna()
    X = adata.X
    ctrl_mean = np.array(X[ctrl_mask.values].mean(axis=0)).ravel() if sp.issparse(X) else np.asarray(X[ctrl_mask.values], dtype=np.float32).mean(axis=0)
    gene_names_all = list(adata.var_names)
    de_results = {}
    for tf in tfs:
        tf_mask = guides == tf
        if tf_mask.sum() < 10: continue
        tf_mean = np.array(X[tf_mask.values].mean(axis=0)).ravel() if sp.issparse(X) else np.asarray(X[tf_mask.values], dtype=np.float32).mean(axis=0)
        de_results[tf] = np.log2(tf_mean + 0.1) - np.log2(ctrl_mean + 0.1)
    print(f"    DE for {len(de_results)} TFs")
    pollen_idx = {g: i for i, g in enumerate(gene_names_all)}
    shared = sorted(set(fleck_genes) & set(gene_names_all))
    tf_list = sorted(de_results.keys())
    K, n = len(tf_list), len(shared)
    inc = np.zeros((n, K), dtype=np.float32)
    eff = np.zeros((K, n), dtype=np.float32)
    for ki, tf in enumerate(tf_list):
        l2fc = de_results[tf]
        for gi, gene in enumerate(shared):
            v = l2fc[pollen_idx[gene]]
            eff[ki, gi] = v
            if abs(v) > 0.25: inc[gi, ki] = 1.0
    print(f"    Shared: {n} genes, incidence ({n},{K})")
    return {"gene_names": shared, "tf_names": tf_list, "incidence": inc, "effects": eff}

def regulon_comparison(fleck, ds, name):
    fg, ft, fi = fleck["gene_names"], fleck["tf_names"], fleck["incidence"]
    fgi = {g: i for i, g in enumerate(fg)}
    dg, dt, di = ds["gene_names"], ds["tf_names"], ds["incidence"]
    dgi = {g: i for i, g in enumerate(dg)}
    de = ds.get("effects")
    shared_tfs = sorted(set(ft) & set(dt))
    shared_genes = sorted(set(fg) & set(dg))
    if not shared_tfs: return {"n_shared_tfs": 0, "name": name}
    results, jaccards, dirs = {}, [], []
    for tf in shared_tfs:
        f_idx, d_idx = ft.index(tf), dt.index(tf)
        fm = {g for g in shared_genes if g in fgi and fi[fgi[g], f_idx] > 0}
        dm = {g for g in shared_genes if g in dgi and di[dgi[g], d_idx] > 0}
        inter, union = fm & dm, fm | dm
        j = len(inter) / len(union) if union else 0
        n = len(shared_genes); a = len(inter); b = len(fm)-a; c = len(dm)-a; d = n-a-b-c
        _, fp = stats.fisher_exact([[a,b],[c,d]], alternative="greater")
        dc = float("nan")
        if len(inter) > 5 and de is not None:
            signs = [np.sign(de[d_idx, dgi[g]]) for g in inter if g in dgi and abs(de[d_idx, dgi[g]]) > 0.1]
            if len(signs) > 3:
                mc = 1.0 if sum(s > 0 for s in signs) > len(signs)/2 else -1.0
                dc = sum(1 for s in signs if s == mc) / len(signs)
        results[tf] = {"fleck": len(fm), "ds": len(dm), "inter": len(inter),
                       "jaccard": j, "fisher_p": fp, "dir_conc": dc}
        jaccards.append(j)
        if not np.isnan(dc): dirs.append(dc)
    ns = sum(1 for r in results.values() if r["fisher_p"] < 0.05)
    nb = sum(1 for r in results.values() if r["fisher_p"] < 0.05/max(len(shared_tfs),1))
    print(f"    {name}: Jaccard={np.mean(jaccards):.4f}, sig={ns}/{len(shared_tfs)}, Bonf={nb}" +
          (f", dir={np.mean(dirs):.1%}" if dirs else ""))
    return {"name": name, "n_shared_tfs": len(shared_tfs), "n_shared_genes": len(shared_genes),
            "mean_jaccard": float(np.mean(jaccards)), "n_sig": ns, "n_bonf": nb,
            "mean_dir": float(np.mean(dirs)) if dirs else None, "per_tf": results}

def main():
    data_dir, fig_dir = Path("data"), Path("figures"); fig_dir.mkdir(exist_ok=True)
    fleck = load_fleck(data_dir / "processed")
    print(f"Fleck: {len(fleck['gene_names'])} genes, {len(fleck['tf_names'])} TFs")
    R = {}
    # 1. Pollen 2D screen
    sd = data_dir / "pollen" / "processed"
    if (sd / "gene_names.json").exists():
        print("\n" + "="*60 + "\n  1. Pollen 2D Screen\n" + "="*60)
        with open(sd/"gene_names.json") as f: sg = json.load(f)
        with open(sd/"tf_names.json") as f: st = json.load(f)
        si, se = np.load(sd/"incidence.npy"), np.load(sd/"perturbation_effects.npy")
        R["Pollen 2D"] = regulon_comparison(fleck, {"gene_names":sg,"tf_names":st,"incidence":si,"effects":se}, "Pollen 2D")
    # 2. Pollen Slice
    if (data_dir / "pollen" / "slice.h5ad").exists():
        print("\n" + "="*60 + "\n  2. Pollen Slice (3D)\n" + "="*60)
        d = preprocess_h5ad(data_dir / "pollen" / "slice.h5ad", fleck["gene_names"])
        if d: R["Pollen Slice"] = regulon_comparison(fleck, d, "Pollen Slice")
    # 3. Pollen IN
    if (data_dir / "pollen" / "IN.h5ad").exists():
        print("\n" + "="*60 + "\n  3. Pollen Interneurons\n" + "="*60)
        d = preprocess_h5ad(data_dir / "pollen" / "IN.h5ad", fleck["gene_names"])
        if d: R["Pollen IN"] = regulon_comparison(fleck, d, "Pollen IN")
    # 4. CHOOSE
    cp = data_dir / "choose" / "CHOOSE_ASD_GE.rds"
    if cp.exists():
        print("\n" + "="*60 + "\n  4. CHOOSE Organoid\n" + "="*60)
        try:
            import pyreadr
            result = pyreadr.read_r(str(cp))
            print(f"    pyreadr keys: {list(result.keys())}")
        except Exception as e:
            print(f"    .rds needs R for conversion: {e}")
    # Summary
    print("\n" + "="*60 + "\n  SUMMARY\n" + "="*60)
    print(f"  {'Dataset':<20} {'TFs':>5} {'Jaccard':>8} {'Sig':>5} {'Bonf':>5} {'Dir':>7}")
    print("  " + "-"*53)
    for n, r in R.items():
        ds = f"{r.get('mean_dir',0):.1%}" if r.get('mean_dir') else "N/A"
        print(f"  {n:<20} {r['n_shared_tfs']:>5} {r['mean_jaccard']:>8.4f} {r['n_sig']:>5} {r['n_bonf']:>5} {ds:>7}")
    # Figure
    if len(R) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(R.keys())
        x = np.arange(len(names))
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"][:len(names)]
        jacs = [R[n]["mean_jaccard"] for n in names]
        bars = ax.bar(x, jacs, color=colors, alpha=0.8)
        for i in range(len(names)):
            ax.text(i, jacs[i]+0.001, f'{R[names[i]]["n_bonf"]} Bonf\n{R[names[i]]["n_sig"]} sig', ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel("Mean Jaccard"); ax.set_title("Fleck Pando Regulon Overlap Across Datasets", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(fig_dir / "phase3_multi_dataset.png", dpi=200, bbox_inches="tight")
        plt.close(fig); print(f"\n  Figure: {fig_dir / 'phase3_multi_dataset.png'}")
    sr = {n: {k:v for k,v in r.items() if k != "per_tf"} for n,r in R.items()}
    with open(fig_dir / "phase3_multi_dataset_results.json", "w") as f:
        json.dump(sr, f, indent=2, default=str)

if __name__ == "__main__":
    main()
