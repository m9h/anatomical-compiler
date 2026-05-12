"""Regenerative Flow — hgx Hypergraph Neural ODE on injury-recovery timecourses.

Solé Open Problem: Self-Repair and Regeneration ("robustness").
Original plan called for GSE138835 (Lawlor 2021 bioprinted kidney) but that
accession contains only one bioprinted-vs-manual sample with no time/condition
structure (verified 2026-05-10 via `obs.columns == [study, system, culture]`,
all single-valued). We therefore make this script **data-agnostic**: point it
at any h5ad whose `obs` includes a `time` (or `stage`) column with at least
three ordered values, and it will fit an `hgx` Hypergraph Neural ODE on the
TF-expression trajectory and return the regulators most predictive of the
regenerative flow.

Recommended timecourse accessions:
  - Balzer et al. 2022 (Nat Commun, doi:10.1038/s41467-022-31772-9) — adaptive
    vs fibrotic kidney regeneration, ~110k cells across multiple post-IRI days.
  - Gupta et al. 2022 (Sci. Transl. Med., doi:10.1126/scitranslmed.abj4772) —
    cisplatin-injured kidney organoids with snRNA-seq at day 0/3/7/14.
  - Subramanian et al. 2020 (Nat Commun) — kidney organoid + cisplatin.

Until one of those is pulled in, the script falls back to running on the
existing Lawlor h5ad as a *single-timepoint* baseline so the rest of the
pipeline (TF panel selection, hypergraph construction, regulon scoring) can be
exercised end-to-end.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

try:
    import jax
    import jax.numpy as jnp
    import hgx
    HAS_HGX = True
except ImportError:
    HAS_HGX = False

try:
    import diffrax  # noqa: F401
    HAS_DIFFRAX = True
except ImportError:
    HAS_DIFFRAX = False


KIDNEY_INJURY_TFS = [
    "HNF4A", "PAX2", "PAX8", "WT1", "LHX1", "SIX1", "SIX2",
    "FOXC1", "FOXC2", "JUN", "FOS", "SOX9", "VCAM1", "HAVCR1",
    "CDH1", "CD44", "ATF3", "MYC", "TP53",
]


def _resolve_symbols(adata, symbols):
    """Resolve gene symbols across human (UPPER) and mouse (Title) conventions."""
    found = []
    for s in symbols:
        if s in adata.var_names:
            found.append(s)
        elif s.title() in adata.var_names:
            found.append(s.title())
        elif s.lower() in adata.var_names:
            found.append(s.lower())
    return found


def _resolve_time_col(adata):
    for col in ("time", "timepoint", "stage", "day", "dpi", "post_injury"):
        if col in adata.obs.columns:
            return col
    return None


def _bin_by_time(adata, time_col, n_bins):
    """Group cells by ordered time value and average TF expression per bin."""
    if pd.api.types.is_numeric_dtype(adata.obs[time_col]):
        order = sorted(adata.obs[time_col].unique())
    else:
        order = sorted(adata.obs[time_col].astype(str).unique(),
                       key=lambda s: (len(s), s))
    bins = order if len(order) <= n_bins else np.array_split(order, n_bins)
    return order, bins


def build_tf_trajectory(adata, tf_panel, time_col, n_bins=5):
    """Return (T, N_TF, 1) trajectory of mean log-expr per timepoint."""
    times, _ = _bin_by_time(adata, time_col, n_bins)
    X = adata[:, tf_panel].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    df = pd.DataFrame(X, columns=tf_panel, index=adata.obs.index)
    df["_t"] = adata.obs[time_col].values
    means = df.groupby("_t")[tf_panel].mean().reindex(times)
    arr = means.values  # (T, N_TF)
    return times, arr[:, :, None].astype(np.float32)


def fit_regenerative_ode(trajectories, epochs, key):
    """Fit Hypergraph Neural ODE on TF trajectory; return per-TF rollout MSE."""
    if not (HAS_HGX and HAS_DIFFRAX):
        return None
    T, N, D = trajectories.shape
    # All-pairs incidence -> single hyperedge containing all TFs (the regulome)
    incidence = jnp.ones((N, 1), dtype=jnp.float32)
    snaps = [hgx.from_incidence(incidence, node_features=jnp.array(trajectories[t])) for t in range(T)]
    times = jnp.linspace(0.0, 1.0, T)
    temp_hg = hgx.from_snapshots(snaps, times=times)

    k1, k2 = jax.random.split(key)
    conv = hgx.UniGCNConv(D, D, key=k1)
    print(f"  Fitting Neural ODE over T={T} timepoints, N={N} TFs (epochs={epochs})...")
    t0 = time.time()
    ode = hgx.fit_neural_ode(temp_hg, conv, key=k2, epochs=epochs, lr=1e-3)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    # Per-TF rollout from t=0
    sol = ode(snaps[0], t0=0.0, t1=1.0, saveat=__import__("diffrax").SaveAt(ts=times[1:]))
    pred = np.asarray(sol.ys)  # (T-1, N, D)
    obs = trajectories[1:]
    per_tf_mse = ((pred - obs) ** 2).mean(axis=(0, 2))
    return {
        "elapsed_sec": elapsed,
        "per_tf_mse": per_tf_mse.tolist(),
        "pred": pred.squeeze(-1).tolist(),
        "obs": obs.squeeze(-1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/bioprinting/lawlor_2021_processed.h5ad",
                        help="h5ad with optional time/timepoint/stage column")
    parser.add_argument("--time-col", default=None,
                        help="Override time column name (auto-detected by default)")
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-fig", default="figures/regenerative_flow.png")
    parser.add_argument("--out-json", default="figures/regenerative_flow_results.json")
    args = parser.parse_args()

    adata_path = Path(args.input)
    if not adata_path.exists():
        print(f"  Error: {adata_path} not found.")
        return

    print(f"Loading {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    time_col = args.time_col or _resolve_time_col(adata)
    tf_panel = _resolve_symbols(adata, KIDNEY_INJURY_TFS)
    print(f"  Time column: {time_col}")
    print(f"  TFs found:   {len(tf_panel)} / {len(KIDNEY_INJURY_TFS)}")
    print(f"  TFs resolved: {tf_panel[:8]}{'...' if len(tf_panel) > 8 else ''}")

    results = {
        "input": str(adata_path),
        "n_cells": int(adata.shape[0]),
        "n_genes": int(adata.shape[1]),
        "time_col": time_col,
        "tfs_resolved": tf_panel,
    }

    if time_col is None or adata.obs[time_col].nunique() < 3:
        print("  No multi-timepoint structure detected.")
        print("  This h5ad is a single-timepoint baseline; regenerative-flow ODE")
        print("  needs a timecourse. See module docstring for candidate accessions.")
        results["status"] = "single_timepoint_baseline"
        results["unique_time_values"] = (
            adata.obs[time_col].unique().tolist() if time_col else []
        )
    elif len(tf_panel) < 3:
        print("  Too few TFs in this dataset for Hypergraph Neural ODE.")
        results["status"] = "tf_panel_underdetermined"
    else:
        results["status"] = "ode_fit"
        times, traj = build_tf_trajectory(adata, tf_panel, time_col, args.n_bins)
        results["timepoints"] = [str(t) for t in times]

        if HAS_HGX and HAS_DIFFRAX:
            key = jax.random.PRNGKey(args.seed)
            ode_out = fit_regenerative_ode(traj, args.epochs, key)
            if ode_out is not None:
                results["ode"] = {
                    "elapsed_sec": ode_out["elapsed_sec"],
                    "per_tf_mse": dict(zip(tf_panel, ode_out["per_tf_mse"])),
                }
                # Lowest MSE = TFs whose flow the ODE captures best (drivers
                # of stable regenerative trajectory)
                ranked = sorted(
                    zip(tf_panel, ode_out["per_tf_mse"]), key=lambda x: x[1]
                )
                results["regenerative_drivers"] = [
                    {"tf": tf, "rollout_mse": float(mse)} for tf, mse in ranked[:8]
                ]
                # Plot
                fig, axes = plt.subplots(1, 2, figsize=(13, 5))
                obs = np.array(ode_out["obs"])  # (T-1, N)
                pred = np.array(ode_out["pred"])
                for i, tf in enumerate(tf_panel[:6]):
                    axes[0].plot(obs[:, i], "o-", label=f"{tf} obs", alpha=0.7)
                    axes[0].plot(pred[:, i], "x--", label=f"{tf} pred", alpha=0.7)
                axes[0].set_xlabel("Timepoint index")
                axes[0].set_ylabel("Log expression")
                axes[0].set_title("Regenerative Flow: ODE rollout vs observed")
                axes[0].legend(fontsize=7, ncol=2)

                top = ranked[:10]
                axes[1].barh([t for t, _ in top][::-1],
                             [m for _, m in top][::-1], color="forestgreen")
                axes[1].set_xlabel("Rollout MSE (lower → better captured)")
                axes[1].set_title("Regenerative drivers")
                plt.tight_layout()
                Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(args.out_fig, dpi=200, bbox_inches="tight")
                print(f"  Figure saved to {args.out_fig}")
        else:
            results["status"] = "missing_hgx_or_diffrax"
            print("  Skipping ODE fit: hgx or diffrax not importable.")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {args.out_json}")


if __name__ == "__main__":
    main()
