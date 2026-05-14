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

# Default driver / stress panels for the multi-component training loss.
# These are the TFs the paper (paper.Rnw §3) calls out as the homeostatic /
# regenerative drivers vs the transient injury responders.  They're used
# both here (when --use-trainer is on) and as the CLI defaults in
# scripts/train_hg_neural_ode.py -- imported from this module so the
# canonical panel definition lives in one place.
DEFAULT_DRIVERS = ["LHX1", "PAX2", "PAX8", "SIX1", "SIX2", "WT1", "FOXC2", "CDH1", "HNF4A"]
DEFAULT_STRESS = ["FOS", "JUN", "ATF3", "CD44"]


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


def _panel_idx_in_tf_panel(tf_panel, names):
    """Return positions within ``tf_panel`` of each name in ``names``.

    Tries the literal symbol, then Title-case, then lower-case, matching
    the gene-symbol resolution policy in :func:`_resolve_symbols`.  The
    output indices are positions in ``tf_panel`` (the model's node axis),
    not adata var indices.
    """
    panel_index = {v: i for i, v in enumerate(tf_panel)}
    out = []
    for s in names:
        for candidate in (s, s.title(), s.lower()):
            if candidate in panel_index:
                out.append(panel_index[candidate])
                break
    return out


def fit_regenerative_ode(
    trajectories,
    epochs,
    key,
    *,
    tf_panel=None,
    drivers=None,
    stress=None,
    lr=1e-3,
    loss_weights=None,
    legacy=False,
):
    """Fit Hypergraph Neural ODE on a TF trajectory and return per-TF rollout MSE.

    Default behaviour now routes through
    :func:`anatomical_compiler.training.train_hg_neural_ode`, which adds
    weighted driver / stress losses, NaN-protected updates, gradient
    clipping, parameter clamping, and a per-epoch components history on
    top of the bare hgx fit.  Pass ``legacy=True`` to fall back to the
    original ``hgx.fit_neural_ode`` call for A/B comparison.

    The return dict preserves the original keys (``elapsed_sec``,
    ``per_tf_mse``, ``pred``, ``obs``) so callers like ``paper.Rnw`` keep
    working unchanged; new keys (``loss_history``, ``final_components``,
    ``driver_panel``, ``stress_panel``) are added on top.

    Args:
        trajectories: ``(T, N, D)`` TF trajectory.
        epochs: Number of training epochs.
        key: JAX PRNG key (forwarded to the conv layer init).
        tf_panel: Optional list of TF symbols ordered along the ``N`` axis;
            required for driver / stress panel resolution.  If ``None``,
            driver and stress weightings degenerate to zero (equivalent
            to the legacy plain-MSE objective).
        drivers: Optional list of TF symbols marked as regenerative
            drivers (extra weight on their MSE).  Defaults to
            :data:`DEFAULT_DRIVERS`.
        stress: Optional list of TF symbols marked as stress responders
            (soft-hinge penalty when their MSE is too small).  Defaults
            to :data:`DEFAULT_STRESS`.
        lr: Adam learning rate.
        loss_weights: Optional dict overriding any of ``w_rollout``,
            ``w_driver``, ``w_stress``, ``w_smooth``, ``w_uniformity``,
            ``w_reg``, ``stress_margin``.  Anything not provided uses the
            ``TrainingConfig`` defaults.
        legacy: If True, run the original bare ``hgx.fit_neural_ode`` and
            return only the legacy keys (no ``loss_history`` etc.).

    Returns:
        dict, or ``None`` if hgx / diffrax are unavailable.
    """
    if not (HAS_HGX and HAS_DIFFRAX):
        return None

    import diffrax  # local import: keeps top-of-module HAS_DIFFRAX gating
    T, N, D = trajectories.shape

    # All-pairs incidence -> single hyperedge containing all TFs (the regulome)
    incidence = jnp.ones((N, 1), dtype=jnp.float32)
    snaps = [
        hgx.from_incidence(incidence, node_features=jnp.array(trajectories[t]))
        for t in range(T)
    ]
    times = jnp.linspace(0.0, 1.0, T)
    temp_hg = hgx.from_snapshots(snaps, times=times)

    k1, k2 = jax.random.split(key)
    conv = hgx.UniGCNConv(D, D, key=k1)

    if legacy:
        print(f"  [legacy fit] Fitting Neural ODE over T={T} timepoints, "
              f"N={N} TFs (epochs={epochs})...")
        t0 = time.time()
        ode = hgx.fit_neural_ode(temp_hg, conv, key=k2, epochs=epochs, lr=lr)
        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.1f}s")
        sol = ode(snaps[0], t0=0.0, t1=1.0,
                  saveat=diffrax.SaveAt(ts=times[1:]))
        pred = np.asarray(sol.ys)
        obs = trajectories[1:]
        per_tf_mse = ((pred - obs) ** 2).mean(axis=(0, 2))
        return {
            "elapsed_sec": elapsed,
            "per_tf_mse": per_tf_mse.tolist(),
            "pred": pred.squeeze(-1).tolist(),
            "obs": obs.squeeze(-1).tolist(),
            "trainer": "legacy",
        }

    # --- New path: multi-component training loop ---
    from hgx._dynamics import HypergraphNeuralODE
    from anatomical_compiler.training import (
        TrainingConfig, train_hg_neural_ode,
    )

    driver_names = list(drivers) if drivers is not None else list(DEFAULT_DRIVERS)
    stress_names = list(stress) if stress is not None else list(DEFAULT_STRESS)
    if tf_panel is None:
        # Without a TF-symbol panel we can't map driver/stress names to
        # node indices; fall back to plain rollout MSE.
        driver_idx, stress_idx = [], []
        driver_resolved, stress_resolved = [], []
    else:
        driver_idx = _panel_idx_in_tf_panel(tf_panel, driver_names)
        stress_idx = _panel_idx_in_tf_panel(tf_panel, stress_names)
        driver_resolved = [tf_panel[i] for i in driver_idx]
        stress_resolved = [tf_panel[i] for i in stress_idx]

    weights = dict(
        w_rollout=1.0, w_driver=1.0, w_stress=0.5,
        w_smooth=0.05, w_uniformity=0.1, w_reg=1e-4,
        stress_margin=0.05,
    )
    if loss_weights:
        weights.update(loss_weights)

    cfg = TrainingConfig(
        n_epochs=epochs,
        learning_rate=lr,
        log_every=max(epochs // 10, 1),
        **weights,
    )

    print(
        f"  Fitting Neural ODE over T={T} timepoints, N={N} TFs "
        f"(epochs={epochs}, drivers={len(driver_idx)}, stress={len(stress_idx)})..."
    )

    model = HypergraphNeuralODE(conv)
    hg0 = snaps[0]
    target = jnp.asarray(trajectories[1:])
    target_times = times[1:]

    def forward(m):
        sol = m(hg0, t0=0.0, t1=1.0,
                saveat=diffrax.SaveAt(ts=target_times))
        return sol.ys

    t0 = time.time()
    result = train_hg_neural_ode(
        model,
        forward=forward,
        target=target,
        driver_idx=jnp.asarray(driver_idx, dtype=jnp.int32),
        stress_idx=jnp.asarray(stress_idx, dtype=jnp.int32),
        config=cfg,
    )
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    pred = np.asarray(forward(result.model))
    obs = np.asarray(target)
    per_tf_mse = ((pred - obs) ** 2).mean(axis=(0, 2))

    # Keep the history compact in JSON: just every record's scalar fields.
    # bl1's train_culture.py logs the whole list -- we do too.
    final_components = result.loss_history[-1] if result.loss_history else {}

    return {
        "elapsed_sec": elapsed,
        "per_tf_mse": per_tf_mse.tolist(),
        "pred": pred.squeeze(-1).tolist(),
        "obs": obs.squeeze(-1).tolist(),
        "trainer": "anatomical_compiler.training.train_hg_neural_ode",
        "loss_history": result.loss_history,
        "final_components": final_components,
        "driver_panel": driver_resolved,
        "stress_panel": stress_resolved,
        "nan_epochs": result.nan_epochs,
        "loss_weights": weights,
    }


def _parse_panel(arg, default):
    """Parse a comma-separated CLI panel argument; fall back to ``default``."""
    if arg is None:
        return list(default)
    return [s.strip() for s in arg.split(",") if s.strip()]


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

    # Multi-component training (anatomical_compiler.training)
    parser.add_argument("--legacy-fit", action="store_true",
                        help="Use the bare hgx.fit_neural_ode path instead of "
                             "the multi-component training loop (A/B comparison).")
    parser.add_argument("--drivers", default=None,
                        help=f"Driver TF panel (comma-separated). Default: "
                             f"{','.join(DEFAULT_DRIVERS)}.")
    parser.add_argument("--stress", default=None,
                        help=f"Stress responder panel (comma-separated). Default: "
                             f"{','.join(DEFAULT_STRESS)}.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate.")
    parser.add_argument("--w-rollout", type=float, default=1.0)
    parser.add_argument("--w-driver", type=float, default=1.0)
    parser.add_argument("--w-stress", type=float, default=0.5)
    parser.add_argument("--w-smooth", type=float, default=0.05)
    parser.add_argument("--w-uniformity", type=float, default=0.1)
    parser.add_argument("--w-reg", type=float, default=1e-4)
    parser.add_argument("--stress-margin", type=float, default=0.05)
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
            drivers = _parse_panel(args.drivers, DEFAULT_DRIVERS)
            stress = _parse_panel(args.stress, DEFAULT_STRESS)
            loss_weights = dict(
                w_rollout=args.w_rollout,
                w_driver=args.w_driver,
                w_stress=args.w_stress,
                w_smooth=args.w_smooth,
                w_uniformity=args.w_uniformity,
                w_reg=args.w_reg,
                stress_margin=args.stress_margin,
            )
            ode_out = fit_regenerative_ode(
                traj, args.epochs, key,
                tf_panel=tf_panel,
                drivers=drivers,
                stress=stress,
                lr=args.lr,
                loss_weights=loss_weights,
                legacy=args.legacy_fit,
            )
            if ode_out is not None:
                # paper.Rnw reads J$regen$ode$per_tf_mse and
                # J$regen$ode$elapsed_sec -- those keys MUST stay.
                results["ode"] = {
                    "elapsed_sec": ode_out["elapsed_sec"],
                    "per_tf_mse": dict(zip(tf_panel, ode_out["per_tf_mse"])),
                    "trainer": ode_out.get("trainer", "legacy"),
                }
                # New multi-component-training fields, added on top.
                for k in ("loss_history", "final_components", "driver_panel",
                          "stress_panel", "nan_epochs", "loss_weights"):
                    if k in ode_out:
                        results["ode"][k] = ode_out[k]

                # Lowest MSE = TFs whose flow the ODE captures best (drivers
                # of stable regenerative trajectory).
                ranked = sorted(
                    zip(tf_panel, ode_out["per_tf_mse"]), key=lambda x: x[1]
                )
                results["regenerative_drivers"] = [
                    {"tf": tf, "rollout_mse": float(mse)} for tf, mse in ranked[:8]
                ]
                # The dual ranking -- highest MSE = transient injury markers
                # the autonomous flow does NOT capture (this is the
                # stress-responder bucket the paper calls out).
                results["transient_responders"] = [
                    {"tf": tf, "rollout_mse": float(mse)} for tf, mse in ranked[-4:]
                ]

                # --- Plot ---
                n_panels = 3 if "loss_history" in ode_out else 2
                fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5))
                if n_panels == 2:
                    axes = list(axes)
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

                if "loss_history" in ode_out and ode_out["loss_history"]:
                    hist = ode_out["loss_history"]
                    epochs_x = [r["epoch"] for r in hist]
                    axes[2].plot(epochs_x, [r["total"] for r in hist],
                                 "k-", lw=2, label="total")
                    axes[2].plot(epochs_x, [r["rollout_mse"] for r in hist],
                                 "C0--", label="rollout")
                    if any(r["driver_mse"] > 0 for r in hist):
                        axes[2].plot(epochs_x, [r["driver_mse"] for r in hist],
                                     "C2--", label="driver")
                    if any(r["stress_hinge"] > 0 for r in hist):
                        axes[2].plot(epochs_x, [r["stress_hinge"] for r in hist],
                                     "C3--", label="stress")
                    axes[2].set_xlabel("Epoch")
                    axes[2].set_ylabel("Loss")
                    axes[2].set_yscale("log")
                    axes[2].set_title("Training loss components")
                    axes[2].legend(fontsize=8)

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
