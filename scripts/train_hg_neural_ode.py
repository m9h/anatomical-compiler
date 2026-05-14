"""Train an hgx Hypergraph Neural ODE on a TF expression timecourse.

CLI counterpart of bl1's ``scripts/train_culture.py``: takes an h5ad with a
time/timepoint/stage column, builds the TF trajectory via the same helpers
used by ``benchmark_regenerative_flow.py``, then runs the proper multi-
component differentiable training loop from
``anatomical_compiler.training.train_hg_neural_ode``.

Compared to ``benchmark_regenerative_flow.py`` (which calls the bare
``hgx.fit_neural_ode``), this script:

  * logs every loss component per epoch into a JSON history;
  * weights a user-supplied driver TF panel and penalises fitting a
    stress-responder panel (so the learned ODE corresponds to the stable
    regenerative flow);
  * uses ``optax.apply_if_finite`` + grad clipping + parameter clamping;
  * writes both the per-epoch history and the final per-TF MSE to the
    output JSON, so downstream R/knitr code can pull a driver/stress
    ranking alongside the trajectory.

Examples::

    # Default kidney-injury defaults (drivers: LHX1, PAX2, ...; stress: FOS, JUN, ATF3)
    python scripts/train_hg_neural_ode.py --input data/bioprinting/lawlor_2021_processed.h5ad

    # Override driver / stress panels
    python scripts/train_hg_neural_ode.py --input my.h5ad \\
        --drivers LHX1,PAX2,SIX1 --stress FOS,JUN,EGR1 --n-epochs 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import util as _ilu
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


# Reuse the trajectory helpers from benchmark_regenerative_flow without
# importing the module as a package (scripts/ isn't on sys.path as one).
def _load_regenerative_flow_helpers():
    here = Path(__file__).parent
    spec = _ilu.spec_from_file_location(
        "_rf", str(here / "benchmark_regenerative_flow.py")
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_RF = _load_regenerative_flow_helpers()


# Default driver / stress panels: imported from benchmark_regenerative_flow
# so the canonical panel definition lives in one place (the benchmark
# defined it first; paper.Rnw §3 references those names).
DEFAULT_DRIVERS = _RF.DEFAULT_DRIVERS
DEFAULT_STRESS = _RF.DEFAULT_STRESS


def _resolve_panel(adata, names: list[str]) -> tuple[list[str], list[int]]:
    """Return (matched_symbols, var_axis_indices) for ``names`` in adata."""
    matched, idx = [], []
    var_names = adata.var_names.tolist()
    var_index = {v: i for i, v in enumerate(var_names)}
    for s in names:
        for candidate in (s, s.title(), s.lower()):
            if candidate in var_index:
                matched.append(candidate)
                idx.append(var_index[candidate])
                break
    return matched, idx


def _panel_idx_in_tf_panel(tf_panel: list[str], names: list[str]) -> list[int]:
    """Return positions within ``tf_panel`` of each name in ``names``.

    The trajectory tensor's node axis is ``tf_panel``, not the full ``adata``,
    so driver/stress indices are positions in ``tf_panel`` (= the model's
    node axis) rather than gene-level adata indices.
    """
    panel_index = {v: i for i, v in enumerate(tf_panel)}
    out = []
    for s in names:
        for candidate in (s, s.title(), s.lower()):
            if candidate in panel_index:
                out.append(panel_index[candidate])
                break
    return out


def _parse_panel(arg: str | None, default: list[str]) -> list[str]:
    if arg is None:
        return list(default)
    return [s.strip() for s in arg.split(",") if s.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Hypergraph Neural ODE on a TF timecourse.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--input", required=True,
                   help="Path to an h5ad with a time/timepoint/stage column.")
    p.add_argument("--time-col", default=None,
                   help="Override time column name (auto-detected by default).")
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument("--tf-panel", default=None,
                   help="Comma-separated list of TFs to include "
                        "(defaults to the kidney-injury panel in "
                        "benchmark_regenerative_flow).")

    # Driver / stress panels (drive the training-loss weighting)
    p.add_argument("--drivers", default=None,
                   help=f"Driver TF panel (comma-separated; default: "
                        f"{','.join(DEFAULT_DRIVERS)}).")
    p.add_argument("--stress", default=None,
                   help=f"Stress responder panel (default: "
                        f"{','.join(DEFAULT_STRESS)}).")

    # Optimisation
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--lr", "--learning-rate", type=float, default=1e-3,
                   dest="learning_rate")
    p.add_argument("--grad-clip-norm", type=float, default=1.0)

    # Loss weights (mirrors anatomical_compiler.training.TrainingConfig)
    p.add_argument("--w-rollout", type=float, default=1.0)
    p.add_argument("--w-driver", type=float, default=1.0)
    p.add_argument("--w-stress", type=float, default=0.5)
    p.add_argument("--w-smooth", type=float, default=0.05)
    p.add_argument("--w-uniformity", type=float, default=0.1)
    p.add_argument("--w-reg", type=float, default=1e-4)
    p.add_argument("--stress-margin", type=float, default=0.05)
    p.add_argument("--max-param-value", type=float, default=5.0)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)

    # Output
    p.add_argument("--out-json", default="figures/train_hg_neural_ode_results.json")
    p.add_argument("--out-fig", default="figures/train_hg_neural_ode.png")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import jax
        import jax.numpy as jnp
        import diffrax
        import hgx
    except ImportError as e:
        print(f"  Error: required package missing -- {e}")
        return 2

    # Local import so the module is loadable even without the runtime deps.
    from anatomical_compiler.training import (
        TrainingConfig, train_hg_neural_ode,
    )

    adata_path = Path(args.input)
    if not adata_path.exists():
        print(f"  Error: {adata_path} not found.")
        return 2

    print(f"Loading {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    time_col = args.time_col or _RF._resolve_time_col(adata)
    if time_col is None or adata.obs[time_col].nunique() < 3:
        print(f"  Error: need >=3 distinct timepoints in column '{time_col}'.")
        return 2

    tf_panel_request = _parse_panel(args.tf_panel, _RF.KIDNEY_INJURY_TFS)
    tf_panel = _RF._resolve_symbols(adata, tf_panel_request)
    if len(tf_panel) < 3:
        print(f"  Error: only {len(tf_panel)} TFs resolved from panel.")
        return 2

    drivers_req = _parse_panel(args.drivers, DEFAULT_DRIVERS)
    stress_req = _parse_panel(args.stress, DEFAULT_STRESS)
    driver_idx = _panel_idx_in_tf_panel(tf_panel, drivers_req)
    stress_idx = _panel_idx_in_tf_panel(tf_panel, stress_req)

    print(f"  Time column: {time_col}")
    print(f"  TFs in panel: {len(tf_panel)} -> {tf_panel}")
    print(f"  Driver panel: {len(driver_idx)} -> "
          f"{[tf_panel[i] for i in driver_idx]}")
    print(f"  Stress panel: {len(stress_idx)} -> "
          f"{[tf_panel[i] for i in stress_idx]}")

    times, traj = _RF.build_tf_trajectory(adata, tf_panel, time_col, args.n_bins)
    T, N, D = traj.shape
    print(f"  Trajectory shape: T={T} timepoints, N={N} TFs, D={D}")

    # All-pairs single-hyperedge incidence (same as benchmark_regenerative_flow).
    incidence = jnp.ones((N, 1), dtype=jnp.float32)
    snaps = [
        hgx.from_incidence(incidence, node_features=jnp.array(traj[t]))
        for t in range(T)
    ]
    t_arr = jnp.linspace(0.0, 1.0, T)
    temp_hg = hgx.from_snapshots(snaps, times=t_arr)

    key = jax.random.PRNGKey(args.seed)
    k_conv, _k_unused = jax.random.split(key)
    conv = hgx.UniGCNConv(D, D, key=k_conv)

    # HypergraphNeuralODE built explicitly so the trainer's `forward` can
    # call it with a SaveAt over the held-out timepoints.
    from hgx._dynamics import HypergraphNeuralODE
    model = HypergraphNeuralODE(conv)

    hg0 = snaps[0]
    target = jnp.asarray(traj[1:])               # (T-1, N, D)
    target_times = t_arr[1:]
    t0_val = float(t_arr[0])
    t1_val = float(t_arr[-1])

    def forward(m):
        sol = m(
            hg0,
            t0=t0_val,
            t1=t1_val,
            saveat=diffrax.SaveAt(ts=target_times),
        )
        return sol.ys  # (T-1, N, D)

    cfg = TrainingConfig(
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
        w_rollout=args.w_rollout,
        w_driver=args.w_driver,
        w_stress=args.w_stress,
        w_smooth=args.w_smooth,
        w_uniformity=args.w_uniformity,
        w_reg=args.w_reg,
        stress_margin=args.stress_margin,
        max_param_value=args.max_param_value,
        log_every=args.log_every,
        seed=args.seed,
    )

    result = train_hg_neural_ode(
        model,
        forward=forward,
        target=target,
        driver_idx=jnp.asarray(driver_idx, dtype=jnp.int32),
        stress_idx=jnp.asarray(stress_idx, dtype=jnp.int32),
        config=cfg,
    )

    # Final per-TF MSE -> ranking (lowest = best-captured = drivers)
    per_tf = (
        np.asarray(result.final_per_tf_mse)
        if result.final_per_tf_mse is not None
        else np.full(N, np.nan)
    )
    ranked = sorted(zip(tf_panel, per_tf.tolist()), key=lambda x: x[1])

    # Final rollout for the figure
    pred_final = np.asarray(forward(result.model))  # (T-1, N, D)
    obs_final = np.asarray(target)

    out = {
        "input": str(adata_path),
        "time_col": time_col,
        "timepoints": [str(t) for t in times],
        "tf_panel": tf_panel,
        "driver_panel": [tf_panel[i] for i in driver_idx],
        "stress_panel": [tf_panel[i] for i in stress_idx],
        "config": {
            k: getattr(cfg, k)
            for k in (
                "n_epochs", "learning_rate", "grad_clip_norm",
                "w_rollout", "w_driver", "w_stress", "w_smooth",
                "w_uniformity", "w_reg", "stress_margin",
                "max_param_value", "log_every", "seed",
            )
        },
        "loss_history": result.loss_history,
        "wall_time_s": result.wall_time_s,
        "nan_epochs": result.nan_epochs,
        "per_tf_mse": dict(zip(tf_panel, per_tf.tolist())),
        "regenerative_drivers": [
            {"tf": tf, "rollout_mse": float(mse)} for tf, mse in ranked[:8]
        ],
        "transient_responders": [
            {"tf": tf, "rollout_mse": float(mse)} for tf, mse in ranked[-4:]
        ],
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=str))
    print(f"  Results saved to {args.out_json}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: loss curves
    epochs = [r["epoch"] for r in result.loss_history]
    axes[0].plot(epochs, [r["total"] for r in result.loss_history],
                 "k-", lw=2, label="total")
    axes[0].plot(epochs, [r["rollout_mse"] for r in result.loss_history],
                 "C0--", label="rollout")
    axes[0].plot(epochs, [r["driver_mse"] for r in result.loss_history],
                 "C2--", label="driver")
    axes[0].plot(epochs, [r["stress_hinge"] for r in result.loss_history],
                 "C3--", label="stress")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Loss components")
    axes[0].legend()

    # Panel 2: rollout vs observed for top-6 TFs by name
    for i, tf in enumerate(tf_panel[:6]):
        axes[1].plot(obs_final[:, i, 0], "o-",
                     color=f"C{i}", label=f"{tf} obs", alpha=0.7)
        axes[1].plot(pred_final[:, i, 0], "x--",
                     color=f"C{i}", label=f"{tf} pred", alpha=0.7)
    axes[1].set_xlabel("Timepoint index")
    axes[1].set_ylabel("Log expression")
    axes[1].set_title("Rollout vs observed")
    axes[1].legend(fontsize=7, ncol=2)

    # Panel 3: regenerative drivers ranking
    top = ranked[:10]
    axes[2].barh([t for t, _ in top][::-1],
                 [m for _, m in top][::-1], color="forestgreen")
    axes[2].set_xlabel("Rollout MSE (lower -> better captured)")
    axes[2].set_title("Regenerative drivers")

    plt.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_fig, dpi=200, bbox_inches="tight")
    print(f"  Figure saved to {args.out_fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
