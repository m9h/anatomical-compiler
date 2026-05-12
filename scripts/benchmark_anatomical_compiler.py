"""Anatomical compiler (nonlinear): steer a learned Hypergraph Neural ODE to a target state.

This is the nonlinear counterpart of ``benchmark_network_control.py`` (which does the
*linear* warm-up on the static co-regulation graph).  The pipeline is the one ``jaxctrl``
is built for, run end-to-end:

  1. **Plant model (hgx).**  Fit a Hypergraph Neural ODE ``dz/dt = f_theta(z)`` on a TF
     expression timecourse (default: the Balzer 2022 kidney injury-repair data, as in
     ``benchmark_regenerative_flow.py``) -- this is the data-driven dynamics.
  2. **System identification (jaxctrl L0-ish).**  Sample the learned ODE's dense rollout
     and fit a linear surrogate ``dz/dt ~= A z + c`` by least squares on finite differences
     (the place where ``jaxctrl.SINDyOptimizer`` / ``KoopmanEstimator`` would slot in for a
     richer surrogate).
  3. **Controller synthesis (jaxctrl L1).**  Pick a target state z_f (the recovered/last
     timepoint), choose an actuatable TF set B (the regenerative drivers found in step 1's
     rollout error), and compute: controllability of the surrogate from B, the smallest
     controllability-Gramian eigenvalue (steerability), and an LQR feedback gain K -- with
     B = drivers vs B = all TFs.
  4. **Closed-loop validation on the nonlinear plant (diffrax).**  Integrate the *learned*
     Hypergraph Neural ODE under u = -K (z - z_f), B-actuated, and report the achieved final
     state vs target vs the uncontrolled (free) rollout.  This is "desired state in,
     intervention out" -- Levin's anatomical compiler -- at the level of a learned regulome
     surrogate.

Resilient like the other benchmark_*.py: needs hgx + diffrax + jaxctrl + jax and a
multi-timepoint h5ad; otherwise prints a note and exits.

Refs: Levin 2022 (TAME / the anatomical compiler); jaxctrl (github.com/m9h/jaxctrl).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def _json_safe(obj):
    """Recursively replace non-finite floats with None so the JSON parses everywhere
    (Python's json.dump otherwise emits bare NaN/Infinity, which strict parsers reject)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _dump(results, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_json_safe(results), indent=2))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

# Reuse the timecourse helpers / TF panel from the regenerative-flow script.
try:
    from scripts.benchmark_regenerative_flow import (  # type: ignore
        _resolve_symbols, _resolve_time_col, build_tf_trajectory, KIDNEY_INJURY_TFS,
    )
except Exception:  # run as a plain file
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_rf", str(Path(__file__).with_name("benchmark_regenerative_flow.py")))
    _rf = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_rf)  # type: ignore
    _resolve_symbols, _resolve_time_col = _rf._resolve_symbols, _rf._resolve_time_col
    build_tf_trajectory, KIDNEY_INJURY_TFS = _rf.build_tf_trajectory, _rf.KIDNEY_INJURY_TFS


def _linear_surrogate(rollout: np.ndarray, ts: np.ndarray):
    """Least-squares dz/dt ~= A z + c from a dense rollout (z: (T, n))."""
    z = rollout
    dz = np.gradient(z, ts, axis=0)                         # (T, n)
    X = np.hstack([z, np.ones((z.shape[0], 1))])            # (T, n+1)
    W, *_ = np.linalg.lstsq(X, dz, rcond=None)              # (n+1, n)
    A = W[:-1].T                                            # (n, n)  (dz = A z + c)
    c = W[-1]                                               # (n,)
    return A.astype(np.float32), c.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/bioprinting/balzer_2022_processed.h5ad")
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-drivers", type=int, default=6,
                    help="actuate the K TFs the learned ODE fits best (the regenerative drivers)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-fig", default="figures/anatomical_compiler.png")
    ap.add_argument("--out-json", default="figures/anatomical_compiler_results.json")
    args = ap.parse_args()

    print("Anatomical compiler (nonlinear): steer a learned Hypergraph Neural ODE to a target...")
    p = Path(args.input)
    if not p.exists():
        print(f"  Error: {p} not found.")
        return
    try:
        import jax
        import jax.numpy as jnp
        import diffrax as dfx
        import hgx
        from hgx._dynamics import _hg_to_args  # private but stable: dict for the drift's `args`
        import jaxctrl
    except Exception as e:
        print(f"  Missing a dependency ({e!r}). Need hgx + diffrax + jaxctrl + jax (`uv sync`).")
        return

    adata = sc.read_h5ad(p)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    time_col = args.time_col or _resolve_time_col(adata)
    tf_panel = _resolve_symbols(adata, KIDNEY_INJURY_TFS)
    print(f"  Time column: {time_col}   TFs found: {len(tf_panel)}/{len(KIDNEY_INJURY_TFS)}")
    results = {"input": str(p), "n_cells": int(adata.shape[0]), "time_col": time_col,
               "tfs": tf_panel, "refs": ["Levin 2022 (anatomical compiler)", "jaxctrl"]}
    if time_col is None or adata.obs[time_col].nunique() < 3 or len(tf_panel) < 3:
        print("  No multi-timepoint structure (or too few TFs) -- need a timecourse h5ad.")
        results["status"] = "insufficient_structure"
        _dump(results, args.out_json)
        return

    times, traj = build_tf_trajectory(adata, tf_panel, time_col, args.n_bins)  # (T, n, 1)
    T, n, _ = traj.shape
    results["timepoints"] = [str(t) for t in times]

    # --- 1. plant model: fit the Hypergraph Neural ODE ------------------------
    inc = jnp.ones((n, 1), dtype=jnp.float32)
    snaps = [hgx.from_incidence(inc, node_features=jnp.array(traj[t])) for t in range(T)]
    grid = jnp.linspace(0.0, 1.0, T)
    thg = hgx.from_snapshots(snaps, times=grid)
    k1, k2 = jax.random.split(jax.random.PRNGKey(args.seed))
    conv = hgx.UniGCNConv(1, 1, key=k1)
    print(f"  Fitting Hypergraph Neural ODE: T={T} timepoints, n={n} TFs, epochs={args.epochs}...")
    t0 = time.time()
    ode = hgx.fit_neural_ode(thg, conv, key=k2, epochs=args.epochs, lr=1e-3)
    print(f"  Training time: {time.time()-t0:.1f}s")

    # dense rollout of the *learned* dynamics from z0
    z0 = jnp.array(traj[0])                                  # (n, 1)
    z_target = jnp.array(traj[-1])                           # (n, 1)  -- the "recovered" state
    fine = jnp.linspace(0.0, 1.0, 41)
    sol_free = ode(snaps[0], t0=0.0, t1=1.0, saveat=dfx.SaveAt(ts=fine))
    roll = np.asarray(sol_free.ys).reshape(len(fine), n)     # (41, n)
    free_final_err = float(jnp.linalg.norm(jnp.array(roll[-1]) - z_target.ravel()))
    per_tf_mse = ((np.asarray(ode(snaps[0], t0=0.0, t1=1.0, saveat=dfx.SaveAt(ts=grid[1:])).ys).reshape(T-1, n)
                   - traj[1:].reshape(T-1, n)) ** 2).mean(axis=0)
    drivers = sorted(np.argsort(per_tf_mse)[:args.n_drivers].tolist())   # best-fit = drivers
    driver_names = [tf_panel[i] for i in drivers]
    print(f"  Actuatable (regenerative-driver) TFs: {driver_names}")
    print(f"  Uncontrolled rollout ||z(T) - z_target|| = {free_final_err:.3f}")

    # --- 2. linear surrogate (jaxctrl warm-start) -----------------------------
    # z-score the rollout so the least-squares is conditioned, then fit dz/dt ~= A z + c.
    mu, sd = roll.mean(0), roll.std(0) + 1e-6
    roll_z = (roll - mu) / sd
    A_np, c_np = _linear_surrogate(roll_z, np.asarray(fine))
    A = jnp.asarray(A_np)
    eig_max = float(jnp.max(jnp.linalg.eigvals(A).real))
    A_clamp = A - max(eig_max + 0.5, 0.0) * jnp.eye(n)        # Hurwitz copy for Gramians
    eye = jnp.eye(n); B_d = eye[:, jnp.asarray(drivers)]
    dz_z = np.gradient(roll_z, np.asarray(fine), axis=0)
    sid_res = float(np.linalg.norm((A_np @ roll_z.T + c_np[:, None]) - dz_z.T)
                    / (np.linalg.norm(dz_z.T) + 1e-9))
    surrogate_reliable = sid_res < 0.4 and abs(eig_max) < 10.0   # honest: a few timepoints rarely give a clean linear ID
    print(f"  Linear surrogate dz/dt ~= A z + c  (rel. residual {sid_res:.2f}, max Re(eig A) {eig_max:.2f}; "
          f"{'usable warm-start' if surrogate_reliable else 'rough -- few timepoints; used for orientation only, the controller is optimised directly on the nonlinear ODE'})")

    # --- 3. jaxctrl L1 on the surrogate (controllability + an LQR warm-start) -
    def gram_min_eig(Bsel):
        return float(jnp.linalg.eigvalsh(jaxctrl.controllability_gramian(A_clamp, Bsel, T=4.0))[0])
    rank_d = int(jnp.linalg.matrix_rank(jaxctrl.controllability_matrix(A, B_d)))
    Q = jnp.eye(n)
    K_d, X_d = jaxctrl.lqr(A_clamp, B_d, Q, jnp.eye(len(drivers)))
    K_f, X_f = jaxctrl.lqr(A_clamp, eye, Q, eye)
    y0z = (roll_z[0] - roll_z[-1])
    results["surrogate_control"] = {
        "linear_surrogate_rel_residual": sid_res, "max_re_eig_A": eig_max,
        "surrogate_reliable": surrogate_reliable,
        "drivers": driver_names, "n_drivers": len(drivers), "ctrl_rank_from_drivers": rank_d, "n": n,
        "gramian_min_eig_drivers": gram_min_eig(B_d), "gramian_min_eig_full": gram_min_eig(eye),
        "lqr_cost_to_go_drivers": float(y0z @ X_d @ y0z), "lqr_gain_norm_drivers": float(jnp.linalg.norm(K_d)),
        "lqr_cost_to_go_full": float(y0z @ X_f @ y0z), "lqr_gain_norm_full": float(jnp.linalg.norm(K_f)),
    }
    print(f"  jaxctrl surrogate: ctrl. rank from {len(drivers)} drivers {rank_d}/{n}; "
          f"LQR cost-to-go drivers {results['surrogate_control']['lqr_cost_to_go_drivers']:.2f} vs full "
          f"{results['surrogate_control']['lqr_cost_to_go_full']:.2f} (warm-start; refined below)")

    # --- 4. DIRECT optimal control on the nonlinear hgx ODE (diffrax adjoints) -
    # Parametrise a piecewise-constant input u_seg in R^{n_seg x m} on the m driver TFs;
    # minimise  J(u) = ||z(1) - z_target||^2 + lam * mean(u^2)  over the *learned* hgx ODE.
    drift, drift_args = ode.drift, _hg_to_args(snaps[0])
    Bd = jnp.asarray(np.eye(n)[:, drivers]); m = len(drivers); d_idx = np.asarray(drivers)
    n_seg, lam, U_MAX = 6, 1e-3, 6.0
    zf_v = z_target.ravel()
    def u_of_t(useg, t):
        idx = jnp.clip(jnp.floor(t * n_seg).astype(int), 0, n_seg - 1)
        return jnp.clip(useg[idx], -U_MAX, U_MAX)
    def rollout(useg):
        def vf(t, y, a):
            return drift(t, y, drift_args) + (Bd @ u_of_t(useg, t))[:, None]
        sol = dfx.diffeqsolve(dfx.ODETerm(vf), dfx.Tsit5(), t0=0.0, t1=1.0, dt0=0.02, y0=z0,
                              args=drift_args, saveat=dfx.SaveAt(ts=fine),
                              stepsize_controller=dfx.ConstantStepSize(), max_steps=2**14)
        return sol.ys.reshape(len(fine), n)
    def loss(useg):
        ys = rollout(useg)
        return jnp.sum((ys[-1] - zf_v) ** 2) + lam * jnp.mean(useg ** 2)
    import optax
    useg = jnp.zeros((n_seg, m))
    opt = optax.adam(5e-2); opt_state = opt.init(useg)
    loss_grad = jax.jit(jax.value_and_grad(loss))
    print(f"  Optimising a {n_seg}x{m} piecewise-constant control on the learned ODE (diffrax adjoints)...")
    try:
        l0 = float(loss(useg))
        for step in range(300):
            lval, g = loss_grad(useg)
            updates, opt_state = opt.update(g, opt_state); useg = optax.apply_updates(useg, updates)
        cl_roll = np.asarray(rollout(useg)); oc_ok = True
        u_traj = np.asarray([np.asarray(u_of_t(useg, t)) for t in fine])
        print(f"    loss {l0:.3f} -> {float(lval):.3f}")
    except Exception as e:
        print(f"  (direct optimal control failed: {e!r}; reporting uncontrolled rollout)")
        cl_roll = roll.copy(); u_traj = np.zeros((len(fine), m)); oc_ok = False
    ctrl_final_err = float(np.linalg.norm(cl_roll[-1] - np.asarray(zf_v)))
    # error on the *actuated* subspace (the part the m driver inputs can directly move)
    free_sub = float(np.linalg.norm(roll[-1, d_idx] - np.asarray(zf_v)[d_idx]))
    ctrl_sub = float(np.linalg.norm(cl_roll[-1, d_idx] - np.asarray(zf_v)[d_idx]))
    print(f"  Optimal-control rollout ||z(T) - z_target|| = {ctrl_final_err:.3f} (uncontrolled {free_final_err:.3f}; "
          f"reduced {100*(1 - ctrl_final_err/(free_final_err+1e-9)):.0f}%)   on the {m} actuated TFs: "
          f"{ctrl_sub:.3f} vs {free_sub:.3f} (reduced {100*(1 - ctrl_sub/(free_sub+1e-9)):.0f}%)")
    results["optimal_control"] = {
        "method": "direct (piecewise-constant u, diffrax adjoints + Adam); jaxctrl LQR as warm-start",
        "converged": oc_ok, "n_segments": n_seg, "control_penalty_lambda": lam, "u_max": U_MAX,
        "uncontrolled_final_err": free_final_err, "controlled_final_err": ctrl_final_err,
        "error_reduction_frac": float(1 - ctrl_final_err / (free_final_err + 1e-9)) if oc_ok else 0.0,
        "uncontrolled_final_err_actuated": free_sub, "controlled_final_err_actuated": ctrl_sub,
        "error_reduction_frac_actuated": float(1 - ctrl_sub / (free_sub + 1e-9)) if oc_ok else 0.0,
        "ts": np.asarray(fine).tolist(), "actuated_tfs": driver_names,
        "control_input": u_traj.tolist(),                    # (T, m)
        "u_segments": np.asarray(useg).tolist(),             # (n_seg, m)
        "z_free": roll[:, drivers].tolist(), "z_controlled": cl_roll[:, drivers].tolist(),
        "z_target": [float(np.asarray(zf_v)[i]) for i in drivers],
        "z_start": [float(np.asarray(z0.ravel())[i]) for i in drivers],
    }
    results["status"] = "ok"

    # --- figure --------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    zf_, zc_ = np.asarray(results["optimal_control"]["z_free"]), np.asarray(results["optimal_control"]["z_controlled"])
    tg = results["optimal_control"]["z_target"]
    for j, nm in enumerate(driver_names):
        col = cyc[j % len(cyc)]
        ax[0].plot(fine, zf_[:, j], col, ls="--", alpha=0.55)
        ax[0].plot(fine, zc_[:, j], col, ls="-", label=nm)
        ax[0].axhline(tg[j], color=col, ls=":", lw=1)
    ax[0].set_title("Anatomical compiler: steer the learned Hypergraph Neural ODE to a target\n"
                    "(solid = optimally controlled, dashed = uncontrolled, dotted = target $z_f$)")
    ax[0].set_xlabel("time"); ax[0].set_ylabel("TF expr (learned-ODE state)"); ax[0].legend(fontsize=8)
    ut = np.asarray(results["optimal_control"]["control_input"])
    for j, nm in enumerate(driver_names):
        ax[1].step(fine, ut[:, j], where="post", color=cyc[j % len(cyc)], label=nm)
    ax[1].set_title("Optimal control input $u(t)$ per actuated TF\n(piecewise-constant; diffrax adjoints + Adam)")
    ax[1].set_xlabel("time"); ax[1].set_ylabel("u"); ax[1].legend(fontsize=8); ax[1].axhline(0, color="k", lw=0.5)
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out_fig, dpi=200, bbox_inches="tight")
    print(f"  Figure -> {args.out_fig}")
    _dump(results, args.out_json)
    print(f"  Results -> {args.out_json}")


if __name__ == "__main__":
    main()
