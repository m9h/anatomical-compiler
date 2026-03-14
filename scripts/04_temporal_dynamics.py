#!/usr/bin/env python3
"""Temporal Dynamics, Poincare Embeddings, and Stochastic Fate.

Produces Figure 3 (Continuous Trajectory) and Figure 4 (Stochastic Fate).
Covers Analyses 2.2 (Neural ODE trajectory), 2.3 (Poincare latent space),
and 2.4 (SDE-based stochastic fate decisions).

Figure 3 (2x2):
    A: PCA of expression colored by pseudotime + fate
    B: Neural ODE predicted vs observed expression for key TFs
    C: Poincare disk embedding colored by fate
    D: Gromov delta bar chart -- Euclidean vs Poincare

Figure 4 (2x2):
    A: Multiple SDE trajectories diverging at branch point (PCA)
    B: Learned diffusion sigma per gene vs expression variance
    C: Phase portrait of ODE trajectories
    D: SDE rollout variance vs pseudotime

Usage:
    uv run python scripts/04_temporal_dynamics.py
    uv run python scripts/04_temporal_dynamics.py --epochs 300 --seed 7
    uv run python scripts/04_temporal_dynamics.py --data-dir data/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import hgx  # noqa: E402

# Dynamics imports (require diffrax)
try:
    import diffrax

    HAS_DIFFRAX = True
except ImportError:
    HAS_DIFFRAX = False
    print(
        "WARNING: diffrax not available. "
        "Neural ODE/SDE analyses will be skipped."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_GENES = 200
NUM_TFS = 20
NUM_MODULES = 15
NUM_FATES = 3
NUM_PSEUDOTIME = 10
FEATURE_DIM = 8
LATENT_DIM = 16

FATE_NAMES = ["Cortical (ctx)", "Ganglionic (GE)", "Neural Tube (NT)"]
FATE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]
TF_NAMES = ["GLI3", "FOXG1", "TBR1"]
TF_INDICES = [0, 1, 2]  # First three TFs used as exemplars


# ---------------------------------------------------------------------------
# Data loading / generation
# ---------------------------------------------------------------------------


def try_load_expression(data_dir: Path):
    """Try loading real expression and pseudotime from AnnData."""
    h5ad_path = data_dir / "expression" / "expression.h5ad"
    if not h5ad_path.exists():
        return None
    try:
        import anndata

        adata = anndata.read_h5ad(h5ad_path)
        X = np.asarray(adata.X)

        # Handle sparse matrices
        try:
            from scipy.sparse import issparse

            if issparse(X):
                X = np.asarray(X.toarray())
        except ImportError:
            pass

        X = X.astype(np.float32)
        print(f"  Loaded expression from {h5ad_path}: {X.shape}")

        # Extract pseudotime if available
        pseudotime = None
        for col in ["pseudotime", "dpt_pseudotime", "monocle3_pseudotime"]:
            if col in adata.obs.columns:
                pseudotime = np.asarray(adata.obs[col].values, dtype=np.float32)
                break

        # Extract fate labels if available
        fates = None
        for col in ["fate", "cell_type", "celltype", "leiden"]:
            if col in adata.obs.columns:
                cats = adata.obs[col].astype("category")
                fates = np.asarray(cats.cat.codes, dtype=np.int32)
                break

        return {"expression": X, "pseudotime": pseudotime, "fates": fates}
    except Exception as e:
        print(f"  Could not load expression data: {e}")
        return None


def generate_synthetic_temporal(*, key):
    """Generate synthetic temporal regulome data with 3 fates.

    Creates trajectories that diverge over pseudotime into three
    cell fates: Cortical, Ganglionic Eminence, and Neural Tube.
    """
    rng_key = key
    seed_val = int(jax.random.randint(rng_key, (), 0, 2**30))
    rng = np.random.RandomState(seed_val)

    # --- Build incidence matrix ---
    incidence = np.zeros((NUM_GENES, NUM_MODULES), dtype=np.float32)
    module_labels = np.full(NUM_GENES, -1, dtype=np.int32)

    target_pool = list(range(NUM_TFS, NUM_GENES))
    rng.shuffle(target_pool)

    idx = 0
    sizes = [20] * 3 + [12] * 7 + [8] * 5
    for m in range(NUM_MODULES):
        tf = min(m, NUM_TFS - 1)
        incidence[tf, m] = 1.0
        module_labels[tf] = m

        n_targets = sizes[m]
        end = min(idx + n_targets, len(target_pool))
        for t_idx in range(idx, end):
            g = target_pool[t_idx]
            incidence[g, m] = 1.0
            if module_labels[g] == -1:
                module_labels[g] = m
        idx = end

    for g in range(NUM_GENES):
        if module_labels[g] == -1:
            m = rng.randint(NUM_MODULES)
            incidence[g, m] = 1.0
            module_labels[g] = m

    # Assign genes to fates (3 fates)
    module_to_fate = np.array(
        [min(m * NUM_FATES // NUM_MODULES, NUM_FATES - 1) for m in range(NUM_MODULES)]
    )
    gene_fates = module_to_fate[module_labels]

    # --- Build expression trajectories over pseudotime ---
    k1, k2, k3 = jax.random.split(key, 3)
    base_expr = jax.random.normal(k1, (NUM_GENES, FEATURE_DIM)) * 0.3

    fate_sigs = jnp.stack(
        [
            jax.random.normal(jax.random.fold_in(k2, f), (FEATURE_DIM,))
            for f in range(NUM_FATES)
        ]
    )

    # Create smoothly diverging trajectories
    trajectories = []
    for t in range(NUM_PSEUDOTIME):
        frac = t / max(NUM_PSEUDOTIME - 1, 1)
        fate_bias = fate_sigs[jnp.array(gene_fates)] * frac
        # Add temporal trend (gradual activation)
        temporal_trend = 0.5 * jnp.sin(
            jnp.pi * frac
            * jnp.arange(FEATURE_DIM)[None, :]
            / FEATURE_DIM
        ) * jnp.ones((NUM_GENES, 1))
        noise = 0.08 * jax.random.normal(
            jax.random.fold_in(k3, t), (NUM_GENES, FEATURE_DIM)
        )
        trajectories.append(base_expr + fate_bias + temporal_trend + noise)

    trajectories = jnp.stack(trajectories)  # (T, N, D)

    # Build cells-by-genes expression matrix for temporal hypergraph API
    # Simulate 50 cells per timepoint
    n_cells_per_t = 50
    cells_expr = []
    cell_times = []
    cell_fates_list = []
    for t in range(NUM_PSEUDOTIME):
        for _ in range(n_cells_per_t):
            # Each cell gets the mean expression + cell-level noise
            k_cell = jax.random.fold_in(k3, t * n_cells_per_t + _)
            cell_noise = 0.1 * jax.random.normal(k_cell, (NUM_GENES,))
            # Take mean over feature dims for a scalar per gene
            gene_means = trajectories[t].mean(axis=-1)
            cells_expr.append(gene_means + cell_noise)
            cell_times.append(t)
            # Assign cell fate based on module fates
            cell_fates_list.append(rng.choice(NUM_FATES, p=[0.4, 0.35, 0.25]))

    expression_matrix = jnp.stack(cells_expr)  # (n_cells, n_genes)
    time_labels = jnp.array(cell_times)
    cell_fates_arr = jnp.array(cell_fates_list)

    return {
        "incidence": jnp.array(incidence),
        "trajectories": trajectories,
        "module_labels": jnp.array(module_labels),
        "gene_fates": jnp.array(gene_fates),
        "expression_matrix": expression_matrix,
        "time_labels": time_labels,
        "cell_fates": cell_fates_arr,
    }


# ---------------------------------------------------------------------------
# Poincare distance and Gromov delta
# ---------------------------------------------------------------------------


def poincare_dist(x, y, c=1.0):
    """Poincare ball distance between two points."""
    diff_sq = jnp.sum((x - y) ** 2)
    x_sq = jnp.sum(x**2)
    y_sq = jnp.sum(y**2)
    denom = jnp.maximum((1 - c * x_sq) * (1 - c * y_sq), 1e-8)
    return jnp.arccosh(jnp.maximum(1 + 2 * c * diff_sq / denom, 1.0 + 1e-7))


def compute_gromov_delta(embeddings, c=1.0, n_samples=25):
    """Estimate Gromov delta-hyperbolicity via the four-point condition.

    Lower delta indicates more tree-like (hyperbolic) structure.
    """
    n = embeddings.shape[0]
    idx = np.random.choice(n, min(n_samples, n), replace=False)
    pts = embeddings[idx]
    k = len(idx)

    dists = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d = float(poincare_dist(pts[i], pts[j], c))
            dists[i, j] = dists[j, i] = d

    delta = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            for ci in range(b + 1, k):
                for di in range(ci + 1, k):
                    s1 = dists[a, b] + dists[ci, di]
                    s2 = dists[a, ci] + dists[b, di]
                    s3 = dists[a, di] + dists[b, ci]
                    sums = sorted([s1, s2, s3], reverse=True)
                    delta = max(delta, (sums[0] - sums[1]) / 2)

    return delta


def euclidean_gromov_delta(embeddings, n_samples=25):
    """Gromov delta using Euclidean distances."""
    n = embeddings.shape[0]
    idx = np.random.choice(n, min(n_samples, n), replace=False)
    pts = np.array(embeddings[idx])
    k = len(idx)

    dists = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            dists[i, j] = dists[j, i] = d

    delta = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            for ci in range(b + 1, k):
                for di in range(ci + 1, k):
                    s1 = dists[a, b] + dists[ci, di]
                    s2 = dists[a, ci] + dists[b, di]
                    s3 = dists[a, di] + dists[b, ci]
                    sums = sorted([s1, s2, s3], reverse=True)
                    delta = max(delta, (sums[0] - sums[1]) / 2)

    return delta


# ---------------------------------------------------------------------------
# Part A: Neural ODE Trajectory
# ---------------------------------------------------------------------------


def run_neural_ode(data, epochs, *, key):
    """Fit Neural ODE to temporal trajectory and evaluate."""
    if not HAS_DIFFRAX:
        print("  Skipping Neural ODE (diffrax not available).")
        return None

    print("\n" + "=" * 60)
    print("  Part A: Neural ODE Trajectory (Figure 3A-B)")
    print("=" * 60)

    k1, k2 = jax.random.split(key)
    incidence = data["incidence"]
    trajs = data["trajectories"]  # (T, N, D)

    # Build temporal hypergraphs
    hgs = []
    for t in range(NUM_PSEUDOTIME):
        hg_t = hgx.from_incidence(incidence, node_features=trajs[t])
        hgs.append(hg_t)
    times = jnp.linspace(0.0, 1.0, NUM_PSEUDOTIME)
    temp_hg = hgx.from_snapshots(hgs, times=times)

    # Build convolution for ODE
    conv = hgx.UniGCNConv(FEATURE_DIM, FEATURE_DIM, key=k1)

    # Fit Neural ODE
    print("  Fitting Neural ODE...")
    t_start = time.time()
    neural_ode = hgx.fit_neural_ode(
        temp_hg, conv, key=k2, epochs=epochs, lr=1e-3
    )
    elapsed = time.time() - t_start
    print(f"  Training time: {elapsed:.1f}s")

    # Evaluate: 1-step MSE
    hg0 = temp_hg[0]
    sol = neural_ode(
        hg0,
        t0=0.0,
        t1=1.0,
        saveat=diffrax.SaveAt(ts=times[1:]),
    )
    pred_trajs = sol.ys  # (T-1, N, D)

    step_mses = []
    for t in range(NUM_PSEUDOTIME - 1):
        mse = float(jnp.mean((pred_trajs[t] - trajs[t + 1]) ** 2))
        step_mses.append(mse)
    avg_mse = float(np.mean(step_mses))

    # Rollout MSE: predict from last training point
    num_train = NUM_PSEUDOTIME - 3
    rollout_pred = trajs[num_train - 1]
    rollout_mses = []
    for t in range(num_train - 1, NUM_PSEUDOTIME - 1):
        hg_t = hgx.from_incidence(incidence, node_features=rollout_pred)
        sol_step = neural_ode(hg_t, t0=0.0, t1=1.0 / (NUM_PSEUDOTIME - 1))
        rollout_pred = sol_step.ys[-1]
        mse = float(jnp.mean((rollout_pred - trajs[t + 1]) ** 2))
        rollout_mses.append(mse)

    avg_rollout_mse = float(np.mean(rollout_mses))
    print(f"  1-step MSE: {avg_mse:.6f}")
    print(f"  Rollout MSE: {avg_rollout_mse:.6f}")

    # Get full trajectory for visualization
    ts_traj, ode_trajectory = hgx.trajectory(
        neural_ode, hg0, t0=0.0, t1=1.0, num_steps=50
    )

    return {
        "neural_ode": neural_ode,
        "pred_trajs": pred_trajs,
        "ode_trajectory": np.array(ode_trajectory),
        "ts_traj": np.array(ts_traj),
        "step_mses": step_mses,
        "rollout_mses": rollout_mses,
        "avg_mse": avg_mse,
        "avg_rollout_mse": avg_rollout_mse,
    }


# ---------------------------------------------------------------------------
# Part B: Poincare Latent Space
# ---------------------------------------------------------------------------


def run_poincare_embedding(data, epochs, *, key):
    """Train LatentHypergraphODE with Poincare conv, measure hyperbolicity."""
    if not HAS_DIFFRAX:
        print("  Skipping Poincare embedding (diffrax not available).")
        return None

    print("\n" + "=" * 60)
    print("  Part B: Poincare Latent Space (Figure 3C-D)")
    print("=" * 60)

    k1, k2, k3 = jax.random.split(key, 3)
    trajs = data["trajectories"]
    incidence = data["incidence"]
    num_train = NUM_PSEUDOTIME - 3
    num_pairs = num_train - 1

    # --- Poincare model ---
    print("  Training Poincare LatentODE...")
    poincare_model = hgx.LatentHypergraphODE(
        obs_dim=FEATURE_DIM,
        latent_dim=LATENT_DIM,
        conv_cls=hgx.PoincareHypergraphConv,
        key=k1,
    )

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(poincare_model, eqx.is_array))

    @eqx.filter_jit
    def step_poincare(model, opt_state, hg, target):
        @eqx.filter_value_and_grad
        def loss_fn(m):
            pred = m(hg, t0=0.0, t1=1.0)
            return jnp.mean((pred - target) ** 2)

        loss, grads = loss_fn(model)
        updates, new_opt = optimizer.update(grads, opt_state)
        return eqx.apply_updates(model, updates), new_opt, loss

    losses_p = []
    t_start = time.time()
    for i in range(epochs):
        t = i % num_pairs
        hg_t = hgx.from_incidence(incidence, node_features=trajs[t])
        poincare_model, opt_state, loss = step_poincare(
            poincare_model, opt_state, hg_t, trajs[t + 1]
        )
        losses_p.append(float(loss))
        if (i + 1) % max(1, epochs // 5) == 0:
            print(f"    Step {i+1:4d}  loss={loss:.6f}")
    elapsed_p = time.time() - t_start
    print(f"  Poincare training time: {elapsed_p:.1f}s")

    # Extract Poincare embeddings
    final_features = trajs[-1]
    z_poincare = jax.vmap(poincare_model.encoder)(final_features)
    c_val = float(jnp.abs(poincare_model.dynamics.drift.conv.c) + 1e-6)

    # Project into Poincare ball
    z_np = np.array(z_poincare)
    max_norm = 1.0 / np.sqrt(c_val) - 1e-5
    norms = np.linalg.norm(z_np, axis=-1, keepdims=True)
    z_ball = z_np * np.minimum(max_norm / np.maximum(norms, 1e-6), 1.0)

    # Compute Poincare Gromov delta
    print("  Computing Poincare Gromov delta-hyperbolicity...")
    delta_poincare = compute_gromov_delta(jnp.array(z_ball), c=c_val)
    print(f"  Poincare Gromov delta: {delta_poincare:.4f}")

    # --- Euclidean model (for comparison) ---
    print("\n  Training Euclidean LatentODE for comparison...")
    euclidean_model = hgx.LatentHypergraphODE(
        obs_dim=FEATURE_DIM,
        latent_dim=LATENT_DIM,
        conv_cls=hgx.UniGCNConv,
        key=k2,
    )

    opt_euc = optax.adam(1e-3)
    opt_state_euc = opt_euc.init(eqx.filter(euclidean_model, eqx.is_array))

    @eqx.filter_jit
    def step_euclidean(model, opt_state, hg, target):
        @eqx.filter_value_and_grad
        def loss_fn(m):
            pred = m(hg, t0=0.0, t1=1.0)
            return jnp.mean((pred - target) ** 2)

        loss, grads = loss_fn(model)
        updates, new_opt = opt_euc.update(grads, opt_state)
        return eqx.apply_updates(model, updates), new_opt, loss

    for i in range(epochs):
        t = i % num_pairs
        hg_t = hgx.from_incidence(incidence, node_features=trajs[t])
        euclidean_model, opt_state_euc, loss = step_euclidean(
            euclidean_model, opt_state_euc, hg_t, trajs[t + 1]
        )
        if (i + 1) % max(1, epochs // 5) == 0:
            print(f"    Step {i+1:4d}  loss={loss:.6f}")

    z_euclidean = jax.vmap(euclidean_model.encoder)(final_features)
    print("  Computing Euclidean Gromov delta-hyperbolicity...")
    delta_euclidean = euclidean_gromov_delta(z_euclidean)
    print(f"  Euclidean Gromov delta: {delta_euclidean:.4f}")

    print(f"\n  Delta improvement (Euclidean - Poincare): "
          f"{delta_euclidean - delta_poincare:.4f}")

    return {
        "z_poincare": z_ball,
        "z_euclidean": np.array(z_euclidean),
        "delta_poincare": delta_poincare,
        "delta_euclidean": delta_euclidean,
        "c_val": c_val,
        "losses_poincare": losses_p,
    }


# ---------------------------------------------------------------------------
# Part C: Stochastic Fate Decisions via SDE
# ---------------------------------------------------------------------------


def run_stochastic_sde(data, epochs, *, key):
    """Train Neural SDE and sample multiple trajectories for fate analysis."""
    if not HAS_DIFFRAX:
        print("  Skipping SDE analysis (diffrax not available).")
        return None

    print("\n" + "=" * 60)
    print("  Part C: Stochastic Fate Decisions via SDE (Figure 4)")
    print("=" * 60)

    k1, k2, k3 = jax.random.split(key, 3)
    incidence = data["incidence"]
    trajs = data["trajectories"]
    num_train = NUM_PSEUDOTIME - 3

    # Build SDE
    conv_sde = hgx.UniGCNConv(FEATURE_DIM, FEATURE_DIM, key=k1)
    sde = hgx.HypergraphNeuralSDE(
        conv=conv_sde,
        num_nodes=NUM_GENES,
        node_dim=FEATURE_DIM,
        sigma_init=0.1,
        dt=0.02,
        key=k2,
    )

    # Train SDE on temporal data
    print("  Training Neural SDE...")
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(sde, eqx.is_array))

    @eqx.filter_jit
    def sde_step(model, opt_state, hg, target, key):
        @eqx.filter_value_and_grad
        def loss_fn(m):
            sol = m(hg, t0=0.0, t1=1.0 / (NUM_PSEUDOTIME - 1), key=key)
            pred = sol.ys[-1].reshape(NUM_GENES, FEATURE_DIM)
            return jnp.mean((pred - target) ** 2)

        loss, grads = loss_fn(model)
        updates, new_opt = optimizer.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), new_opt, loss

    losses_sde = []
    t_start = time.time()
    for i in range(epochs):
        t = i % (num_train - 1)
        hg_t = hgx.from_incidence(incidence, node_features=trajs[t])
        k_step = jax.random.fold_in(k3, i)
        sde, opt_state, loss = sde_step(
            sde, opt_state, hg_t, trajs[t + 1], k_step
        )
        losses_sde.append(float(loss))
        if (i + 1) % max(1, epochs // 5) == 0:
            print(f"    Step {i+1:4d}  loss={loss:.6f}")

    elapsed = time.time() - t_start
    print(f"  SDE training time: {elapsed:.1f}s")

    # --- Sample multiple trajectories from same initial condition ---
    print("  Sampling 20 stochastic trajectories...")
    hg_init = hgx.from_incidence(incidence, node_features=trajs[0])
    n_samples = 20
    n_traj_steps = 50
    save_ts = jnp.linspace(0.0, 1.0, n_traj_steps)

    trajectories_list = []
    for i in range(n_samples):
        k_sample = jax.random.fold_in(key, i + 1000)
        sol = sde(
            hg_init,
            t0=0.0,
            t1=1.0,
            key=k_sample,
            saveat=diffrax.SaveAt(ts=save_ts),
        )
        traj = sol.ys.reshape(n_traj_steps, NUM_GENES, FEATURE_DIM)
        trajectories_list.append(np.array(traj))

    trajectories_arr = np.stack(trajectories_list)  # (n_samples, T, N, D)

    # Compute variance across samples at each timepoint
    traj_variance = np.var(trajectories_arr, axis=0)  # (T, N, D)
    mean_var_over_time = traj_variance.mean(axis=(1, 2))  # (T,)

    # --- Extract learned diffusion sigma ---
    sigma_raw = sde.diffusion.log_sigma  # (D,)
    sigma = np.array(jnp.exp(sigma_raw))  # (D,)
    # Tile to per-gene: each gene has the same sigma per dim
    sigma_per_gene = np.tile(sigma, NUM_GENES)  # (N*D,)
    sigma_mean_per_gene = np.mean(sigma) * np.ones(NUM_GENES)

    # Expression variance along pseudotime (per gene)
    trajs_np = np.array(trajs)  # (T, N, D)
    expr_var_per_gene = np.var(trajs_np, axis=0).mean(axis=-1)  # (N,)

    print(f"  Learned sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
    print(f"  Mean rollout variance at t=1: {mean_var_over_time[-1]:.6f}")

    return {
        "sde": sde,
        "trajectories": trajectories_arr,
        "save_ts": np.array(save_ts),
        "traj_variance": mean_var_over_time,
        "sigma": sigma,
        "sigma_per_gene": sigma_mean_per_gene,
        "expr_var_per_gene": expr_var_per_gene,
        "losses_sde": losses_sde,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def plot_figure_3(data, ode_results, poincare_results, fig_dir):
    """Generate Figure 3: Continuous Trajectory (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    trajs = np.array(data["trajectories"])  # (T, N, D)
    gene_fates = np.array(data["gene_fates"])

    # --- 3A: PCA of expression colored by pseudotime + fate ---
    ax = axes[0, 0]
    # Stack all timepoints for joint PCA
    all_expr = trajs.reshape(-1, FEATURE_DIM)  # (T*N, D)
    centered = all_expr - all_expr.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pc = centered @ Vt[:2].T  # (T*N, 2)

    for t_idx in [0, NUM_PSEUDOTIME // 2, NUM_PSEUDOTIME - 1]:
        start = t_idx * NUM_GENES
        end = start + NUM_GENES
        pc_t = pc[start:end]
        alpha = 0.2 + 0.8 * (t_idx / (NUM_PSEUDOTIME - 1))
        for f in range(NUM_FATES):
            mask = gene_fates == f
            label = FATE_NAMES[f] if t_idx == NUM_PSEUDOTIME - 1 else None
            ax.scatter(
                pc_t[mask, 0],
                pc_t[mask, 1],
                c=FATE_COLORS[f],
                alpha=alpha,
                s=8,
                label=label,
            )

    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title("A. Gene Expression PCA by Pseudotime & Fate", fontsize=12)
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.3)

    # --- 3B: Neural ODE predicted vs observed for key TFs ---
    ax = axes[0, 1]
    if ode_results is not None:
        pred_trajs = np.array(ode_results["pred_trajs"])  # (T-1, N, D)
        pseudotime_obs = np.linspace(0, 1, NUM_PSEUDOTIME)
        pseudotime_pred = pseudotime_obs[1:]

        for i, (tf_idx, tf_name) in enumerate(zip(TF_INDICES, TF_NAMES)):
            color = FATE_COLORS[i % len(FATE_COLORS)]
            # Observed: mean over feature dims
            obs_vals = trajs[:, tf_idx, :].mean(axis=-1)
            pred_vals = pred_trajs[:, tf_idx, :].mean(axis=-1)

            ax.plot(
                pseudotime_obs,
                obs_vals,
                "-",
                color=color,
                linewidth=2,
                label=f"{tf_name} (obs)",
            )
            ax.plot(
                pseudotime_pred,
                pred_vals,
                "--",
                color=color,
                linewidth=2,
                label=f"{tf_name} (pred)",
            )

        ax.set_xlabel("Pseudotime", fontsize=11)
        ax.set_ylabel("Mean Expression", fontsize=11)
        ax.set_title("B. Neural ODE: Predicted vs Observed", fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "diffrax required", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("B. Neural ODE (unavailable)", fontsize=12)

    # --- 3C: Poincare disk embedding colored by fate ---
    ax = axes[1, 0]
    if poincare_results is not None:
        z = poincare_results["z_poincare"]
        # Project to 2D via PCA of Poincare embeddings
        z_centered = z - z.mean(axis=0)
        _, _, Vt_z = np.linalg.svd(z_centered, full_matrices=False)
        z_2d = z_centered @ Vt_z[:2].T

        # Normalize to fit in unit disk
        max_r = np.max(np.linalg.norm(z_2d, axis=-1))
        if max_r > 0:
            z_2d = z_2d / (max_r * 1.05)

        # Draw unit disk boundary
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1.5, alpha=0.5)

        for f in range(NUM_FATES):
            mask = gene_fates == f
            ax.scatter(
                z_2d[mask, 0],
                z_2d[mask, 1],
                c=FATE_COLORS[f],
                s=15,
                alpha=0.7,
                label=FATE_NAMES[f],
            )

        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.set_xlabel("Poincare dim 1", fontsize=11)
        ax.set_ylabel("Poincare dim 2", fontsize=11)
        ax.set_title("C. Poincare Disk Embedding", fontsize=12)
        ax.legend(fontsize=9, markerscale=2)
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, "diffrax required", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("C. Poincare Disk (unavailable)", fontsize=12)

    # --- 3D: Gromov delta bar chart ---
    ax = axes[1, 1]
    if poincare_results is not None:
        delta_e = poincare_results["delta_euclidean"]
        delta_p = poincare_results["delta_poincare"]

        bars = ax.bar(
            [0, 1],
            [delta_e, delta_p],
            color=["#4393c3", "#d6604d"],
            edgecolor="black",
            linewidth=0.8,
            width=0.5,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Euclidean\n(UniGCNConv)", "Poincare\n(PoincareConv)"],
                           fontsize=10)
        ax.set_ylabel("Gromov delta", fontsize=11)
        ax.set_title("D. Gromov delta-Hyperbolicity", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, [delta_e, delta_p]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.text(
            0.5,
            0.05,
            "(lower = more hyperbolic / tree-like)",
            ha="center",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
        )
    else:
        ax.text(0.5, 0.5, "diffrax required", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("D. Gromov delta (unavailable)", fontsize=12)

    fig.tight_layout()
    fig_path = fig_dir / "figure_03_trajectory.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 3 saved to {fig_path}")


def plot_figure_4(data, sde_results, ode_results, fig_dir):
    """Generate Figure 4: Stochastic Fate (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    trajs = np.array(data["trajectories"])

    # --- 4A: Multiple SDE trajectories diverging at branch point ---
    ax = axes[0, 0]
    if sde_results is not None:
        all_trajs = sde_results["trajectories"]  # (n_samples, T, N, D)
        n_samples = all_trajs.shape[0]
        n_traj_steps = all_trajs.shape[1]

        # PCA of all trajectories jointly for consistent projection
        flat = all_trajs.reshape(-1, FEATURE_DIM)  # (n_samples*T*N, D)
        centered = flat - flat.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered[:1000], full_matrices=False)  # subsample SVD

        # Plot each trajectory's mean gene expression projected to 2D
        cmap = matplotlib.colormaps["viridis"]
        for i in range(min(n_samples, 20)):
            traj_i = all_trajs[i]  # (T, N, D)
            # Mean over genes -> (T, D) then project to 2D
            mean_traj = traj_i.mean(axis=1)  # (T, D)
            pc = (mean_traj - flat.mean(axis=0)) @ Vt[:2].T
            color = cmap(i / n_samples)
            ax.plot(pc[:, 0], pc[:, 1], color=color, alpha=0.6, linewidth=1.0)
            ax.plot(pc[0, 0], pc[0, 1], "o", color=color, markersize=4)
            ax.plot(pc[-1, 0], pc[-1, 1], "s", color=color, markersize=4)

        ax.set_xlabel("PC1", fontsize=11)
        ax.set_ylabel("PC2", fontsize=11)
        ax.set_title("A. SDE Trajectory Ensemble (PCA)", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add annotation for branch point
        ax.annotate(
            "branch\npoint",
            xy=(0, 0),
            fontsize=9,
            color="dimgray",
            ha="center",
        )
    else:
        ax.text(0.5, 0.5, "diffrax required", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("A. SDE Trajectories (unavailable)", fontsize=12)

    # --- 4B: Learned diffusion sigma vs expression variance ---
    ax = axes[0, 1]
    if sde_results is not None:
        sigma_per_gene = sde_results["sigma_per_gene"]  # (N,)
        expr_var = sde_results["expr_var_per_gene"]  # (N,)

        ax.scatter(expr_var, sigma_per_gene, s=10, alpha=0.6, c="steelblue")

        # Highlight TFs
        for i, (tf_idx, tf_name) in enumerate(zip(TF_INDICES, TF_NAMES)):
            ax.scatter(
                expr_var[tf_idx],
                sigma_per_gene[tf_idx],
                s=60,
                c=FATE_COLORS[i],
                edgecolors="black",
                zorder=5,
            )
            ax.annotate(
                tf_name,
                xy=(expr_var[tf_idx], sigma_per_gene[tf_idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )

        # Fit and draw trend line
        if len(expr_var) > 2:
            z = np.polyfit(expr_var, sigma_per_gene, 1)
            p = np.poly1d(z)
            x_line = np.linspace(expr_var.min(), expr_var.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=1.5)
            corr = np.corrcoef(expr_var, sigma_per_gene)[0, 1]
            ax.text(
                0.05,
                0.95,
                f"r = {corr:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                va="top",
            )

        ax.set_xlabel("Expression Variance (across pseudotime)", fontsize=11)
        ax.set_ylabel("Learned Diffusion sigma", fontsize=11)
        ax.set_title("B. Diffusion vs Expression Variability", fontsize=12)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "diffrax required", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("B. Diffusion sigma (unavailable)", fontsize=12)

    # --- 4C: Phase portrait ---
    ax = axes[1, 0]
    if ode_results is not None and ode_results.get("ode_trajectory") is not None:
        ode_traj = ode_results["ode_trajectory"]  # (T, N, D)
        # Use hgx.draw_phase_portrait if available, else manual
        try:
            # Show a few representative genes from different fates
            gene_fates_np = np.array(data["gene_fates"])
            show_nodes = []
            for f in range(NUM_FATES):
                fate_genes = np.where(gene_fates_np == f)[0]
                if len(fate_genes) > 0:
                    show_nodes.append(int(fate_genes[0]))

            hgx.draw_phase_portrait(
                ode_traj,
                node_indices=show_nodes,
                dims=(0, 1),
                ax=ax,
                title="C. Phase Portrait (ODE)",
            )
        except Exception:
            # Manual fallback
            for f in range(NUM_FATES):
                gene_fates_np = np.array(data["gene_fates"])
                fate_genes = np.where(gene_fates_np == f)[0]
                if len(fate_genes) > 0:
                    g = fate_genes[0]
                    ax.plot(
                        ode_traj[:, g, 0],
                        ode_traj[:, g, 1],
                        color=FATE_COLORS[f],
                        linewidth=1.5,
                        label=FATE_NAMES[f],
                    )
                    ax.plot(
                        ode_traj[0, g, 0],
                        ode_traj[0, g, 1],
                        "o",
                        color=FATE_COLORS[f],
                        markersize=6,
                    )

            ax.set_xlabel("Feature dim 0", fontsize=11)
            ax.set_ylabel("Feature dim 1", fontsize=11)
            ax.set_title("C. Phase Portrait (ODE)", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "diffrax required", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("C. Phase Portrait (unavailable)", fontsize=12)

    # --- 4D: SDE rollout variance vs pseudotime ---
    ax = axes[1, 1]
    if sde_results is not None:
        save_ts = sde_results["save_ts"]
        traj_var = sde_results["traj_variance"]

        ax.plot(save_ts, traj_var, "b-", linewidth=2, label="Mean variance")
        ax.fill_between(save_ts, 0, traj_var, alpha=0.15, color="steelblue")

        ax.set_xlabel("Pseudotime", fontsize=11)
        ax.set_ylabel("Trajectory Variance", fontsize=11)
        ax.set_title("D. SDE Rollout Variance vs Pseudotime", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Mark the approximate branch point
        # Find where variance starts increasing rapidly
        if len(traj_var) > 10:
            grad_var = np.gradient(traj_var)
            branch_idx = np.argmax(grad_var > np.percentile(grad_var, 75))
            if branch_idx > 0:
                ax.axvline(
                    save_ts[branch_idx],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Branch point (~t={save_ts[branch_idx]:.2f})",
                )
                ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "diffrax required", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("D. Rollout Variance (unavailable)", fontsize=12)

    fig.tight_layout()
    fig_path = fig_dir / "figure_04_stochastic.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 4 saved to {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Temporal dynamics, Poincare embeddings, and stochastic fate "
            "(Figures 3-4)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Analyses:\n"
            "  2.2 Neural ODE trajectory fitting\n"
            "  2.3 Poincare latent space and Gromov hyperbolicity\n"
            "  2.4 SDE-based stochastic fate decisions\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (default: ../data relative to script)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Training epochs (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir.parent / "data"
    fig_dir = script_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Temporal Dynamics, Poincare Embeddings, Stochastic Fate")
    print("  Figure 3 (Trajectory) + Figure 4 (Stochastic)")
    print("=" * 64)

    key = jax.random.PRNGKey(args.seed)
    k_data, k_ode, k_poincare, k_sde = jax.random.split(key, 4)

    # --- Load or generate data ---
    print("\nPreparing data...")
    real_data = try_load_expression(data_dir)

    if real_data is not None and real_data.get("pseudotime") is not None:
        print("  Using real expression data with pseudotime.")
        # Build synthetic incidence for the real expression data
        n_genes = real_data["expression"].shape[1]
        n_mods = min(NUM_MODULES, n_genes // 5)
        rng = np.random.RandomState(args.seed)
        incidence = np.zeros((n_genes, n_mods), dtype=np.float32)
        for g in range(n_genes):
            m = rng.randint(n_mods)
            incidence[g, m] = 1.0

        # Build temporal hypergraphs from real expression
        hgs = hgx.grn_to_temporal_hypergraphs(
            real_data["expression"],
            real_data["pseudotime"],
            incidence,
            num_timepoints=NUM_PSEUDOTIME,
        )
        times = jnp.linspace(0, 1, len(hgs))

        # Build trajectories array from hypergraph features
        # Note: grn_to_temporal_hypergraphs produces (n_genes, 1) features
        feat_dim = hgs[0].node_features.shape[1]
        trajectories = jnp.stack([hg.node_features for hg in hgs])

        gene_fates = jnp.array(
            [g % NUM_FATES for g in range(n_genes)], dtype=jnp.int32
        )
        if real_data.get("fates") is not None:
            gene_fates = jnp.array(real_data["fates"][:n_genes] % NUM_FATES)

        data = {
            "incidence": jnp.array(incidence),
            "trajectories": trajectories,
            "gene_fates": gene_fates,
        }
        # Adjust globals for real data dimensions
        # (scripts uses the data dict shape for everything)
    else:
        print("  Real expression data not found; generating synthetic data...")
        data = generate_synthetic_temporal(key=k_data)

    n_genes = data["incidence"].shape[0]
    n_modules = data["incidence"].shape[1]
    n_time = data["trajectories"].shape[0]
    feat_dim = data["trajectories"].shape[2]
    print(f"  Genes: {n_genes}  Modules: {n_modules}  "
          f"Timepoints: {n_time}  Features: {feat_dim}")
    print(f"  Trajectories shape: {data['trajectories'].shape}")

    # --- Run analyses ---
    total_start = time.time()

    ode_results = run_neural_ode(data, args.epochs, key=k_ode)
    poincare_results = run_poincare_embedding(data, args.epochs, key=k_poincare)
    sde_results = run_stochastic_sde(data, args.epochs, key=k_sde)

    total_elapsed = time.time() - total_start

    # --- Summary ---
    print("\n" + "=" * 64)
    print("  Results Summary")
    print("=" * 64)

    if ode_results is not None:
        print(f"  Neural ODE 1-step MSE:    {ode_results['avg_mse']:.6f}")
        print(f"  Neural ODE rollout MSE:   {ode_results['avg_rollout_mse']:.6f}")

    if poincare_results is not None:
        print(f"  Poincare Gromov delta:     {poincare_results['delta_poincare']:.4f}")
        print(f"  Euclidean Gromov delta:    {poincare_results['delta_euclidean']:.4f}")
        improvement = (
            poincare_results["delta_euclidean"] - poincare_results["delta_poincare"]
        )
        print(f"  Delta improvement:         {improvement:+.4f}")

    if sde_results is not None:
        print(f"  Learned sigma range:       "
              f"[{sde_results['sigma'].min():.4f}, "
              f"{sde_results['sigma'].max():.4f}]")
        print(f"  Final rollout variance:    "
              f"{sde_results['traj_variance'][-1]:.6f}")

    print(f"\n  Total analysis time: {total_elapsed:.1f}s")

    # --- Generate figures ---
    print("\nGenerating figures...")
    plot_figure_3(data, ode_results, poincare_results, fig_dir)
    plot_figure_4(data, sde_results, ode_results, fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
