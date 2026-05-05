#!/usr/bin/env python3
"""11_fleck_forward_ode.py — Forward modeling of Fleck et al. dynamics.

Trains a Hypergraph Neural ODE to match the observed temporal expression
trajectory of the Fleck et al. organoid dataset.
"""

import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import hgx
    import diffrax
except ImportError:
    sys.exit("ERROR: hgx or diffrax not found. Run: uv pip install -e ../hgx diffrax")


def main():
    # 1. Load preprocessed Fleck data
    data_dir = PROJECT_ROOT / "data" / "processed"
    if not data_dir.exists():
        sys.exit(f"ERROR: Data directory {data_dir} not found. Run scripts/00_preprocess.py first.")

    print(f"Loading data from {data_dir}...")
    incidence = jnp.array(np.load(data_dir / "incidence.npy"))
    obs_expr = jnp.array(np.load(data_dir / "temporal_expression.npy")) # (T, N)
    times = jnp.array(np.load(data_dir / "pseudotime_centers.npy")) # (T,)
    
    with open(data_dir / "gene_names.json") as f:
        import json
        gene_names = json.load(f)
    with open(data_dir / "tf_names.json") as f:
        tf_names = json.load(f)
    with open(data_dir / "key_tf_indices.json") as f:
        key_tf_indices = json.load(f)

    n_genes, n_edges = incidence.shape
    n_times = obs_expr.shape[0]
    
    print(f"  Genes: {n_genes}, Hyperedges: {n_edges}")
    print(f"  Timepoints: {n_times} (pseudotime {times[0]:.2f} to {times[-1]:.2f})")

    # 2. Initial state: expression at first timepoint
    y0 = obs_expr[0][:, None] # (N, 1)
    hg = hgx.from_incidence(incidence, node_features=y0)

    # 3. Model: Neural ODE with Hypergraph Convolution
    # We use a 2-layer HGNNStack to allow more complex dynamics than a single conv
    # We must ensure the input and output dims are both 1 to preserve expression state
    conv_dims = [(1, 16), (16, 1)]
    stack = hgx.HGNNStack(
        conv_dims=conv_dims,
        conv_cls=hgx.UniGCNConv,
        activation=jax.nn.tanh, # Tanh helps stability in ODEs
        key=jax.random.PRNGKey(42)
    )
    
    # Wrap in HypergraphNeuralODE
    # Note: HGNNStack is duck-typed as a conv here
    model = hgx.HypergraphNeuralODE(
        stack, 
        activation=lambda x: x # Stack already has activation
    )

    # 4. Training loop
    optimizer = optax.adam(5e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, hg, targets, ts):
        def loss_fn(m):
            # Integrate from t_start to t_end, saving at all observed points
            sol = m(hg, t0=ts[0], t1=ts[-1], 
                    saveat=diffrax.SaveAt(ts=ts))
            
            # sol.ys has shape (T, N, 1)
            preds = sol.ys.squeeze(-1) # (T, N)
            
            # MSE loss against observed expression
            return jnp.mean((preds - targets) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt = optimizer.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), new_opt, loss

    print("\nTraining Hypergraph Neural ODE...")
    t_start = time.time()
    losses = []
    epochs = 200
    for i in range(epochs):
        model, opt_state, loss = train_step(model, opt_state, hg, obs_expr, times)
        losses.append(float(loss))
        if (i + 1) % 40 == 0:
            print(f"  Epoch {i+1:3d} | Loss: {loss:.6f}")

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f}s")

    # 5. Final rollout for visualization
    print("\nPerforming final rollout...")
    save_ts = jnp.linspace(times[0], times[-1], 50)
    sol = model(hg, t0=times[0], t1=times[-1], 
                saveat=diffrax.SaveAt(ts=save_ts))
    traj = sol.ys.squeeze(-1) # (50, N)

    # 6. Visualization: Key TFs
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Selection of TFs to plot
    plot_tfs = ["GLI3", "TBR1", "NEUROD6", "EOMES"]
    for i, tf_name in enumerate(plot_tfs):
        if tf_name not in key_tf_indices: continue
        idx = key_tf_indices[tf_name]
        
        ax = axes[i]
        ax.plot(times, obs_expr[:, idx], 'ko', label="Observed", alpha=0.6)
        ax.plot(save_ts, traj[:, idx], 'r-', label="ODE Pred")
        ax.set_title(f"TF: {tf_name}")
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Expression")
        ax.legend()

    plt.tight_layout()
    fig_path = PROJECT_ROOT / "figures" / "fleck_forward_ode.png"
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")

    # Save results summary
    results = {
        "final_loss": float(losses[-1]),
        "training_time": elapsed,
        "epochs": epochs
    }
    with open(PROJECT_ROOT / "figures" / "fleck_forward_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
