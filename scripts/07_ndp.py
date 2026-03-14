#!/usr/bin/env python3
"""07 -- Neural Developmental Programs (NDP) on organoid hypergraphs.

Demonstrates hgx's NDP capability: growing hypergraphs that model cell
division and topology growth, applied to a synthetic cerebral organoid
seed. A shared CellProgram acts as "cellular DNA", deciding state
updates and division for each node at every developmental step.

Supplementary analysis producing figure_supp_ndp.png with four panels:
  A: Initial seed hypergraph
  B: Final developed hypergraph colored by fate (PCA cluster)
  C: Node and edge count vs developmental step
  D: Feature diversity (std of node features) vs step
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import optax

import hgx
from hgx._dynamic import preallocate


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neural Developmental Programs on organoid hypergraphs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="PRNG seed (default: 42)",
    )
    parser.add_argument(
        "--num-steps", type=int, default=20,
        help="Number of NDP developmental steps (default: 20)",
    )
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

GENE_DIM = 16  # synthetic gene-expression feature dimension
MAX_NODES = 500
MAX_EDGES = 100
INITIAL_NODES = 5
INITIAL_EDGES = 3


def create_seed_hypergraph(key: jax.Array) -> hgx.Hypergraph:
    """Create a small progenitor 'organoid seed'.

    5 progenitor cells with random gene-expression features and
    3 initial regulatory interactions (hyperedges).
    """
    features = jax.random.normal(key, (INITIAL_NODES, GENE_DIM))
    # Three initial hyperedges of mixed arity
    hg = hgx.from_edge_list(
        [(0, 1, 2), (1, 2, 3), (3, 4)],
        num_nodes=INITIAL_NODES,
        node_features=features,
    )
    return hg


def generate_target_trajectory(
    num_steps: int, key: jax.Array,
) -> jnp.ndarray:
    """Synthetic target: features diverge over time (cell differentiation).

    Returns an array of shape (num_steps, MAX_NODES, GENE_DIM) where the
    first INITIAL_NODES rows carry smoothly diverging expression profiles
    and the remaining rows are zero (representing cells not yet born in
    the target).
    """
    k1, k2 = jax.random.split(key)
    # Base directions -- each progenitor drifts in its own direction
    directions = jax.random.normal(k1, (INITIAL_NODES, GENE_DIM))
    directions = directions / (jnp.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)

    ts = jnp.linspace(0.0, 1.0, num_steps)
    # (T, n_init, d) -- linear drift + small noise
    noise = jax.random.normal(k2, (num_steps, INITIAL_NODES, GENE_DIM)) * 0.05
    target = directions[None, :, :] * ts[:, None, None] + noise

    # Pad to MAX_NODES
    padded = jnp.zeros((num_steps, MAX_NODES, GENE_DIM))
    padded = padded.at[:, :INITIAL_NODES, :].set(target)
    return padded


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)
    num_steps = args.num_steps

    # ---- 1. Create seed hypergraph ----------------------------------------
    k_seed, k_prog, k_conv, k_dev, k_traj, k_target, k_train = jax.random.split(key, 7)
    initial_hg = create_seed_hypergraph(k_seed)
    print(f"Seed: {initial_hg.num_nodes} nodes, {initial_hg.num_edges} edges")

    # Pre-allocate for NDP growth
    initial_hg = preallocate(initial_hg, MAX_NODES, MAX_EDGES)

    # ---- 2. Define CellProgram (shared "DNA") -----------------------------
    program = hgx.CellProgram(state_dim=GENE_DIM, hidden_dim=32, key=k_prog)

    # ---- 3. Build NDP -----------------------------------------------------
    conv = hgx.UniGCNConv(GENE_DIM, GENE_DIM, key=k_conv)
    ndp = hgx.HypergraphNDP(program, conv, max_nodes=MAX_NODES, max_edges=MAX_EDGES)

    # ---- 4. Run development -----------------------------------------------
    final_hg = ndp.develop(initial_hg, num_steps=num_steps, key=k_dev)
    print(
        f"Grew from {INITIAL_NODES} to {final_hg.num_nodes} nodes, "
        f"{INITIAL_EDGES} to {final_hg.num_edges} edges"
    )

    # ---- 5. Record full trajectory ----------------------------------------
    ts, features_traj = hgx.develop_trajectory(ndp, initial_hg, num_steps=num_steps, key=k_traj)
    # features_traj: (T, MAX_NODES, GENE_DIM)
    ts_np = np.asarray(ts)
    features_np = np.asarray(features_traj)

    # ---- 6. Train NDP to match a target trajectory (demo) -----------------
    import equinox as eqx

    target = generate_target_trajectory(num_steps, k_target)

    # Simple MSE loss over the initial progenitor features only (the
    # NDP may create additional nodes that have no target counterpart).
    # Use a short trajectory (3 steps) for training to keep it tractable.
    train_steps = min(3, num_steps)
    target_short = target[:train_steps]

    def loss_fn(ndp_model):
        _, feats = hgx.develop_trajectory(ndp_model, initial_hg, num_steps=train_steps, key=k_train)
        pred = feats[:, :INITIAL_NODES, :]
        tgt = target_short[:, :INITIAL_NODES, :]
        return jnp.mean((pred - tgt) ** 2)

    opt = optax.adam(1e-3)
    opt_state = opt.init(eqx.filter(ndp, eqx.is_array))

    @eqx.filter_jit
    def train_step(ndp_model, opt_state):
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(ndp_model)
        updates, new_opt_state = opt.update(grads, opt_state, ndp_model)
        return eqx.apply_updates(ndp_model, updates), new_opt_state, loss_val

    print("Training NDP to match target trajectory (5 epochs, demo)...")
    for epoch in range(5):
        ndp, opt_state, loss_val = train_step(ndp, opt_state)
        print(f"  epoch {epoch + 1}: loss = {float(loss_val):.6f}")

    # ---- 7. Growth statistics ---------------------------------------------
    # Count active nodes/edges at each step from the trajectory.
    # node_mask after preallocate: first INITIAL_NODES are True. During
    # development the NDP adds nodes -- but since we only store features
    # in develop_trajectory, we approximate active count by checking which
    # rows have non-zero norm.
    node_counts = []
    edge_counts = []
    feature_stds = []

    # Re-run development step by step to track masks
    hg_step = preallocate(create_seed_hypergraph(k_seed), MAX_NODES, MAX_EDGES)
    step_keys = jax.random.split(k_dev, num_steps)
    for step_i in range(num_steps):
        hg_step = ndp(hg_step, key=step_keys[step_i])  # single step
        node_counts.append(int(hg_step.num_nodes))
        edge_counts.append(int(hg_step.num_edges))
        # Feature diversity: std over active node features
        mask = np.asarray(hg_step.node_mask)
        active_feats = np.asarray(hg_step.node_features)[mask]
        if len(active_feats) > 1:
            feature_stds.append(float(np.std(active_feats)))
        else:
            feature_stds.append(0.0)

    steps = np.arange(1, num_steps + 1)

    print("\nGrowth statistics per step:")
    for s, nc, ec, fstd in zip(steps, node_counts, edge_counts, feature_stds):
        print(f"  step {s:3d}: {nc:4d} nodes, {ec:3d} edges, feat std = {fstd:.4f}")

    # ---- 8. Figure --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: initial hypergraph
    ax_a = axes[0, 0]
    hg_init_viz = create_seed_hypergraph(k_seed)
    hgx.draw_hypergraph(hg_init_viz, ax=ax_a, title="A: Initial Organoid Seed")

    # Panel B: final hypergraph colored by fate (PCA cluster)
    ax_b = axes[0, 1]
    # Build a small hypergraph from final active nodes for visualization
    mask_final = np.asarray(final_hg.node_mask)
    n_active = int(mask_final.sum())
    active_feats = np.asarray(final_hg.node_features)[mask_final]

    # Fate assignment: PCA -> k-means-like clustering via feature sign
    if n_active > 2 and active_feats.shape[1] >= 2:
        from numpy.linalg import svd as np_svd
        centered = active_feats - active_feats.mean(axis=0, keepdims=True)
        U, S, Vt = np_svd(centered, full_matrices=False)
        pc1 = U[:, 0]
        # Simple 3-cluster assignment by terciles
        thresholds = np.percentile(pc1, [33, 66])
        fate_labels = np.digitize(pc1, thresholds)
        fate_colors = ["#e74c3c", "#2ecc71", "#3498db"]
        colors = [fate_colors[l] for l in fate_labels]
    else:
        colors = "steelblue"

    # Reconstruct a compact hypergraph from active nodes for drawing
    active_idx = np.where(mask_final)[0]
    inc_full = np.asarray(final_hg.incidence)
    edge_mask_final = np.asarray(final_hg.edge_mask)
    active_edge_idx = np.where(edge_mask_final)[0]

    if len(active_idx) > 0 and len(active_edge_idx) > 0:
        # Remap to compact indices
        inc_compact = inc_full[np.ix_(active_idx, active_edge_idx)]
        # Keep only edges that still have >= 2 members
        valid_edges = inc_compact.sum(axis=0) >= 2
        inc_compact = inc_compact[:, valid_edges]
        if inc_compact.shape[1] > 0:
            hg_final_viz = hgx.from_incidence(
                jnp.array(inc_compact),
                node_features=jnp.array(active_feats),
            )
            hgx.draw_hypergraph(
                hg_final_viz, ax=ax_b, node_color=colors,
                title=f"B: Final Hypergraph ({n_active} cells, colored by fate)",
            )
        else:
            ax_b.text(0.5, 0.5, f"Final: {n_active} nodes\n(no multi-member edges)",
                      ha="center", va="center", transform=ax_b.transAxes)
            ax_b.set_title("B: Final Hypergraph")
    else:
        ax_b.text(0.5, 0.5, "No active nodes/edges", ha="center", va="center",
                  transform=ax_b.transAxes)
        ax_b.set_title("B: Final Hypergraph")
    ax_b.axis("off")

    # Panel C: node count and edge count vs step
    ax_c = axes[1, 0]
    ax_c.plot(steps, node_counts, "o-", color="#2c3e50", label="Nodes")
    ax_c2 = ax_c.twinx()
    ax_c2.plot(steps, edge_counts, "s--", color="#e67e22", label="Edges")
    ax_c.set_xlabel("Developmental Step")
    ax_c.set_ylabel("Node Count", color="#2c3e50")
    ax_c2.set_ylabel("Edge Count", color="#e67e22")
    ax_c.set_title("C: Topology Growth")
    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Panel D: feature diversity vs step
    ax_d = axes[1, 1]
    ax_d.plot(steps, feature_stds, "D-", color="#8e44ad")
    ax_d.set_xlabel("Developmental Step")
    ax_d.set_ylabel("Feature Std (diversity)")
    ax_d.set_title("D: Feature Diversity Over Development")
    ax_d.axhline(y=feature_stds[0], color="gray", linestyle=":", alpha=0.5, label="Initial")
    ax_d.legend()

    fig.suptitle("Supplementary: Neural Developmental Programs on Organoid Hypergraph", fontsize=14)
    plt.tight_layout()

    out_path = Path(__file__).resolve().parent.parent / "figures" / "figure_supp_ndp.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
