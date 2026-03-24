#!/usr/bin/env python3
"""Higher-Order Regulatory Interactions: convolution layer comparison.

Compares five convolution layers on regulatory module detection to
demonstrate that higher-order interactions (THNNConv) surpass pairwise
methods (UniGCNConv).  Produces Figure 2 panels C-D.

Convolution layers tested:
    1. UniGCNConv   -- 1st-order, symmetric-normalized (baseline)
    2. UniGATConv   -- attention-based
    3. UniGINConv   -- GIN expressiveness
    4. THNNConv     -- tensorized higher-order (CP decomposition)
    5. SheafDiffusion -- sheaf-based diffusion (preserves dimension)

Usage:
    uv run python scripts/03_higher_order.py
    uv run python scripts/03_higher_order.py --epochs 200 --seed 7
    uv run python scripts/03_higher_order.py --data-dir data/pando
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_GENES = 200
NUM_TFS = 20
NUM_MODULES = 15
NUM_FATES = 4
FEATURE_DIM = 8
NUM_MASTER = 3
NUM_INTERMEDIATE = 7


# ---------------------------------------------------------------------------
# Data loading / generation
# ---------------------------------------------------------------------------


def try_load_pando(data_dir: Path):
    """Attempt to load real GRN data from Pando coefficients TSV."""
    coefs_path = data_dir / "pando" / "coefs.tsv"
    if not coefs_path.exists():
        return None
    try:
        hg = hgx.load_grn_from_csv(
            coefs_path,
            tf_col="tf",
            target_col="target",
            weight_col="estimate",
        )
        print(f"  Loaded Pando GRN from {coefs_path}")
        print(f"  Incidence shape: {hg.incidence.shape}")
        return hg
    except Exception as e:
        print(f"  Could not load Pando data: {e}")
        return None


def generate_synthetic_regulome(*, key):
    """Generate synthetic cerebral organoid regulome data.

    Creates a hierarchical GRN with 200 genes (20 TFs, 180 targets),
    15 regulatory modules, and 8-dimensional feature vectors.
    """
    seed_val = int(jax.random.randint(key, (), 0, 2**30))
    rng = np.random.RandomState(seed_val)

    # --- Build regulatory modules (hyperedges) ---
    incidence = np.zeros((NUM_GENES, NUM_MODULES), dtype=np.float32)
    module_labels = np.full(NUM_GENES, -1, dtype=np.int32)

    target_pool = list(range(NUM_TFS, NUM_GENES))
    rng.shuffle(target_pool)

    idx = 0
    for m in range(NUM_MODULES):
        if m < NUM_MASTER:
            tf, n_targets = m, 25
        elif m < NUM_MASTER + NUM_INTERMEDIATE:
            tf, n_targets = m, 15
        else:
            tf = (m - NUM_MASTER - NUM_INTERMEDIATE) + 10
            n_targets = 10

        incidence[tf, m] = 1.0
        module_labels[tf] = m

        end = min(idx + n_targets, len(target_pool))
        for t_idx in range(idx, end):
            g = target_pool[t_idx]
            incidence[g, m] = 1.0
            if module_labels[g] == -1:
                module_labels[g] = m
        idx = end

    # Assign remaining unassigned genes
    for g in range(NUM_GENES):
        if module_labels[g] == -1:
            m = rng.randint(NUM_MODULES)
            incidence[g, m] = 1.0
            module_labels[g] = m

    # Hierarchical cross-module connections
    for m_tf in range(NUM_MASTER):
        for int_mod in range(NUM_MASTER, NUM_MASTER + 3):
            incidence[m_tf, int_mod] = 1.0
    for i_tf in range(NUM_MASTER, NUM_MASTER + NUM_INTERMEDIATE):
        for d_mod in range(NUM_MASTER + NUM_INTERMEDIATE, NUM_MODULES):
            if rng.random() < 0.3:
                incidence[i_tf, d_mod] = 1.0

    # Gene-to-fate mapping based on module
    module_to_fate = np.array(
        [min(m * NUM_FATES // NUM_MODULES, NUM_FATES - 1) for m in range(NUM_MODULES)]
    )
    gene_fates = module_to_fate[module_labels]

    # Node features: fate-biased random features
    k1, k2 = jax.random.split(key)
    base_expr = jax.random.normal(k1, (NUM_GENES, FEATURE_DIM)) * 0.3
    fate_sigs = jnp.stack(
        [
            jax.random.normal(jax.random.fold_in(k2, f), (FEATURE_DIM,))
            for f in range(NUM_FATES)
        ]
    )
    features = base_expr + fate_sigs[jnp.array(gene_fates)] * 0.8

    return {
        "incidence": jnp.array(incidence),
        "features": features,
        "module_labels": jnp.array(module_labels),
    }


# ---------------------------------------------------------------------------
# Macro F1 computation
# ---------------------------------------------------------------------------


def compute_macro_f1(preds, labels, num_classes):
    """Compute macro-averaged F1 score."""
    f1s = []
    for c in range(num_classes):
        tp = float(jnp.sum((preds == c) & (labels == c)))
        fp = float(jnp.sum((preds == c) & (labels != c)))
        fn = float(jnp.sum((preds != c) & (labels == c)))
        prec = tp / max(tp + fp, 1e-8)
        rec = tp / max(tp + fn, 1e-8)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-8))
    return float(np.mean(f1s))


# ---------------------------------------------------------------------------
# Stratified train / val / test split
# ---------------------------------------------------------------------------


def make_split(
    n: int,
    labels: np.ndarray,
    seed: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for train/val/test.

    Uses stratified splitting: for each class, allocate 60/20/20 of its
    members.  Falls back gracefully for classes with < 3 samples.
    """
    rng = np.random.RandomState(seed)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    unique_labels = np.unique(labels)
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        n_lab = len(idx)
        n_train = max(1, int(n_lab * train_frac))
        n_val = max(0, int(n_lab * val_frac))
        # Ensure at least one test sample when possible
        if n_lab > 2:
            n_val = min(n_val, n_lab - n_train - 1)
            n_val = max(0, n_val)
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train : n_train + n_val]] = True
        test_mask[idx[n_train + n_val :]] = True

    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_hgnn_model(conv_name, incidence, *, key):
    """Build an HGNNStack model for the given convolution type.

    Returns (model, is_sheaf) where is_sheaf indicates whether the model
    uses SheafDiffusion (which requires a different forward pass).
    """
    k1, k2 = jax.random.split(key)

    if conv_name == "SheafDiffusion":
        # SheafDiffusion preserves dimension (output = in_dim), so we
        # use it as a feature extractor followed by a linear readout.
        nnz = int(jnp.sum(incidence > 0))
        sheaf = hgx.SheafDiffusion(
            num_steps=3,
            in_dim=FEATURE_DIM,
            edge_stalk_dim=FEATURE_DIM,
            num_incidences=nnz,
            key=k1,
        )
        readout = eqx.nn.Linear(FEATURE_DIM, NUM_MODULES, key=k2)
        return (sheaf, readout), True

    conv_kwargs = {}
    if conv_name == "THNNConv":
        conv_cls = hgx.THNNConv
        conv_kwargs = {"rank": 32}
    elif conv_name == "UniGCNConv":
        conv_cls = hgx.UniGCNConv
    elif conv_name == "UniGATConv":
        conv_cls = hgx.UniGATConv
    elif conv_name == "UniGINConv":
        conv_cls = hgx.UniGINConv
    else:
        raise ValueError(f"Unknown conv: {conv_name}")

    model = hgx.HGNNStack(
        conv_dims=[(FEATURE_DIM, 64), (64, 32)],
        conv_cls=conv_cls,
        readout_dim=NUM_MODULES,
        activation=jax.nn.relu,
        dropout_rate=0.0,
        conv_kwargs=conv_kwargs if conv_kwargs else None,
        key=k1,
    )
    return model, False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(model, is_sheaf, hg, labels, epochs, masks, *, key, patience=20):
    """Train a model with cross-entropy loss on train split.

    Uses val split for early stopping and reports final metrics on test split.

    Parameters
    ----------
    masks : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Boolean arrays (train_mask, val_mask, test_mask).
    patience : int
        Number of epochs without val loss improvement before early stopping.

    Returns
    -------
    model, losses, val_losses, test_macro_f1, elapsed
    """
    train_mask, val_mask, test_mask = masks
    optimizer = optax.adam(3e-3)

    if is_sheaf:
        sheaf, readout = model

        # Flatten for optax: use the combined model as a single pytree
        class _SheafModel(eqx.Module):
            sheaf: hgx.SheafDiffusion
            readout: eqx.nn.Linear

            def __call__(self, hg):
                x = self.sheaf(hg)
                return jax.vmap(self.readout)(x)

        model = _SheafModel(sheaf=sheaf, readout=readout)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    if is_sheaf:

        def loss_fn(m, hg, labels, mask):
            logits = m(hg)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            one_hot = jax.nn.one_hot(labels, num_classes=NUM_MODULES)
            per_node = -jnp.sum(one_hot * log_probs, axis=-1)
            return jnp.sum(per_node * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    else:

        def loss_fn(m, hg, labels, mask):
            logits = m(hg, inference=True)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            one_hot = jax.nn.one_hot(labels, num_classes=NUM_MODULES)
            per_node = -jnp.sum(one_hot * log_probs, axis=-1)
            return jnp.sum(per_node * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    @eqx.filter_jit
    def step(model, opt_state, hg, labels, mask):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, hg, labels, mask)
        updates, new_opt = optimizer.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), new_opt, loss

    @eqx.filter_jit
    def eval_loss(model, hg, labels, mask):
        return loss_fn(model, hg, labels, mask)

    jnp_train = jnp.array(train_mask, dtype=jnp.float32)
    jnp_val = jnp.array(val_mask, dtype=jnp.float32)

    losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model = model
    epochs_no_improve = 0

    t_start = time.time()
    for epoch in range(epochs):
        model, opt_state, loss = step(model, opt_state, hg, labels, jnp_train)
        losses.append(float(loss))

        # Evaluate on validation set
        v_loss = float(eval_loss(model, hg, labels, jnp_val))
        val_losses.append(v_loss)

        # Early stopping / model selection on val loss
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_model = model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % max(1, epochs // 5) == 0:
            if is_sheaf:
                preds = jnp.argmax(model(hg), axis=-1)
            else:
                preds = jnp.argmax(model(hg, inference=True), axis=-1)
            train_acc = float(
                jnp.sum((preds == labels) * jnp_train) / jnp.maximum(jnp.sum(jnp_train), 1.0)
            )
            val_acc = float(
                jnp.sum((preds == labels) * jnp_val) / jnp.maximum(jnp.sum(jnp_val), 1.0)
            )
            print(
                f"    Epoch {epoch+1:4d}  train_loss={loss:.4f}"
                f"  val_loss={v_loss:.4f}  train_acc={train_acc:.1%}"
                f"  val_acc={val_acc:.1%}"
            )

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1} (patience={patience})")
            break

    elapsed = time.time() - t_start

    # Final predictions on best model, evaluated on test set
    if is_sheaf:
        preds = jnp.argmax(best_model(hg), axis=-1)
    else:
        preds = jnp.argmax(best_model(hg, inference=True), axis=-1)

    test_preds = preds[test_mask]
    test_labels = labels[test_mask]
    test_macro_f1 = compute_macro_f1(test_preds, test_labels, NUM_MODULES)

    return best_model, losses, val_losses, test_macro_f1, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


CONV_NAMES = ["UniGCNConv", "UniGATConv", "UniGINConv", "THNNConv", "SheafDiffusion"]
CONV_DISPLAY = {
    "UniGCNConv": "UniGCN\n(1st-order)",
    "UniGATConv": "UniGAT\n(attention)",
    "UniGINConv": "UniGIN\n(expressive)",
    "THNNConv": "THNN\n(higher-order)",
    "SheafDiffusion": "Sheaf\n(diffusion)",
}
CONV_COLORS = {
    "UniGCNConv": "#7fbf7f",
    "UniGATConv": "#4393c3",
    "UniGINConv": "#b2abd2",
    "THNNConv": "#d6604d",
    "SheafDiffusion": "#fdb863",
}


def main():
    parser = argparse.ArgumentParser(
        description="Higher-order regulatory interaction comparison (Figure 2C-D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        default=100,
        help="Training epochs per model (default: 100)",
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
    print("  Higher-Order Regulatory Interactions")
    print("  Convolution Layer Comparison (Figure 2C-D)")
    print("=" * 64)

    key = jax.random.PRNGKey(args.seed)

    # --- Load or generate data ---
    print("\nPreparing data...")
    pando_hg = try_load_pando(data_dir)

    if pando_hg is not None:
        incidence = pando_hg.incidence
        features = pando_hg.node_features
        n_genes = features.shape[0]
        n_modules = incidence.shape[1]
        # If loaded features are low-dim (e.g. degree vectors), expand to FEATURE_DIM
        if features.shape[1] < FEATURE_DIM:
            k_feat, key = jax.random.split(key)
            # Use incidence structure + noise as richer features
            inc_feats = incidence[:, :min(FEATURE_DIM, incidence.shape[1])]
            if inc_feats.shape[1] < FEATURE_DIM:
                pad = jax.random.normal(k_feat, (n_genes, FEATURE_DIM - inc_feats.shape[1])) * 0.1
                features = jnp.concatenate([inc_feats, pad], axis=1)
            else:
                features = inc_feats[:, :FEATURE_DIM]
            print(f"  Expanded features from {pando_hg.node_features.shape[1]} to {FEATURE_DIM} dims")
        # Use simple module assignment: argmax of incidence row
        module_labels = jnp.argmax(incidence, axis=1)
        # Update globals to match loaded data
        global NUM_GENES, NUM_MODULES
        NUM_GENES = n_genes
        NUM_MODULES = n_modules
        print(f"  {n_genes} genes, {n_modules} modules from Pando data")
    else:
        print("  Pando data not found; generating synthetic regulome...")
        k_data, key = jax.random.split(key)
        data = generate_synthetic_regulome(key=k_data)
        incidence = data["incidence"]
        features = data["features"]
        module_labels = data["module_labels"]
        print(
            f"  {NUM_GENES} genes ({NUM_TFS} TFs), "
            f"{NUM_MODULES} modules, {FEATURE_DIM} features"
        )

    hg = hgx.from_incidence(incidence, node_features=features)
    print(f"  Incidence shape: {incidence.shape}")
    print(f"  Feature shape:   {features.shape}")

    # --- Stratified train / val / test split ---
    n_nodes = int(features.shape[0])
    labels_np = np.array(module_labels)
    train_mask, val_mask, test_mask = make_split(n_nodes, labels_np, args.seed)
    masks = (train_mask, val_mask, test_mask)
    print(
        f"  Split: {train_mask.sum()} train, {val_mask.sum()} val, "
        f"{test_mask.sum()} test"
    )

    # --- Train each convolution model ---
    results = {}
    for conv_name in CONV_NAMES:
        print(f"\n{'─' * 48}")
        print(f"  Training: {conv_name}")
        print(f"{'─' * 48}")

        k_model, key = jax.random.split(key)
        model, is_sheaf = build_hgnn_model(conv_name, incidence, key=k_model)
        _, losses, val_losses, macro_f1, elapsed = train_model(
            model, is_sheaf, hg, module_labels, args.epochs, masks, key=k_model
        )
        results[conv_name] = {
            "losses": losses,
            "val_losses": val_losses,
            "macro_f1": macro_f1,
            "time": elapsed,
        }
        print(f"  Test Macro F1: {macro_f1:.3f}  |  Time: {elapsed:.1f}s")

    # --- Quantify higher-order benefit ---
    baseline_f1 = results["UniGCNConv"]["macro_f1"]
    thnn_f1 = results["THNNConv"]["macro_f1"]
    ho_gap = thnn_f1 - baseline_f1

    print("\n" + "=" * 64)
    print("  Summary  (test-set metrics)")
    print("=" * 64)
    for name in CONV_NAMES:
        r = results[name]
        gap = r["macro_f1"] - baseline_f1
        marker = " ***" if name == "THNNConv" else ""
        print(
            f"  {name:20s}  Test F1={r['macro_f1']:.3f}"
            f"  (delta={gap:+.3f})  {r['time']:.1f}s{marker}"
        )

    print(f"\n  Higher-order benefit (THNNConv - UniGCNConv): {ho_gap:+.3f} test F1")

    # --- Figure 2C: Bar chart of macro F1 per convolution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x_pos = np.arange(len(CONV_NAMES))
    f1_vals = [results[n]["macro_f1"] for n in CONV_NAMES]
    colors = [CONV_COLORS[n] for n in CONV_NAMES]
    bars = ax.bar(x_pos, f1_vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([CONV_DISPLAY[n] for n in CONV_NAMES], fontsize=9)
    ax.set_ylabel("Test Macro F1", fontsize=12)
    ax.set_title("C. Module Detection: Convolution Comparison", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate bars with values
    for bar, val in zip(bars, f1_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Mark the higher-order benefit gap
    if ho_gap > 0:
        ax.annotate(
            f"HO benefit\n+{ho_gap:.2f}",
            xy=(0, baseline_f1),
            xytext=(0.5, baseline_f1 + 0.15),
            fontsize=9,
            color="firebrick",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.5),
        )

    # --- Figure 2D: Training & validation loss curves ---
    ax = axes[1]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for i, conv_name in enumerate(CONV_NAMES):
        losses = results[conv_name]["losses"]
        val_losses = results[conv_name]["val_losses"]
        ax.plot(
            losses,
            label=f"{conv_name} (train)",
            color=CONV_COLORS[conv_name],
            linestyle=linestyles[i],
            linewidth=1.8,
        )
        ax.plot(
            val_losses,
            label=f"{conv_name} (val)",
            color=CONV_COLORS[conv_name],
            linestyle=linestyles[i],
            linewidth=1.0,
            alpha=0.5,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss (log scale)", fontsize=12)
    ax.set_title("D. Training & Validation Convergence", fontsize=13)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = fig_dir / "figure_02cd_conv_comparison.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
