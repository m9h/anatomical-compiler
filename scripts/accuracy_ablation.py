#!/usr/bin/env python3
"""accuracy_ablation.py — Diagnose and fix the hgx accuracy gap vs HyperGCN.

The benchmark shows:
    HyperGCN (DHG/PyTorch): 37.8% accuracy
    hgx UniGATConv:         18.4% accuracy
    hgx UniGCNConv:         16.6% accuracy

on a 720-class node classification task (2,792 nodes, 720 TF regulon
hyperedges, 97-dim PCA features).

This script:
    1. Diagnoses the problem (class distribution, feature stats, gradient norms)
    2. Runs ablations on hgx UniGATConv (lr, hidden_dim, num_layers, dropout)
    3. Tries alternative classification tasks (spectral clusters, TF-vs-target,
       lineage assignment)
    4. Compares hgx convolutions (UniGCNConv, UniGATConv, UniGINConv) per task
    5. Outputs a results table and figure to figures/accuracy_ablation.png

Usage:
    uv run python scripts/accuracy_ablation.py
    uv run python scripts/accuracy_ablation.py --data-dir data/processed --epochs 300
    uv run python scripts/accuracy_ablation.py --seed 0
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

try:
    import hgx
except ImportError:
    sys.exit("ERROR: hgx not installed. Run: uv pip install -e ../hgx")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONV_MAP = {
    "UniGCNConv": hgx.UniGCNConv,
    "UniGATConv": hgx.UniGATConv,
    "UniGINConv": hgx.UniGINConv,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def detect_data_dir() -> Path:
    """Auto-detect data/processed directory."""
    for candidate in [
        Path("/workspace/benchmark/data/processed"),
        PROJECT_ROOT / "data" / "processed",
    ]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find data/processed directory. Pass --data-dir explicitly."
    )


def load_data(data_dir: Path) -> dict:
    """Load preprocessed numpy arrays and JSON metadata."""
    print(f"Loading data from {data_dir}")
    d = {}

    # Numpy arrays
    for name in [
        "incidence", "node_features_pca", "module_labels",
        "temporal_expression", "lineage_fractions", "fate_probabilities",
    ]:
        path = data_dir / f"{name}.npy"
        if path.exists():
            d[name] = np.load(path)
            print(f"  {name}: {d[name].shape} {d[name].dtype}")
        else:
            print(f"  WARNING: {name}.npy not found")
            d[name] = None

    # JSON files
    for name in ["gene_names", "tf_names"]:
        path = data_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                d[name] = json.load(f)
            print(f"  {name}: {len(d[name])} entries")
        else:
            d[name] = None

    return d


# ---------------------------------------------------------------------------
# Train/val/test split (stratified when possible)
# ---------------------------------------------------------------------------

def make_split(
    n: int,
    labels: np.ndarray,
    seed: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for train/val/test.

    Uses stratified splitting when feasible: for each class, allocate
    60/20/20 of its members.  Falls back to random shuffle for classes
    with < 3 samples.
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
# Part 1: Diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(data: dict) -> None:
    """Print class distribution, feature statistics, and label balance info."""
    print("\n" + "=" * 72)
    print("  PART 1: DIAGNOSTICS")
    print("=" * 72)

    features = data["node_features_pca"]
    labels = data["module_labels"]
    incidence = data["incidence"]

    if features is None or labels is None or incidence is None:
        print("  SKIPPED: required data arrays not found.")
        return

    n_nodes, feat_dim = features.shape
    n_edges = incidence.shape[1]

    # --- Feature statistics ---
    print(f"\n  Feature matrix: {n_nodes} nodes x {feat_dim} dims")
    print(f"    mean:  {features.mean():.6f}")
    print(f"    std:   {features.std():.6f}")
    print(f"    min:   {features.min():.6f}")
    print(f"    max:   {features.max():.6f}")
    per_node_norm = np.linalg.norm(features, axis=1)
    print(f"    per-node L2 norm: mean={per_node_norm.mean():.4f}, "
          f"std={per_node_norm.std():.4f}, "
          f"min={per_node_norm.min():.4f}, max={per_node_norm.max():.4f}")

    # Fraction of zero features
    zero_frac = (features == 0).mean()
    print(f"    fraction zero entries: {zero_frac:.4f}")

    # --- Class distribution ---
    unique, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique)
    print(f"\n  Labels: {num_classes} classes")
    print(f"    samples/class: mean={counts.mean():.1f}, "
          f"median={np.median(counts):.1f}, "
          f"min={counts.min()}, max={counts.max()}")

    # Show distribution histogram
    bins_hist = [1, 2, 3, 4, 5, 10, 20, 50, 100, 500]
    print("    class-size histogram:")
    for i in range(len(bins_hist)):
        lo = bins_hist[i - 1] if i > 0 else 0
        hi = bins_hist[i]
        n_in_range = np.sum((counts >= lo) & (counts < hi))
        if n_in_range > 0:
            print(f"      [{lo:>4d}, {hi:>4d}): {n_in_range} classes")
    n_large = np.sum(counts >= bins_hist[-1])
    if n_large > 0:
        print(f"      [{bins_hist[-1]:>4d},  inf): {n_large} classes")

    # Classes with only 1 sample (impossible to train+test)
    n_singleton = np.sum(counts == 1)
    n_doubleton = np.sum(counts == 2)
    print(f"    singletons (1 sample):  {n_singleton} classes")
    print(f"    doubletons (2 samples): {n_doubleton} classes")

    # --- Label balance ---
    # Effective number of classes (entropy-based)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    effective_classes = np.exp(entropy)
    print(f"\n    Shannon entropy: {entropy:.2f}")
    print(f"    Effective num classes (exp(H)): {effective_classes:.1f}")
    print(f"    Actual num classes: {num_classes}")
    print(f"    Imbalance ratio (max/min): {counts.max()/max(counts.min(),1):.1f}")

    # Random-guess baseline
    random_acc = 1.0 / num_classes
    majority_acc = counts.max() / n_nodes
    print(f"\n    Random-guess accuracy: {random_acc:.4f} ({random_acc*100:.2f}%)")
    print(f"    Majority-class accuracy: {majority_acc:.4f} ({majority_acc*100:.2f}%)")

    # --- Incidence structure ---
    row_sums = incidence.sum(axis=1)
    col_sums = incidence.sum(axis=0)
    print(f"\n  Incidence matrix: {incidence.shape}")
    print(f"    genes per regulon: mean={col_sums.mean():.1f}, "
          f"median={np.median(col_sums):.1f}, "
          f"min={col_sums.min():.0f}, max={col_sums.max():.0f}")
    print(f"    regulons per gene: mean={row_sums.mean():.2f}, "
          f"median={np.median(row_sums):.1f}, "
          f"min={row_sums.min():.0f}, max={row_sums.max():.0f}")
    n_unassigned = np.sum(row_sums == 0)
    print(f"    unassigned genes (row sum = 0): {n_unassigned}")


# ---------------------------------------------------------------------------
# Part 1b: Gradient diagnostics during a short training run
# ---------------------------------------------------------------------------

def run_gradient_diagnostics(
    data: dict, seed: int, epochs: int = 50
) -> None:
    """Train UniGATConv for a few epochs and print gradient norms."""
    print("\n  --- Gradient Diagnostics (UniGATConv, 50 epochs) ---")

    features = data["node_features_pca"]
    labels_np = data["module_labels"]
    incidence_np = data["incidence"]

    if features is None or labels_np is None or incidence_np is None:
        print("  SKIPPED: data not available.")
        return

    in_dim = features.shape[1]
    num_classes = int(labels_np.max()) + 1

    incidence_jnp = jnp.array(incidence_np)
    features_jnp = jnp.array(features)
    labels_jnp = jnp.array(labels_np)

    hg = hgx.from_incidence(incidence_jnp, node_features=features_jnp)

    key = jax.random.PRNGKey(seed)
    model = hgx.HGNNStack(
        conv_dims=[(in_dim, 64), (64, 32)],
        conv_cls=hgx.UniGATConv,
        readout_dim=num_classes,
        activation=jax.nn.relu,
        dropout_rate=0.0,
        key=key,
    )

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def loss_fn(m, hg, labels):
        logits = m(hg, inference=True)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(labels, num_classes=num_classes)
        return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    @eqx.filter_jit
    def step_with_grads(model, opt_state, hg, labels):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, hg, labels)
        updates, new_opt = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt, loss, grads

    print(f"    {'Epoch':>6s}  {'Loss':>10s}  {'Grad L2':>12s}  "
          f"{'Grad Max':>12s}  {'Acc':>8s}")
    print("    " + "-" * 56)

    for epoch in range(min(epochs, 50)):
        model, opt_state, loss, grads = step_with_grads(
            model, opt_state, hg, labels_jnp
        )

        # Compute gradient statistics across all parameters
        leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
        all_grads_flat = jnp.concatenate([l.ravel() for l in leaves])
        grad_l2 = float(jnp.linalg.norm(all_grads_flat))
        grad_max = float(jnp.max(jnp.abs(all_grads_flat)))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            preds = jnp.argmax(model(hg, inference=True), axis=-1)
            acc = float(jnp.mean(preds == labels_jnp))
            print(f"    {epoch+1:6d}  {float(loss):10.4f}  {grad_l2:12.6f}  "
                  f"{grad_max:12.6f}  {acc:8.4f}")

    # Check for vanishing/exploding gradients
    if grad_l2 < 1e-6:
        print("    WARNING: Gradients appear to be vanishing!")
    elif grad_l2 > 100:
        print("    WARNING: Gradients appear to be exploding!")
    else:
        print(f"    Gradient norms look reasonable (L2={grad_l2:.6f}).")


# ---------------------------------------------------------------------------
# Build alternative classification labels
# ---------------------------------------------------------------------------

def build_task_labels(data: dict, seed: int) -> dict[str, tuple[np.ndarray, int, str]]:
    """Build label arrays for different classification tasks.

    Returns dict mapping task_name -> (labels, num_classes, description).
    """
    tasks = {}
    incidence = data["incidence"]
    module_labels = data["module_labels"]
    n_nodes = incidence.shape[0]

    # --- Task 1: 720 regulons (original) ---
    num_classes_720 = int(module_labels.max()) + 1
    tasks["720_regulons"] = (
        module_labels.copy(),
        num_classes_720,
        f"720-class regulon assignment (~{n_nodes/num_classes_720:.1f} samples/class)",
    )

    # --- Task 2: 20 spectral clusters ---
    # Compute hypergraph Laplacian eigenvectors and K-means into 20 clusters
    n_clusters = 20
    try:
        # Normalized Laplacian from incidence: L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
        H = incidence.astype(np.float64)
        d_v = H.sum(axis=1)  # node degree
        d_e = H.sum(axis=0)  # edge degree
        d_v_inv_sqrt = np.where(d_v > 0, 1.0 / np.sqrt(d_v), 0.0)
        d_e_inv = np.where(d_e > 0, 1.0 / d_e, 0.0)

        # L = I - D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
        D_v_isqrt = np.diag(d_v_inv_sqrt)
        D_e_inv = np.diag(d_e_inv)
        Theta = D_v_isqrt @ H @ D_e_inv @ H.T @ D_v_isqrt
        L = np.eye(n_nodes) - Theta

        # Smallest eigenvectors (excluding trivial)
        eigvals, eigvecs = np.linalg.eigh(L)
        spectral_feats = eigvecs[:, 1 : n_clusters + 1]  # skip Fiedler-0

        # K-means
        from scipy.cluster.vq import kmeans2
        _, spectral_labels = kmeans2(
            spectral_feats.astype(np.float64), n_clusters,
            minit="++", seed=seed,
        )
        spectral_labels = spectral_labels.astype(np.int32)
        tasks["20_spectral"] = (
            spectral_labels,
            n_clusters,
            f"20-class spectral clustering (~{n_nodes/n_clusters:.0f} samples/class)",
        )
    except Exception as e:
        print(f"  WARNING: spectral clustering failed: {e}")

    # --- Task 3: Binary TF vs target ---
    # TFs are genes that appear as a TF in at least one regulon (genes where
    # the diagonal-like pattern holds: gene i is in regulon i).
    # Simple heuristic: a gene is a TF if it is listed in tf_names.
    tf_names = data.get("tf_names")
    gene_names = data.get("gene_names")
    if tf_names is not None and gene_names is not None:
        tf_set = set(tf_names)
        binary_labels = np.array(
            [0 if g in tf_set else 1 for g in gene_names], dtype=np.int32
        )
        n_tfs = int((binary_labels == 0).sum())
        n_targets = int((binary_labels == 1).sum())
        tasks["binary_tf_target"] = (
            binary_labels,
            2,
            f"Binary TF vs target ({n_tfs} TFs, {n_targets} targets)",
        )
    else:
        # Fallback: genes in many regulons = TFs, genes in few = targets
        row_sums = incidence.sum(axis=1)
        median_membership = np.median(row_sums[row_sums > 0])
        binary_labels = np.where(row_sums > median_membership, 0, 1).astype(np.int32)
        tasks["binary_tf_target"] = (
            binary_labels,
            2,
            f"Binary high-connectivity vs low-connectivity",
        )

    # --- Task 4: Lineage (3 classes) ---
    # Assign each gene to its dominant lineage fate based on temporal
    # expression correlation with lineage fractions.
    lineage_fracs = data.get("lineage_fractions")  # (T, 3)
    temporal_expr = data.get("temporal_expression")  # (T, n_genes)

    if lineage_fracs is not None and temporal_expr is not None:
        try:
            T, n_genes_temp = temporal_expr.shape
            assert n_genes_temp == n_nodes, (
                f"temporal_expression has {n_genes_temp} genes but incidence "
                f"has {n_nodes} nodes"
            )

            # Correlation of each gene's temporal profile with each fate
            # lineage_fracs: (T, 3), temporal_expr: (T, n_genes)
            lineage_labels = np.zeros(n_nodes, dtype=np.int32)
            for gi in range(n_nodes):
                gene_profile = temporal_expr[:, gi]
                gene_std = gene_profile.std()
                if gene_std < 1e-8:
                    # Flat expression -- assign to class 0 as default
                    lineage_labels[gi] = 0
                    continue
                corrs = []
                for fi in range(3):
                    fate_profile = lineage_fracs[:, fi]
                    fate_std = fate_profile.std()
                    if fate_std < 1e-8:
                        corrs.append(0.0)
                    else:
                        r = np.corrcoef(gene_profile, fate_profile)[0, 1]
                        corrs.append(r if np.isfinite(r) else 0.0)
                lineage_labels[gi] = int(np.argmax(corrs))

            unique_lin, counts_lin = np.unique(lineage_labels, return_counts=True)
            desc = ", ".join(f"class{l}={c}" for l, c in zip(unique_lin, counts_lin))
            tasks["lineage_3class"] = (
                lineage_labels,
                3,
                f"3-class lineage assignment ({desc})",
            )
        except Exception as e:
            print(f"  WARNING: lineage label construction failed: {e}")
    else:
        print("  WARNING: temporal data not available for lineage task.")

    return tasks


# ---------------------------------------------------------------------------
# Build model with configurable architecture
# ---------------------------------------------------------------------------

def build_model(
    conv_name: str,
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_classes: int,
    dropout: float,
    key: jax.Array,
):
    """Build an hgx HGNNStack with the given hyperparameters."""
    conv_cls = CONV_MAP[conv_name]

    # Build conv_dims list: (in_dim, hidden) -> (hidden, hidden) x (num_layers-1)
    dims = [(in_dim, hidden_dim)]
    for _ in range(num_layers - 1):
        dims.append((hidden_dim, hidden_dim))

    model = hgx.HGNNStack(
        conv_dims=dims,
        conv_cls=conv_cls,
        readout_dim=num_classes,
        activation=jax.nn.relu,
        dropout_rate=dropout,
        key=key,
    )
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(
    model,
    hg,
    labels: jnp.ndarray,
    train_mask: jnp.ndarray,
    val_mask: jnp.ndarray,
    test_mask: jnp.ndarray,
    num_classes: int,
    lr: float,
    epochs: int,
    verbose: bool = False,
) -> dict:
    """Train an hgx model and return train/val/test metrics.

    Returns dict with keys: test_acc, val_acc, train_acc, final_loss,
                            train_time, best_epoch
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def loss_fn(m, hg, labels, mask):
        logits = m(hg, inference=True)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(labels, num_classes=num_classes)
        per_node = -jnp.sum(one_hot * log_probs, axis=-1)
        # Mean only over masked (training) nodes
        return jnp.sum(jnp.where(mask, per_node, 0.0)) / jnp.maximum(
            jnp.sum(mask), 1.0
        )

    @eqx.filter_jit
    def step(model, opt_state, hg, labels, mask):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, hg, labels, mask)
        updates, new_opt = optimizer.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), new_opt, loss

    @eqx.filter_jit
    def predict(model, hg):
        return model(hg, inference=True)

    best_val_acc = -1.0
    best_model = model
    best_epoch = 0

    t_start = time.perf_counter()
    for epoch in range(epochs):
        model, opt_state, loss = step(model, opt_state, hg, labels, train_mask)

        if (epoch + 1) % max(1, epochs // 5) == 0:
            logits = predict(model, hg)
            preds = jnp.argmax(logits, axis=-1)
            val_correct = jnp.sum(jnp.where(val_mask, preds == labels, False))
            val_total = jnp.sum(val_mask)
            val_acc = float(val_correct / jnp.maximum(val_total, 1.0))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                best_epoch = epoch + 1
            if verbose:
                print(f"      Epoch {epoch+1:4d}  loss={float(loss):.4f}  "
                      f"val_acc={val_acc:.4f}")

    jax.block_until_ready(eqx.filter(model, eqx.is_array))
    train_time = time.perf_counter() - t_start

    # Evaluate best model on all splits
    logits = predict(best_model, hg)
    preds = jnp.argmax(logits, axis=-1)

    def acc_on_mask(mask):
        total = jnp.sum(mask)
        if float(total) == 0:
            return 0.0
        correct = jnp.sum(jnp.where(mask, preds == labels, False))
        return float(correct / total)

    return {
        "test_acc": acc_on_mask(test_mask),
        "val_acc": acc_on_mask(val_mask),
        "train_acc": acc_on_mask(train_mask),
        "final_loss": float(loss),
        "train_time": train_time,
        "best_epoch": best_epoch,
    }


# ---------------------------------------------------------------------------
# Part 2: hgx UniGATConv ablation
# ---------------------------------------------------------------------------

def run_ablation(
    data: dict, seed: int, epochs: int
) -> list[dict]:
    """Run hyperparameter ablation on UniGATConv for the 720-regulon task."""
    print("\n" + "=" * 72)
    print("  PART 2: hgx UniGATConv ABLATION (720-regulon task)")
    print("=" * 72)

    features = data["node_features_pca"]
    labels_np = data["module_labels"]
    incidence_np = data["incidence"]

    if features is None or labels_np is None or incidence_np is None:
        print("  SKIPPED: data not available.")
        return []

    in_dim = features.shape[1]
    n_nodes = features.shape[0]
    num_classes = int(labels_np.max()) + 1

    # Build hypergraph
    incidence_jnp = jnp.array(incidence_np)
    features_jnp = jnp.array(features)
    labels_jnp = jnp.array(labels_np)
    hg = hgx.from_incidence(incidence_jnp, node_features=features_jnp)

    # Stratified split
    train_mask_np, val_mask_np, test_mask_np = make_split(
        n_nodes, labels_np, seed
    )
    train_mask = jnp.array(train_mask_np)
    val_mask = jnp.array(val_mask_np)
    test_mask = jnp.array(test_mask_np)

    # Ablation grid
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    hidden_dims = [32, 64, 128, 256]
    num_layers_list = [1, 2, 3, 4]
    dropouts = [0.0, 0.1, 0.3, 0.5]

    results = []
    base_key = jax.random.PRNGKey(seed)

    # --- Sweep learning rate (hold others at defaults) ---
    print("\n  --- Learning Rate Sweep ---")
    print(f"  (hidden=64, layers=2, dropout=0.0)")
    for lr in learning_rates:
        key = jax.random.fold_in(base_key, abs(hash(("lr", lr))) % (2**31))
        model = build_model("UniGATConv", in_dim, 64, 2, num_classes, 0.0, key)
        metrics = train_and_evaluate(
            model, hg, labels_jnp, train_mask, val_mask, test_mask,
            num_classes, lr, epochs,
        )
        row = {"ablation": "lr", "lr": lr, "hidden": 64, "layers": 2,
               "dropout": 0.0, **metrics}
        results.append(row)
        print(f"    lr={lr:.0e}  test_acc={metrics['test_acc']:.4f}  "
              f"val_acc={metrics['val_acc']:.4f}  time={metrics['train_time']:.1f}s")

    # --- Sweep hidden dim ---
    print("\n  --- Hidden Dim Sweep ---")
    print(f"  (lr=1e-3, layers=2, dropout=0.0)")
    for hd in hidden_dims:
        key = jax.random.fold_in(base_key, abs(hash(("hd", hd))) % (2**31))
        model = build_model("UniGATConv", in_dim, hd, 2, num_classes, 0.0, key)
        metrics = train_and_evaluate(
            model, hg, labels_jnp, train_mask, val_mask, test_mask,
            num_classes, 1e-3, epochs,
        )
        row = {"ablation": "hidden", "lr": 1e-3, "hidden": hd, "layers": 2,
               "dropout": 0.0, **metrics}
        results.append(row)
        print(f"    hidden={hd:4d}  test_acc={metrics['test_acc']:.4f}  "
              f"val_acc={metrics['val_acc']:.4f}  time={metrics['train_time']:.1f}s")

    # --- Sweep num layers ---
    print("\n  --- Num Layers Sweep ---")
    print(f"  (lr=1e-3, hidden=64, dropout=0.0)")
    for nl in num_layers_list:
        key = jax.random.fold_in(base_key, abs(hash(("nl", nl))) % (2**31))
        model = build_model("UniGATConv", in_dim, 64, nl, num_classes, 0.0, key)
        metrics = train_and_evaluate(
            model, hg, labels_jnp, train_mask, val_mask, test_mask,
            num_classes, 1e-3, epochs,
        )
        row = {"ablation": "layers", "lr": 1e-3, "hidden": 64, "layers": nl,
               "dropout": 0.0, **metrics}
        results.append(row)
        print(f"    layers={nl}  test_acc={metrics['test_acc']:.4f}  "
              f"val_acc={metrics['val_acc']:.4f}  time={metrics['train_time']:.1f}s")

    # --- Sweep dropout ---
    print("\n  --- Dropout Sweep ---")
    print(f"  (lr=1e-3, hidden=64, layers=2)")
    for dp in dropouts:
        key = jax.random.fold_in(base_key, abs(hash(("dp", dp))) % (2**31))
        model = build_model("UniGATConv", in_dim, 64, 2, num_classes, dp, key)
        metrics = train_and_evaluate(
            model, hg, labels_jnp, train_mask, val_mask, test_mask,
            num_classes, 1e-3, epochs,
        )
        row = {"ablation": "dropout", "lr": 1e-3, "hidden": 64, "layers": 2,
               "dropout": dp, **metrics}
        results.append(row)
        print(f"    dropout={dp:.1f}  test_acc={metrics['test_acc']:.4f}  "
              f"val_acc={metrics['val_acc']:.4f}  time={metrics['train_time']:.1f}s")

    # Find best config
    if results:
        best = max(results, key=lambda r: r["test_acc"])
        print(f"\n  BEST ablation config:")
        print(f"    lr={best['lr']}, hidden={best['hidden']}, "
              f"layers={best['layers']}, dropout={best['dropout']}")
        print(f"    test_acc={best['test_acc']:.4f}, "
              f"val_acc={best['val_acc']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Part 3 & 4: Alternative tasks x convolution comparison
# ---------------------------------------------------------------------------

def run_task_comparison(
    data: dict, seed: int, epochs: int
) -> list[dict]:
    """Run all tasks x all convolutions and return results."""
    print("\n" + "=" * 72)
    print("  PART 3-4: TASK x CONVOLUTION COMPARISON")
    print("=" * 72)

    features = data["node_features_pca"]
    incidence_np = data["incidence"]

    if features is None or incidence_np is None:
        print("  SKIPPED: data not available.")
        return []

    in_dim = features.shape[1]
    n_nodes = features.shape[0]

    incidence_jnp = jnp.array(incidence_np)
    features_jnp = jnp.array(features)
    hg = hgx.from_incidence(incidence_jnp, node_features=features_jnp)

    # Build all task labels
    tasks = build_task_labels(data, seed)
    print(f"\n  Tasks: {list(tasks.keys())}")
    for tname, (labels, nc, desc) in tasks.items():
        print(f"    {tname}: {desc}")

    conv_names = ["UniGCNConv", "UniGATConv", "UniGINConv"]
    base_key = jax.random.PRNGKey(seed)

    # Hyperparameters: use reasonable defaults (can be refined from ablation)
    lr = 1e-3
    hidden = 128
    n_layers = 2
    dropout = 0.1

    results = []

    for task_name, (labels_np, num_classes, desc) in tasks.items():
        print(f"\n  === Task: {task_name} ({num_classes} classes) ===")

        labels_jnp = jnp.array(labels_np)
        train_mask_np, val_mask_np, test_mask_np = make_split(
            n_nodes, labels_np, seed
        )
        train_mask = jnp.array(train_mask_np)
        val_mask = jnp.array(val_mask_np)
        test_mask = jnp.array(test_mask_np)

        # Random baseline for this task
        random_acc = 1.0 / num_classes
        print(f"    Random baseline: {random_acc:.4f}")

        for conv_name in conv_names:
            key = jax.random.fold_in(
                base_key, hash((task_name, conv_name))) % (2**31)
            )
            model = build_model(
                conv_name, in_dim, hidden, n_layers, num_classes, dropout, key
            )
            metrics = train_and_evaluate(
                model, hg, labels_jnp, train_mask, val_mask, test_mask,
                num_classes, lr, epochs, verbose=False,
            )
            row = {
                "task": task_name,
                "num_classes": num_classes,
                "conv": conv_name,
                "lr": lr,
                "hidden": hidden,
                "layers": n_layers,
                "dropout": dropout,
                **metrics,
            }
            results.append(row)
            print(f"    {conv_name:14s}  test={metrics['test_acc']:.4f}  "
                  f"val={metrics['val_acc']:.4f}  "
                  f"train={metrics['train_acc']:.4f}  "
                  f"time={metrics['train_time']:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Part 5: Output table and figure
# ---------------------------------------------------------------------------

def print_results_table(
    ablation_results: list[dict],
    task_results: list[dict],
) -> None:
    """Print formatted results tables."""
    print("\n" + "=" * 72)
    print("  PART 5: RESULTS SUMMARY")
    print("=" * 72)

    # --- Ablation table ---
    if ablation_results:
        print("\n  --- UniGATConv Ablation (720-regulon task) ---")
        header = (
            f"  {'Sweep':<10s} {'LR':>8s} {'Hidden':>7s} {'Layers':>7s} "
            f"{'Drop':>6s} {'Test':>8s} {'Val':>8s} {'Train':>8s} "
            f"{'Time':>7s}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in ablation_results:
            print(
                f"  {r['ablation']:<10s} {r['lr']:>8.0e} {r['hidden']:>7d} "
                f"{r['layers']:>7d} {r['dropout']:>6.1f} "
                f"{r['test_acc']:>8.4f} {r['val_acc']:>8.4f} "
                f"{r['train_acc']:>8.4f} {r['train_time']:>6.1f}s"
            )

        best = max(ablation_results, key=lambda r: r["test_acc"])
        print(f"\n  Best: lr={best['lr']}, hidden={best['hidden']}, "
              f"layers={best['layers']}, dropout={best['dropout']} "
              f"-> test_acc={best['test_acc']:.4f}")

    # --- Task comparison table ---
    if task_results:
        print("\n  --- Task x Convolution Comparison ---")
        header = (
            f"  {'Task':<20s} {'Classes':>8s} {'Conv':>14s} "
            f"{'Test':>8s} {'Val':>8s} {'Train':>8s} {'Time':>7s}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in task_results:
            print(
                f"  {r['task']:<20s} {r['num_classes']:>8d} "
                f"{r['conv']:>14s} {r['test_acc']:>8.4f} "
                f"{r['val_acc']:>8.4f} {r['train_acc']:>8.4f} "
                f"{r['train_time']:>6.1f}s"
            )

        # Best per task
        print("\n  Best per task:")
        task_names = sorted(set(r["task"] for r in task_results))
        for tname in task_names:
            task_rows = [r for r in task_results if r["task"] == tname]
            best = max(task_rows, key=lambda r: r["test_acc"])
            print(f"    {tname:<20s}  {best['conv']:>14s}  "
                  f"test_acc={best['test_acc']:.4f}")


def save_figure(
    ablation_results: list[dict],
    task_results: list[dict],
    fig_path: Path,
) -> None:
    """Generate and save the ablation summary figure."""
    n_panels = 0
    has_ablation = bool(ablation_results)
    has_tasks = bool(task_results)
    if has_ablation:
        n_panels += 4  # 4 ablation sweep panels
    if has_tasks:
        n_panels += 1  # 1 task comparison panel

    if n_panels == 0:
        print("  No results to plot.")
        return

    fig, axes = plt.subplots(
        2, 3, figsize=(18, 10), constrained_layout=True
    )
    axes = axes.ravel()

    panel_idx = 0
    sweep_colors = {
        "lr": "#e41a1c",
        "hidden": "#377eb8",
        "layers": "#4daf4a",
        "dropout": "#984ea3",
    }

    if has_ablation:
        # --- Panel A: LR sweep ---
        ax = axes[panel_idx]
        lr_rows = [r for r in ablation_results if r["ablation"] == "lr"]
        if lr_rows:
            xs = [r["lr"] for r in lr_rows]
            ys_test = [r["test_acc"] for r in lr_rows]
            ys_val = [r["val_acc"] for r in lr_rows]
            ax.semilogx(xs, ys_test, "o-", color=sweep_colors["lr"],
                        label="Test", linewidth=2, markersize=6)
            ax.semilogx(xs, ys_val, "s--", color=sweep_colors["lr"],
                        alpha=0.6, label="Val", linewidth=1.5, markersize=5)
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Accuracy")
            ax.set_title("A. Learning Rate Sweep", fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        panel_idx += 1

        # --- Panel B: Hidden dim sweep ---
        ax = axes[panel_idx]
        hd_rows = [r for r in ablation_results if r["ablation"] == "hidden"]
        if hd_rows:
            xs = [r["hidden"] for r in hd_rows]
            ys_test = [r["test_acc"] for r in hd_rows]
            ys_val = [r["val_acc"] for r in hd_rows]
            ax.plot(xs, ys_test, "o-", color=sweep_colors["hidden"],
                    label="Test", linewidth=2, markersize=6)
            ax.plot(xs, ys_val, "s--", color=sweep_colors["hidden"],
                    alpha=0.6, label="Val", linewidth=1.5, markersize=5)
            ax.set_xlabel("Hidden Dimension")
            ax.set_ylabel("Accuracy")
            ax.set_title("B. Hidden Dimension Sweep", fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        panel_idx += 1

        # --- Panel C: Layers sweep ---
        ax = axes[panel_idx]
        nl_rows = [r for r in ablation_results if r["ablation"] == "layers"]
        if nl_rows:
            xs = [r["layers"] for r in nl_rows]
            ys_test = [r["test_acc"] for r in nl_rows]
            ys_val = [r["val_acc"] for r in nl_rows]
            ax.plot(xs, ys_test, "o-", color=sweep_colors["layers"],
                    label="Test", linewidth=2, markersize=6)
            ax.plot(xs, ys_val, "s--", color=sweep_colors["layers"],
                    alpha=0.6, label="Val", linewidth=1.5, markersize=5)
            ax.set_xlabel("Number of Layers")
            ax.set_ylabel("Accuracy")
            ax.set_title("C. Depth Sweep", fontweight="bold")
            ax.set_xticks(xs)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        panel_idx += 1

        # --- Panel D: Dropout sweep ---
        ax = axes[panel_idx]
        dp_rows = [r for r in ablation_results if r["ablation"] == "dropout"]
        if dp_rows:
            xs = [r["dropout"] for r in dp_rows]
            ys_test = [r["test_acc"] for r in dp_rows]
            ys_val = [r["val_acc"] for r in dp_rows]
            ax.plot(xs, ys_test, "o-", color=sweep_colors["dropout"],
                    label="Test", linewidth=2, markersize=6)
            ax.plot(xs, ys_val, "s--", color=sweep_colors["dropout"],
                    alpha=0.6, label="Val", linewidth=1.5, markersize=5)
            ax.set_xlabel("Dropout Rate")
            ax.set_ylabel("Accuracy")
            ax.set_title("D. Dropout Sweep", fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        panel_idx += 1

    # --- Panel E: Task x Convolution grouped bar chart ---
    if has_tasks:
        ax = axes[panel_idx]
        task_names = sorted(set(r["task"] for r in task_results))
        conv_names_plot = sorted(set(r["conv"] for r in task_results))
        n_tasks = len(task_names)
        n_convs = len(conv_names_plot)

        conv_colors = {
            "UniGCNConv": "#7fbf7f",
            "UniGATConv": "#4393c3",
            "UniGINConv": "#b2abd2",
        }

        x = np.arange(n_tasks)
        bar_width = 0.8 / n_convs
        for i, conv_name in enumerate(conv_names_plot):
            vals = []
            for tname in task_names:
                matching = [
                    r for r in task_results
                    if r["task"] == tname and r["conv"] == conv_name
                ]
                vals.append(matching[0]["test_acc"] if matching else 0.0)
            offset = (i - n_convs / 2 + 0.5) * bar_width
            ax.bar(
                x + offset, vals, bar_width * 0.9,
                label=conv_name,
                color=conv_colors.get(conv_name, "#999999"),
                edgecolor="black", linewidth=0.5,
            )

        # Add random baselines
        for ti, tname in enumerate(task_names):
            matching = [r for r in task_results if r["task"] == tname]
            if matching:
                random_baseline = 1.0 / matching[0]["num_classes"]
                ax.axhline(y=random_baseline, color="gray", linestyle=":",
                           alpha=0.3)
                ax.annotate(
                    f"random={random_baseline:.2f}",
                    xy=(ti, random_baseline),
                    fontsize=7, color="gray", ha="center",
                    va="bottom",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [t.replace("_", "\n") for t in task_names],
            fontsize=9,
        )
        ax.set_ylabel("Test Accuracy")
        ax.set_title("E. Task x Convolution Comparison", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        panel_idx += 1

    # Hide unused panels
    for i in range(panel_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "hgx Accuracy Ablation: Organoid GRN Node Classification",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved to {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose and ablate hgx accuracy gap on organoid GRN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to data/processed/ directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Training epochs per ablation run (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else detect_data_dir()

    print("=" * 72)
    print("  hgx Accuracy Ablation: Organoid GRN")
    print("=" * 72)
    print(f"  Data dir:  {data_dir}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Seed:      {args.seed}")
    print(f"  JAX:       {jax.__version__} (devices: {jax.devices()})")
    print()

    # --- Load data ---
    data = load_data(data_dir)

    # --- Part 1: Diagnostics ---
    run_diagnostics(data)
    run_gradient_diagnostics(data, seed=args.seed)

    # --- Part 2: Ablation ---
    ablation_results = run_ablation(data, seed=args.seed, epochs=args.epochs)

    # --- Part 3-4: Task x Convolution comparison ---
    task_results = run_task_comparison(
        data, seed=args.seed, epochs=args.epochs
    )

    # --- Part 5: Output ---
    print_results_table(ablation_results, task_results)
    fig_path = FIG_DIR / "accuracy_ablation.png"
    save_figure(ablation_results, task_results, fig_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
