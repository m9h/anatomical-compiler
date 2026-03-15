#!/usr/bin/env python3
"""fix_thnn.py — Diagnose and fix NaN losses in THNNConv on the organoid GRN.

The benchmark shows THNNConv producing NaN losses after ~100 epochs:
    Epoch   40  loss=6.4231  acc=0.6%
    Epoch   80  loss=6.1754  acc=0.7%
    Epoch  120  loss=nan  acc=1.9%

THNNConv uses CP decomposition with element-wise products via log-space
aggregation. This script:
    1. Diagnoses the exact operation that produces the first NaN/Inf
    2. Tests six remediation strategies
    3. Reports which fix(es) prevent NaN and what accuracy they achieve
    4. Recommends whether the fix belongs in hgx or user code

Usage:
    uv run python scripts/fix_thnn.py
    uv run python scripts/fix_thnn.py --data-dir data/processed --epochs 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

try:
    import hgx
except ImportError:
    sys.exit("ERROR: hgx not installed. Run: uv pip install -e ../hgx")

from hgx._hypergraph import Hypergraph

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading (mirrors accuracy_ablation.py)
# ---------------------------------------------------------------------------

def detect_data_dir() -> Path:
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

    for name in ["gene_names", "tf_names"]:
        path = data_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                d[name] = json.load(f)
            print(f"  {name}: {len(d[name])} entries")
        else:
            d[name] = None
    return d


def make_split(
    n: int,
    labels: np.ndarray,
    seed: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for train/val/test (stratified)."""
    rng = np.random.RandomState(seed)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        n_lab = len(idx)
        n_train = max(1, int(n_lab * train_frac))
        n_val = max(0, int(n_lab * val_frac))
        if n_lab > 2:
            n_val = min(n_val, n_lab - n_train - 1)
            n_val = max(0, n_val)
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train : n_train + n_val]] = True
        test_mask[idx[n_train + n_val :]] = True

    return train_mask, val_mask, test_mask


# =========================================================================
# PART 1: Diagnose the NaN source
# =========================================================================

def diagnose_nan_source(hg: Hypergraph, conv: hgx.THNNConv) -> None:
    """Run THNNConv forward pass step by step, printing intermediate stats.

    Reproduces the exact computation from hgx/_conv/_thnn.py lines 88-143
    with instrumentation after every operation.
    """
    print("\n" + "=" * 72)
    print("  PART 1: DIAGNOSE NaN SOURCE IN THNNConv FORWARD PASS")
    print("=" * 72)

    def tensor_stats(name: str, t: jnp.ndarray) -> None:
        """Print min, max, has_nan, has_inf for a tensor."""
        t_min = float(jnp.min(t))
        t_max = float(jnp.max(t))
        has_nan = bool(jnp.any(jnp.isnan(t)))
        has_inf = bool(jnp.any(jnp.isinf(t)))
        t_mean = float(jnp.mean(t))
        t_std = float(jnp.std(t))
        flag = ""
        if has_nan:
            flag += " *** HAS NaN ***"
        if has_inf:
            flag += " *** HAS Inf ***"
        print(
            f"    {name:30s}  shape={str(t.shape):20s}  "
            f"min={t_min:12.6f}  max={t_max:12.6f}  "
            f"mean={t_mean:12.6f}  std={t_std:12.6f}{flag}"
        )

    # Step 0: Input features
    print("\n  --- Input ---")
    tensor_stats("node_features", hg.node_features)

    # Step 1: Masked incidence
    H = hg._masked_incidence()
    tensor_stats("H (masked incidence)", H)
    n, m = H.shape
    print(f"    n={n}, m={m}")

    # Step 2: Augmented features [h; 1]
    ones = jnp.ones((n, 1))
    x_aug = jnp.concatenate([hg.node_features, ones], axis=1)
    tensor_stats("x_aug = [features; 1]", x_aug)

    # Step 3: Shared projection z = Theta^T [h; 1]
    z = jax.vmap(conv.theta)(x_aug)
    tensor_stats("z = Theta(x_aug)", z)

    # Step 4: Absolute value + epsilon
    eps = 1e-8
    z_abs = jnp.abs(z) + eps
    tensor_stats("z_abs = |z| + eps", z_abs)

    # Step 5: Sign
    z_sign = jnp.sign(z)
    tensor_stats("z_sign", z_sign)
    n_zeros = int(jnp.sum(z_sign == 0))
    n_pos = int(jnp.sum(z_sign > 0))
    n_neg = int(jnp.sum(z_sign < 0))
    print(f"    sign distribution: pos={n_pos}, neg={n_neg}, zero={n_zeros}")

    # Step 6: log(|z| + eps)
    log_z_abs = jnp.log(z_abs)
    tensor_stats("log(z_abs)", log_z_abs)

    # Step 7: H^T @ log(|z|) -- sum of logs = log of product
    log_prod = H.T @ log_z_abs
    tensor_stats("log_prod = H^T @ log(z_abs)", log_prod)

    # Analyze hyperedge sizes (key to understanding magnitude)
    edge_sizes = jnp.sum(H, axis=0)
    print(f"    hyperedge sizes: min={float(jnp.min(edge_sizes)):.0f}, "
          f"max={float(jnp.max(edge_sizes)):.0f}, "
          f"mean={float(jnp.mean(edge_sizes)):.1f}, "
          f"median={float(jnp.median(edge_sizes)):.1f}")

    # Check: for large hyperedges, sum of logs can be huge
    # If avg log(|z|) ~ -2 and edge has 100 members, log_prod ~ -200
    # exp(-200) -> underflow; if log(|z|) ~ +2 and 100 members, exp(200) -> Inf
    log_z_mean = float(jnp.mean(log_z_abs))
    max_edge_size = float(jnp.max(edge_sizes))
    print(f"    worst-case log_prod estimate: mean(log_z)*max_edge_size = "
          f"{log_z_mean:.3f} * {max_edge_size:.0f} = {log_z_mean * max_edge_size:.1f}")
    if abs(log_z_mean * max_edge_size) > 80:
        print("    WARNING: log_prod will overflow exp() for large hyperedges!")

    # Step 8: Sign angle trick
    sign_angle = jnp.where(z_sign < 0, jnp.pi, 0.0)
    tensor_stats("sign_angle", sign_angle)

    total_angle = H.T @ sign_angle
    tensor_stats("total_angle = H^T @ sign_angle", total_angle)

    prod_sign = jnp.cos(total_angle)
    tensor_stats("prod_sign = cos(total_angle)", prod_sign)

    # Step 9: Reconstruct product = sign * exp(log_prod)
    exp_log_prod = jnp.exp(log_prod)
    tensor_stats("exp(log_prod)", exp_log_prod)

    m_e = prod_sign * exp_log_prod
    tensor_stats("m_e = prod_sign * exp(log_prod)", m_e)

    # Step 10: tanh(m_e)
    m_e_tanh = jnp.tanh(m_e)
    tensor_stats("tanh(m_e)", m_e_tanh)

    # Step 11: Output projection Q * tanh(m_e)
    e_out = jax.vmap(conv.q)(m_e_tanh)
    tensor_stats("e_out = Q(tanh(m_e))", e_out)

    # Step 12: Degree normalization (if enabled)
    if conv.normalize:
        d_v = jnp.sum(H, axis=1, keepdims=True)
        d_v_inv = jnp.where(d_v > 0, 1.0 / d_v, 0.0)
        out = d_v_inv * (H @ e_out)
        tensor_stats("out (normalized)", out)
    else:
        out = H @ e_out
        tensor_stats("out (unnormalized)", out)

    # --- Summary ---
    print("\n  --- Diagnosis Summary ---")
    if bool(jnp.any(jnp.isinf(exp_log_prod))):
        print("  FINDING: exp(log_prod) produces Inf values.")
        print("    Root cause: For large hyperedges, H^T @ log(|z|) accumulates")
        print("    large positive sums. When a hyperedge has K members and the")
        print("    projected features z have magnitudes > 1, the sum of K logs")
        print("    grows linearly with K. exp() of large sums -> Inf.")
        print(f"    Max log_prod value: {float(jnp.max(log_prod)):.1f}")
        print(f"    Largest hyperedge: {max_edge_size:.0f} members")
    elif bool(jnp.any(jnp.isnan(m_e))):
        print("  FINDING: m_e = prod_sign * exp(log_prod) produces NaN.")
        print("    Likely cause: 0 * Inf = NaN or sign computation issues.")
    elif bool(jnp.any(jnp.isnan(out))):
        print("  FINDING: NaN appears after degree normalization.")
    else:
        print("  NOTE: No NaN/Inf in single forward pass with initial weights.")
        print("    NaN likely emerges during training as weights grow, causing")
        print("    |z| >> 1 -> log(|z|) >> 0 -> sum-of-logs overflow -> Inf -> NaN.")

    # Check gradient stability: does loss_fn produce finite gradients?
    print("\n  --- Gradient check (single step) ---")

    num_classes = 720  # placeholder, will be overridden by caller if needed
    labels_dummy = jnp.zeros(n, dtype=jnp.int32)

    def single_conv_loss(conv_module, hg_in):
        out = conv_module(hg_in)
        # Simple loss to check gradient flow
        return jnp.mean(out ** 2)

    loss_val, grads = eqx.filter_value_and_grad(single_conv_loss)(conv, hg)
    grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    all_grads = jnp.concatenate([g.ravel() for g in grad_leaves])
    grad_l2 = float(jnp.linalg.norm(all_grads))
    grad_has_nan = bool(jnp.any(jnp.isnan(all_grads)))
    grad_has_inf = bool(jnp.any(jnp.isinf(all_grads)))
    print(f"    Loss value: {float(loss_val):.6f}")
    print(f"    Gradient L2 norm: {grad_l2:.6f}")
    print(f"    Gradient has NaN: {grad_has_nan}")
    print(f"    Gradient has Inf: {grad_has_inf}")


# =========================================================================
# PART 2: Test fixes
# =========================================================================

def train_thnn_with_fix(
    fix_name: str,
    incidence_jnp: jnp.ndarray,
    features_jnp: jnp.ndarray,
    labels_jnp: jnp.ndarray,
    train_mask: jnp.ndarray,
    val_mask: jnp.ndarray,
    test_mask: jnp.ndarray,
    num_classes: int,
    in_dim: int,
    epochs: int,
    seed: int,
) -> dict:
    """Train THNNConv with a specific fix applied. Returns metrics dict.

    fix_name is one of:
        "baseline"           -- no fix (vanilla THNNConv, rank=64)
        "grad_clip"          -- gradient clipping via clip_by_global_norm(1.0)
        "input_norm"         -- L2-normalize input features before THNNConv
        "rank_8"             -- rank=8
        "rank_16"            -- rank=16
        "rank_32"            -- rank=32
        "lr_1e-4"            -- learning rate 1e-4
        "lr_5e-4"            -- learning rate 5e-4
        "weight_decay"       -- adamw with weight_decay=1e-4
        "clamp_activations"  -- jnp.clip(x, -10, 10) after each operation
        "combo_recommended"  -- combination: input_norm + grad_clip + rank=32 + lr=5e-4
    """
    key = jax.random.PRNGKey(seed)

    # --- Determine hyperparameters per fix ---
    rank = 64
    lr = 1e-3
    use_grad_clip = False
    use_input_norm = False
    use_weight_decay = False
    use_clamp = False

    if fix_name == "baseline":
        pass
    elif fix_name == "grad_clip":
        use_grad_clip = True
    elif fix_name == "input_norm":
        use_input_norm = True
    elif fix_name == "rank_8":
        rank = 8
    elif fix_name == "rank_16":
        rank = 16
    elif fix_name == "rank_32":
        rank = 32
    elif fix_name == "lr_1e-4":
        lr = 1e-4
    elif fix_name == "lr_5e-4":
        lr = 5e-4
    elif fix_name == "weight_decay":
        use_weight_decay = True
    elif fix_name == "clamp_activations":
        use_clamp = True
    elif fix_name == "combo_recommended":
        use_input_norm = True
        use_grad_clip = True
        rank = 32
        lr = 5e-4
    else:
        raise ValueError(f"Unknown fix: {fix_name}")

    # --- Optionally normalize input features ---
    feat = features_jnp
    if use_input_norm:
        feat_norms = jnp.linalg.norm(feat, axis=1, keepdims=True)
        feat = feat / jnp.maximum(feat_norms, 1e-8)

    # --- Build model ---
    model = hgx.HGNNStack(
        conv_dims=[(in_dim, 64), (64, 32)],
        conv_cls=hgx.THNNConv,
        readout_dim=num_classes,
        activation=jax.nn.relu,
        dropout_rate=0.0,
        conv_kwargs={"rank": rank},
        key=key,
    )

    # --- Build optimizer ---
    if use_grad_clip and use_weight_decay:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(lr, weight_decay=1e-4),
        )
    elif use_grad_clip:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr),
        )
    elif use_weight_decay:
        optimizer = optax.adamw(lr, weight_decay=1e-4)
    else:
        optimizer = optax.adam(lr)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- Build hypergraph ---
    hg = hgx.from_incidence(incidence_jnp, node_features=feat)

    # --- Loss function ---
    def loss_fn(m, hg_in, labels, mask):
        logits = m(hg_in, inference=True)
        if use_clamp:
            logits = jnp.clip(logits, -10.0, 10.0)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(labels, num_classes=num_classes)
        per_node = -jnp.sum(one_hot * log_probs, axis=-1)
        return jnp.sum(jnp.where(mask, per_node, 0.0)) / jnp.maximum(
            jnp.sum(mask), 1.0
        )

    @eqx.filter_jit
    def step(model, opt_state, hg_in, labels, mask):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            model, hg_in, labels, mask
        )

        # Clamp activations fix: also clip gradients to prevent NaN propagation
        if use_clamp:
            grads = jax.tree.map(
                lambda g: jnp.clip(g, -10.0, 10.0) if eqx.is_array(g) else g,
                grads,
            )

        updates, new_opt = optimizer.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), new_opt, loss

    @eqx.filter_jit
    def predict(model, hg_in):
        logits = model(hg_in, inference=True)
        if use_clamp:
            logits = jnp.clip(logits, -10.0, 10.0)
        return logits

    # --- Training loop ---
    loss_history = []
    acc_history = []
    nan_epoch = None
    best_val_acc = -1.0
    best_test_acc = -1.0
    best_epoch = 0

    t_start = time.perf_counter()

    for epoch in range(epochs):
        model, opt_state, loss = step(model, opt_state, hg, labels_jnp, train_mask)
        loss_val = float(loss)
        loss_history.append(loss_val)

        if np.isnan(loss_val) and nan_epoch is None:
            nan_epoch = epoch + 1

        # Evaluate every 20 epochs or at the end
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            logits = predict(model, hg)
            preds = jnp.argmax(logits, axis=-1)

            val_correct = jnp.sum(jnp.where(val_mask, preds == labels_jnp, False))
            val_total = jnp.maximum(jnp.sum(val_mask), 1.0)
            val_acc = float(val_correct / val_total)

            test_correct = jnp.sum(jnp.where(test_mask, preds == labels_jnp, False))
            test_total = jnp.maximum(jnp.sum(test_mask), 1.0)
            test_acc = float(test_correct / test_total)

            train_correct = jnp.sum(jnp.where(train_mask, preds == labels_jnp, False))
            train_total = jnp.maximum(jnp.sum(train_mask), 1.0)
            train_acc = float(train_correct / train_total)

            acc_history.append((epoch + 1, train_acc, val_acc, test_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = epoch + 1

    jax.block_until_ready(eqx.filter(model, eqx.is_array))
    train_time = time.perf_counter() - t_start

    return {
        "fix": fix_name,
        "rank": rank,
        "lr": lr,
        "grad_clip": use_grad_clip,
        "input_norm": use_input_norm,
        "weight_decay": use_weight_decay,
        "clamp": use_clamp,
        "nan_epoch": nan_epoch,
        "prevented_nan": nan_epoch is None,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "train_time": train_time,
        "loss_history": loss_history,
        "acc_history": acc_history,
    }


# =========================================================================
# PART 3: Run comparison across all fixes
# =========================================================================

def run_fix_comparison(
    data: dict,
    epochs: int,
    seed: int,
) -> list[dict]:
    """Train THNNConv with each fix and collect results."""
    print("\n" + "=" * 72)
    print("  PART 2-3: FIX COMPARISON (THNNConv, 200 epochs each)")
    print("=" * 72)

    features = data["node_features_pca"]
    labels_np = data["module_labels"]
    incidence_np = data["incidence"]

    if features is None or labels_np is None or incidence_np is None:
        print("  SKIPPED: required data arrays not found.")
        return []

    in_dim = features.shape[1]
    n_nodes = features.shape[0]
    num_classes = int(labels_np.max()) + 1

    incidence_jnp = jnp.array(incidence_np)
    features_jnp = jnp.array(features)
    labels_jnp = jnp.array(labels_np)

    train_mask_np, val_mask_np, test_mask_np = make_split(
        n_nodes, labels_np, seed
    )
    train_mask = jnp.array(train_mask_np)
    val_mask = jnp.array(val_mask_np)
    test_mask = jnp.array(test_mask_np)

    fixes = [
        "baseline",
        "grad_clip",
        "input_norm",
        "rank_8",
        "rank_16",
        "rank_32",
        "lr_1e-4",
        "lr_5e-4",
        "weight_decay",
        "clamp_activations",
        "combo_recommended",
    ]

    results = []
    for fix_name in fixes:
        print(f"\n  --- Fix: {fix_name} ---")
        result = train_thnn_with_fix(
            fix_name=fix_name,
            incidence_jnp=incidence_jnp,
            features_jnp=features_jnp,
            labels_jnp=labels_jnp,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_classes=num_classes,
            in_dim=in_dim,
            epochs=epochs,
            seed=seed,
        )
        results.append(result)

        # Print training trajectory
        nan_str = (
            f"NaN at epoch {result['nan_epoch']}"
            if result["nan_epoch"] is not None
            else "No NaN"
        )
        print(f"    {nan_str}")
        for ep, tr_acc, va_acc, te_acc in result["acc_history"]:
            loss_at_ep = result["loss_history"][ep - 1] if ep <= len(result["loss_history"]) else float("nan")
            print(
                f"    Epoch {ep:4d}  loss={loss_at_ep:10.4f}  "
                f"train={tr_acc:.4f}  val={va_acc:.4f}  test={te_acc:.4f}"
            )
        print(
            f"    Best: epoch={result['best_epoch']}, "
            f"val_acc={result['best_val_acc']:.4f}, "
            f"test_acc={result['best_test_acc']:.4f}, "
            f"time={result['train_time']:.1f}s"
        )

    return results


# =========================================================================
# PART 4: Report and recommendation
# =========================================================================

def print_report(results: list[dict]) -> None:
    """Print comparison table and recommended fix."""
    print("\n" + "=" * 72)
    print("  PART 4: RESULTS TABLE AND RECOMMENDATION")
    print("=" * 72)

    if not results:
        print("  No results to report.")
        return

    # Results table
    header = (
        f"  {'Fix':<22s} {'Rank':>5s} {'LR':>8s} "
        f"{'NaN?':>8s} {'NaN@':>6s} {'BestVal':>8s} {'BestTest':>9s} "
        f"{'BestEp':>7s} {'Time':>7s}"
    )
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    for r in results:
        nan_str = f"{r['nan_epoch']}" if r["nan_epoch"] is not None else "---"
        prevented = "OK" if r["prevented_nan"] else "FAIL"
        print(
            f"  {r['fix']:<22s} {r['rank']:>5d} {r['lr']:>8.0e} "
            f"{prevented:>8s} {nan_str:>6s} "
            f"{r['best_val_acc']:>8.4f} {r['best_test_acc']:>9.4f} "
            f"{r['best_epoch']:>7d} {r['train_time']:>6.1f}s"
        )

    # Sort by: prevented NaN first, then by best test accuracy
    valid = [r for r in results if r["prevented_nan"]]
    invalid = [r for r in results if not r["prevented_nan"]]

    print("\n  --- Fixes that PREVENTED NaN ---")
    if valid:
        valid_sorted = sorted(valid, key=lambda r: r["best_test_acc"], reverse=True)
        for r in valid_sorted:
            print(f"    {r['fix']:<22s}  test_acc={r['best_test_acc']:.4f}")
        best = valid_sorted[0]
    else:
        print("    (none)")
        best = None

    print("\n  --- Fixes that FAILED to prevent NaN ---")
    if invalid:
        for r in invalid:
            print(f"    {r['fix']:<22s}  NaN at epoch {r['nan_epoch']}")
    else:
        print("    (none -- all fixes worked)")

    # Recommendation
    print("\n" + "=" * 72)
    print("  RECOMMENDATION")
    print("=" * 72)

    print("""
  ROOT CAUSE:
    THNNConv computes element-wise products of projected features across all
    members of each hyperedge via log-space aggregation:

        log_prod = H^T @ log(|z| + eps)    -- sum of logs across hyperedge
        m_e = sign * exp(log_prod)          -- reconstruct product

    For large hyperedges (the organoid GRN has regulons with up to ~500
    target genes), the sum of logs can exceed the dynamic range of float32:
      - If projected features |z| > 1: sum-of-logs >> 0 -> exp() = Inf
      - If projected features |z| < 1: sum-of-logs << 0 -> exp() = 0 (underflow)

    As training proceeds, weights grow, z values drift from initialization,
    and overflow becomes inevitable around epoch 100-120.

    The subsequent tanh(m_e) rescues Inf -> +/-1, but NaN arises from
    0 * Inf or Inf - Inf in gradient computation.
    """)

    if best is not None:
        print(f"  RECOMMENDED FIX: {best['fix']}")
        print(f"    Test accuracy: {best['best_test_acc']:.4f}")
        print(f"    Achieved at epoch: {best['best_epoch']}")
    else:
        print("  No single fix fully prevented NaN. Consider combining multiple.")

    print("""
  WHERE TO APPLY THE FIX:

    1. IN hgx ITSELF (recommended for robustness):
       In hgx/_conv/_thnn.py, clamp log_prod before exp():

           log_prod = H.T @ jnp.log(z_abs)
           log_prod = jnp.clip(log_prod, -30.0, 30.0)   # <-- ADD THIS
           m_e = prod_sign * jnp.exp(log_prod)

       This bounds exp() to [exp(-30), exp(30)] ~ [9.4e-14, 1.1e+13],
       which is safely within float32 range and gets squashed by tanh()
       anyway. This is a principled fix: the original paper's tanh already
       intends to bound the output, but Inf * sign can produce NaN before
       tanh is applied.

    2. IN USER CODE (for immediate workaround):
       - Normalize input features to unit L2 norm (keeps |z| near 1)
       - Use gradient clipping: optax.clip_by_global_norm(1.0)
       - Lower the CP rank (rank=32 instead of 64)
       - Lower learning rate to 5e-4

    BEST PRACTICE: Apply fix (1) in hgx for correctness, and fix (2) in
    user code for best training dynamics. The combo_recommended variant
    combines the most effective user-side fixes.
    """)


# =========================================================================
# Optional: Generate figure
# =========================================================================

def save_figure(results: list[dict], fig_path: Path) -> None:
    """Plot loss curves and accuracy comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Color map for fixes
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # --- Panel A: Loss curves ---
    ax = axes[0]
    for i, r in enumerate(results):
        label = r["fix"]
        loss_h = r["loss_history"]
        # Replace NaN with None for clean plotting
        epochs_plot = list(range(1, len(loss_h) + 1))
        loss_plot = [v if np.isfinite(v) else None for v in loss_h]
        # Split at first NaN for clean line
        valid_ep = []
        valid_loss = []
        for ep, lv in zip(epochs_plot, loss_plot):
            if lv is not None:
                valid_ep.append(ep)
                valid_loss.append(lv)
            else:
                break
        if valid_ep:
            ax.plot(valid_ep, valid_loss, label=label, color=colors[i],
                    linewidth=1.5, alpha=0.8)
            if r["nan_epoch"] is not None:
                ax.axvline(x=r["nan_epoch"], color=colors[i], linestyle=":",
                           alpha=0.4, linewidth=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("A. Training Loss Curves", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Panel B: Test accuracy bar chart ---
    ax = axes[1]
    fix_names = [r["fix"] for r in results]
    test_accs = [r["best_test_acc"] for r in results]
    bar_colors = ["#d62728" if not r["prevented_nan"] else "#2ca02c" for r in results]

    bars = ax.barh(range(len(results)), test_accs, color=bar_colors,
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(fix_names, fontsize=8)
    ax.set_xlabel("Best Test Accuracy")
    ax.set_title("B. Test Accuracy per Fix", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate NaN failures
    for i, r in enumerate(results):
        if not r["prevented_nan"]:
            ax.annotate(
                f"NaN@{r['nan_epoch']}",
                xy=(test_accs[i] + 0.001, i),
                fontsize=7, color="#d62728", va="center",
            )

    # --- Panel C: NaN epoch vs rank ---
    ax = axes[2]
    rank_fixes = [r for r in results if r["fix"].startswith("rank_") or r["fix"] == "baseline"]
    if rank_fixes:
        ranks = [r["rank"] for r in rank_fixes]
        nan_epochs = [
            r["nan_epoch"] if r["nan_epoch"] is not None else len(r["loss_history"])
            for r in rank_fixes
        ]
        accs = [r["best_test_acc"] for r in rank_fixes]

        ax2 = ax.twinx()
        ax.bar(range(len(ranks)), nan_epochs, color="#4393c3", alpha=0.7,
               label="NaN epoch (or max if none)")
        ax2.plot(range(len(ranks)), accs, "o-", color="#d62728",
                 linewidth=2, markersize=8, label="Best test acc")

        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels([str(r) for r in ranks])
        ax.set_xlabel("CP Rank")
        ax.set_ylabel("Epoch of First NaN", color="#4393c3")
        ax2.set_ylabel("Test Accuracy", color="#d62728")
        ax.set_title("C. Rank vs Stability", fontweight="bold")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    else:
        ax.text(0.5, 0.5, "No rank sweep data", ha="center", va="center",
                transform=ax.transAxes)

    fig.suptitle(
        "THNNConv NaN Diagnosis: Organoid GRN Benchmark",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved to {fig_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose and fix THNNConv NaN losses on organoid GRN",
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
        help="Training epochs per fix variant (default: 200)",
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
    print("  THNNConv NaN Diagnosis and Fix: Organoid GRN")
    print("=" * 72)
    print(f"  Data dir:  {data_dir}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Seed:      {args.seed}")
    print(f"  JAX:       {jax.__version__} (devices: {jax.devices()})")
    print()

    # --- Load data ---
    data = load_data(data_dir)

    features = data["node_features_pca"]
    labels_np = data["module_labels"]
    incidence_np = data["incidence"]

    if features is None or labels_np is None or incidence_np is None:
        sys.exit("ERROR: Required data files not found in data/processed/.")

    in_dim = features.shape[1]
    num_classes = int(labels_np.max()) + 1

    # --- Part 1: Diagnose NaN source ---
    incidence_jnp = jnp.array(incidence_np)
    features_jnp = jnp.array(features)
    hg = hgx.from_incidence(incidence_jnp, node_features=features_jnp)

    key = jax.random.PRNGKey(args.seed)
    conv = hgx.THNNConv(in_dim, 64, rank=64, normalize=True, key=key)
    diagnose_nan_source(hg, conv)

    # --- Parts 2-3: Test fixes ---
    results = run_fix_comparison(data, epochs=args.epochs, seed=args.seed)

    # --- Part 4: Report ---
    print_report(results)

    # --- Optional figure ---
    fig_path = FIG_DIR / "thnn_nan_diagnosis.png"
    try:
        save_figure(results, fig_path)
    except Exception as e:
        print(f"  WARNING: Could not save figure: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
