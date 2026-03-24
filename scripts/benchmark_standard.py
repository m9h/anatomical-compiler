#!/usr/bin/env python3
"""Benchmark hgx on standard cocitation hypergraph datasets.

Downloads Cora, Citeseer, and Pubmed citation graphs via DHG's data loader,
then constructs cocitation hypergraphs manually (1-hop neighborhoods) to
avoid DHG's higher-level API issues.  Trains hgx models (UniGCNConv,
UniGATConv, UniGINConv) and compares against published results from the
UniGNN paper (Huang & Yang, IJCAI 2021).

Metrics per model per dataset:
    - Test accuracy (node classification, mean +/- std over multiple seeds)
    - Training time per epoch
    - Inference time (single forward pass, averaged over 100 runs)

Usage:
    uv run python scripts/benchmark_standard.py
    uv run python scripts/benchmark_standard.py --dataset cora --epochs 200
    uv run python scripts/benchmark_standard.py --dataset all --seed 42
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

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

DATASET_NAMES = ["cora", "citeseer", "pubmed"]

CONV_MAP = {
    "UniGCNConv": hgx.UniGCNConv,
    "UniGATConv": hgx.UniGATConv,
    "UniGINConv": hgx.UniGINConv,
}

HIDDEN_DIM = 64
INFERENCE_RUNS = 100
NUM_SEEDS = 5

# Published results from the UniGNN paper (Huang & Yang, IJCAI 2021) and
# related works.  All values are test accuracy (%) on standard semi-supervised
# node classification splits.
PUBLISHED = {
    "cora": {
        "HGNN": 79.39,
        "HyperGCN": 78.45,
        "UniGCN": 78.95,
        "AllSet": 78.58,
    },
    "citeseer": {
        "HGNN": 72.01,
        "HyperGCN": 71.22,
        "UniGCN": 71.63,
        "AllSet": 70.83,
    },
    "pubmed": {
        "HGNN": 86.44,
        "HyperGCN": 82.80,
        "UniGCN": 79.28,
        "AllSet": 78.58,
    },
}


# ---------------------------------------------------------------------------
# Data loading: DHG for download, manual cocitation hypergraph construction
# ---------------------------------------------------------------------------


def load_dataset(name: str) -> dict[str, Any]:
    """Load a citation dataset via DHG and build a cocitation hypergraph.

    Uses DHG only for downloading the raw data (features, labels, edge_list).
    Constructs the cocitation hypergraph incidence matrix manually: each
    node's 1-hop neighborhood (including itself) forms one hyperedge.

    Parameters
    ----------
    name : str
        Dataset name: "cora", "citeseer", or "pubmed".

    Returns
    -------
    dict with keys:
        features : np.ndarray (n, d), float32
        labels : np.ndarray (n,), int32
        num_classes : int
        incidence : np.ndarray (n, m), float32 -- dense incidence matrix
        train_mask, val_mask, test_mask : np.ndarray (n,), bool
    """
    try:
        import dhg
        _loaders = {"cora": dhg.data.Cora, "citeseer": dhg.data.Citeseer, "pubmed": dhg.data.Pubmed}
    except ImportError:
        from planetoid_loader import Cora, Citeseer, Pubmed
        _loaders = {"cora": Cora, "citeseer": Citeseer, "pubmed": Pubmed}

    print(f"  Loading dataset: {name}")

    if name not in _loaders:
        raise ValueError(f"Unknown dataset: {name}")
    data = _loaders[name]()

    features = np.array(data["features"], dtype=np.float32)
    labels = np.array(data["labels"], dtype=np.int32)
    num_classes = data["num_classes"]
    edge_list = data["edge_list"]
    n = len(labels)

    print(f"    Nodes: {n}, Features: {features.shape[1]}, Classes: {num_classes}")
    print(f"    Citation edges: {len(edge_list)}")

    # --- Build cocitation hypergraph ---
    # For each node, collect its neighbors from the citation edge_list.
    # Each node's 1-hop neighborhood (node + neighbors) becomes a hyperedge.
    neighbors = defaultdict(set)
    for src, dst in edge_list:
        neighbors[src].add(dst)
        neighbors[dst].add(src)

    hyperedges = []
    for node in range(n):
        he = sorted(neighbors[node] | {node})
        if len(he) >= 2:
            hyperedges.append(he)

    m = len(hyperedges)
    print(f"    Cocitation hyperedges: {m}")

    # Build dense incidence matrix
    incidence = np.zeros((n, m), dtype=np.float32)
    for j, he in enumerate(hyperedges):
        for i in he:
            incidence[i, j] = 1.0

    # Hyperedge size statistics
    he_sizes = np.array([len(he) for he in hyperedges])
    print(
        f"    Hyperedge sizes: mean={he_sizes.mean():.1f}, "
        f"median={np.median(he_sizes):.1f}, "
        f"min={he_sizes.min()}, max={he_sizes.max()}"
    )

    # --- Train/val/test masks ---
    # Try to use DHG's built-in masks; fall back to random 60/20/20 split.
    try:
        _tm = data["train_mask"]
        if _tm is not None:
            train_mask = np.array(_tm, dtype=bool)
            val_mask = np.array(data["val_mask"], dtype=bool)
            test_mask = np.array(data["test_mask"], dtype=bool)
            print(
                f"    Split (from DHG): train={train_mask.sum()}, "
                f"val={val_mask.sum()}, test={test_mask.sum()}"
            )
        else:
            raise KeyError("train_mask is None")
    except (AssertionError, KeyError):
        train_mask, val_mask, test_mask = _make_random_split(n, seed=0)
        print(
            f"    Split (random 60/20/20): train={train_mask.sum()}, "
            f"val={val_mask.sum()}, test={test_mask.sum()}"
        )

    return {
        "features": features,
        "labels": labels,
        "num_classes": num_classes,
        "incidence": incidence,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


def _make_random_split(
    n: int,
    seed: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for a random train/val/test split."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train : n_train + n_val]] = True
    test_mask[perm[n_train + n_val :]] = True

    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_model(
    conv_name: str,
    in_dim: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
    key: jax.Array,
) -> hgx.HGNNStack:
    """Build an hgx HGNNStack with 2 conv layers + readout."""
    conv_cls = CONV_MAP[conv_name]
    model = hgx.HGNNStack(
        conv_dims=[(in_dim, hidden_dim), (hidden_dim, hidden_dim)],
        conv_cls=conv_cls,
        readout_dim=num_classes,
        activation=jax.nn.relu,
        dropout_rate=dropout,
        key=key,
    )
    return model


# ---------------------------------------------------------------------------
# Training loop with early stopping
# ---------------------------------------------------------------------------


def train_and_evaluate(
    model: hgx.HGNNStack,
    hg: hgx.Hypergraph,
    labels: jnp.ndarray,
    train_mask: jnp.ndarray,
    val_mask: jnp.ndarray,
    test_mask: jnp.ndarray,
    num_classes: int,
    lr: float,
    epochs: int,
    patience: int,
    verbose: bool = False,
) -> dict[str, float]:
    """Train an hgx model with early stopping and return metrics.

    Returns
    -------
    dict with keys: test_acc, val_acc, train_acc, final_loss, train_time,
                    best_epoch, time_per_epoch
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def loss_fn(m, hg, labels, mask):
        logits = m(hg, inference=True)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(labels, num_classes=num_classes)
        per_node = -jnp.sum(one_hot * log_probs, axis=-1)
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

    def compute_acc(logits, labels, mask):
        preds = jnp.argmax(logits, axis=-1)
        correct = jnp.sum(jnp.where(mask, preds == labels, False))
        total = jnp.sum(mask)
        return float(correct / jnp.maximum(total, 1.0))

    best_val_acc = -1.0
    best_model = model
    best_epoch = 0
    epochs_no_improve = 0

    t_start = time.perf_counter()

    for epoch in range(epochs):
        model, opt_state, loss = step(model, opt_state, hg, labels, train_mask)

        # Evaluate every epoch for early stopping
        logits = predict(model, hg)
        val_acc = compute_acc(logits, labels, val_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and ((epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0):
            train_acc = compute_acc(logits, labels, train_mask)
            print(
                f"      Epoch {epoch+1:4d}  loss={float(loss):.4f}  "
                f"val_acc={val_acc:.4f}  train_acc={train_acc:.4f}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(
                    f"      Early stopping at epoch {epoch+1} "
                    f"(best val_acc={best_val_acc:.4f} at epoch {best_epoch})"
                )
            break

    # Ensure JAX computation is complete for timing
    jax.block_until_ready(eqx.filter(model, eqx.is_array))
    train_time = time.perf_counter() - t_start
    actual_epochs = epoch + 1

    # Evaluate best model on all splits
    logits = predict(best_model, hg)
    test_acc = compute_acc(logits, labels, test_mask)
    val_acc_final = compute_acc(logits, labels, val_mask)
    train_acc = compute_acc(logits, labels, train_mask)

    return {
        "test_acc": test_acc,
        "val_acc": val_acc_final,
        "train_acc": train_acc,
        "final_loss": float(loss),
        "train_time": train_time,
        "best_epoch": best_epoch,
        "actual_epochs": actual_epochs,
        "time_per_epoch": train_time / max(actual_epochs, 1),
    }


# ---------------------------------------------------------------------------
# Inference timing
# ---------------------------------------------------------------------------


def measure_inference_time(
    model: hgx.HGNNStack,
    hg: hgx.Hypergraph,
    n_runs: int = INFERENCE_RUNS,
) -> float:
    """Measure average inference time (seconds) over n_runs forward passes."""

    @eqx.filter_jit
    def predict(model, hg):
        return model(hg, inference=True)

    # Warm-up (JIT compilation)
    logits = predict(model, hg)
    jax.block_until_ready(logits)

    t_start = time.perf_counter()
    for _ in range(n_runs):
        logits = predict(model, hg)
        jax.block_until_ready(logits)
    t_total = time.perf_counter() - t_start

    return t_total / n_runs


# ---------------------------------------------------------------------------
# Run one dataset
# ---------------------------------------------------------------------------


def run_dataset(
    dataset_name: str,
    epochs: int,
    patience: int,
    lr: float,
    dropout: float,
    seeds: list[int],
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Run all hgx models on one dataset over multiple seeds.

    Returns a list of result dicts (one per model per seed).
    """
    data = load_dataset(dataset_name)

    features_jnp = jnp.array(data["features"])
    labels_jnp = jnp.array(data["labels"])
    incidence_jnp = jnp.array(data["incidence"])
    train_mask_jnp = jnp.array(data["train_mask"])
    val_mask_jnp = jnp.array(data["val_mask"])
    test_mask_jnp = jnp.array(data["test_mask"])

    in_dim = data["features"].shape[1]
    num_classes = data["num_classes"]

    # Build hgx Hypergraph
    hg = hgx.from_incidence(incidence_jnp, node_features=features_jnp)

    all_results = []

    for conv_name in CONV_MAP:
        print(f"\n    Model: {conv_name}")
        seed_results = []

        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            model = build_model(
                conv_name, in_dim, HIDDEN_DIM, num_classes, dropout, key
            )

            metrics = train_and_evaluate(
                model=model,
                hg=hg,
                labels=labels_jnp,
                train_mask=train_mask_jnp,
                val_mask=val_mask_jnp,
                test_mask=test_mask_jnp,
                num_classes=num_classes,
                lr=lr,
                epochs=epochs,
                patience=patience,
                verbose=verbose and seed == seeds[0],
            )

            # Measure inference time using the best model (re-build for clean
            # timing; the JIT cache makes this fast after first seed).
            inference_time = measure_inference_time(model, hg)

            row = {
                "dataset": dataset_name,
                "model": conv_name,
                "seed": seed,
                "inference_time": inference_time,
                **metrics,
            }
            seed_results.append(row)
            all_results.append(row)

            print(
                f"      seed={seed}  test_acc={metrics['test_acc']:.4f}  "
                f"val_acc={metrics['val_acc']:.4f}  "
                f"best_epoch={metrics['best_epoch']}  "
                f"time={metrics['train_time']:.1f}s  "
                f"infer={inference_time*1000:.2f}ms"
            )

        # Summary for this model
        test_accs = [r["test_acc"] for r in seed_results]
        mean_acc = np.mean(test_accs)
        std_acc = np.std(test_accs)
        print(
            f"      => {conv_name} mean test_acc: "
            f"{mean_acc:.4f} +/- {std_acc:.4f}"
        )

    # Clean up to save memory
    gc.collect()

    return all_results


# ---------------------------------------------------------------------------
# Summary table and comparison with published results
# ---------------------------------------------------------------------------


def build_summary(all_results: list[dict]) -> pd.DataFrame:
    """Aggregate per-seed results into mean +/- std summaries."""
    df = pd.DataFrame(all_results)

    summary_rows = []
    for (dataset, model), group in df.groupby(["dataset", "model"]):
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "test_acc_mean": group["test_acc"].mean(),
                "test_acc_std": group["test_acc"].std(),
                "val_acc_mean": group["val_acc"].mean(),
                "train_acc_mean": group["train_acc"].mean(),
                "time_per_epoch_mean": group["time_per_epoch"].mean(),
                "inference_time_mean": group["inference_time"].mean(),
                "best_epoch_mean": group["best_epoch"].mean(),
                "n_seeds": len(group),
            }
        )

    return pd.DataFrame(summary_rows)


def print_comparison_table(summary: pd.DataFrame) -> None:
    """Print hgx results alongside published baselines."""
    print("\n" + "=" * 90)
    print("  RESULTS: hgx vs Published Baselines (test accuracy %)")
    print("=" * 90)

    for dataset in DATASET_NAMES:
        ds_summary = summary[summary["dataset"] == dataset]
        if ds_summary.empty:
            continue

        published = PUBLISHED.get(dataset, {})

        print(f"\n  --- {dataset.upper()} ---")
        header = f"  {'Model':<18s} {'Test Acc (%)':>14s} {'Source':>12s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        # Published baselines first
        for model_name, acc in published.items():
            print(f"  {model_name:<18s} {acc:>11.2f}%    {'published':>10s}")

        print("  " + "." * (len(header) - 2))

        # hgx results
        for _, row in ds_summary.iterrows():
            acc_str = f"{row['test_acc_mean']*100:.2f} +/- {row['test_acc_std']*100:.2f}%"
            print(f"  {row['model']:<18s} {acc_str:>14s}    {'hgx':>10s}")

    # Timing summary
    print(f"\n\n  --- Timing Summary ---")
    header = (
        f"  {'Dataset':<12s} {'Model':<18s} "
        f"{'ms/epoch':>10s} {'Infer (ms)':>12s} {'Best Epoch':>12s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in summary.iterrows():
        print(
            f"  {row['dataset']:<12s} {row['model']:<18s} "
            f"{row['time_per_epoch_mean']*1000:>10.1f} "
            f"{row['inference_time_mean']*1000:>12.2f} "
            f"{row['best_epoch_mean']:>12.0f}"
        )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def save_figures(
    summary: pd.DataFrame,
    all_results: list[dict],
    fig_dir: Path,
) -> None:
    """Generate and save benchmark figures."""
    df_all = pd.DataFrame(all_results)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    conv_colors = {
        "UniGCNConv": "#7fbf7f",
        "UniGATConv": "#4393c3",
        "UniGINConv": "#b2abd2",
    }
    published_color = "#d9d9d9"

    for ax_idx, dataset in enumerate(DATASET_NAMES):
        ax = axes[ax_idx]
        ds_summary = summary[summary["dataset"] == dataset]
        published = PUBLISHED.get(dataset, {})

        if ds_summary.empty and not published:
            ax.set_visible(False)
            continue

        # Collect all model names and their accuracies
        model_names = []
        means = []
        stds = []
        colors = []

        # Published baselines
        for model_name, acc in published.items():
            model_names.append(model_name)
            means.append(acc)
            stds.append(0.0)
            colors.append(published_color)

        # hgx models
        for _, row in ds_summary.iterrows():
            model_names.append(row["model"])
            means.append(row["test_acc_mean"] * 100)
            stds.append(row["test_acc_std"] * 100)
            colors.append(conv_colors.get(row["model"], "#999999"))

        x = np.arange(len(model_names))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(f"{dataset.upper()}", fontweight="bold", fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, mean_val, std_val in zip(bars, means, stds):
            label = f"{mean_val:.1f}"
            if std_val > 0:
                label += f"\n({std_val:.1f})"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.1 + 0.3,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle(
        "hgx Benchmark on Standard Cocitation Hypergraphs\n"
        "(gray = published baselines, colored = hgx)",
        fontsize=14,
        fontweight="bold",
    )

    fig_path = fig_dir / "benchmark_standard.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved to {fig_path}")

    # --- Timing figure ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Panel A: time per epoch
    ax = axes2[0]
    for dataset in DATASET_NAMES:
        ds = summary[summary["dataset"] == dataset]
        if ds.empty:
            continue
        x = np.arange(len(ds))
        ax.bar(
            x + DATASET_NAMES.index(dataset) * 0.25,
            ds["time_per_epoch_mean"].values * 1000,
            width=0.2,
            label=dataset.upper(),
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(np.arange(len(CONV_MAP)))
    ax.set_xticklabels(list(CONV_MAP.keys()), rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Time per Epoch (ms)")
    ax.set_title("A. Training Speed", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: inference time
    ax = axes2[1]
    for dataset in DATASET_NAMES:
        ds = summary[summary["dataset"] == dataset]
        if ds.empty:
            continue
        x = np.arange(len(ds))
        ax.bar(
            x + DATASET_NAMES.index(dataset) * 0.25,
            ds["inference_time_mean"].values * 1000,
            width=0.2,
            label=dataset.upper(),
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(np.arange(len(CONV_MAP)))
    ax.set_xticklabels(list(CONV_MAP.keys()), rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("B. Inference Speed", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig2_path = fig_dir / "benchmark_standard_timing.png"
    fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Timing figure saved to {fig2_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark hgx on standard cocitation hypergraph datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=DATASET_NAMES + ["all"],
        help="Dataset to benchmark (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum training epochs (default: 200)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience based on val accuracy (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate (default: 0.5, standard for citation benchmarks)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42). Runs NUM_SEEDS seeds starting here.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=NUM_SEEDS,
        help=f"Number of random seeds to average over (default: {NUM_SEEDS})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-epoch training details for the first seed",
    )
    args = parser.parse_args()

    datasets = DATASET_NAMES if args.dataset == "all" else [args.dataset]
    seeds = list(range(args.seed, args.seed + args.num_seeds))

    print("=" * 72)
    print("  hgx Standard Cocitation Benchmark")
    print("=" * 72)
    print(f"  Datasets:  {datasets}")
    print(f"  Models:    {list(CONV_MAP.keys())}")
    print(f"  Epochs:    {args.epochs} (patience={args.patience})")
    print(f"  LR:        {args.lr}")
    print(f"  Dropout:   {args.dropout}")
    print(f"  Seeds:     {seeds}")
    print(f"  Hidden:    {HIDDEN_DIM}")
    print(f"  JAX:       {jax.__version__} (devices: {jax.devices()})")
    print()

    all_results = []

    for dataset_name in datasets:
        print(f"\n{'='*72}")
        print(f"  Dataset: {dataset_name.upper()}")
        print(f"{'='*72}")

        results = run_dataset(
            dataset_name=dataset_name,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            dropout=args.dropout,
            seeds=seeds,
            verbose=args.verbose,
        )
        all_results.extend(results)

    # Build summary
    summary = build_summary(all_results)

    # Print comparison table
    print_comparison_table(summary)

    # Save CSV
    csv_path = FIG_DIR / "benchmark_standard.csv"
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(csv_path, index=False)
    print(f"\n  Raw results saved to {csv_path}")

    csv_summary_path = FIG_DIR / "benchmark_standard_summary.csv"
    summary.to_csv(csv_summary_path, index=False)
    print(f"  Summary saved to {csv_summary_path}")

    # Generate figures
    save_figures(summary, all_results, FIG_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
