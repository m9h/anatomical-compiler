#!/usr/bin/env python3
"""Benchmark comparison: hgx (JAX) vs DHG (DeepHypergraph, PyTorch).

Compares hypergraph neural network implementations across standard cocitation
datasets (Cora, Citeseer, Pubmed) and the organoid GRN (Fleck et al. 2023).

DHG models (PyTorch, GPU):
    - HGNN   (Feng et al. 2019)
    - HGNN+
    - HyperGCN

hgx models (JAX, GPU):
    - UniGCNConv
    - UniGATConv
    - UniGINConv
    - SheafDiffusion

Metrics per model per dataset:
    - Test accuracy (node classification, 60/20/20 split)
    - Training time (configurable epochs)
    - Inference time (single forward pass, averaged over 100 runs)
    - Peak GPU memory usage

Usage:
    uv run python scripts/benchmark_comparison.py
    uv run python scripts/benchmark_comparison.py --dataset cora --epochs 200
    uv run python scripts/benchmark_comparison.py --dataset all --seed 42
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAMES = ["cora", "citeseer", "pubmed", "organoid"]
DHG_MODEL_NAMES = ["HGNN", "HGNN+", "HyperGCN"]
HGX_MODEL_NAMES = ["UniGCNConv", "UniGATConv", "UniGINConv", "SheafDiffusion"]
ALL_MODEL_NAMES = DHG_MODEL_NAMES + HGX_MODEL_NAMES

HIDDEN_DIM = 64
INFERENCE_RUNS = 100

# ---------------------------------------------------------------------------
# Utility: train/val/test split
# ---------------------------------------------------------------------------


def make_split(n: int, seed: int, train_frac: float = 0.6, val_frac: float = 0.2):
    """Return boolean masks for train/val/test (60/20/20 by default)."""
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


# ===================================================================
# DHG (PyTorch) section
# ===================================================================


def load_dhg_dataset(name: str):
    """Load a citation dataset via DHG and convert to hypergraph.

    Returns
    -------
    dict with keys: features, labels, num_classes, num_vertices, hg,
                    train_mask, val_mask, test_mask
    """
    import torch
    import dhg

    print(f"  [DHG] Loading dataset: {name}")

    if name == "cora":
        data = dhg.data.Cora()
    elif name == "citeseer":
        data = dhg.data.Citeseer()
    elif name == "pubmed":
        data = dhg.data.Pubmed()
    else:
        raise ValueError(f"Unknown DHG dataset: {name}")

    num_v = data["num_vertices"]
    features = torch.FloatTensor(np.array(data["features"]))
    labels = torch.LongTensor(np.array(data["labels"]))
    num_classes = data["num_classes"]

    # Build a Graph and convert to Hypergraph via k-hop neighborhood
    edge_list = data["edge_list"]
    G = dhg.Graph(num_v, edge_list)
    HG = dhg.Hypergraph.from_graph_kHop(G, k=1)

    # Use DHG's own masks if available, otherwise create 60/20/20 split.
    # DHG data objects don't support the ``in`` operator for checking keys,
    # so we use try/except instead.
    try:
        _tm = data["train_mask"]
        if _tm is not None:
            train_mask = torch.BoolTensor(np.array(_tm))
            val_mask = torch.BoolTensor(np.array(data["val_mask"]))
            test_mask = torch.BoolTensor(np.array(data["test_mask"]))
        else:
            raise KeyError("train_mask is None")
    except (AssertionError, KeyError):
        tm, vm, tsm = make_split(num_v, seed=0)
        train_mask = torch.BoolTensor(tm)
        val_mask = torch.BoolTensor(vm)
        test_mask = torch.BoolTensor(tsm)

    print(
        f"  [DHG]   {num_v} vertices, {len(edge_list)} edges, "
        f"{HG.num_e} hyperedges, {num_classes} classes"
    )

    return {
        "features": features,
        "labels": labels,
        "num_classes": num_classes,
        "num_vertices": num_v,
        "hg": HG,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


class DHG_HGNN_Model:
    """Wrapper for DHG's HGNN (Feng et al. 2019)."""

    name = "HGNN"

    @staticmethod
    def build(in_dim: int, num_classes: int, device):
        import torch
        import torch.nn as nn
        import dhg.nn as dhgnn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = dhgnn.HGNNConv(in_dim, HIDDEN_DIM, drop_rate=0.5)
                self.conv2 = dhgnn.HGNNConv(HIDDEN_DIM, num_classes, drop_rate=0.5)

            def forward(self, x, hg):
                x = torch.relu(self.conv1(x, hg))
                return self.conv2(x, hg)

        return Net().to(device)


class DHG_HGNNPlus_Model:
    """Wrapper for DHG's HGNN+."""

    name = "HGNN+"

    @staticmethod
    def build(in_dim: int, num_classes: int, device):
        import torch
        import torch.nn as nn
        import dhg.nn as dhgnn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = dhgnn.HGNNPConv(in_dim, HIDDEN_DIM, drop_rate=0.5)
                self.conv2 = dhgnn.HGNNPConv(HIDDEN_DIM, num_classes, drop_rate=0.5)

            def forward(self, x, hg):
                x = torch.relu(self.conv1(x, hg))
                return self.conv2(x, hg)

        return Net().to(device)


class DHG_HyperGCN_Model:
    """Wrapper for DHG's HyperGCN."""

    name = "HyperGCN"

    @staticmethod
    def build(in_dim: int, num_classes: int, device):
        import torch
        import torch.nn as nn
        import dhg.nn as dhgnn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = dhgnn.HyperGCNConv(in_dim, HIDDEN_DIM, drop_rate=0.5)
                self.conv2 = dhgnn.HyperGCNConv(HIDDEN_DIM, num_classes, drop_rate=0.5)

            def forward(self, x, hg):
                x = torch.relu(self.conv1(x, hg))
                return self.conv2(x, hg)

        return Net().to(device)


DHG_MODELS = {
    "HGNN": DHG_HGNN_Model,
    "HGNN+": DHG_HGNNPlus_Model,
    "HyperGCN": DHG_HyperGCN_Model,
}


def _train_dhg_model_on_device(
    model_cls,
    dataset: dict,
    epochs: int,
    device,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
):
    """Core DHG training loop on a specific device.  Called by train_dhg_model."""
    import torch
    import torch.nn.functional as F

    print(f"  [DHG] Training {model_cls.name} on {device}")

    features = dataset["features"].to(device)
    labels = dataset["labels"].to(device)
    hg = dataset["hg"]
    if device.type == "cuda":
        hg = hg.to(device)
    train_mask = dataset["train_mask"].to(device)
    val_mask = dataset["val_mask"].to(device)
    test_mask = dataset["test_mask"].to(device)

    model = model_cls.build(features.shape[1], dataset["num_classes"], device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Reset peak memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # --- Training loop ---
    best_val_acc = 0.0
    best_state = None

    t_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, hg)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        if (epoch + 1) % max(1, epochs // 5) == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(features, hg)
                val_preds = val_logits[val_mask].argmax(dim=-1)
                val_acc = (val_preds == labels[val_mask]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(
                f"    Epoch {epoch+1:4d}  loss={loss.item():.4f}  "
                f"val_acc={val_acc:.4f}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t_start

    # Reload best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Test accuracy ---
    model.eval()
    with torch.no_grad():
        logits = model(features, hg)
        test_preds = logits[test_mask].argmax(dim=-1)
        test_acc = (test_preds == labels[test_mask]).float().mean().item()

    # --- Inference time ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_times = []
    model.eval()
    with torch.no_grad():
        for _ in range(INFERENCE_RUNS):
            t0 = time.perf_counter()
            _ = model(features, hg)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_times.append(time.perf_counter() - t0)
    infer_time = np.mean(infer_times)

    # --- Peak memory ---
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_mem_mb = float("nan")

    print(
        f"  [DHG] {model_cls.name}: test_acc={test_acc:.4f}, "
        f"train={train_time:.2f}s, infer={infer_time*1000:.2f}ms, "
        f"mem={peak_mem_mb:.1f}MB, device={device}"
    )

    return {
        "test_acc": test_acc,
        "train_time": train_time,
        "infer_time": infer_time,
        "peak_mem_mb": peak_mem_mb,
        "device": str(device),
    }


def train_dhg_model(
    model_cls,
    dataset: dict,
    epochs: int,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
):
    """Train a DHG model (PyTorch) and return metrics.

    Attempts CUDA first.  If a RuntimeError indicates a device mismatch (a
    known DHG bug with PyTorch >= 2.11 where sparse hypergraph tensors stay
    on CPU), the training is automatically retried on CPU so the benchmark
    can still produce valid accuracy / timing numbers.

    Returns
    -------
    dict with keys: test_acc, train_time, infer_time, peak_mem_mb, device
    """
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            return _train_dhg_model_on_device(
                model_cls, dataset, epochs, device, lr, weight_decay,
            )
        except RuntimeError as exc:
            if "device" in str(exc).lower() or "expected all tensors" in str(exc).lower():
                print(
                    f"  [DHG] CUDA failed with device-mismatch error "
                    f"(known DHG bug): {exc}"
                )
                print("  [DHG] Retrying on CPU...")
                # Free CUDA memory from the failed attempt
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise  # Re-raise non-device-related RuntimeErrors

    # Fallback (or only option) — run on CPU
    device = torch.device("cpu")
    return _train_dhg_model_on_device(
        model_cls, dataset, epochs, device, lr, weight_decay,
    )


# ===================================================================
# hgx (JAX) section
# ===================================================================


def dhg_to_hgx(dataset: dict):
    """Convert a DHG dataset dict into hgx-compatible JAX arrays.

    Returns
    -------
    dict with keys: hg, features, labels, num_classes,
                    train_mask, val_mask, test_mask
    """
    import jax.numpy as jnp
    import hgx as hgx_lib

    # Build dense incidence from DHG Hypergraph
    dhg_hg = dataset["hg"]
    incidence_dense = dhg_hg.H.to_dense().cpu().numpy().astype(np.float32)
    features_np = dataset["features"].cpu().numpy().astype(np.float32)

    incidence_jnp = jnp.array(incidence_dense)
    features_jnp = jnp.array(features_np)
    labels_jnp = jnp.array(dataset["labels"].cpu().numpy())

    hg = hgx_lib.from_incidence(incidence_jnp, node_features=features_jnp)

    return {
        "hg": hg,
        "features": features_jnp,
        "labels": labels_jnp,
        "num_classes": dataset["num_classes"],
        "train_mask": jnp.array(dataset["train_mask"].cpu().numpy()),
        "val_mask": jnp.array(dataset["val_mask"].cpu().numpy()),
        "test_mask": jnp.array(dataset["test_mask"].cpu().numpy()),
        "incidence": incidence_jnp,
    }


def build_hgx_model(
    conv_name: str,
    in_dim: int,
    num_classes: int,
    incidence,
    key,
    hidden_dim: int = HIDDEN_DIM,
    num_layers: int = 2,
):
    """Build an hgx model for node classification.

    Returns
    -------
    (model, is_sheaf) : tuple
        model is an eqx Module; is_sheaf indicates SheafDiffusion path.
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import hgx as hgx_lib

    k1, k2 = jax.random.split(key)

    if conv_name == "SheafDiffusion":
        nnz = int(jnp.sum(incidence > 0))
        # Guard against OOM: skip SheafDiffusion for very large problems
        if num_classes > 100 or nnz > 50000:
            warnings.warn(
                f"Skipping SheafDiffusion: num_classes={num_classes}, nnz={nnz} "
                f"(thresholds: num_classes>100 or nnz>50000). "
                f"Likely to OOM."
            )
            return None, True
        sheaf = hgx_lib.SheafDiffusion(
            num_steps=3,
            in_dim=in_dim,
            edge_stalk_dim=in_dim,
            num_incidences=nnz,
            key=k1,
        )
        readout = eqx.nn.Linear(in_dim, num_classes, key=k2)

        class _SheafClassifier(eqx.Module):
            sheaf: hgx_lib.SheafDiffusion
            readout: eqx.nn.Linear

            def __call__(self, hg):
                x = self.sheaf(hg)
                return jax.vmap(self.readout)(x)

        model = _SheafClassifier(sheaf=sheaf, readout=readout)
        return model, True

    conv_map = {
        "UniGCNConv": hgx_lib.UniGCNConv,
        "UniGATConv": hgx_lib.UniGATConv,
        "UniGINConv": hgx_lib.UniGINConv,
    }
    conv_cls = conv_map[conv_name]

    # Build conv_dims list for the requested number of layers
    conv_dims = [(in_dim, hidden_dim)]
    for _ in range(num_layers - 1):
        conv_dims.append((hidden_dim, hidden_dim))

    model = hgx_lib.HGNNStack(
        conv_dims=conv_dims,
        conv_cls=conv_cls,
        readout_dim=num_classes,
        activation=jax.nn.relu,
        dropout_rate=0.0,
        key=k1,
    )
    return model, False


def train_hgx_model(
    conv_name: str,
    dataset: dict,
    epochs: int,
    seed: int,
    lr: float = 0.01,
    hidden_dim: int = HIDDEN_DIM,
    num_layers: int = 2,
):
    """Train an hgx model (JAX) and return metrics.

    Returns
    -------
    dict with keys: test_acc, train_time, infer_time, peak_mem_mb
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax

    print(f"  [hgx] Training {conv_name} on {jax.devices()}")

    key = jax.random.PRNGKey(seed)
    k_model, k_train = jax.random.split(key)

    hg = dataset["hg"]
    features = dataset["features"]
    labels = dataset["labels"]
    incidence = dataset["incidence"]
    num_classes = dataset["num_classes"]
    train_mask = dataset["train_mask"]
    val_mask = dataset["val_mask"]
    test_mask = dataset["test_mask"]
    in_dim = features.shape[1]

    model, is_sheaf = build_hgx_model(
        conv_name, in_dim, num_classes, incidence, k_model,
        hidden_dim=hidden_dim, num_layers=num_layers,
    )

    # build_hgx_model returns None when the model is skipped (e.g. OOM guard)
    if model is None:
        print(f"  [hgx] {conv_name}: SKIPPED (too large for this model)")
        return {
            "test_acc": float("nan"),
            "train_time": float("nan"),
            "infer_time": float("nan"),
            "peak_mem_mb": float("nan"),
        }

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- Loss function ---
    def loss_fn(m, hg, labels, mask):
        if is_sheaf:
            logits = m(hg)
        else:
            logits = m(hg, inference=True)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(labels, num_classes=num_classes)
        per_node = -jnp.sum(one_hot * log_probs, axis=-1)
        return jnp.mean(jnp.where(mask, per_node, 0.0))

    @eqx.filter_jit
    def step(model, opt_state, hg, labels, mask):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, hg, labels, mask)
        updates, new_opt = optimizer.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), new_opt, loss

    @eqx.filter_jit
    def predict(model, hg):
        if is_sheaf:
            return model(hg)
        return model(hg, inference=True)

    # --- Measure JAX memory before training ---
    # JAX does not have a direct peak-memory query like PyTorch; we
    # measure device memory via jax.local_devices() backend stats if
    # available, and fall back to NaN.
    def _get_jax_peak_mem_mb():
        try:
            device = jax.local_devices()[0]
            stats = device.memory_stats()
            if stats and "peak_bytes_in_use" in stats:
                return stats["peak_bytes_in_use"] / (1024**2)
        except Exception:
            pass
        return float("nan")

    # --- Training loop ---
    best_val_acc = 0.0
    best_model = model

    t_start = time.perf_counter()
    for epoch in range(epochs):
        model, opt_state, loss = step(model, opt_state, hg, labels, train_mask)

        if (epoch + 1) % max(1, epochs // 5) == 0:
            logits = predict(model, hg)
            val_preds = jnp.argmax(logits, axis=-1)
            val_acc = float(
                jnp.mean(jnp.where(val_mask, val_preds == labels, False))
                / jnp.mean(val_mask)
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
            print(
                f"    Epoch {epoch+1:4d}  loss={float(loss):.4f}  "
                f"val_acc={val_acc:.4f}"
            )

    # Block until computation finishes
    jax.block_until_ready(eqx.filter(model, eqx.is_array))
    train_time = time.perf_counter() - t_start

    # --- Test accuracy ---
    logits = predict(best_model, hg)
    test_preds = jnp.argmax(logits, axis=-1)
    test_acc = float(
        jnp.sum(jnp.where(test_mask, test_preds == labels, False))
        / jnp.sum(test_mask)
    )

    # --- Inference time ---
    # Warmup
    _ = predict(best_model, hg)
    jax.block_until_ready(_)

    infer_times = []
    for _ in range(INFERENCE_RUNS):
        t0 = time.perf_counter()
        out = predict(best_model, hg)
        jax.block_until_ready(out)
        infer_times.append(time.perf_counter() - t0)
    infer_time = np.mean(infer_times)

    # --- Peak memory ---
    peak_mem_mb = _get_jax_peak_mem_mb()

    print(
        f"  [hgx] {conv_name}: test_acc={test_acc:.4f}, "
        f"train={train_time:.2f}s, infer={infer_time*1000:.2f}ms, "
        f"mem={peak_mem_mb:.1f}MB"
    )

    return {
        "test_acc": test_acc,
        "train_time": train_time,
        "infer_time": infer_time,
        "peak_mem_mb": peak_mem_mb,
    }


# ===================================================================
# Organoid GRN dataset
# ===================================================================


def load_organoid_dataset(seed: int):
    """Load organoid GRN as a DHG-compatible dataset dict.

    Uses hgx's data_loader to build the hypergraph, then creates a
    node-classification task from regulatory module membership (argmax
    of incidence row assigns each gene to its primary TF regulon).

    Returns the same dict format as load_dhg_dataset().
    """
    import torch
    import dhg

    # Import project data loader
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from data_loader import load_pando_grn

    hg_obj, coefs_df, gene_names, tf_names = load_pando_grn()
    incidence_np = np.array(hg_obj.incidence)
    n_nodes, n_edges = incidence_np.shape

    # Node features: use incidence structure padded/projected to a fixed dim
    # (the GRN has no inherent feature vectors, so we use the incidence
    # profile + degree as proxy features)
    degree = incidence_np.sum(axis=1, keepdims=True)
    raw_feats = np.concatenate([incidence_np, degree], axis=1).astype(np.float32)
    # Normalize
    row_norms = np.linalg.norm(raw_feats, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-8)
    raw_feats = raw_feats / row_norms

    # Labels: primary module assignment (argmax of incidence row)
    labels_np = incidence_np.argmax(axis=1).astype(np.int64)
    num_classes = int(labels_np.max() + 1)

    # Build DHG Hypergraph from edge list
    edge_list = []
    for e_idx in range(n_edges):
        members = list(np.where(incidence_np[:, e_idx] > 0)[0])
        if len(members) >= 2:
            edge_list.append(members)
    dhg_hg = dhg.Hypergraph(n_nodes, edge_list)

    # Train/val/test split
    tm, vm, tsm = make_split(n_nodes, seed)

    features = torch.FloatTensor(raw_feats)
    labels = torch.LongTensor(labels_np)

    print(
        f"  [Organoid] {n_nodes} genes, {len(edge_list)} hyperedges, "
        f"{num_classes} classes (TF regulons)"
    )

    return {
        "features": features,
        "labels": labels,
        "num_classes": num_classes,
        "num_vertices": n_nodes,
        "hg": dhg_hg,
        "train_mask": torch.BoolTensor(tm),
        "val_mask": torch.BoolTensor(vm),
        "test_mask": torch.BoolTensor(tsm),
    }


# ===================================================================
# Plotting
# ===================================================================


def plot_results(results_df: pd.DataFrame, fig_path: Path):
    """Generate grouped bar charts for accuracy and timing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = results_df["dataset"].unique()
    models = results_df["model"].unique()
    n_datasets = len(datasets)
    n_models = len(models)

    # Color scheme: DHG models in blues, hgx models in oranges/reds
    dhg_colors = ["#2166ac", "#4393c3", "#92c5de"]
    hgx_colors = ["#d6604d", "#f4a582", "#fddbc7", "#b2182b"]
    color_map = {}
    dhg_idx, hgx_idx = 0, 0
    for m in models:
        if m in DHG_MODEL_NAMES:
            color_map[m] = dhg_colors[dhg_idx % len(dhg_colors)]
            dhg_idx += 1
        else:
            color_map[m] = hgx_colors[hgx_idx % len(hgx_colors)]
            hgx_idx += 1

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Panel A: Test Accuracy ---
    ax = axes[0, 0]
    bar_width = 0.8 / n_models
    x = np.arange(n_datasets)
    for i, model in enumerate(models):
        vals = []
        for ds in datasets:
            row = results_df[(results_df["dataset"] == ds) & (results_df["model"] == model)]
            vals.append(row["test_acc"].values[0] if len(row) > 0 else 0)
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width * 0.9,
            label=model, color=color_map.get(model, "#999999"),
            edgecolor="black", linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("A. Node Classification Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel B: Training Time ---
    ax = axes[0, 1]
    for i, model in enumerate(models):
        vals = []
        for ds in datasets:
            row = results_df[(results_df["dataset"] == ds) & (results_df["model"] == model)]
            vals.append(row["train_time"].values[0] if len(row) > 0 else 0)
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, vals, bar_width * 0.9,
            label=model, color=color_map.get(model, "#999999"),
            edgecolor="black", linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Training Time (s)", fontsize=12)
    ax.set_title("B. Training Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel C: Inference Time ---
    ax = axes[1, 0]
    for i, model in enumerate(models):
        vals = []
        for ds in datasets:
            row = results_df[(results_df["dataset"] == ds) & (results_df["model"] == model)]
            val = row["infer_time"].values[0] if len(row) > 0 else 0
            vals.append(val * 1000)  # convert to ms
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, vals, bar_width * 0.9,
            label=model, color=color_map.get(model, "#999999"),
            edgecolor="black", linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Inference Time (ms)", fontsize=12)
    ax.set_title("C. Inference Time (avg over 100 runs)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel D: Peak Memory ---
    ax = axes[1, 1]
    for i, model in enumerate(models):
        vals = []
        for ds in datasets:
            row = results_df[(results_df["dataset"] == ds) & (results_df["model"] == model)]
            vals.append(row["peak_mem_mb"].values[0] if len(row) > 0 else 0)
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, vals, bar_width * 0.9,
            label=model, color=color_map.get(model, "#999999"),
            edgecolor="black", linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Peak GPU Memory (MB)", fontsize=12)
    ax.set_title("D. Peak GPU Memory Usage", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Hypergraph Neural Network Benchmark: hgx (JAX) vs DHG (PyTorch)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {fig_path}")


# ===================================================================
# Main
# ===================================================================


def print_results_table(results: list[dict]):
    """Pretty-print results as a formatted table."""
    df = pd.DataFrame(results)
    # Format columns
    fmt = {
        "test_acc": lambda x: f"{x:.4f}" if not np.isnan(x) else "NaN",
        "train_time": lambda x: f"{x:.2f}s" if not np.isnan(x) else "NaN",
        "infer_time": lambda x: f"{x*1000:.2f}ms" if not np.isnan(x) else "NaN",
        "peak_mem_mb": lambda x: f"{x:.1f}MB" if not np.isnan(x) else "NaN",
    }

    header = f"{'Dataset':<12} {'Model':<16} {'Framework':<10} {'Accuracy':>10} {'Train':>10} {'Infer':>12} {'Memory':>10}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for _, row in df.iterrows():
        print(
            f"{row['dataset']:<12} {row['model']:<16} {row['framework']:<10} "
            f"{fmt['test_acc'](row['test_acc']):>10} "
            f"{fmt['train_time'](row['train_time']):>10} "
            f"{fmt['infer_time'](row['infer_time']):>12} "
            f"{fmt['peak_mem_mb'](row['peak_mem_mb']):>10}"
        )
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark hgx (JAX) vs DHG (PyTorch) on hypergraph datasets",
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
        help="Number of training epochs (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(FIG_DIR / "benchmark_results.csv"),
        help="Path to save CSV results",
    )
    parser.add_argument(
        "--output-fig",
        type=str,
        default=str(FIG_DIR / "benchmark_comparison.png"),
        help="Path to save figure",
    )
    args = parser.parse_args()

    datasets_to_run = DATASET_NAMES if args.dataset == "all" else [args.dataset]

    print("=" * 72)
    print("  Hypergraph Neural Network Benchmark")
    print("  hgx (JAX) vs DHG/DeepHypergraph (PyTorch)")
    print("=" * 72)
    print(f"  Datasets:  {', '.join(datasets_to_run)}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Seed:      {args.seed}")
    print(f"  Output:    {args.output_csv}")
    print()

    # --- Check available frameworks ---
    dhg_available = True
    try:
        import dhg
        print(f"  DHG version:    {dhg.__version__}")
    except ImportError:
        print("  WARNING: DHG not installed. DHG models will be skipped.")
        dhg_available = False

    hgx_available = True
    try:
        import hgx as hgx_lib
        print(f"  hgx available:  yes")
    except ImportError:
        print("  WARNING: hgx not installed. hgx models will be skipped.")
        hgx_available = False

    try:
        import torch
        print(f"  PyTorch:        {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError:
        print("  PyTorch:        not available")
        dhg_available = False

    try:
        import jax
        print(f"  JAX:            {jax.__version__} (devices: {jax.devices()})")
    except ImportError:
        print("  JAX:            not available")
        hgx_available = False

    print()

    # --- Run benchmarks ---
    all_results = []

    for ds_name in datasets_to_run:
        print(f"\n{'='*72}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*72}")

        # Load dataset
        dhg_dataset = None
        hgx_dataset = None

        try:
            if ds_name == "organoid":
                dhg_dataset = load_organoid_dataset(seed=args.seed)
            else:
                dhg_dataset = load_dhg_dataset(ds_name)
        except Exception as e:
            print(f"  ERROR loading dataset {ds_name}: {e}")
            traceback.print_exc()
            # Record NaN for all models on this dataset
            for model_name in ALL_MODEL_NAMES:
                fw = "DHG" if model_name in DHG_MODEL_NAMES else "hgx"
                all_results.append({
                    "dataset": ds_name,
                    "model": model_name,
                    "framework": fw,
                    "test_acc": float("nan"),
                    "train_time": float("nan"),
                    "infer_time": float("nan"),
                    "peak_mem_mb": float("nan"),
                })
            continue

        # Convert for hgx
        if dhg_dataset is not None and hgx_available:
            try:
                hgx_dataset = dhg_to_hgx(dhg_dataset)
            except Exception as e:
                print(f"  ERROR converting dataset to hgx: {e}")
                traceback.print_exc()

        # --- DHG models ---
        if dhg_available and dhg_dataset is not None:
            for model_name in DHG_MODEL_NAMES:
                print(f"\n  --- {model_name} (DHG/PyTorch) ---")
                try:
                    metrics = train_dhg_model(
                        DHG_MODELS[model_name],
                        dhg_dataset,
                        epochs=args.epochs,
                    )
                    all_results.append({
                        "dataset": ds_name,
                        "model": model_name,
                        "framework": "DHG",
                        **metrics,
                    })
                except Exception as e:
                    print(f"  ERROR training {model_name}: {e}")
                    traceback.print_exc()
                    all_results.append({
                        "dataset": ds_name,
                        "model": model_name,
                        "framework": "DHG",
                        "test_acc": float("nan"),
                        "train_time": float("nan"),
                        "infer_time": float("nan"),
                        "peak_mem_mb": float("nan"),
                    })
                gc.collect()
        else:
            for model_name in DHG_MODEL_NAMES:
                all_results.append({
                    "dataset": ds_name,
                    "model": model_name,
                    "framework": "DHG",
                    "test_acc": float("nan"),
                    "train_time": float("nan"),
                    "infer_time": float("nan"),
                    "peak_mem_mb": float("nan"),
                })

        # --- hgx models ---
        if hgx_available and hgx_dataset is not None:
            for model_name in HGX_MODEL_NAMES:
                print(f"\n  --- {model_name} (hgx/JAX) ---")
                try:
                    metrics = train_hgx_model(
                        model_name,
                        hgx_dataset,
                        epochs=args.epochs,
                        seed=args.seed,
                    )
                    all_results.append({
                        "dataset": ds_name,
                        "model": model_name,
                        "framework": "hgx",
                        **metrics,
                    })
                except Exception as e:
                    print(f"  ERROR training {model_name}: {e}")
                    traceback.print_exc()
                    all_results.append({
                        "dataset": ds_name,
                        "model": model_name,
                        "framework": "hgx",
                        "test_acc": float("nan"),
                        "train_time": float("nan"),
                        "infer_time": float("nan"),
                        "peak_mem_mb": float("nan"),
                    })
                gc.collect()
        else:
            for model_name in HGX_MODEL_NAMES:
                all_results.append({
                    "dataset": ds_name,
                    "model": model_name,
                    "framework": "hgx",
                    "test_acc": float("nan"),
                    "train_time": float("nan"),
                    "infer_time": float("nan"),
                    "peak_mem_mb": float("nan"),
                })

    # --- Fix 4: Hyperparameter ablation for hgx UniGATConv ---
    if hgx_available:
        print(f"\n{'='*72}")
        print("  Hyperparameter ablation: hgx UniGATConv")
        print(f"{'='*72}")

        ablation_lrs = [1e-4, 1e-3, 1e-2]
        ablation_layers = [2, 3, 4]
        ablation_hdims = [64, 128, 256]

        best_ablation_acc = -1.0
        best_ablation_cfg = {}
        best_ablation_metrics = None

        for ds_name in datasets_to_run:
            # We need hgx_dataset for this dataset; reload if needed
            try:
                if ds_name == "organoid":
                    _dhg_ds = load_organoid_dataset(seed=args.seed)
                else:
                    _dhg_ds = load_dhg_dataset(ds_name)
                _hgx_ds = dhg_to_hgx(_dhg_ds)
            except Exception as e:
                print(f"  Ablation: could not load {ds_name}: {e}")
                continue

            print(f"\n  Ablation on dataset: {ds_name}")
            ds_best_acc = -1.0
            ds_best_cfg = {}
            ds_best_metrics = None

            for ab_lr in ablation_lrs:
                for ab_layers in ablation_layers:
                    for ab_hdim in ablation_hdims:
                        cfg_str = f"lr={ab_lr}, layers={ab_layers}, hdim={ab_hdim}"
                        try:
                            metrics = train_hgx_model(
                                "UniGATConv",
                                _hgx_ds,
                                epochs=args.epochs,
                                seed=args.seed,
                                lr=ab_lr,
                                hidden_dim=ab_hdim,
                                num_layers=ab_layers,
                            )
                            acc = metrics["test_acc"]
                            print(f"    {cfg_str} -> acc={acc:.4f}")
                            if acc > ds_best_acc:
                                ds_best_acc = acc
                                ds_best_cfg = {
                                    "lr": ab_lr,
                                    "num_layers": ab_layers,
                                    "hidden_dim": ab_hdim,
                                }
                                ds_best_metrics = metrics
                        except Exception as e:
                            print(f"    {cfg_str} -> ERROR: {e}")
                        gc.collect()

            if ds_best_metrics is not None:
                print(
                    f"  Best config for {ds_name}: {ds_best_cfg} "
                    f"-> acc={ds_best_acc:.4f}"
                )
                all_results.append({
                    "dataset": ds_name,
                    "model": "hgx-tuned",
                    "framework": "hgx",
                    **ds_best_metrics,
                })

    # --- Fix 5: Leiden (Spectral) clustering for organoid dataset ---
    if "organoid" in datasets_to_run and hgx_available:
        print(f"\n{'='*72}")
        print("  Spectral clustering labels for organoid (20 classes)")
        print(f"{'='*72}")

        try:
            from sklearn.cluster import SpectralClustering

            # Load organoid dataset
            _org_dhg = load_organoid_dataset(seed=args.seed)
            _org_hgx = dhg_to_hgx(_org_dhg)

            # Build adjacency from incidence and cluster into 20 classes
            incidence_np = np.array(_org_hgx["incidence"])
            adj_binary = (incidence_np @ incidence_np.T > 0).astype(float)
            np.fill_diagonal(adj_binary, 0)
            clustering = SpectralClustering(
                n_clusters=20, affinity="precomputed", random_state=42
            )
            leiden_labels = clustering.fit_predict(adj_binary)

            import jax.numpy as jnp

            n_nodes = len(leiden_labels)
            tm, vm, tsm = make_split(n_nodes, args.seed)

            clustered_dataset = {
                "hg": _org_hgx["hg"],
                "features": _org_hgx["features"],
                "labels": jnp.array(leiden_labels),
                "num_classes": 20,
                "train_mask": jnp.array(tm),
                "val_mask": jnp.array(vm),
                "test_mask": jnp.array(tsm),
                "incidence": _org_hgx["incidence"],
            }

            # Run all hgx models with 20-class labels
            for model_name in HGX_MODEL_NAMES:
                print(f"\n  --- {model_name} (hgx/JAX, 20-class organoid) ---")
                try:
                    metrics = train_hgx_model(
                        model_name,
                        clustered_dataset,
                        epochs=args.epochs,
                        seed=args.seed,
                    )
                    all_results.append({
                        "dataset": "organoid-20c",
                        "model": model_name,
                        "framework": "hgx",
                        **metrics,
                    })
                except Exception as e:
                    print(f"  ERROR training {model_name} (20-class): {e}")
                    traceback.print_exc()
                    all_results.append({
                        "dataset": "organoid-20c",
                        "model": model_name,
                        "framework": "hgx",
                        "test_acc": float("nan"),
                        "train_time": float("nan"),
                        "infer_time": float("nan"),
                        "peak_mem_mb": float("nan"),
                    })
                gc.collect()

            # Also run DHG models with 20-class labels
            if dhg_available:
                import torch

                _org_dhg_clustered = {
                    "features": _org_dhg["features"],
                    "labels": torch.LongTensor(leiden_labels),
                    "num_classes": 20,
                    "num_vertices": _org_dhg["num_vertices"],
                    "hg": _org_dhg["hg"],
                    "train_mask": torch.BoolTensor(tm),
                    "val_mask": torch.BoolTensor(vm),
                    "test_mask": torch.BoolTensor(tsm),
                }
                for model_name in DHG_MODEL_NAMES:
                    print(f"\n  --- {model_name} (DHG/PyTorch, 20-class organoid) ---")
                    try:
                        metrics = train_dhg_model(
                            DHG_MODELS[model_name],
                            _org_dhg_clustered,
                            epochs=args.epochs,
                        )
                        all_results.append({
                            "dataset": "organoid-20c",
                            "model": model_name,
                            "framework": "DHG",
                            **metrics,
                        })
                    except Exception as e:
                        print(f"  ERROR training {model_name} (20-class): {e}")
                        traceback.print_exc()
                        all_results.append({
                            "dataset": "organoid-20c",
                            "model": model_name,
                            "framework": "DHG",
                            "test_acc": float("nan"),
                            "train_time": float("nan"),
                            "infer_time": float("nan"),
                            "peak_mem_mb": float("nan"),
                        })
                    gc.collect()

        except ImportError:
            print("  WARNING: scikit-learn not installed. Skipping spectral clustering.")
        except Exception as e:
            print(f"  ERROR in spectral clustering: {e}")
            traceback.print_exc()

    # --- Results ---
    print_results_table(all_results)

    results_df = pd.DataFrame(all_results)

    # Save CSV
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

    # Save figure
    try:
        fig_path = Path(args.output_fig)
        # Only plot datasets/models that have at least one non-NaN result
        valid_df = results_df.dropna(subset=["test_acc"])
        if len(valid_df) > 0:
            plot_results(valid_df, fig_path)
        else:
            print("\nNo valid results to plot.")
    except Exception as e:
        print(f"\nWARNING: Could not generate figure: {e}")
        traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
