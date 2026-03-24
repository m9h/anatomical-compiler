"""Lightweight Planetoid dataset loader (Cora, Citeseer, Pubmed).

Drop-in replacement for dhg.data.{Cora,Citeseer,Pubmed} that does NOT
require PyTorch.  Downloads raw .npz files from the Kimura/Planetoid
GitHub mirrors and caches them locally.

Each dataset object is dict-like with keys:
    features, labels, num_classes, edge_list, train_mask, val_mask, test_mask
"""

from __future__ import annotations

import os
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

_PLANETOID_URL = (
    "https://github.com/kimiyoung/planetoid/raw/master/data/"
)

_CACHE_DIR = Path(os.environ.get(
    "PLANETOID_CACHE",
    Path.home() / ".cache" / "planetoid",
))


def _download(name: str) -> Path:
    """Download all Planetoid files for *name* and return the cache dir."""
    out = _CACHE_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    suffixes = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]
    for s in suffixes:
        fname = f"ind.{name}.{s}"
        dest = out / fname
        if dest.exists():
            continue
        url = _PLANETOID_URL + fname
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    return out


def _load_planetoid(name: str) -> dict:
    """Load a Planetoid dataset and return a dict compatible with DHG."""
    cache = _download(name)

    def _pkl(suffix):
        with open(cache / f"ind.{name}.{suffix}", "rb") as f:
            return pickle.load(f, encoding="latin1")

    x = _pkl("x")        # training features (sparse)
    y = _pkl("y")        # training labels (one-hot)
    tx = _pkl("tx")       # test features (sparse)
    ty = _pkl("ty")       # test labels (one-hot)
    allx = _pkl("allx")   # all training features (sparse)
    ally = _pkl("ally")    # all training labels (one-hot)
    graph = _pkl("graph")  # adjacency dict {node: [neighbours]}

    test_idx = []
    with open(cache / f"ind.{name}.test.index") as f:
        for line in f:
            test_idx.append(int(line.strip()))
    test_idx = np.array(test_idx)

    # --- Combine features and labels (standard Planetoid procedure) ---
    # allx contains features for train+val nodes (indices 0..len(allx)-1).
    # tx contains features for test nodes (in test_idx order).
    # After vstack, rows len(allx)..len(allx)+len(tx)-1 hold test features.
    # We reorder so that features[test_idx[i]] = tx[i].
    features = sp.vstack([allx, tx]).toarray()
    labels_onehot = np.vstack([ally, ty])

    # Sort test indices and reorder tx/ty accordingly
    test_idx_reorder = np.argsort(test_idx)
    test_idx_sorted = test_idx[test_idx_reorder]

    # Citeseer has non-contiguous test indices with gaps.  Extend the total
    # number of nodes to max(test_idx)+1 if needed.
    n_total = max(features.shape[0], int(test_idx_sorted.max()) + 1)
    if n_total > features.shape[0]:
        extra = n_total - features.shape[0]
        features = np.vstack([features, np.zeros((extra, features.shape[1]), dtype=features.dtype)])
        labels_onehot = np.vstack([labels_onehot, np.zeros((extra, labels_onehot.shape[1]), dtype=labels_onehot.dtype)])

    # Place test data at the correct row indices
    tx_dense = tx.toarray() if sp.issparse(tx) else tx
    features[test_idx_sorted] = tx_dense[test_idx_reorder]
    labels_onehot[test_idx_sorted] = ty[test_idx_reorder]

    test_idx = test_idx_sorted
    labels = np.argmax(labels_onehot, axis=1).astype(np.int32)

    num_classes = labels_onehot.shape[1]
    n = features.shape[0]

    # --- Edge list ---
    edge_list = []
    for src, dsts in graph.items():
        for dst in dsts:
            if src < n and dst < n:
                edge_list.append((src, dst))

    # --- Standard masks ---
    n_train = x.shape[0]
    n_val = 500
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[test_idx] = True

    return {
        "features": features.astype(np.float32),
        "labels": labels,
        "num_classes": int(num_classes),
        "edge_list": edge_list,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


# ---------------------------------------------------------------------------
# Public API (mimics dhg.data.*)
# ---------------------------------------------------------------------------

class Cora(dict):
    def __init__(self):
        super().__init__(_load_planetoid("cora"))

class Citeseer(dict):
    def __init__(self):
        super().__init__(_load_planetoid("citeseer"))

class Pubmed(dict):
    def __init__(self):
        super().__init__(_load_planetoid("pubmed"))
