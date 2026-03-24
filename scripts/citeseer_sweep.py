#!/usr/bin/env python3
"""Citeseer hyperparameter sweep to close the gap vs published HGNN (72.01%)."""
import numpy as np, time
from collections import defaultdict
import jax, jax.numpy as jnp, equinox as eqx, optax, hgx
try:
    import dhg; data = dhg.data.Citeseer()
except ImportError:
    from planetoid_loader import Citeseer; data = Citeseer()
feats = np.array(data["features"], dtype=np.float32)
labels = np.array(data["labels"], dtype=np.int32)
edges = data["edge_list"]; n, fd = feats.shape; nc = data["num_classes"]
train_mask = np.array(data["train_mask"], dtype=bool)
val_mask = np.array(data["val_mask"], dtype=bool)
test_mask = np.array(data["test_mask"], dtype=bool)

nbrs = defaultdict(set)
for s, d in edges: nbrs[s].add(d); nbrs[d].add(s)
hes = [sorted(nbrs[node] | {node}) for node in range(n) if len(nbrs[node]) > 0]
m = len(hes)
inc = np.zeros((n, m), dtype=np.float32)
for j, he in enumerate(hes):
    for i in he: inc[i, j] = 1.0
iso = (inc.sum(1) == 0).sum()
print(f"Citeseer: {n} nodes, {fd} feats, {nc} classes, {m} edges, {iso} isolated")

lj = jnp.array(labels); trj = jnp.array(train_mask); vj = jnp.array(val_mask); tj = jnp.array(test_mask)

configs = [
    ("baseline",    0.01,  0.5, 64,  2, None, False),
    ("lr=0.005",    0.005, 0.5, 64,  2, None, False),
    ("lr=0.001",    0.001, 0.5, 64,  2, None, False),
    ("drop=0.6",    0.01,  0.6, 64,  2, None, False),
    ("drop=0.7",    0.01,  0.7, 64,  2, None, False),
    ("hidden=128",  0.01,  0.5, 128, 2, None, False),
    ("hidden=256",  0.01,  0.5, 256, 2, None, False),
    ("3-layer",     0.01,  0.5, 64,  3, None, False),
    ("wd=5e-4",     0.01,  0.5, 64,  2, 5e-4, False),
    ("wd=1e-3",     0.01,  0.5, 64,  2, 1e-3, False),
    ("feat_norm",   0.01,  0.5, 64,  2, None, True),
    ("combo",       0.005, 0.6, 128, 2, 5e-4, True),
]

print(f"{'Config':15s} {'Test':>8s} {'Val':>8s} {'Epoch':>6s} {'Time':>6s}")
print("-" * 50)

for name, lr, drop, hid, nlayers, wd, feat_norm in configs:
    f_in = feats.copy()
    if feat_norm:
        norms = np.linalg.norm(f_in, axis=1, keepdims=True)
        f_in = f_in / np.maximum(norms, 1e-8)
    hg = hgx.from_incidence(jnp.array(inc), node_features=jnp.array(f_in))
    dims = [(fd, hid)] + [(hid, hid)] * (nlayers - 1)
    k = jax.random.PRNGKey(42)
    model = hgx.HGNNStack(conv_dims=dims, conv_cls=hgx.UniGCNConv, readout_dim=nc, dropout_rate=drop, key=k)
    opt = optax.adamw(lr, weight_decay=wd) if wd else optax.adam(lr)
    ost = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, o, hg, lab, mask):
        def lf(m):
            logits = m(hg, inference=True)
            return jnp.where(mask, -jnp.sum(jax.nn.one_hot(lab, nc) * jax.nn.log_softmax(logits, -1), -1), 0.0).sum() / mask.sum()
        l, g = eqx.filter_value_and_grad(lf)(m)
        u, no = opt.update(g, o, m)
        return eqx.apply_updates(m, u), no, l

    bv = 0; be = 0; bm = model
    t0 = time.time()
    for ep in range(300):
        model, ost, loss = step(model, ost, hg, lj, trj)
        if (ep + 1) % 10 == 0:
            va = float(jnp.sum((jnp.argmax(model(hg, inference=True), -1) == lj) & vj) / vj.sum())
            if va > bv: bv = va; be = ep + 1; bm = model
            elif (ep + 1) - be >= 100: break
    elapsed = time.time() - t0
    ta = float(jnp.sum((jnp.argmax(bm(hg, inference=True), -1) == lj) & tj) / tj.sum())
    print(f"{name:15s} {ta*100:7.2f}% {bv*100:7.1f}% {be:5d} {elapsed:5.1f}s")

print("Published HGNN: 72.01%")
