#!/usr/bin/env python3
"""Test HGNN symmetric convolution (D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}) on Citeseer."""
import numpy as np, time
from collections import defaultdict
import jax, jax.numpy as jnp, equinox as eqx, optax, hgx, dhg

# Custom HGNNConv with symmetric normalization
class HGNNConv(eqx.Module):
    """HGNN convolution: X' = sigma(D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X Theta)"""
    linear: eqx.nn.Linear

    def __init__(self, in_dim, out_dim, *, key):
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)

    def __call__(self, hg):
        H = hg.incidence
        x = jax.vmap(self.linear)(hg.node_features)
        d_v = jnp.sum(H, axis=1)
        d_e = jnp.sum(H, axis=0)
        d_v_sqrt_inv = jnp.where(d_v > 0, 1.0 / jnp.sqrt(d_v), 0.0)
        d_e_inv = jnp.where(d_e > 0, 1.0 / d_e, 0.0)
        # HGNN: D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} X
        x_norm = x * d_v_sqrt_inv[:, None]           # D_v^{-1/2} X
        e = H.T @ x_norm                              # H^T D_v^{-1/2} X
        e = e * d_e_inv[:, None]                       # D_e^{-1} H^T D_v^{-1/2} X
        out = H @ e                                    # H D_e^{-1} H^T D_v^{-1/2} X
        out = out * d_v_sqrt_inv[:, None]              # D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} X
        return out

class HGNNModel(eqx.Module):
    conv1: HGNNConv
    conv2: HGNNConv
    readout: eqx.nn.Linear
    dropout_rate: float = eqx.field(static=True)

    def __init__(self, in_dim, hid, out_dim, dropout, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv1 = HGNNConv(in_dim, hid, key=k1)
        self.conv2 = HGNNConv(hid, hid, key=k2)
        self.readout = eqx.nn.Linear(hid, out_dim, key=k3)
        self.dropout_rate = dropout

    def __call__(self, hg, *, inference=True):
        x = jax.nn.relu(self.conv1(hg))
        hg2 = hgx.from_incidence(hg.incidence, node_features=x)
        x2 = jax.nn.relu(self.conv2(hg2))
        return jax.vmap(self.readout)(x2)

def run_dataset(name, data_cls, graph_cls):
    data = data_cls()
    graph_data = graph_cls()
    feats = np.array(data["features"], dtype=np.float32)
    labels = np.array(data["labels"], dtype=np.int32)
    edges = data["edge_list"]
    n, fd = feats.shape; nc = data["num_classes"]
    train_mask = np.array(graph_data["train_mask"], dtype=bool)
    val_mask = np.array(graph_data["val_mask"], dtype=bool)
    test_mask = np.array(graph_data["test_mask"], dtype=bool)

    nbrs = defaultdict(set)
    for s, d in edges: nbrs[s].add(d); nbrs[d].add(s)
    hes = [sorted(nbrs[node] | {node}) for node in range(n) if len(nbrs[node]) > 0]
    m = len(hes)
    inc = np.zeros((n, m), dtype=np.float32)
    for j, he in enumerate(hes):
        for i in he: inc[i, j] = 1.0

    print(f"\n{name}: {n} nodes, {fd} feats, {nc} classes, {m} edges")
    print(f"  train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    hg = hgx.from_incidence(jnp.array(inc), node_features=jnp.array(feats))
    lj = jnp.array(labels)
    trj = jnp.array(train_mask); vj = jnp.array(val_mask); tj = jnp.array(test_mask)

    # Test both UniGCNConv and HGNNConv
    for conv_label, model_fn in [
        ("UniGCNConv (hgx)", lambda k: hgx.HGNNStack(
            conv_dims=[(fd, 64), (64, 64)], conv_cls=hgx.UniGCNConv,
            readout_dim=nc, dropout_rate=0.5, key=k)),
        ("HGNNConv (symmetric)", lambda k: HGNNModel(fd, 64, nc, 0.5, key=k)),
    ]:
        accs = []
        for seed in [42, 43, 44, 45, 46]:
            k = jax.random.PRNGKey(seed)
            model = model_fn(k)
            opt = optax.adam(0.01)
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
            accs.append(ta)

        mean_a = np.mean(accs) * 100; std_a = np.std(accs) * 100
        print(f"  {conv_label:30s}  {mean_a:.2f} +/- {std_a:.2f}%")

# Run on both Cora and Citeseer
run_dataset("Cora", dhg.data.Cora, dhg.data.Cora)
run_dataset("Citeseer", dhg.data.Citeseer, dhg.data.Citeseer)

print("\nPublished: Cora HGNN 79.39%, Citeseer HGNN 72.01%")
