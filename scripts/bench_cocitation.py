#!/usr/bin/env python3
"""Benchmark hgx on all DHG cocitation/coauthorship datasets."""
import numpy as np, time
import jax, jax.numpy as jnp, equinox as eqx, optax, hgx, dhg

PUBLISHED = {
    "CocitationCora": {"HGNN": 79.39, "HyperGCN": 78.45, "UniGCN": 78.95, "AllSet": 78.58},
    "CocitationCiteseer": {"HGNN": 72.01, "HyperGCN": 71.22, "UniGCN": 71.63, "AllSet": 70.83},
    "CocitationPubmed": {"HGNN": 86.44, "HyperGCN": 82.80, "UniGCN": 79.28, "AllSet": 78.58},
    "CoauthorshipCora": {"HGNN": 82.64, "HyperGCN": 79.92, "UniGCN": 81.08, "AllSet": 79.75},
}

# Use cocitation hyperedges but Planetoid splits from graph versions
DATASETS = [
    ("CocitationCora", dhg.data.CocitationCora, dhg.data.Cora),
    ("CocitationCiteseer", dhg.data.CocitationCiteseer, dhg.data.Citeseer),
    ("CocitationPubmed", dhg.data.CocitationPubmed, dhg.data.Pubmed),
    ("CoauthorshipCora", dhg.data.CoauthorshipCora, dhg.data.Cora),
]

CONVS = [
    ("UniGCNConv", hgx.UniGCNConv),
    ("UniGATConv", hgx.UniGATConv),
    ("UniGINConv", hgx.UniGINConv),
]

print(f"JAX {jax.__version__} on {jax.devices()}")
results = []

for ds_name, ds_cls, graph_cls in DATASETS:
    print(f"\n{'='*60}")
    print(f"  {ds_name}")
    print("=" * 60)

    try:
        data = ds_cls()
        feats = np.array(data["features"], dtype=np.float32)
        labels = np.array(data["labels"], dtype=np.int32)
        edge_list = data["edge_list"]
        nc = data["num_classes"]
        n, fd = feats.shape

        # Use Planetoid splits from graph version (standard: train=140, val=500, test=1000)
        try:
            graph_data = graph_cls()
            train_mask = np.array(graph_data["train_mask"], dtype=bool)
            val_mask = np.array(graph_data["val_mask"], dtype=bool)
            test_mask = np.array(graph_data["test_mask"], dtype=bool)
            split_source = "Planetoid"
        except Exception:
            try:
                train_mask = np.array(data["train_mask"], dtype=bool)
                val_mask = np.array(data["val_mask"], dtype=bool)
                test_mask = np.array(data["test_mask"], dtype=bool)
                split_source = "DHG cocitation"
            except Exception:
                idx = np.random.permutation(n)
                s1, s2 = int(0.6*n), int(0.8*n)
                train_mask = np.zeros(n, dtype=bool); train_mask[idx[:s1]] = True
                val_mask = np.zeros(n, dtype=bool); val_mask[idx[s1:s2]] = True
                test_mask = np.zeros(n, dtype=bool); test_mask[idx[s2:]] = True
                split_source = "random 60/20/20"

        # Build incidence from edge_list
        m = len(edge_list)
        inc = np.zeros((n, m), dtype=np.float32)
        for j, he in enumerate(edge_list):
            for i in he:
                inc[i, j] = 1.0

        print(f"  {n} nodes, {fd} feats, {nc} classes, {m} edges")
        print(f"  train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()} ({split_source})")

        hg = hgx.from_incidence(jnp.array(inc), node_features=jnp.array(feats))
        labels_j = jnp.array(labels)
        train_j = jnp.array(train_mask)
        val_j = jnp.array(val_mask)
        test_j = jnp.array(test_mask)

        for conv_name, conv_cls in CONVS:
            seed_accs = []
            infer_ms = 0

            for seed in [42, 43, 44]:
                k = jax.random.PRNGKey(seed)
                model = hgx.HGNNStack(
                    conv_dims=[(fd, 64), (64, 64)],
                    conv_cls=conv_cls,
                    readout_dim=nc,
                    dropout_rate=0.5,
                    key=k,
                )
                opt = optax.adam(0.01)
                opt_state = opt.init(eqx.filter(model, eqx.is_array))

                @eqx.filter_jit
                def step(m, o, hg, lab, mask):
                    def lf(m):
                        logits = m(hg, inference=True)
                        lp = jax.nn.log_softmax(logits, -1)
                        oh = jax.nn.one_hot(lab, nc)
                        per_node = -jnp.sum(oh * lp, -1)
                        return jnp.where(mask, per_node, 0.0).sum() / mask.sum()
                    l, g = eqx.filter_value_and_grad(lf)(m)
                    u, no = opt.update(g, o, m)
                    return eqx.apply_updates(m, u), no, l

                best_val_ep = 0
                best_val_acc = 0.0
                best_model = model
                t0 = time.time()
                for ep in range(200):
                    model, opt_state, loss = step(model, opt_state, hg, labels_j, train_j)
                    if (ep + 1) % 10 == 0:
                        logits = model(hg, inference=True)
                        preds = jnp.argmax(logits, -1)
                        va = float(jnp.sum((preds == labels_j) & val_j) / val_j.sum())
                        if va > best_val_acc:
                            best_val_acc = va
                            best_val_ep = ep + 1
                            best_model = model
                        elif (ep + 1) - best_val_ep >= 50:
                            break
                elapsed = time.time() - t0

                logits = best_model(hg, inference=True)
                preds = jnp.argmax(logits, -1)
                ta = float(jnp.sum((preds == labels_j) & test_j) / test_j.sum())
                seed_accs.append(ta)

            # Inference timing on last model
            for _ in range(10):
                _ = model(hg, inference=True)
            t0 = time.time()
            for _ in range(100):
                logits = model(hg, inference=True)
                logits.block_until_ready()
            infer_ms = (time.time() - t0) / 100 * 1000

            mean_acc = np.mean(seed_accs) * 100
            std_acc = np.std(seed_accs) * 100
            print(f"  {conv_name:15s}  {mean_acc:.2f} +/- {std_acc:.2f}%  infer={infer_ms:.2f}ms")
            results.append({"ds": ds_name, "model": conv_name, "acc": mean_acc, "std": std_acc, "infer": infer_ms})

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("  COMPLETE RESULTS: hgx vs Published Baselines")
print("=" * 80)
print(f"{'Dataset':25s} {'Model':15s} {'hgx':>14s}  {'Best Published':>20s}")
print("-" * 80)
for r in results:
    pub = PUBLISHED.get(r["ds"], {})
    if pub:
        best_name = max(pub, key=pub.get)
        best_val = pub[best_name]
        gap = r["acc"] - best_val
        marker = "MATCH" if abs(gap) < 2 else ("ABOVE" if gap > 0 else "BELOW")
        print(f"{r['ds']:25s} {r['model']:15s} {r['acc']:5.2f}+/-{r['std']:.2f}%  {best_name:>8s} {best_val:.2f}%  [{marker} {gap:+.2f}]")
    else:
        print(f"{r['ds']:25s} {r['model']:15s} {r['acc']:5.2f}+/-{r['std']:.2f}%")
