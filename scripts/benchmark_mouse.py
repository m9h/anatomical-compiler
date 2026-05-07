import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import hgx
    import jax
    import jax.numpy as jnp
    import devograph
    from devograph import PerturbationPredictor
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

def load_dataset(data_dir: Path, name: str) -> dict | None:
    d = {"name": name, "dir": data_dir}
    for fname in ["gene_names", "tf_names"]:
        path = data_dir / f"{fname}.json"
        if path.exists():
            with open(path) as f:
                d[fname] = json.load(f)
        else:
            return None
    for fname in ["incidence", "node_features_pca"]:
        path = data_dir / f"{fname}.npy"
        if path.exists():
            d[fname] = np.load(path)
    return d

def main():
    parser = argparse.ArgumentParser(description="Benchmark hgx on Mouse neocortex dataset")
    parser.add_argument("--fleck-dir", type=str, default="data/processed", help="Fleck dir")
    parser.add_argument("--mouse-dir", type=str, default="data/mouse/processed", help="Mouse dir")
    args = parser.parse_args()

    fleck = load_dataset(Path(args.fleck_dir), "Fleck")
    mouse = load_dataset(Path(args.mouse_dir), "Mouse")

    if not fleck or not mouse:
        print("Missing datasets")
        return

    print(f"Fleck: {len(fleck['gene_names'])} genes, {len(fleck['tf_names'])} TFs")
    print(f"Mouse: {len(mouse['gene_names'])} genes, {len(mouse['tf_names'])} TFs")

    # 1. Map shared genes (Human UPPER -> Mouse TitleCase)
    f_genes = [g.upper() for g in fleck["gene_names"]]
    m_genes = [g.upper() for g in mouse["gene_names"]]
    shared_genes = sorted(set(f_genes) & set(m_genes))
    print(f"\nShared genes: {len(shared_genes)}")
    
    f_idx = [f_genes.index(g) for g in shared_genes]
    m_idx = [m_genes.index(g) for g in shared_genes]
    
    # 2. Shared TFs
    f_tfs = [t.upper() for t in fleck["tf_names"]]
    m_tfs = [t.upper() for t in mouse["tf_names"]]
    shared_tfs = sorted(set(f_tfs) & set(m_tfs))
    print(f"Shared TFs: {shared_tfs}")

    # 3. Cross-Species Transfer (Human Fleck -> Mouse Neocortex)
    print("\nBenchmark: Cross-Species (Fleck Human -> Loo Mouse)")
    results = []
    
    if HAS_LIBS:
        # Load human hypergraph
        feat_f = jnp.array(fleck["node_features_pca"])
        hg_f = hgx.from_incidence(jnp.array(fleck["incidence"]), node_features=feat_f)
        
        # In this benchmark, we don't have real Mouse perturbations, 
        # so we validate Fleck's predicted Mouse topology against Mouse correlation logic.
        
        for tf_sym in shared_tfs:
            # Human prediction (topology row)
            fi = f_tfs.index(tf_sym)
            pred_f = fleck["incidence"][:, fi] # (n_human_genes)
            pred_shared = pred_f[f_idx]        # (n_shared_genes)
            
            # Mouse observation (correlation row)
            mi = m_tfs.index(tf_sym)
            obs_m = mouse["incidence"][:, mi]  # (n_mouse_genes)
            obs_shared = obs_m[m_idx]          # (n_shared_genes)
            
            mask = (pred_shared != 0) | (obs_shared != 0)
            if mask.any():
                concord = (pred_shared[mask] == obs_shared[mask]).mean()
                results.append({"tf": tf_sym, "concord": float(concord)})
                print(f"  {tf_sym:<10}: Topology Concordance {concord:>6.1%}")

    # 4. Summary & Plot
    if results:
        mean_concord = np.mean([r["concord"] for r in results])
        print(f"\nMean Human-Mouse Topology Concordance: {mean_concord:.1%}")
        
        plt.figure(figsize=(10, 5))
        tfs = [r["tf"] for r in results]
        concords = [r["concord"] for r in results]
        plt.bar(tfs, concords, color="mediumseagreen")
        plt.axhline(y=0.5, color="gray", linestyle="--")
        plt.ylabel("Topology Concordance")
        plt.title("Human (Organoid) -> Mouse (Neocortex) GRN Transfer")
        plt.savefig(PROJECT_ROOT / "figures/mouse_benchmark.png")
        print(f"Figure saved to figures/mouse_benchmark.png")

if __name__ == "__main__":
    main()
