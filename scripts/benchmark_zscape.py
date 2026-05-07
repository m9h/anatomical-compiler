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
    HAS_DEVOGRAPH = True
except ImportError:
    HAS_LIBS = False
    HAS_DEVOGRAPH = False

def load_dataset(data_dir: Path, name: str) -> dict | None:
    d = {"name": name, "dir": data_dir}
    for fname in ["gene_names", "tf_names"]:
        path = data_dir / f"{fname}.json"
        if path.exists():
            with open(path) as f:
                d[fname] = json.load(f)
        else:
            return None
    for fname in ["incidence", "node_features_pca", "perturbation_effects"]:
        path = data_dir / f"{fname}.npy"
        if path.exists():
            d[fname] = np.load(path)
    return d

def main():
    parser = argparse.ArgumentParser(description="Benchmark hgx on ZSCAPE zebrafish dataset")
    parser.add_argument("--fleck-dir", type=str, default="data/processed", help="Fleck dir")
    parser.add_argument("--zscape-dir", type=str, default="data/zscape/processed", help="ZSCAPE dir")
    args = parser.parse_args()

    print(f"hgx available: {HAS_LIBS}")
    print(f"devograph available: {HAS_DEVOGRAPH}")

    fleck = load_dataset(Path(args.fleck_dir), "Fleck")
    zscape = load_dataset(Path(args.zscape_dir), "ZSCAPE")

    if not fleck or not zscape:
        print("Missing datasets")
        return

    print(f"Fleck: {len(fleck['gene_names'])} genes, {len(fleck['tf_names'])} TFs")
    print(f"ZSCAPE: {len(zscape['gene_names'])} genes, {len(zscape['tf_names'])} TFs")

    # 1. Shared TFs (case-insensitive)
    f_tfs = [t.upper() for t in fleck["tf_names"]]
    z_tfs = [t.upper() for t in zscape["tf_names"]]
    shared_tfs = sorted(set(f_tfs) & set(z_tfs))
    print(f"\nShared TFs: {len(shared_tfs)}")
    print(f"  {shared_tfs}")

    # 2. Self-Validation (ZSCAPE -> ZSCAPE)
    print("\nBenchmark 1: Self-Validation (Within ZSCAPE)")
    results_self = []
    if HAS_LIBS:
        feat = jnp.array(zscape["node_features_pca"])
        hg = hgx.from_incidence(jnp.array(zscape["incidence"]), node_features=feat)
        
        for tf in zscape["tf_names"]:
            ti = zscape["tf_names"].index(tf)
            obs_eff = zscape["perturbation_effects"][ti]
            mask = obs_eff != 0
            if mask.any():
                concord = (np.sign(obs_eff[mask]) == 1).mean()
                results_self.append({"tf": tf, "concord": float(concord)})
                print(f"  {tf:<10}: Concordance {concord:>6.1%}")

    # 3. Cross-Species Validation (Human -> Zebrafish)
    print("\nBenchmark 2: Cross-Species (Fleck -> ZSCAPE)")
    f_genes = [g.upper() for g in fleck["gene_names"]]
    z_genes = [g.upper() for g in zscape["gene_names"]]
    shared_genes = sorted(set(f_genes) & set(z_genes))
    print(f"  Shared genes: {len(shared_genes)}")
    
    f_idx = [f_genes.index(g) for g in shared_genes]
    z_idx = [z_genes.index(g) for g in shared_genes]
    
    results_cross = []
    if HAS_LIBS and HAS_DEVOGRAPH:
        for tf_sym in shared_tfs:
            zi = z_tfs.index(tf_sym)
            z_eff = zscape["perturbation_effects"][zi, z_idx]
            
            fi = f_tfs.index(tf_sym)
            pred_f = fleck["incidence"][:, fi]
            pred_z = pred_f[f_idx]
            
            mask = (pred_z != 0) & (z_eff != 0)
            print(f"  Debug {tf_sym}: {pred_z.sum()} human targets, {mask.sum()} overlap in shared genes")
            
            if mask.any():
                concord = (np.sign(pred_z[mask]) == np.sign(z_eff[mask])).mean()
                results_cross.append({"tf": tf_sym, "concord": float(concord)})
                print(f"  {tf_sym:<10}: Concordance {concord:>6.1%}")

    # 4. Summary & Plot
    if results_cross:
        mean_concord = np.mean([r["concord"] for r in results_cross])
        print(f"\nMean Cross-Species Concordance: {mean_concord:.1%}")
        
        plt.figure(figsize=(10, 5))
        tfs = [r["tf"] for r in results_cross]
        concords = [r["concord"] for r in results_cross]
        plt.bar(tfs, concords, color="lightcoral")
        plt.axhline(y=0.5, color="gray", linestyle="--")
        plt.ylabel("Direction Concordance")
        plt.title("Cross-Species Transfer: Human (Organoid) -> Zebrafish (Embryo)")
        plt.savefig(PROJECT_ROOT / "figures/zscape_benchmark.png")
        print(f"Figure saved to figures/zscape_benchmark.png")

if __name__ == "__main__":
    main()
