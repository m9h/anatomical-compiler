#!/usr/bin/env python3
"""08 -- Cross-Species & Cross-Organoid Showcase (Figure 7).

Compares cerebral organoid GRNs with C. elegans developmental datasets
available through hgx's built-in data loaders. Produces a 2x2 figure:
  A: C. elegans cell lineage as 3-uniform hypergraph
  B: DevoGraph 3D positions at early/mid/late timepoints
  C: Neural ODE trajectory prediction on DevoGraph
  D: Persistence diagram comparison (organoid GRN vs C. elegans lineage)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-species hypergraph comparison (Figure 7)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="PRNG seed (default: 42)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Neural ODE training epochs (default: 100)",
    )
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_import_hgx_data():
    """Import hgx data loaders, handling potential import failures."""
    try:
        from hgx import load_cell_lineage, load_devograph
        return load_cell_lineage, load_devograph
    except (ImportError, AttributeError) as exc:
        print(f"Warning: hgx data loaders unavailable: {exc}", file=sys.stderr)
        return None, None


def _build_organoid_grn() -> "hgx.Hypergraph":
    """Build a synthetic organoid GRN hypergraph for comparison.

    Uses data from the project's data directory if available, otherwise
    creates a synthetic small GRN.
    """
    import hgx

    data_dir = Path(__file__).resolve().parent.parent / "data"

    # Try loading real Pando GRN via load_pando_modules
    pando_dir = data_dir / "pando"
    coefs_file = pando_dir / "coefs.tsv"
    modules_file = pando_dir / "modules.tsv"
    if coefs_file.exists():
        try:
            hg = hgx.load_pando_modules(
                coef_csv=str(coefs_file),
                modules_csv=str(modules_file) if modules_file.exists() else None,
                padj_threshold=0.05,
            )
            print(f"  Loaded organoid GRN from Pando output ({hg.num_nodes} nodes, {hg.num_edges} edges)")
            return hg
        except Exception as exc:
            print(f"  Could not load Pando data: {exc}")

    # Fallback: synthetic organoid-like GRN
    print("  Using synthetic organoid GRN for comparison")
    key = jax.random.PRNGKey(123)
    n_genes = 50
    edges = []
    # Create regulatory modules (hyperedges of size 3-6)
    k = key
    for i in range(15):
        k, subkey = jax.random.split(k)
        size = int(jax.random.randint(subkey, (), 3, 7))
        k, subkey = jax.random.split(k)
        members = tuple(sorted(set(
            int(x) for x in jax.random.randint(subkey, (size,), 0, n_genes)
        )))
        if len(members) >= 2:
            edges.append(members)

    k, subkey = jax.random.split(k)
    features = jax.random.normal(subkey, (n_genes, 8))
    hg = hgx.from_edge_list(edges, num_nodes=n_genes, node_features=features)
    return hg


# ── Part A: Cell Lineage ─────────────────────────────────────────────────────

def panel_a(ax: matplotlib.axes.Axes) -> None:
    """C. elegans cell lineage as 3-uniform hypergraph."""
    import hgx

    load_cell_lineage, _ = _safe_import_hgx_data()
    if load_cell_lineage is None:
        ax.text(0.5, 0.5, "Cell lineage data\nunavailable",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("A: C. elegans Cell Lineage")
        ax.axis("off")
        return

    try:
        hg_lineage = load_cell_lineage(max_depth=4)
    except Exception as exc:
        ax.text(0.5, 0.5, f"Load failed:\n{exc}",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("A: C. elegans Cell Lineage")
        ax.axis("off")
        return

    n_nodes = hg_lineage.num_nodes
    n_edges = hg_lineage.num_edges
    degrees = np.asarray(hg_lineage.node_degrees)
    print(f"\nPanel A -- C. elegans cell lineage (depth 4):")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges (divisions): {n_edges}")
    print(f"  Degree distribution: mean={degrees.mean():.2f}, "
          f"min={degrees.min()}, max={degrees.max()}")

    hgx.draw_hypergraph(hg_lineage, ax=ax, title="A: C. elegans Cell Lineage (3-uniform)")
    ax.axis("off")


# ── Part B: DevoGraph 3D Positions ───────────────────────────────────────────

def panel_b(ax: matplotlib.axes.Axes) -> None:
    """DevoGraph 3D scatter at early/mid/late timepoints."""
    _, load_devograph = _safe_import_hgx_data()
    if load_devograph is None:
        ax.text(0.5, 0.5, "DevoGraph data\nunavailable",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("B: DevoGraph 3D Positions")
        ax.axis("off")
        return

    stages = {"Early (t=10)": 10, "Mid (t=95)": 95, "Late (t=180)": 180}
    colors_map = {"Early (t=10)": "#3498db", "Mid (t=95)": "#2ecc71", "Late (t=180)": "#e74c3c"}
    marker_map = {"Early (t=10)": "o", "Mid (t=95)": "s", "Late (t=180)": "^"}

    print("\nPanel B -- DevoGraph 3D positions:")
    for label, t in stages.items():
        try:
            hg = load_devograph(time_step=t, k_neighbors=5)
            pos = np.asarray(hg.positions)
            n = hg.num_nodes
            print(f"  {label}: {n} cells, {hg.num_edges} hyperedges")
            ax.scatter(
                pos[:, 0], pos[:, 1],
                c=colors_map[label], marker=marker_map[label],
                s=20, alpha=0.7, label=f"{label} ({n} cells)",
                edgecolors="none",
            )
        except Exception as exc:
            print(f"  {label}: failed -- {exc}")

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("B: DevoGraph Cell Positions (2D projection)")
    ax.legend(fontsize=8, loc="upper right")


# ── Part C: Neural ODE on DevoGraph ──────────────────────────────────────────

def panel_c(ax: matplotlib.axes.Axes, key: jax.Array, epochs: int) -> None:
    """Fit Neural ODE to DevoGraph trajectory, plot predicted vs observed."""
    import hgx

    _, load_devograph = _safe_import_hgx_data()
    if load_devograph is None:
        ax.text(0.5, 0.5, "DevoGraph data\nunavailable",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("C: Neural ODE Trajectory")
        ax.axis("off")
        return

    steps = list(range(10, 60, 5))  # 10 timepoints
    print(f"\nPanel C -- Neural ODE on DevoGraph (steps {steps[0]}-{steps[-1]}):")

    try:
        snapshots = [load_devograph(time_step=s, k_neighbors=5) for s in steps]
    except Exception as exc:
        ax.text(0.5, 0.5, f"Data load failed:\n{exc}",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("C: Neural ODE Trajectory")
        ax.axis("off")
        return

    # Align topologies (snapshots may have different numbers of cells)
    try:
        temp_hg = hgx.align_topologies(snapshots, times=jnp.array(steps, dtype=float))
    except Exception as exc:
        print(f"  align_topologies failed: {exc}")
        ax.text(0.5, 0.5, f"Topology alignment failed:\n{exc}",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("C: Neural ODE Trajectory")
        ax.axis("off")
        return

    print(f"  Temporal hypergraph: {len(temp_hg)} snapshots, "
          f"features shape {temp_hg.features.shape}")

    # Feature dim is 4: (x, y, z, size)
    feat_dim = temp_hg.features.shape[-1]

    # Observed feature norms per timepoint (mean over active nodes)
    observed_norms = []
    for t_idx in range(len(temp_hg)):
        feats_t = np.asarray(temp_hg.features[t_idx])
        if temp_hg.node_mask is not None:
            mask_t = np.asarray(temp_hg.node_mask[t_idx])
            active = feats_t[mask_t]
        else:
            active = feats_t
        if len(active) > 0:
            observed_norms.append(float(np.mean(np.linalg.norm(active, axis=1))))
        else:
            observed_norms.append(0.0)

    # Fit Neural ODE
    try:
        k1, k2 = jax.random.split(key)
        conv = hgx.UniGCNConv(feat_dim, feat_dim, key=k1)
        print(f"  Training Neural ODE for {epochs} epochs...")
        ode_model = hgx.fit_neural_ode(temp_hg, conv, key=k2, epochs=epochs, lr=1e-3)
        print("  Training complete.")

        # Predict trajectory using the trained ODE
        import diffrax
        hg0 = temp_hg[0]
        t0 = float(temp_hg.times[0])
        t1 = float(temp_hg.times[-1])
        target_times = temp_hg.times[1:]

        sol = ode_model(
            hg0, t0=t0, t1=t1,
            saveat=diffrax.SaveAt(ts=target_times),
        )
        pred_features = np.asarray(sol.ys)  # (T-1, n, d)

        # Predicted norms (prepend initial observation)
        predicted_norms = [observed_norms[0]]  # initial is the same
        for t_idx in range(pred_features.shape[0]):
            pf = pred_features[t_idx]
            if temp_hg.node_mask is not None:
                mask_t = np.asarray(temp_hg.node_mask[t_idx + 1])
                active = pf[mask_t]
            else:
                active = pf
            if len(active) > 0:
                predicted_norms.append(float(np.mean(np.linalg.norm(active, axis=1))))
            else:
                predicted_norms.append(0.0)

        ax.plot(steps, observed_norms, "o-", color="#2c3e50", label="Observed", linewidth=2)
        ax.plot(steps, predicted_norms, "s--", color="#e74c3c", label="Predicted (NeuralODE)", linewidth=2)
        ax.fill_between(steps, observed_norms, predicted_norms, alpha=0.15, color="#e74c3c")

    except Exception as exc:
        print(f"  Neural ODE training/prediction failed: {exc}")
        ax.plot(steps, observed_norms, "o-", color="#2c3e50", label="Observed", linewidth=2)
        ax.text(0.5, 0.3, f"NeuralODE failed:\n{type(exc).__name__}",
                ha="center", transform=ax.transAxes, fontsize=9, color="red")

    ax.set_xlabel("DevoGraph Timestep")
    ax.set_ylabel("Mean Feature Norm")
    ax.set_title("C: Neural ODE -- Predicted vs Observed")
    ax.legend(fontsize=9)


# ── Part D: Persistence Diagram Comparison ───────────────────────────────────

def panel_d(ax: matplotlib.axes.Axes) -> None:
    """Compare persistence diagrams: organoid GRN vs C. elegans lineage."""
    import hgx

    # Check for persistence computation support
    has_persistence = hasattr(hgx, "compute_persistence")
    if not has_persistence:
        ax.text(0.5, 0.5,
                "Persistence unavailable\n(install giotto-tda or ripser)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("D: Persistence Diagram Comparison")
        ax.axis("off")
        return

    print("\nPanel D -- Persistence diagram comparison:")

    # Organoid GRN
    organoid_diagrams = None
    try:
        hg_organoid = _build_organoid_grn()
        organoid_diagrams = hgx.compute_persistence(hg_organoid, filtration="weight", max_dim=1)
        n_pairs = sum(len(d) for d in organoid_diagrams)
        print(f"  Organoid GRN: {n_pairs} persistence pairs across {len(organoid_diagrams)} dims")
    except Exception as exc:
        print(f"  Organoid persistence failed: {exc}")

    # C. elegans lineage
    lineage_diagrams = None
    load_cell_lineage, _ = _safe_import_hgx_data()
    if load_cell_lineage is not None:
        try:
            hg_lineage = load_cell_lineage(max_depth=4)
            lineage_diagrams = hgx.compute_persistence(hg_lineage, filtration="weight", max_dim=1)
            n_pairs = sum(len(d) for d in lineage_diagrams)
            print(f"  C. elegans lineage: {n_pairs} persistence pairs across {len(lineage_diagrams)} dims")
        except Exception as exc:
            print(f"  C. elegans persistence failed: {exc}")

    # Plot birth-death scatter
    has_data = False

    if organoid_diagrams is not None:
        for dim, dgm in enumerate(organoid_diagrams):
            if len(dgm) > 0:
                has_data = True
                marker = "o" if dim == 0 else "^"
                ax.scatter(
                    dgm[:, 0], dgm[:, 1],
                    c="#3498db", marker=marker, s=40, alpha=0.7,
                    label=f"Organoid H{dim}" if dim <= 1 else None,
                    edgecolors="white", linewidth=0.5,
                )

    if lineage_diagrams is not None:
        for dim, dgm in enumerate(lineage_diagrams):
            if len(dgm) > 0:
                has_data = True
                marker = "o" if dim == 0 else "^"
                ax.scatter(
                    dgm[:, 0], dgm[:, 1],
                    c="#e74c3c", marker=marker, s=40, alpha=0.7,
                    label=f"C. elegans H{dim}" if dim <= 1 else None,
                    edgecolors="white", linewidth=0.5,
                )

    if has_data:
        # Diagonal reference line
        all_vals = []
        if organoid_diagrams is not None:
            for d in organoid_diagrams:
                if len(d) > 0:
                    all_vals.extend(d.flatten().tolist())
        if lineage_diagrams is not None:
            for d in lineage_diagrams:
                if len(d) > 0:
                    all_vals.extend(d.flatten().tolist())
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            margin = (hi - lo) * 0.05 if hi > lo else 0.5
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    "k--", alpha=0.3, linewidth=1)
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.text(0.5, 0.5, "No persistence data computed",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("D: Persistence Diagrams (organoid vs C. elegans)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)

    print("=" * 60)
    print("Cross-Species & Cross-Organoid Hypergraph Comparison")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel A: C. elegans cell lineage
    panel_a(axes[0, 0])

    # Panel B: DevoGraph 3D positions
    panel_b(axes[0, 1])

    # Panel C: Neural ODE on DevoGraph
    k_c, _ = jax.random.split(key)
    panel_c(axes[1, 0], k_c, epochs=args.epochs)

    # Panel D: Persistence diagram comparison
    panel_d(axes[1, 1])

    fig.suptitle("Figure 7: Cross-Species Hypergraph Comparison", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(__file__).resolve().parent.parent / "figures" / "figure_07_cross_species.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
