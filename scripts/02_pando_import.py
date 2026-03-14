#!/usr/bin/env python3
"""02_pando_import.py — Import Pando GRN into hgx and run Phase 1 analyses.

Loads Pando output (coefs.tsv + modules.tsv) into an hgx Hypergraph,
trains an HGNNStack for module detection, computes TF centrality
metrics, and produces Figure 1 (GRN Architecture) and Figure 2
(Module Detection Performance).

Usage:
    python scripts/02_pando_import.py
    python scripts/02_pando_import.py --data-dir data --epochs 200 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

try:
    import hgx
except ImportError:
    sys.exit(
        "ERROR: hgx is not installed. Install it with:\n"
        "  uv pip install -e ../hgx\n"
        "or add it as a path dependency in pyproject.toml."
    )

try:
    import networkx as nx
except ImportError:
    nx = None
    print("WARNING: networkx not installed. Panel A (pairwise graph) will be skipped.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_MASTER_REGULATORS = [
    "GLI3", "FOXG1", "TBR1", "DLX1", "DLX2", "EMX1", "EOMES", "NEUROD6",
]


# ---------------------------------------------------------------------------
# 1.1  Load GRN into hgx
# ---------------------------------------------------------------------------


def load_grn(data_dir: Path) -> tuple:
    """Load the Pando GRN into an hgx Hypergraph.

    Tries load_pando_modules() with coefs.tsv + modules.tsv first.
    Falls back to load_grn_from_csv() with just coefs.tsv.

    Returns:
        (hg, gene_names, module_labels, coefs_df, modules_df)
    """
    coefs_path = data_dir / "pando" / "coefs.tsv"
    modules_path = data_dir / "pando" / "modules.tsv"

    if not coefs_path.exists():
        sys.exit(
            f"ERROR: {coefs_path} not found.\n"
            "Run 01_prepare_data.py first to generate synthetic data."
        )

    # Load raw data for gene names and module labels
    import pandas as pd
    coefs_df = pd.read_csv(coefs_path, sep="\t")
    print(f"  Loaded coefs: {len(coefs_df)} rows, {coefs_df['tf'].nunique()} TFs, "
          f"{coefs_df['target'].nunique()} targets")

    modules_df = None
    if modules_path.exists():
        modules_df = pd.read_csv(modules_path, sep="\t")
        print(f"  Loaded modules: {len(modules_df)} gene-module pairs, "
              f"{modules_df['module'].nunique()} modules")

    # Build hypergraph
    if modules_path.exists():
        print("  Building hypergraph with load_pando_modules()...")
        hg = hgx.load_pando_modules(
            coef_csv=str(coefs_path),
            modules_csv=str(modules_path),
            padj_threshold=0.05,
            tf_col="tf",
            target_col="target",
            estimate_col="estimate",
            padj_col="padj",
            module_col="module",
            gene_col="gene",
        )
    else:
        print("  Building hypergraph with load_grn_from_csv()...")
        hg = hgx.load_grn_from_csv(
            str(coefs_path),
            tf_col="tf",
            target_col="target",
            weight_col="estimate",
        )

    # Extract gene names in order (reconstruct from the edge list logic)
    sig_coefs = coefs_df[coefs_df["padj"] < 0.05]
    genes: dict[str, int] = {}
    for _, row in sig_coefs.iterrows():
        genes.setdefault(row["tf"], len(genes))
        genes.setdefault(row["target"], len(genes))

    if modules_df is not None:
        for _, row in modules_df.iterrows():
            genes.setdefault(row["gene"], len(genes))

    gene_names = [""] * len(genes)
    for name, idx in genes.items():
        gene_names[idx] = name

    # Build module label array
    num_nodes = hg.incidence.shape[0]
    module_labels = None
    if modules_df is not None:
        gene_to_mod = dict(zip(modules_df["gene"], modules_df["module"]))
        unique_modules = sorted(modules_df["module"].unique())
        mod_to_idx = {m: i for i, m in enumerate(unique_modules)}

        module_labels = np.full(num_nodes, -1, dtype=np.int32)
        for gname, gidx in genes.items():
            if gidx < num_nodes and gname in gene_to_mod:
                module_labels[gidx] = mod_to_idx[gene_to_mod[gname]]

        # Assign unassigned genes to the nearest module based on incidence
        unassigned = np.where(module_labels == -1)[0]
        if len(unassigned) > 0:
            H = np.array(hg.incidence)
            for idx in unassigned:
                edge_membership = H[idx]
                if edge_membership.sum() > 0:
                    # Find the most common module among co-members
                    best_mod = 0
                    best_count = 0
                    for e in range(H.shape[1]):
                        if edge_membership[e] > 0:
                            members = np.where(H[:, e] > 0)[0]
                            for m_idx in members:
                                if module_labels[m_idx] >= 0:
                                    mod = module_labels[m_idx]
                                    count = np.sum(module_labels[members] == mod)
                                    if count > best_count:
                                        best_count = count
                                        best_mod = mod
                    module_labels[idx] = best_mod
                else:
                    module_labels[idx] = 0

    print(f"\n  Hypergraph summary:")
    print(f"    Nodes: {hg.num_nodes}")
    print(f"    Hyperedges: {hg.num_edges}")
    print(f"    Incidence shape: {hg.incidence.shape}")

    node_deg = np.array(hg.node_degrees)
    edge_deg = np.array(hg.edge_degrees)
    print(f"    Node degree: mean={node_deg.mean():.1f}, "
          f"median={np.median(node_deg):.1f}, max={node_deg.max():.0f}")
    print(f"    Edge degree: mean={edge_deg.mean():.1f}, "
          f"median={np.median(edge_deg):.1f}, max={edge_deg.max():.0f}")

    return hg, gene_names, module_labels, coefs_df, modules_df


# ---------------------------------------------------------------------------
# 1.2  Module detection with HGNNStack
# ---------------------------------------------------------------------------


def run_module_detection(
    hg, module_labels, num_modules, epochs, seed,
):
    """Train HGNNStack + UniGATConv to classify genes into modules.

    Returns dict with macro_f1, attn_corr, losses, per_module_acc, preds, model.
    """
    print("\n" + "=" * 60)
    print("  Analysis 1.2: Module Detection with HGNNStack")
    print("=" * 60)

    key = jax.random.PRNGKey(seed)
    labels = jnp.array(module_labels)
    in_dim = hg.node_features.shape[-1]

    model = hgx.HGNNStack(
        conv_dims=[(in_dim, 64), (64, 32)],
        conv_cls=hgx.UniGATConv,
        readout_dim=num_modules,
        activation=jax.nn.relu,
        dropout_rate=0.0,
        key=key,
    )

    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, hg, labels):
        def loss_fn(model, hg, labels):
            logits = model(hg, inference=True)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            one_hot = jax.nn.one_hot(labels, num_classes=num_modules)
            return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, hg, labels)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), new_opt_state, loss

    losses = []
    t_start = time.time()
    for epoch in range(epochs):
        model, opt_state, loss = step(model, opt_state, hg, labels)
        losses.append(float(loss))
        if (epoch + 1) % max(1, epochs // 5) == 0:
            preds = jnp.argmax(model(hg, inference=True), axis=-1)
            acc = float(jnp.mean(preds == labels))
            print(f"    Epoch {epoch+1:4d}  loss={loss:.4f}  acc={acc:.1%}")

    elapsed = time.time() - t_start

    # Final predictions
    preds = jnp.argmax(model(hg, inference=True), axis=-1)

    # Macro F1
    f1s = []
    per_module_acc = []
    for c in range(num_modules):
        tp = float(jnp.sum((preds == c) & (labels == c)))
        fp = float(jnp.sum((preds == c) & (labels != c)))
        fn = float(jnp.sum((preds != c) & (labels == c)))
        prec = tp / max(tp + fp, 1e-8)
        rec = tp / max(tp + fn, 1e-8)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-8))
        n_in_class = float(jnp.sum(labels == c))
        per_module_acc.append(tp / max(n_in_class, 1e-8))
    macro_f1 = float(np.mean(f1s))

    # Attention-incidence correlation from first UniGATConv layer
    conv0 = model.convs[0]
    H = hg._masked_incidence()
    features = hg.node_features
    x_proj = jax.vmap(conv0.linear)(features)
    out_dim = x_proj.shape[-1]

    if conv0.normalize:
        d_e = jnp.sum(H, axis=0, keepdims=True)
        e_repr = (H * jnp.where(d_e > 0, 1.0 / d_e, 0.0)).T @ x_proj
    else:
        e_repr = H.T @ x_proj

    v_sc = x_proj @ conv0.attn[:out_dim]
    e_sc = e_repr @ conv0.attn[out_dim:]
    raw = v_sc[:, None] + e_sc[None, :]
    raw = jnp.where(raw >= 0, raw, conv0.negative_slope * raw)
    raw = jnp.where(H > 0, raw, -1e9)
    attn = jax.nn.softmax(raw, axis=1) * H

    incidence_np = np.array(hg.incidence)
    attn_np = np.array(attn)
    attn_corr = float(
        np.corrcoef(attn_np.ravel(), incidence_np.ravel())[0, 1]
    )

    print(f"\n    Time: {elapsed:.1f}s")
    print(f"    Macro F1: {macro_f1:.3f}")
    print(f"    Attention-incidence Pearson r: {attn_corr:.3f}")

    return {
        "macro_f1": macro_f1,
        "attn_corr": attn_corr,
        "losses": losses,
        "per_module_acc": per_module_acc,
        "preds": np.array(preds),
        "model": model,
        "attn": attn_np,
    }


# ---------------------------------------------------------------------------
# 1.3  TF centrality comparison
# ---------------------------------------------------------------------------


def compute_centrality(hg, gene_names):
    """Compute centrality metrics for known master regulators.

    Computes:
    - Node degree (from hgx)
    - Eigenvector centrality (from hypergraph Laplacian)
    - Betweenness centrality (via clique expansion + networkx)

    Returns dict mapping TF name -> {degree, eigvec, betweenness}.
    """
    print("\n" + "=" * 60)
    print("  Analysis 1.3: TF Centrality Comparison")
    print("=" * 60)

    # Node degree
    node_deg = np.array(hg.node_degrees)

    # Eigenvector centrality from Laplacian
    lap = np.array(hgx.hypergraph_laplacian(hg, normalized=True))
    # Eigenvector centrality: use the Fiedler vector (2nd smallest eigenvalue)
    # but for centrality, use the dominant eigenvector of (I - L) = D^{-1/2} H D_e^{-1} H^T D^{-1/2}
    eigenvalues, eigenvectors = np.linalg.eigh(lap)
    # The eigenvector corresponding to the smallest non-zero eigenvalue
    # gives connectivity structure; for centrality, use element-wise magnitude
    # of eigenvectors weighted by inverse eigenvalues
    # A simpler approach: use the adjacency-based eigenvector centrality
    adj = np.array(hgx.clique_expansion(hg))
    adj_eigenvalues, adj_eigenvectors = np.linalg.eigh(adj)
    # Dominant eigenvector (largest eigenvalue)
    dominant_idx = np.argmax(adj_eigenvalues)
    eigvec_centrality = np.abs(adj_eigenvectors[:, dominant_idx])
    eigvec_centrality = eigvec_centrality / max(eigvec_centrality.max(), 1e-8)

    # Betweenness centrality via clique expansion + networkx
    betweenness = np.zeros(len(gene_names))
    if nx is not None:
        G = nx.from_numpy_array(np.array(adj))
        bw = nx.betweenness_centrality(G)
        for node_idx, cent in bw.items():
            betweenness[node_idx] = cent

    # Build gene name -> index lookup
    name_to_idx = {}
    for i, name in enumerate(gene_names):
        if name:
            name_to_idx[name] = i

    # Report known master regulators
    print(f"\n    {'TF':<12} {'Degree':>8} {'Eigvec':>10} {'Betweenness':>12}")
    print("    " + "-" * 44)

    centrality_results = {}
    for tf in KNOWN_MASTER_REGULATORS:
        if tf in name_to_idx:
            idx = name_to_idx[tf]
            deg = float(node_deg[idx])
            eig = float(eigvec_centrality[idx])
            bw = float(betweenness[idx])
            print(f"    {tf:<12} {deg:>8.1f} {eig:>10.4f} {bw:>12.6f}")
            centrality_results[tf] = {
                "degree": deg,
                "eigvec": eig,
                "betweenness": bw,
                "index": idx,
            }

    # Rankings
    deg_rank = np.argsort(-node_deg)
    eig_rank = np.argsort(-eigvec_centrality)
    bw_rank = np.argsort(-betweenness)

    print(f"\n    Top 5 by degree: ", end="")
    print(", ".join(gene_names[i] or f"node_{i}" for i in deg_rank[:5]))
    print(f"    Top 5 by eigvec: ", end="")
    print(", ".join(gene_names[i] or f"node_{i}" for i in eig_rank[:5]))
    if nx is not None:
        print(f"    Top 5 by betweenness: ", end="")
        print(", ".join(gene_names[i] or f"node_{i}" for i in bw_rank[:5]))

    return centrality_results, node_deg, eigvec_centrality, betweenness


# ---------------------------------------------------------------------------
# Figure 1: GRN Architecture Comparison
# ---------------------------------------------------------------------------


def plot_figure_1(hg, gene_names, coefs_df, node_deg, fig_dir):
    """Generate Figure 1: GRN Architecture Comparison (4 panels).

    A: Pando's pairwise graph (networkx subgraph of top TFs)
    B: hgx hypergraph via draw_hypergraph() with module colors
    C: Incidence matrix heatmap via draw_incidence()
    D: Degree distributions comparison
    """
    print("\n  Generating Figure 1: GRN Architecture Comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # --- Panel A: Pairwise graph from Pando ---
    ax = axes[0, 0]
    if nx is not None:
        sig_coefs = coefs_df[coefs_df["padj"] < 0.05].copy()
        # Subset to known TFs and their top targets for visibility
        focus_tfs = KNOWN_MASTER_REGULATORS[:6]
        sub = sig_coefs[sig_coefs["tf"].isin(focus_tfs)]
        # Take top 5 targets per TF by absolute estimate
        sub["abs_est"] = sub["estimate"].abs()
        sub = sub.sort_values("abs_est", ascending=False).groupby("tf").head(5)

        G = nx.DiGraph()
        for _, row in sub.iterrows():
            G.add_edge(
                row["tf"], row["target"],
                weight=abs(row["estimate"]),
                sign="+" if row["estimate"] > 0 else "-",
            )

        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, seed=42, k=2.0)
            # Color TF nodes differently
            node_colors = []
            for node in G.nodes():
                if node in focus_tfs:
                    node_colors.append("#e41a1c")
                else:
                    node_colors.append("#377eb8")

            edge_colors = ["green" if G[u][v]["sign"] == "+" else "red"
                           for u, v in G.edges()]
            edge_widths = [G[u][v]["weight"] * 3 for u, v in G.edges()]

            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                   node_size=300, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                                   width=edge_widths, alpha=0.6,
                                   arrows=True, arrowsize=10, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

    ax.set_title("A. Pando Pairwise Graph (top TF regulons)", fontsize=11)
    ax.axis("off")

    # --- Panel B: hgx hypergraph ---
    ax = axes[0, 1]
    # Draw a subset (first 30 nodes, all edges) for visibility
    n_sub = min(30, hg.incidence.shape[0])
    H_sub = hg.incidence[:n_sub, :]
    # Keep only edges that have >= 2 members in the subset
    edge_mask = jnp.sum(H_sub > 0, axis=0) >= 2
    m_sub = int(jnp.sum(edge_mask))
    if m_sub > 0:
        edge_indices = jnp.where(edge_mask)[0]
        # Limit to at most 15 edges for readability
        if m_sub > 15:
            edge_indices = edge_indices[:15]
            m_sub = 15
        H_sub_filtered = H_sub[:, edge_indices]
        sub_features = hg.node_features[:n_sub]
        hg_sub = hgx.from_incidence(H_sub_filtered, node_features=sub_features)

        # Use gene names as labels
        sub_labels = [gene_names[i] if i < len(gene_names) and gene_names[i]
                      else str(i) for i in range(n_sub)]
        # Color nodes by degree
        sub_deg = np.array(hg_sub.node_degrees)
        node_colors = plt.cm.YlOrRd(sub_deg / max(sub_deg.max(), 1))

        hgx.draw_hypergraph(
            hg_sub, ax=ax,
            node_color=node_colors,
            node_labels=sub_labels,
            title="B. hgx Hypergraph (subset)",
            node_size=200,
        )
    else:
        ax.text(0.5, 0.5, "No multi-member edges\nin subset",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("B. hgx Hypergraph (subset)", fontsize=11)

    # --- Panel C: Incidence matrix ---
    ax = axes[1, 0]
    # Show a manageable slice of the incidence matrix
    n_show = min(80, hg.incidence.shape[0])
    m_show = min(hg.incidence.shape[1], 20)
    H_show = np.array(hg.incidence[:n_show, :m_show])
    im = ax.imshow(H_show, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Hyperedges (modules)")
    ax.set_ylabel("Genes")
    ax.set_title("C. Incidence Matrix (subset)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Panel D: Degree distributions ---
    ax = axes[1, 1]
    node_degrees = np.array(hg.node_degrees)
    edge_degrees = np.array(hg.edge_degrees)

    ax.hist(node_degrees, bins=30, alpha=0.7, label="Node degree",
            color="steelblue", edgecolor="white")
    ax2 = ax.twinx()
    ax2.hist(edge_degrees, bins=30, alpha=0.5, label="Edge degree",
             color="coral", edgecolor="white")

    ax.set_xlabel("Degree")
    ax.set_ylabel("Node count", color="steelblue")
    ax2.set_ylabel("Edge count", color="coral")
    ax.set_title("D. Degree Distributions", fontsize=11)

    # Combined legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.7, label="Node degree"),
        Patch(facecolor="coral", alpha=0.5, label="Edge degree"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    path = fig_dir / "figure_01_grn_architecture.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved to {path}")


# ---------------------------------------------------------------------------
# Figure 2: Module Detection Performance
# ---------------------------------------------------------------------------


def plot_figure_2(hg, detection_results, module_names, fig_dir):
    """Generate Figure 2: Module Detection Performance (4 panels).

    A: Per-module classification accuracy bar chart
    B: Attention weights vs ground-truth incidence heatmap
    C: Placeholder for convolution comparison (03_higher_order.py)
    D: Training loss curve
    """
    print("\n  Generating Figure 2: Module Detection Performance...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Panel A: Per-module accuracy ---
    ax = axes[0, 0]
    per_mod_acc = detection_results["per_module_acc"]
    num_modules = len(per_mod_acc)
    x_pos = np.arange(num_modules)
    colors = plt.cm.Set3(np.linspace(0, 1, num_modules))
    bars = ax.bar(x_pos, per_mod_acc, color=colors, edgecolor="gray", linewidth=0.5)

    ax.set_xlabel("Module")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("A. Per-Module Classification Accuracy", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=float(np.mean(per_mod_acc)), color="red", linestyle="--",
               linewidth=1, label=f"Mean = {np.mean(per_mod_acc):.2f}")
    ax.legend()
    ax.set_xticks(x_pos)
    if module_names is not None and len(module_names) == num_modules:
        ax.set_xticklabels(module_names, rotation=45, ha="right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel B: Attention vs incidence heatmap ---
    ax = axes[0, 1]
    attn = detection_results["attn"]
    incidence = np.array(hg.incidence)
    # Show subset for visibility
    n_show = min(50, attn.shape[0])
    m_show = min(20, attn.shape[1])

    # Side-by-side comparison: left half is incidence, right half is attention
    combined = np.zeros((n_show, m_show * 2))
    combined[:, :m_show] = incidence[:n_show, :m_show]
    combined[:, m_show:] = attn[:n_show, :m_show]

    im = ax.imshow(combined, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.axvline(x=m_show - 0.5, color="red", linewidth=2, linestyle="--")
    ax.text(m_show / 2, -2, "Ground Truth", ha="center", fontsize=9, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    ax.text(m_show + m_show / 2, -2, "Learned Attention", ha="center", fontsize=9,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    ax.set_xlabel("Hyperedges")
    ax.set_ylabel("Genes")
    ax.set_title(
        f"B. Attention vs Incidence (r = {detection_results['attn_corr']:.3f})",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Panel C: Placeholder ---
    ax = axes[1, 0]
    ax.text(
        0.5, 0.5,
        "Panel reserved for\nconvolution comparison\n(03_higher_order.py)",
        ha="center", va="center", fontsize=14, color="gray",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.8),
    )
    ax.set_title("C. Convolution Comparison (placeholder)", fontsize=11)
    ax.axis("off")

    # --- Panel D: Training loss curve ---
    ax = axes[1, 1]
    losses = detection_results["losses"]
    ax.plot(losses, linewidth=1.2, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("D. Training Loss", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Annotate final values
    final_loss = losses[-1]
    ax.annotate(
        f"Final: {final_loss:.4f}",
        xy=(len(losses) - 1, final_loss),
        xytext=(-80, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
    )

    # Add summary text
    fig.text(
        0.5, 0.01,
        f"Macro F1 = {detection_results['macro_f1']:.3f}  |  "
        f"Attention-Incidence r = {detection_results['attn_corr']:.3f}  |  "
        f"{len(losses)} epochs",
        ha="center", fontsize=11, style="italic",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    path = fig_dir / "figure_02_module_detection.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Import Pando GRN into hgx and run Phase 1 analyses "
            "(module detection, centrality)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/02_pando_import.py\n"
            "  python scripts/02_pando_import.py --epochs 200 --seed 123\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Directory containing prepared data (default: ../data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs for module detection (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / args.data_dir).resolve()
    fig_dir = (script_dir / ".." / "figures").resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Organoid Regulome Benchmark — Phase 1: Reproducibility")
    print("=" * 60)
    print(f"  Data directory: {data_dir}")
    print(f"  Figure directory: {fig_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Seed: {args.seed}")

    # --- 1.1: Load GRN ---
    print("\n" + "=" * 60)
    print("  Analysis 1.1: Import Pando GRN into hgx")
    print("=" * 60)

    hg, gene_names, module_labels, coefs_df, modules_df = load_grn(data_dir)

    if module_labels is None:
        print("\n  WARNING: No module labels available. Skipping module detection.")
        print("  Only centrality analysis and partial figures will be generated.")
        num_modules = 0
    else:
        num_modules = int(module_labels.max()) + 1
        print(f"  Module labels: {num_modules} modules, "
              f"{(module_labels >= 0).sum()} genes assigned")

    # --- 1.2: Module detection ---
    detection_results = None
    if module_labels is not None and num_modules > 1:
        detection_results = run_module_detection(
            hg, module_labels, num_modules, args.epochs, args.seed,
        )

    # --- 1.3: TF centrality ---
    centrality_results, node_deg, eigvec_cent, betweenness = compute_centrality(
        hg, gene_names,
    )

    # --- Figures ---
    print("\n" + "=" * 60)
    print("  Generating figures")
    print("=" * 60)

    plot_figure_1(hg, gene_names, coefs_df, node_deg, fig_dir)

    if detection_results is not None:
        # Get module names if available
        module_name_list = None
        if modules_df is not None:
            unique_modules = sorted(modules_df["module"].unique())
            module_name_list = unique_modules

        plot_figure_2(hg, detection_results, module_name_list, fig_dir)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Phase 1 Results Summary")
    print("=" * 60)
    print(f"  GRN: {hg.num_nodes} nodes, {hg.num_edges} hyperedges")

    if detection_results is not None:
        print(f"  Module detection:")
        print(f"    Macro F1:                {detection_results['macro_f1']:.3f}")
        print(f"    Attention-incidence r:   {detection_results['attn_corr']:.3f}")

    print(f"  TF centrality (top regulators):")
    for tf in ["GLI3", "FOXG1", "TBR1"]:
        if tf in centrality_results:
            c = centrality_results[tf]
            print(f"    {tf}: degree={c['degree']:.0f}, "
                  f"eigvec={c['eigvec']:.4f}, "
                  f"betweenness={c['betweenness']:.6f}")

    print(f"\n  Figures saved to: {fig_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
