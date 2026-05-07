import os
import sys
import json
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import hgx

def load_real_de(gene):
    path = PROJECT_ROOT / f"data/cropseq/{gene.lower()}_ko_de.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def main():
    fig_dir = PROJECT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load human GRN
    data_dir = PROJECT_ROOT / "data/processed"
    with open(data_dir / "gene_names.json") as f: gene_names = json.load(f)
    with open(data_dir / "tf_names.json") as f: tf_names = json.load(f)
    incidence = np.load(data_dir / "incidence.npy")
    features = np.load(data_dir / "node_features_pca.npy")
    
    hg = hgx.from_incidence(jnp.array(incidence), node_features=jnp.array(features))
    
    # 2. Target TFs with real data
    targets = ["GLI3", "TBR1", "NEUROD6", "PAX6", "HES1"]
    
    # 3. Train a quick predictor on the real data
    # We'll use 4 for training and 1 (GLI3) for testing
    train_tfs = ["TBR1", "NEUROD6", "PAX6", "HES1"]
    test_tf = "GLI3"
    
    # Prepare training targets
    masks = []
    expr_targets = []
    gene_dim = features.shape[1] # Use actual dim
    
    for tf in train_tfs:
        de = load_real_de(tf)
        # Map DE to our gene space
        log2fc = np.zeros(len(gene_names), dtype=np.float32)
        de_map = dict(zip(de["gene"], de["log2fc"]))
        for i, g in enumerate(gene_names):
            log2fc[i] = de_map.get(g, 0.0)
            
        mask = np.zeros(len(gene_names), dtype=bool)
        mask[tf_names.index(tf)] = True
        
        masks.append(mask)
        # Expand log2fc to feature dim by tile
        expr_targets.append(np.tile(log2fc[:, None], (1, gene_dim)))
        
    masks = jnp.array(masks)
    expr_targets = jnp.array(expr_targets)
    
    print(f"Training PerturbationPredictor on {len(train_tfs)} real KOs...")
    predictor = hgx.PerturbationPredictor(
        gene_dim=gene_dim,
        hidden_dim=64,
        num_fates=3,
        conv_cls=hgx.UniGCNConv,
        num_layers=2,
        key=jax.random.PRNGKey(42)
    )
    
    predictor = hgx.train_perturbation_predictor(
        predictor, hg, masks, (expr_targets, jnp.zeros((len(train_tfs), 3))),
        epochs=100,
        key=jax.random.PRNGKey(43)
    )
    
    # 4. Predict GLI3 KO
    print(f"Predicting {test_tf} KO (Zero-shot test)...")
    pred_expr, _ = hgx.in_silico_knockout(predictor, hg, tf_names.index(test_tf))
    pred_mean = np.array(pred_expr).mean(axis=-1)
    
    # Load observed GLI3
    obs_de = load_real_de(test_tf)
    obs_map = dict(zip(obs_de["gene"], obs_de["log2fc"]))
    obs_mean = np.array([obs_map.get(g, 0.0) for g in gene_names])
    
    # 5. Figure 5: Real Validation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: GLI3 Scatter
    ax = axes[0]
    mask = (obs_mean != 0)
    r, p = stats.pearsonr(obs_mean[mask], pred_mean[mask])
    ax.scatter(obs_mean[mask], pred_mean[mask], alpha=0.5, s=10, color="darkblue")
    ax.set_xlabel("Observed log2FC (CROP-seq)")
    ax.set_ylabel("Predicted expression change (hgx)")
    ax.set_title(f"GLI3 KO Validation (r={r:.2f}, p={p:.1e})")
    
    # Panel B: Top targets concordance
    ax = axes[1]
    # Check key genes
    key_genes = ["DLX1", "DLX2", "GAD1", "GAD2", "FEZF2", "TBR1", "EMX1"]
    vals_pred = []
    vals_obs = []
    labels = []
    for g in key_genes:
        if g in gene_names:
            idx = gene_names.index(g)
            vals_pred.append(pred_mean[idx])
            vals_obs.append(obs_mean[idx])
            labels.append(g)
            
    x = np.arange(len(labels))
    ax.bar(x - 0.2, vals_pred, 0.4, label="Predicted", color="skyblue")
    ax.bar(x + 0.2, vals_obs, 0.4, label="Observed", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("log2FC")
    ax.set_title("Key Target Concordance")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_05_real_validation.png")
    print(f"Figure saved to {fig_dir / 'figure_05_real_validation.png'}")

if __name__ == "__main__":
    main()
