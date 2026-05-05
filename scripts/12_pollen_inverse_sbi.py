#!/usr/bin/env python3
"""12_pollen_inverse_sbi.py — Inverse Modeling (SBI) via CellFlow.

This script demonstrates using CellFlow to learn velocity fields from 
CRISPRi perturbation data and inverting them to infer regulatory links.

Logic:
    1. Learn a generative flow matching model: Control -> Perturbed.
    2. Compute the Jacobian of the learned velocity field dV/dX.
    3. The Jacobian entries J_ij represent the inferred influence of 
       gene i on the rate of change of gene j.
    4. Compare this inferred matrix against the biological GRN (incidence).
"""

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from cellflow.model import CellFlow
    from cellflow.data._dataloader import PredictionSampler
    import hgx
    from hgx_prep import io
except ImportError:
    sys.exit("ERROR: missing dependencies. Run: uv pip install -e ../cellflow ../hgx")


def generate_mock_pollen_data(n_cells=2000, n_genes=50):
    """Generate mock single-cell data if real h5ad is missing."""
    print(f"Generating mock Pollen data ({n_cells} cells, {n_genes} genes)...")
    
    # 1. Base features (PCA)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n_cells, 8))
    
    # 2. Conditions: Control + 3 TFs
    tfs = ["TF1", "TF2", "TF3"]
    perturbations = ["control"] * (n_cells // 2)
    for i, tf in enumerate(tfs):
        perturbations.extend([tf] * (n_cells // (2 * len(tfs))))
    
    # Fill remaining
    while len(perturbations) < n_cells:
        perturbations.append("control")
        
    obs = pd.DataFrame({
        "perturbation": perturbations,
        "is_control": [p == "control" for p in perturbations]
    })
    
    # 3. Add effect to perturbed cells
    # Each TF affects a few 'target' dimensions in PCA space
    for i, tf in enumerate(tfs):
        mask = obs["perturbation"] == tf
        x[mask, i % 8] += 1.5
        x[mask, (i + 1) % 8] -= 1.0
        
    adata = ad.AnnData(x.astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = adata.X
    return adata


def main():
    parser = argparse.ArgumentParser(description="Inverse Modeling (SBI) via CellFlow")
    parser.add_argument("--h5ad", type=str, help="Path to real screen.h5ad")
    parser.add_argument("--epochs", type=int, default=100, help="Training iterations")
    parser.add_argument("--dry-run", action="store_true", help="Run with mock data")
    args = parser.parse_args()

    # 1. Load Data
    if args.h5ad:
        print(f"Loading real data from {args.h5ad}...")
        adata = ad.read_h5ad(args.h5ad)
        # Use existing detection logic if available, or specify columns
        guide_col = "Gene_target_single" 
        adata.obs["is_control"] = adata.obs[guide_col].isin(["non-targeting", "NT", "CTRL"])
    else:
        print("Real h5ad not provided. Using mock data.")
        adata = generate_mock_pollen_data()
        guide_col = "perturbation"

    # 2. Train CellFlow
    print("\nTraining CellFlow (Flow Matching)...")
    model = CellFlow(adata)
    model.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={guide_col: [guide_col]}
    )
    model.prepare_model()
    model.train(num_iterations=args.epochs, valid_freq=args.epochs // 2)

    # 3. Inverse Modeling (SBI): Jacobian Attribution
    print("\nComputing Jacobian-based SBI importance...")
    
    # We want to compute dV/dX for each perturbed condition
    # This tells us which input features drive the predicted velocity
    
    vf = model.solver.vf
    state = model.solver.vf_state_inference
    
    @jax.jit
    def get_velocity(x, condition_dict):
        # x: (D,), condition_dict: {col: array([idx])}
        # time t=0.5 for mid-flow velocity
        t = jnp.array([0.5])
        # CellFlow expects batched x: (1, D)
        v, _, _ = vf.apply_fn(
            {"params": state.params},
            t,
            x[None, :],
            condition_dict,
            inference=True
        )
        return v[0]

    # Compute Jacobian for each TF
    tf_list = [p for p in adata.obs[guide_col].unique() if p != "control" and not str(p).startswith("NT")]
    sbi_results = {}
    
    # We need to get the condition encoder and params from the model
    # but CellFlow class encapsulates them. 
    # The solver has the vf_state_inference.
    
    for tf in tf_list[:3]:  # Demonstrate for first 3 TFs
        print(f"  Processing {tf}...")
        
        # Get condition_dict for this TF
        # We create a dummy covariate dataframe for this TF
        # It must include all columns used in training, including the control key.
        # We use a separate condition_id_key to avoid column name overlap in DataManager.
        cov_df = pd.DataFrame({
            guide_col: [tf], 
            "is_control": [False],
            "cond_id": [tf]
        })
        pred_data = model._dm.get_prediction_data(
            adata[adata.obs["is_control"]], # source cells
            sample_rep="X_pca",
            covariate_data=cov_df,
            condition_id_key="cond_id"
        )
        
        # Use the PredictionSampler to get the properly formatted condition dict
        pred_loader = PredictionSampler(pred_data)
        cond_dict_raw = pred_loader._get_condition_data(0)
        cond_dict = {k: jnp.array(v[0]) for k, v in cond_dict_raw.items()}
        
        # Compute Jacobian at the mean of control cells
        ctrl_mean = jnp.array(adata[adata.obs["is_control"]].X.mean(axis=0))
        
        # We'll use a wrapper that takes x and returns the predicted velocity
        def get_velocity(x):
            # t=0.5 for mid-flow velocity
            # Using (1, 1) shape to ensure correct broadcasting in time encoder
            t = jnp.array([[0.5]])
            # We need to provide encoder_noise, though it's not used in deterministic mode
            noise_dim = model.vf.condition_embedding_dim
            encoder_noise = jnp.zeros((1, noise_dim))
            
            v, _, _ = state.apply_fn(
                {"params": state.params},
                t,
                x[None, :],
                {k: v[None, :] for k, v in cond_dict.items()},
                encoder_noise,
                train=False
            )
            return v[0]
        
        jac_fn = jax.jacobian(get_velocity)
        J = jac_fn(ctrl_mean)
        
        sbi_results[tf] = np.array(J)

    # 4. Compare with Reference GRN (Incidence)
    print("\nComparing SBI importance with reference GRN...")
    # Load reference incidence if available
    pollen_processed = PROJECT_ROOT / "data" / "pollen" / "processed"
    if pollen_processed.exists():
        ref_incidence = np.load(pollen_processed / "incidence.npy")
        print(f"  Reference GRN found: {ref_incidence.shape}")
        # (Real comparison would involve mapping PCA dims back to genes)
    else:
        print("  Reference GRN not found. Skipping quantitative comparison.")

    # 5. Visualization: Jacobian Heatmap
    tf_to_plot = tf_list[0]
    J = sbi_results[tf_to_plot]
    
    plt.figure(figsize=(8, 6))
    vabs = np.max(np.abs(J))
    plt.imshow(J, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    plt.colorbar(label="dV_j / dX_i")
    plt.title(f"SBI Jacobian: Influence of Gene i on Gene j\nCondition: {tf_to_plot} KO")
    plt.xlabel("Input Dimension i")
    plt.ylabel("Velocity Dimension j")
    
    fig_path = PROJECT_ROOT / "figures" / "pollen_sbi_jacobian.png"
    plt.savefig(fig_path)
    print(f"\nSBI Jacobian figure saved to {fig_path}")
    print("Inverse modeling complete.")


if __name__ == "__main__":
    main()
