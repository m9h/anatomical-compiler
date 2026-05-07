import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from hgx_prep.sctab_models import TabnetClassifier

def load_model(model_dir):
    model_dir = Path(model_dir)
    ckpt_path = list(model_dir.glob("*.ckpt"))[0]
    hparams_path = model_dir / "hparams.yaml"
    
    with open(hparams_path, "r") as f:
        # hparams.yaml contains python tags that safe_load can't handle.
        # We only need the basic types for inference.
        content = f.read()
        # Remove !!python/name:... '' tags
        import re
        content = re.sub(r"!!python/name:[\w\.]+\s+''", "null", content)
        hparams = yaml.safe_load(content)
        
    # Load buffers from disk (some are ignored in hparams but needed for init)
    class_weights = np.load(model_dir / "merlin_cxg_2023_05_15_sf-log1p_minimal/class_weights.npy")
    child_matrix = np.load(model_dir / "merlin_cxg_2023_05_15_sf-log1p_minimal/cell_type_hierarchy/child_matrix.npy")
    # Augmentations are not needed for inference, but let's provide a dummy with correct shape
    # Checkpoint has (5000, 19331)
    augmentations = np.zeros((5000, hparams["gene_dim"]))
    
    # Load model from checkpoint
    # Note: We pass the buffers to __init__ so load_from_checkpoint doesn't complain
    model = TabnetClassifier.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
        gene_dim=hparams["gene_dim"],
        type_dim=hparams["type_dim"],
        class_weights=class_weights,
        child_matrix=child_matrix,
        augmentations=augmentations,
        # Other hparams are loaded from the ckpt itself or provided here
        train_set_size=hparams["train_set_size"],
        val_set_size=hparams["val_set_size"],
        batch_size=hparams["batch_size"]
    )
    model.eval()
    return model, hparams

def annotate_adata_backed(h5ad_path, model, hparams, var_df, output_path, batch_size=2000, gene_col=None):
    print(f"Opening {h5ad_path} in backed mode...")
    adata = sc.read_h5ad(h5ad_path, backed='r')
    
    # 1. Map genes
    sctab_genes = var_df["feature_name"].tolist()
    gene_map = {g: i for i, g in enumerate(sctab_genes)}
    
    # Use specified column for gene symbols if provided, otherwise use index
    if gene_col and gene_col in adata.var.columns:
        print(f"  Using .var['{gene_col}'] for gene symbols")
        adata_genes = adata.var[gene_col].astype(str).tolist()
    else:
        print("  Using .var_names for gene symbols")
        adata_genes = adata.var_names.tolist()
        
    overlap = [i for i, g in enumerate(adata_genes) if g in gene_map]
    print(f"  Overlap with scTab genes: {len(overlap)} / {len(sctab_genes)}")
    
    if len(overlap) == 0:
        print("ERROR: No gene overlap found. Check --gene-col or var_names.")
        return
        
    sctab_indices = [gene_map[adata_genes[i]] for i in overlap]
    adata_indices = overlap # Use pre-found indices
    
    # 2. Prepare storage for results
    all_preds = []
    all_probs = []
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    
    # 3. Process in chunks
    with torch.no_grad():
        for i in tqdm(range(0, adata.shape[0], batch_size), desc="scTab Inference"):
            # Load chunk into memory
            chunk = adata[i:i+batch_size].to_memory()
            
            # Normalize chunk
            # Note: scTab expects log1p(sf=1e4)
            sc.pp.normalize_total(chunk, target_sum=1e4)
            sc.pp.log1p(chunk)
            
            # Prepare dense batch
            x_batch_dense = np.zeros((chunk.shape[0], hparams["gene_dim"]), dtype=np.float32)
            # 'adata_indices' are the indices of overlapping genes in the original var
            vals = chunk.X[:, adata_indices]
            if hasattr(vals, "toarray"):
                vals = vals.toarray()
            x_batch_dense[:, sctab_indices] = vals
            
            x_batch = torch.tensor(x_batch_dense).to(device)
            logits, _ = model.classifier(x_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy().max(axis=1))
            
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    
    # 4. Save results (requires loading metadata, but we can do it cell-by-cell if needed)
    # For now, let's load the full metadata and add the columns
    print("Finalizing results...")
    adata_out = adata.to_memory()
    cell_type_names = pd.read_parquet(Path("models/scTab") / "merlin_cxg_2023_05_15_sf-log1p_minimal/categorical_lookup/cell_type.parquet")["label"].tolist()
    adata_out.obs["sctab_cell_type"] = [cell_type_names[p] for p in all_preds]
    adata_out.obs["sctab_probability"] = all_probs
    
    print(f"Saving to {output_path}...")
    adata_out.write_h5ad(output_path)
    return adata_out

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Annotate cell types using scTab")
    parser.add_argument("--input", type=str, required=True, help="Input h5ad file")
    parser.add_argument("--output", type=str, help="Output h5ad file")
    parser.add_argument("--model-dir", type=str, default="models/scTab", help="Model directory")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for inference")
    parser.add_argument("--gene-col", type=str, help="Column in .var for gene symbol mapping")
    args = parser.parse_args()
    
    print(f"Loading scTab model from {args.model_dir}...")
    model, hparams = load_model(args.model_dir)
    
    print(f"Loading gene metadata...")
    var_df = pd.read_parquet(Path(args.model_dir) / "merlin_cxg_2023_05_15_sf-log1p_minimal/var.parquet")
    
    out_path = args.output if args.output else Path(args.input).with_suffix(".sctab.h5ad")
    
    annotate_adata_backed(args.input, model, hparams, var_df, out_path, batch_size=args.batch_size, gene_col=args.gene_col)
    
    print("Done!")

if __name__ == "__main__":
    main()
