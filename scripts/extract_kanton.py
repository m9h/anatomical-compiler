import scanpy as sc
import os

print("Opening hnoca_extended.h5ad in backed mode...")
adata = sc.read_h5ad("data/azbukina/hnoca_extended.h5ad", backed='r')

print("Filtering for Kanton, 2019 cells...")
mask = adata.obs['publication'] == 'Kanton, 2019'
n_cells = mask.sum()
print(f"Found {n_cells} cells from Kanton, 2019")

# Extract to memory
kanton = adata[mask].to_memory()

# Save to a new file
out_path = "data/azbukina/kanton_2019.h5ad"
print(f"Saving to {out_path}...")
kanton.write_h5ad(out_path)
print("Done!")
