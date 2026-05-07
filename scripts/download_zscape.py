import argparse
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description="Download ZSCAPE/ZESTA zebrafish perturbation atlas")
    parser.add_argument("--data-dir", type=str, default="data/zscape", help="Output directory")
    args = parser.parse_args()

    dest_dir = Path(args.data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ZSCAPE/ZESTA dataset (zesta.h5ad) from Hugging Face...")
    dest = hf_hub_download(
        repo_id="theislab/cellflow-datasets",
        filename="zesta.h5ad",
        repo_type="dataset",
        local_dir=dest_dir
    )
    
    print(f"Downloaded to {dest}")
    print("\nNext steps:")
    print(f"  uv run python scripts/preprocess_zscape.py --input {dest}")

if __name__ == "__main__":
    main()
