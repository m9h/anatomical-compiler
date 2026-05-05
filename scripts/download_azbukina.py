#!/usr/bin/env python3
"""Download Azbukina et al. 2025 Posterior Brain data from Zenodo.

Paper: "Multi-omic human neural organoid cell atlas of the posterior brain"
  Azbukina et al. (Treutlein lab)
  bioRxiv 2025, doi: 10.1101/2025.03.20.644368
  Zenodo: 10.5281/zenodo.14901345
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add project root to path so we can import hgx_prep
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from hgx_prep import download, registry
except ImportError:
    print("ERROR: hgx_prep not found. Run from project root.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download Azbukina 2025 data from Zenodo")
    parser.add_argument("--data-dir", type=str, default="data/azbukina", help="Output directory")
    args = parser.parse_args()

    dest_dir = Path(args.data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    doi = "10.5281/zenodo.14901345"
    print(f"Resolving Zenodo DOI: {doi} ...")
    record = download.resolve_zenodo_doi(doi)

    if not record:
        print("FAILED: Could not resolve Zenodo record")
        sys.exit(1)

    print(f"Record: {record['title']}")
    for f in record["files"]:
        dest = dest_dir / f["filename"]
        if dest.exists():
            print(f"  {f['filename']} already exists, skipping")
            continue
        
        download.download_file(f["url"], dest, desc=f["filename"])

    print(f"\nDone. Files downloaded to {dest_dir}")


if __name__ == "__main__":
    main()
