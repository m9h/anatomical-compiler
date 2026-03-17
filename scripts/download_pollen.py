#!/usr/bin/env python3
"""Download Pollen/Ding et al. 2026 CRISPRi Perturb-seq data from GEO.

Paper: "Dissecting gene regulatory networks governing human cortical cell fate"
  Ding, Kim, Ostrowski et al. (Pollen lab, UCSF)
  Nature 2026, doi: 10.1038/s41586-025-09997-7
  GEO: GSE284197

Downloads h5ad files from GEO supplementary data. The main file needed
for the perturbation comparison is GSE284197_screen.h5ad (4.1 GB) which
contains the 44-TF CRISPRi screen in 2D cortical cultures.

Usage:
    uv run python scripts/download_pollen.py
    uv run python scripts/download_pollen.py --data-dir data --screen-only
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path
from urllib.request import urlretrieve

# ---------------------------------------------------------------------------
# GEO supplementary file URLs (GSE284197)
# ---------------------------------------------------------------------------

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE284nnn/GSE284197/suppl"

FILES = {
    "screen": {
        "url": f"{GEO_BASE}/GSE284197_screen.h5ad.gz",
        "filename": "GSE284197_screen.h5ad.gz",
        "output": "screen.h5ad",
        "size_gb": 4.1,
        "description": "Main CRISPRi screen: 44 TFs in 2D cortical cultures",
    },
    "merged": {
        "url": f"{GEO_BASE}/GSE284197_merged.h5ad.gz",
        "filename": "GSE284197_merged.h5ad.gz",
        "output": "merged.h5ad",
        "size_gb": 3.2,
        "description": "Merged dataset (all conditions)",
    },
    "IN": {
        "url": f"{GEO_BASE}/GSE284197_IN.h5ad.gz",
        "filename": "GSE284197_IN.h5ad.gz",
        "output": "IN.h5ad",
        "size_gb": 0.7,
        "description": "Interneuron-focused subset",
    },
    "slice": {
        "url": f"{GEO_BASE}/GSE284197_slice.h5ad.gz",
        "filename": "GSE284197_slice.h5ad.gz",
        "output": "slice.h5ad",
        "size_gb": 0.5,
        "description": "Slice culture experiments",
    },
    "clones": {
        "url": f"{GEO_BASE}/GSE284197_clones.h5ad.gz",
        "filename": "GSE284197_clones.h5ad.gz",
        "output": "clones.h5ad",
        "size_gb": 0.01,
        "description": "Clonal/lineage tracing data",
    },
}


def _progress_hook(count, block_size, total_size):
    """Print download progress."""
    pct = count * block_size * 100 / total_size if total_size > 0 else 0
    mb = count * block_size / 1e6
    total_mb = total_size / 1e6
    print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end="", flush=True)


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress reporting. Returns True on success."""
    print(f"  Downloading {url}")
    try:
        urlretrieve(url, str(dest), reporthook=_progress_hook)
        print()  # newline after progress
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def decompress_gz(gz_path: Path, out_path: Path) -> None:
    """Decompress a .gz file."""
    print(f"  Decompressing -> {out_path.name}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Download Pollen/Ding 2026 CRISPRi data from GEO (GSE284197)"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Root data directory (default: ../data)",
    )
    parser.add_argument(
        "--screen-only", action="store_true",
        help="Only download the main screen.h5ad (4.1 GB)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    pollen_dir = data_dir / "pollen"
    pollen_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Pollen/Ding et al. 2026 — CRISPRi Perturb-seq Download")
    print("  GEO: GSE284197")
    print("  doi: 10.1038/s41586-025-09997-7")
    print("=" * 65)
    print(f"  Output: {pollen_dir}")
    print()

    to_download = ["screen"] if args.screen_only else list(FILES.keys())
    total_gb = sum(FILES[k]["size_gb"] for k in to_download)

    print(f"Files to download ({total_gb:.1f} GB total):")
    for key in to_download:
        info = FILES[key]
        out_path = pollen_dir / info["output"]
        exists = out_path.exists()
        status = " (EXISTS)" if exists else ""
        print(f"  {info['output']:<20s} {info['size_gb']:>5.1f} GB  "
              f"{info['description']}{status}")

    if args.dry_run:
        print("\n  --dry-run: no files downloaded")
        return

    print()
    n_ok = 0
    for key in to_download:
        info = FILES[key]
        out_path = pollen_dir / info["output"]

        if out_path.exists():
            print(f"  {info['output']} already exists, skipping")
            n_ok += 1
            continue

        gz_path = pollen_dir / info["filename"]
        if download_file(info["url"], gz_path):
            if gz_path.suffix == ".gz":
                decompress_gz(gz_path, out_path)
            n_ok += 1
        else:
            print(f"  FAILED to download {info['output']}")

    print()
    print("=" * 65)
    print(f"  Downloaded {n_ok}/{len(to_download)} files to {pollen_dir}")
    print()
    print("Next steps:")
    print("  uv run python scripts/preprocess_pollen.py")
    print("  uv run python scripts/compare_pollen.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
