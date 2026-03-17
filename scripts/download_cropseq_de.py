#!/usr/bin/env python3
"""Download / verify CROP-seq differential expression data.

Checks that the required cropseq DE files exist in data/cropseq/.
If missing, falls back to extract_cropseq.py to generate them from the
Pando GRN.

Usage:
    uv run python scripts/download_cropseq_de.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REQUIRED_FILES = [
    "cropseq_de.csv",
    "gli3_ko_de.csv",
]


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    cropseq_dir = data_dir / "cropseq"

    missing = [f for f in REQUIRED_FILES if not (cropseq_dir / f).exists()]

    if not missing:
        print("CROP-seq DE data already present:")
        for f in REQUIRED_FILES:
            path = cropseq_dir / f
            size = path.stat().st_size
            print(f"  {path}  ({size:,} bytes)")
        return

    print(f"Missing CROP-seq files: {missing}")
    print("Running extract_cropseq.py to generate them ...")

    extract_script = Path(__file__).resolve().parent / "extract_cropseq.py"
    result = subprocess.run(
        [sys.executable, str(extract_script), "--data-dir", str(data_dir)],
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"extract_cropseq.py failed with code {result.returncode}")

    # Verify
    still_missing = [f for f in REQUIRED_FILES if not (cropseq_dir / f).exists()]
    if still_missing:
        sys.exit(f"Still missing after extraction: {still_missing}")

    print("CROP-seq DE data ready.")


if __name__ == "__main__":
    main()
