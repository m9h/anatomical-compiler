"""Datasets for the Solé "Synthetic Agency / Bio-Computing" track.

The Kagan 2022 DishBrain paper (Cortical Labs / Pong) is multi-electrode array
electrophysiology only — it has no transcriptomic readout, and the GEO accession
GSE207577 sometimes attached to it is private until 2026-12-31. So instead of
gating this benchmark on a private record, we use the canonical
activity-induced single-cell dataset:

    Hrvatin et al. (2018) Nat. Neurosci.
    "Single-cell analysis of experience-dependent transcriptomic states in the
    mouse visual cortex." doi:10.1038/s41593-017-0029-5
    GEO: GSE102827 — 48,266 inDrops cells from mouse V1 after 0h / 1h / 4h
    of light stimulation following dark housing.

The "Learning Regulome" task becomes: identify TFs whose regulon activity is
selectively induced by sensory drive (the transcriptomic analog of
"actively-processing-feedback vs static-controls" cells in DishBrain).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
from download_large import download_file

# Hrvatin et al. (2018) — Sensory experience scRNA-seq, mouse V1
HRVATIN_URLS = {
    "counts": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE102nnn/GSE102827/suppl/GSE102827_merged_all_raw.csv.gz",
    "celltypes": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE102nnn/GSE102827/suppl/GSE102827_cell_type_assignments.csv.gz",
}


def main():
    parser = argparse.ArgumentParser(description="Download datasets for the synthetic-agency / learning-regulome track")
    parser.add_argument("--hrvatin", action="store_true",
                        help="Download Hrvatin 2018 GSE102827 (V1 light-stimulation scRNA-seq)")
    parser.add_argument("--out-dir", type=str, default="data/agency", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.hrvatin:
        print("\n--- Downloading Hrvatin et al. (2018) GSE102827 ---")
        dest = out_dir / "hrvatin_2018"
        for key, url in HRVATIN_URLS.items():
            fname = url.split("/")[-1]
            download_file(url, dest / fname)

    if not any([args.hrvatin]):
        parser.print_help()
        print("\nNote: GSE207577 (Kagan/DishBrain) is private until 2026-12-31 and is")
        print("MEA-only. The Hrvatin V1 dataset is the canonical scRNA-seq proxy")
        print("for the activity-induced 'Learning Regulome' analysis.")


if __name__ == "__main__":
    main()
