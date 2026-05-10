import argparse
import os
import sys
from pathlib import Path
from download_large import download_file

# Study: Tang et al. (2020) - Glioblastoma Bioprinting
# GEO: GSE147147
TANG_URLS = {
    "processed_xlsx": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147147/suppl/GSE147147_Bioprinting_RNAseq_Batch1_Processed_Data.xlsx",
    "raw_counts": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147147/suppl/GSE147147_RAW.tar"
}

# Study: Lawlor et al. (2021) - Kidney Organoid Bioprinting
# GEO: GSE138835
LAWLOR_URLS = {
    "raw_counts": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE138nnn/GSE138835/suppl/GSE138835_RAW.tar"
}

# Study: Zhang Lab (2025) - Hepatorganoid Bioprinting
# GEO: GSE298708
ZHANG_URLS = {
    "raw_counts": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE298nnn/GSE298708/suppl/GSE298708_RAW.tar"
}

# Study: Yan et al. (2024) - Human Brain Tissue Bioprinting
# GEO: GSE234774
YAN_URLS = {
    "mtx": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE234nnn/GSE234774/suppl/GSE234774_rnaseq_filtered_scRNA.mtx.gz",
    "features": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE234nnn/GSE234774/suppl/GSE234774_rnaseq_features.txt.gz",
    "barcodes": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE234nnn/GSE234774/suppl/GSE234774_rnaseq_barcodes.txt.gz",
    "meta": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE234nnn/GSE234774/suppl/GSE234774_rnaseq_meta.txt.gz"
}

# Study: Gumuskaya et al. (2023) - Anthrobots (Human Biobots)
# GEO: GSE249581
ANTHRO_URLS = {
    "raw_tar": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE249nnn/GSE249581/suppl/GSE249581_RAW.tar"
}

# Reference: Human Fetal Kidney (Combes et al. 2019 / GSE102596)
# Used as blueprint for RIFLE and Lawlor kidney benchmarks
KIDNEY_REF_URLS = {
    "processed_counts": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2741nnn/GSM2741551/suppl/GSM2741551_count-table-human16w.tsv.gz"
}

def main():
    parser = argparse.ArgumentParser(description="Download bioprinting datasets for GRN analysis")
    parser.add_argument("--tang", action="store_true", help="Download Tang et al. (2020) Glioblastoma dataset")
    parser.add_argument("--lawlor", action="store_true", help="Download Lawlor et al. (2021) Kidney dataset")
    parser.add_argument("--zhang", action="store_true", help="Download Zhang et al. (2025) Liver dataset")
    parser.add_argument("--yan", action="store_true", help="Download Yan et al. (2024) Human Brain dataset")
    parser.add_argument("--anthro", action="store_true", help="Download Anthrobots (2023) dataset")
    parser.add_argument("--kidney-ref", action="store_true", help="Download Human Fetal Kidney reference (GSE102596)")
    parser.add_argument("--out-dir", type=str, default="data/bioprinting", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Add project root to path to import download_file
    sys.path.append(os.getcwd())

    if args.tang:
        print("\n--- Downloading Tang et al. (2020) Glioblastoma ---")
        dest = out_dir / "tang_2020"
        download_file(TANG_URLS["processed_xlsx"], dest / "GSE147147_processed.xlsx")
        # download_file(TANG_URLS["raw_counts"], dest / "GSE147147_RAW.tar")

    if args.lawlor:
        print("\n--- Downloading Lawlor et al. (2021) Kidney ---")
        dest = out_dir / "lawlor_2021"
        download_file(LAWLOR_URLS["raw_counts"], dest / "GSE138835_RAW.tar")

    if args.zhang:
        print("\n--- Downloading Zhang et al. (2025) Liver ---")
        dest = out_dir / "zhang_2025"
        download_file(ZHANG_URLS["raw_counts"], dest / "GSE298708_RAW.tar")

    if args.yan:
        print("\n--- Downloading Yan et al. (2024) Human Brain ---")
        dest = out_dir / "yan_2024"
        for key, url in YAN_URLS.items():
            fname = url.split("/")[-1]
            download_file(url, dest / fname)

    if args.anthro:
        print("\n--- Downloading Anthrobots (2023) ---")
        dest = out_dir / "anthro_2023"
        download_file(ANTHRO_URLS["raw_tar"], dest / "GSE249581_RAW.tar")

    if args.kidney_ref:
        print("\n--- Downloading Human Fetal Kidney reference (GSE102596) ---")
        dest = out_dir / "kidney_ref"
        download_file(KIDNEY_REF_URLS["processed_counts"], dest / "GSE102596_all_cell_counts.txt.gz")

    if not any([args.tang, args.lawlor, args.zhang, args.yan, args.anthro, args.kidney_ref]):
        parser.print_help()

if __name__ == "__main__":
    main()
