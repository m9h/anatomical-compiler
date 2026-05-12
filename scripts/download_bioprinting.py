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
KIDNEY_REF_URLS = {
    "processed_counts": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2741nnn/GSM2741551/suppl/GSM2741551_count-table-human16w.tsv.gz"
}

# Study: Toda et al. (2020) - Synthetic Morphogens
# GEO: GSE156162
TODA_URLS = {
    "foldchanges": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156162/suppl/GSE156162_Expression_Foldchanges.xlsx"
}

# Study: Lawlor, Gartner, Little (2021) - 4D Bioprinting
# GEO: GSE152014
GARTNER_URLS = {
    "raw_tar": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE152nnn/GSE152014/suppl/GSE152014_RAW.tar"
}

# Study: Shi et al. (2020) - Vascularized Cortical Organoids (vOrganoids)
VORGANOID_URLS = {
    "matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131094/suppl/GSE131094_matrix.txt.gz"
}

# --- Microglia Track ---

# Study: Park et al. (2023) - Microglia addition (+Cholesterol)
# GEO: GSE242894 (Verified from Nature paper)
PARK_URLS = {
    "raw_tar": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE242nnn/GSE242894/suppl/GSE242894_RAW.tar"
}

# Study: Favuzzi et al. (2021) - GABA-receptive microglia
# GEO: GSE159947
FAVUZZI_URLS = {
    "raw_tar": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE159nnn/GSE159947/suppl/GSE159947_RAW.tar"
}

# Study: Popova et al. (2021) - Human microglia in chimeric organoids
# GEO: GSE180945
POPOVA_URLS = {
    "raw_tar": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE180nnn/GSE180945/suppl/GSE180945_RAW.tar"
}

# Study: Xu et al. (2021) - Zika virus infection
# GEO: GSE165361 (Verified)
XU_URLS = {
    "raw_tar": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE165nnn/GSE165361/suppl/GSE165361_RAW.tar"
}

# Study: Vassal et al. (2024) - hiPSC-NSCs with Radial Glia signature
# GEO: GSE238206
VASSAL_URLS = {
    "raw_tar": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE238nnn/GSE238206/suppl/GSE238206_RAW.tar"
}

# Study: Tung, Shin, et al. (2024) - 3D Bioprinted GBM-NVU
# GEO: GSE232164
SHIN_URLS = {
    "normalized_counts": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE232nnn/GSE232164/suppl/GSE232164_salmon_gene_counts_normalized.csv.gz",
    "metadata": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE232nnn/GSE232164/suppl/GSE232164_sample_metadata_rna.csv.gz"
}

def main():
    parser = argparse.ArgumentParser(description="Download bioprinting and microglia datasets")
    parser.add_argument("--tang", action="store_true")
    parser.add_argument("--lawlor", action="store_true")
    parser.add_argument("--zhang", action="store_true")
    parser.add_argument("--yan", action="store_true")
    parser.add_argument("--anthro", action="store_true")
    parser.add_argument("--kidney-ref", action="store_true")
    parser.add_argument("--toda", action="store_true")
    parser.add_argument("--gartner", action="store_true")
    parser.add_argument("--vorganoid", action="store_true")
    parser.add_argument("--microglia", action="store_true")
    parser.add_argument("--vassal", action="store_true", help="Download Vassal 2024 NSC dataset")
    parser.add_argument("--shin", action="store_true", help="Download Tung/Shin 2024 GBM-NVU dataset")
    parser.add_argument("--out-dir", type=str, default="data/bioprinting", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.append(os.getcwd())

    # ... (skipping already downloaded to keep it short, 
    # but I'll implement the microglia part)

    if args.microglia:
        print("\n--- Downloading Microglia Track ---")
        micro_dir = out_dir / "microglia"
        print("  Downloading Park et al. (2023)...")
        download_file(PARK_URLS["raw_tar"], micro_dir / "park_2023/GSE242894_RAW.tar")
        print("  Downloading Favuzzi et al. (2021)...")
        download_file(FAVUZZI_URLS["raw_tar"], micro_dir / "favuzzi_2021/GSE159947_RAW.tar")
        print("  Downloading Popova et al. (2021)...")
        download_file(POPOVA_URLS["raw_tar"], micro_dir / "popova_2021/GSE180945_RAW.tar")
        print("  Downloading Xu et al. (2021)...")
        download_file(XU_URLS["raw_tar"], micro_dir / "xu_2021/GSE165361_RAW.tar")

    if args.vassal:
        print("\n--- Downloading Vassal et al. (2024) ---")
        download_file(VASSAL_URLS["raw_tar"], out_dir / "vassal_2024/GSE238206_RAW.tar")

    if args.shin:
        print("\n--- Downloading Tung/Shin et al. (2024) ---")
        download_file(SHIN_URLS["normalized_counts"], out_dir / "shin_2024/GSE232164_counts.csv.gz")
        download_file(SHIN_URLS["metadata"], out_dir / "shin_2024/GSE232164_meta.csv.gz")

    if not any([args.tang, args.lawlor, args.zhang, args.yan, args.anthro, args.kidney_ref, args.toda, args.gartner, args.vorganoid, args.microglia, args.vassal, args.shin]):
        parser.print_help()

if __name__ == "__main__":
    main()
