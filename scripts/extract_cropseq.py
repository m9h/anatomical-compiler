#!/usr/bin/env python
"""Extract real CROP-seq differential expression data from Fleck et al. Seurat objects.

Reads the Seurat .rds files from data/zenodo/seurat_objects.tar.gz (17 GB),
extracts per-knockout differential expression results for GLI3, TBR1, and
EOMES, and saves them as CSVs for perturbation prediction validation.

The Seurat objects contain CROP-seq guide assignments and inferred KO
probabilities (see organoid_regulomes/crop_seq/ko_inference.R for the
original R analysis pipeline). This script tries three extraction
strategies in order:

    1. pyreadr  — pure-Python RDS reader (works for data frames, not
       full Seurat objects)
    2. rpy2     — call R/Seurat directly to read the objects and run
       FindMarkers for each KO
    3. fallback — generate biologically-informed synthetic DE data from
       the Pando GRN, flagged for replacement with real data

Produces:
    data/cropseq/gli3_ko_de.csv
    data/cropseq/tbr1_ko_de.csv
    data/cropseq/eomes_ko_de.csv
    data/cropseq/cropseq_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tarfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Target knockouts ─────────────────────────────────────────────────────

KO_TARGETS = [
    "GLI3", "TBR1", "EOMES", "ASCL1", "NEUROD6", 
    "PAX6", "SOX2", "EMX1", "DLX1", "DLX2", "HES1"
]

# Expected downstream effects based on Fleck et al. 2023 biology:
#   GLI3 KO  -> ventral shift, DLX1/DLX2/GAD1 down in dorsal context
#   TBR1 KO  -> cortical neuron defects, loss of layer markers
#   EOMES KO -> intermediate progenitor loss, reduced neurogenesis
KO_BIOLOGY = {
    "GLI3": {
        "expected_down": ["DLX1", "DLX2", "GAD1", "GAD2", "NKX2-1", "LHX6"],
        "expected_up": ["TBR1", "NEUROD6", "EMX1", "FOXG1"],
        "description": "GLI3 KO: Sonic hedgehog pathway disruption, dorsal-ventral patterning",
    },
    "TBR1": {
        "expected_down": ["FEZF2", "BCL11B", "TLE4", "SOX5", "FOXP2"],
        "expected_up": ["SATB2", "CUX1", "CUX2", "BRN2"],
        "description": "TBR1 KO: Deep-layer cortical neuron specification defects",
    },
    "EOMES": {
        "expected_down": ["NEUROD1", "NEUROD2", "NEUROD6", "TBR1"],
        "expected_up": ["PAX6", "SOX2", "HES1", "NOTCH1"],
        "description": "EOMES KO: Intermediate progenitor loss, reduced neurogenesis",
    },
    "ASCL1": {
        "expected_down": ["DLX1", "DLX2", "GAD1"],
        "expected_up": ["PAX6"],
        "description": "ASCL1 KO: Primal proneuronal factor in interneuron/ventral lineage",
    },
    "NEUROD6": {
        "expected_down": ["TBR1", "SLC17A7"],
        "expected_up": ["PAX6"],
        "description": "NEUROD6 KO: Excitatory neuron differentiation",
    },
    "PAX6": {
        "expected_down": ["EOMES", "TBR1"],
        "expected_up": ["SOX2", "HES1"],
        "description": "PAX6 KO: Radial glia maintenance / neurogenesis initiation",
    },
    "SOX2": {
        "expected_down": ["PAX6", "EOMES"],
        "expected_up": ["GFAP"],
        "description": "SOX2 KO: Multipotency loss in neural progenitors",
    },
    "EMX1": {
        "expected_down": ["TBR1", "NEUROD6"],
        "expected_up": ["DLX1"],
        "description": "EMX1 KO: Dorsal telencephalon specification",
    },
}
# Default biology for any missing TFs
DEFAULT_BIOLOGY = {
    "expected_down": [],
    "expected_up": [],
    "description": "Generic TF knockout",
}


# ── Step 1: Extract tarball ──────────────────────────────────────────────


def extract_tarball(data_dir: Path) -> Path | None:
    """Extract seurat_objects.tar.gz if not already done.

    Returns the extraction directory, or None if the tarball is missing.
    """
    tar_path = data_dir / "zenodo" / "seurat_objects.tar.gz"
    extract_dir = data_dir / "zenodo" / "seurat_objects"

    if not tar_path.exists():
        log.warning("Tarball not found: %s", tar_path)
        log.info(
            "  This file is 17 GB and must be downloaded from Zenodo first."
        )
        log.info(
            "  See BENCHMARK_DATASETS.md for download instructions."
        )
        return None

    if extract_dir.exists():
        rds_files = list(extract_dir.rglob("*.rds"))
        if rds_files:
            log.info(
                "Seurat objects already extracted to %s (%d .rds files)",
                extract_dir,
                len(rds_files),
            )
            for f in rds_files:
                log.info("  %s (%d MB)", f.name, f.stat().st_size // 1_000_000)
            return extract_dir

    log.info("Extracting %s ...", tar_path)
    log.info("  (This is 17 GB and may take several minutes.)")
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        rds_files = list(extract_dir.rglob("*.rds"))
        log.info("Extraction complete. Found %d .rds files:", len(rds_files))
        for f in rds_files:
            log.info("  %s (%d MB)", f.name, f.stat().st_size // 1_000_000)
        return extract_dir
    except Exception:
        log.error("Failed to extract tarball:\n%s", traceback.format_exc())
        return None


# ── Step 2: Try pyreadr ──────────────────────────────────────────────────


def try_pyreadr(extract_dir: Path) -> dict[str, pd.DataFrame] | None:
    """Attempt to read RDS files with pyreadr.

    pyreadr can read simple R data frames saved as .rds, but typically
    cannot deserialize full Seurat S4 objects. We try anyway because
    some .rds files in the tarball may be standalone DE result tables.

    Returns a dict of {ko_gene: DataFrame} if successful, else None.
    """
    try:
        import pyreadr
    except ImportError:
        log.info("pyreadr not installed. Install with: uv pip install pyreadr")
        return None

    log.info("Attempting to read .rds files with pyreadr ...")
    results = {}

    for rds_file in sorted(extract_dir.rglob("*.rds")):
        log.info("  Reading %s (%d MB) ...", rds_file.name, rds_file.stat().st_size // 1_000_000)
        try:
            result = pyreadr.read_r(str(rds_file))
            for key, df in result.items():
                log.info("    %s/%s: shape=%s", rds_file.name, key, df.shape)
                log.info("    columns: %s", list(df.columns)[:20])

                # Check if this looks like a DE results table
                de_cols = {"gene", "log2fc", "pval", "padj", "p_val", "avg_log2FC", "p_val_adj"}
                col_set = set(c.lower() for c in df.columns)
                if col_set & de_cols:
                    log.info("    -> Looks like DE results!")

                    # Try to identify which KO this belongs to
                    for ko in KO_TARGETS:
                        if ko.lower() in rds_file.name.lower() or ko.lower() in key.lower():
                            results[ko] = _normalize_de_df(df, ko)
                            log.info("    -> Assigned to %s KO", ko)
                            break
                    else:
                        # Check for a column that identifies the KO gene
                        for col in df.columns:
                            if "ko" in col.lower() or "target" in col.lower() or "guide" in col.lower():
                                for ko in KO_TARGETS:
                                    if ko in df[col].values:
                                        ko_df = df[df[col] == ko].copy()
                                        results[ko] = _normalize_de_df(ko_df, ko)
                                        log.info("    -> Found %s KO in column '%s'", ko, col)

        except Exception as e:
            log.info("    Failed: %s", e)
            # Seurat objects are S4 classes that pyreadr can't handle — expected.
            continue

    if results:
        log.info("pyreadr extracted DE results for: %s", list(results.keys()))
        return results

    log.info("pyreadr could not extract DE results (Seurat S4 objects are not supported).")
    return None


# ── Step 3: Try rpy2 ─────────────────────────────────────────────────────


def try_rpy2(extract_dir: Path) -> dict[str, pd.DataFrame] | None:
    """Use rpy2 to call R/Seurat and extract CROP-seq DE results.

    This requires R, Seurat, and rpy2 to be installed. On the DGX Spark
    this should be available in the R environment.

    The extraction strategy mirrors the organoid_regulomes pipeline:
      1. Load Seurat object with readRDS()
      2. Check for guide_assignments and perturb assays
      3. Run Seurat::FindMarkers() comparing KO cells vs non-targeting controls
      4. Return per-KO DE tables
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
    except ImportError:
        log.info("rpy2 not installed. Install with: uv pip install rpy2")
        return None

    log.info("Attempting to read Seurat objects with rpy2 ...")

    # Check R and Seurat availability
    try:
        ro.r("library(Seurat)")
        log.info("  Seurat library loaded successfully in R.")
    except Exception as e:
        log.warning("  Could not load Seurat in R: %s", e)
        log.info("  Install Seurat in R with: install.packages('Seurat')")
        return None

    results = {}

    for rds_file in sorted(extract_dir.rglob("*.rds")):
        log.info("  Loading %s with R ...", rds_file.name)
        try:
            rds_path_r = str(rds_file).replace("\\", "/")

            # Load the object and inspect its class
            ro.r(f'obj <- readRDS("{rds_path_r}")')
            obj_class = str(ro.r("class(obj)")[0])
            log.info("    Class: %s", obj_class)

            if obj_class != "Seurat":
                log.info("    Not a Seurat object, skipping.")
                ro.r("rm(obj); gc()")
                continue

            # Inspect assays and metadata
            assays = list(ro.r("names(obj@assays)"))
            log.info("    Assays: %s", assays)

            meta_cols = list(ro.r("colnames(obj@meta.data)"))
            log.info("    Metadata columns: %s", meta_cols[:20])

            n_cells = int(ro.r("ncol(obj)")[0])
            log.info("    Cells: %d", n_cells)

            # Check for CROP-seq-related assays
            has_guide = "guide_assignments" in assays
            has_perturb = "perturb" in assays
            has_target = "target_genes_cons" in assays
            log.info("    guide_assignments assay: %s", has_guide)
            log.info("    perturb assay: %s", has_perturb)
            log.info("    target_genes_cons assay: %s", has_target)

            # Metadata check for specific KOs (e.g. GLI3_KO column)
            for ko in KO_TARGETS:
                ko_col = f"{ko}_KO"
                if ko_col in meta_cols:
                    log.info("    Found metadata column for %s KO", ko)
                    ro.r(f"""
                        # Join layers if Seurat v5 (MUST be on the object, not the assay)
                        if (as.numeric(substr(packageVersion('Seurat'), 1, 1)) >= 5) {{
                            obj <- JoinLayers(obj)
                        }}
                        obj$cropseq_ident <- ifelse(obj[['{ko_col}']] == TRUE, 'KO', 'CTRL')
                        Idents(obj) <- 'cropseq_ident'
                        de_results <- FindMarkers(obj, ident.1 = 'KO', ident.2 = 'CTRL', verbose = FALSE)
                        de_results$gene <- rownames(de_results)
                    """)
                    with localconverter(ro.default_converter + pandas2ri.converter):
                        de_df = ro.r("as.data.frame(de_results)")
                    results[ko] = _normalize_de_df(de_df, ko)

            if not has_guide and not has_perturb and not has_target:
                # Check misc slot for pre-computed DE results
                log.info("    No CROP-seq assays found. Checking misc slot ...")
                misc_names = list(ro.r("names(obj@misc)"))
                log.info("    Misc slot contents: %s", misc_names)

                for misc_name in misc_names:
                    for ko in KO_TARGETS:
                        if ko.lower() in misc_name.lower() and "de" in misc_name.lower():
                            log.info("    Found pre-computed DE: %s", misc_name)
                            try:
                                with localconverter(ro.default_converter + pandas2ri.converter):
                                    de_df = ro.r(f'as.data.frame(obj@misc${misc_name})')
                                results[ko] = _normalize_de_df(de_df, ko)
                            except Exception as e:
                                log.warning("    Failed to extract %s: %s", misc_name, e)

                ro.r("rm(obj); gc()")
                continue

            # Extract DE for each KO target using FindMarkers
            # Identify the guide/KO identity column
            guide_assay = "target_genes_cons" if has_target else ("perturb" if has_perturb else "guide_assignments")

            # Get guide names
            guide_names = list(ro.r(f'rownames(GetAssayData(obj, assay="{guide_assay}"))'))
            log.info("    Guide/Target features (%d): %s", len(guide_names), guide_names[:15])

            for ko in KO_TARGETS:
                if ko in results: continue # Already found in metadata
                
                ko_guides = [g for g in guide_names if g == ko or g.startswith(f"{ko}-")]
                if not ko_guides:
                    continue

                log.info("    Extracting DE for %s KO (targets: %s) ...", ko, ko_guides)

                try:
                    guides_r = "c(" + ",".join(f'"{g}"' for g in ko_guides) + ")"
                    ro.r(f"""
                        guide_data <- GetAssayData(obj, assay='{guide_assay}')
                        if ('{guide_assay}' == 'target_genes_cons') {{
                            ko_scores <- colSums(guide_data[{guides_r}, , drop=FALSE])
                            ko_cells <- names(ko_scores[ko_scores > 0])
                            # Controls are cells with NO target gene assignments
                            total_targets <- colSums(guide_data)
                            ctrl_cells <- names(total_targets[total_targets == 0])
                        }} else {{
                            ko_scores <- colSums(guide_data[{guides_r}, , drop=FALSE])
                            ko_cells <- names(ko_scores[ko_scores > 0])
                            nt_guides <- grep('(NT|CTRL|DUMMY|non.targeting|scramble)',
                                             rownames(guide_data), value=TRUE, ignore.case=TRUE)
                            if (length(nt_guides) > 0) {{
                                nt_scores <- colSums(guide_data[nt_guides, , drop=FALSE])
                                ctrl_cells <- names(nt_scores[nt_scores > 0])
                            }} else {{
                                total_guides <- colSums(guide_data)
                                ctrl_cells <- names(total_guides[total_guides == 0])
                            }}
                        }}
                    """)

                    n_ko = int(ro.r("length(ko_cells)")[0])
                    n_ctrl = int(ro.r("length(ctrl_cells)")[0])
                    log.info("    %s: %d KO cells, %d control cells", ko, n_ko, n_ctrl)

                    if n_ko < 10:
                        log.warning("    Too few KO cells for %s (%d), skipping.", ko, n_ko)
                        continue

                    if n_ctrl < 10:
                        log.warning("    Too few control cells (%d), skipping.", n_ctrl)
                        continue

                    # Set identity classes and run FindMarkers
                    ro.r(f"""
                        # Create identity vector
                        ident_vec <- rep('other', ncol(obj))
                        names(ident_vec) <- colnames(obj)
                        ident_vec[ko_cells] <- 'KO'
                        ident_vec[ctrl_cells] <- 'CTRL'
                        obj$cropseq_ident <- ident_vec

                        Idents(obj) <- 'cropseq_ident'
                        DefaultAssay(obj) <- 'RNA'

                        de_results <- FindMarkers(
                            obj,
                            ident.1 = 'KO',
                            ident.2 = 'CTRL',
                            test.use = 'wilcox',
                            min.pct = 0.1,
                            logfc.threshold = 0.1,
                            verbose = FALSE
                        )
                        de_results$gene <- rownames(de_results)
                    """)

                    with localconverter(ro.default_converter + pandas2ri.converter):
                        de_df = ro.r("as.data.frame(de_results)")

                    log.info(
                        "    %s DE results: %d genes (columns: %s)",
                        ko, len(de_df), list(de_df.columns),
                    )
                    results[ko] = _normalize_de_df(de_df, ko)

                except Exception as e:
                    log.warning("    FindMarkers failed for %s: %s", ko, e)
                    log.debug(traceback.format_exc())

            ro.r("rm(obj); gc()")

        except Exception as e:
            log.warning("  Failed to process %s: %s", rds_file.name, e)
            log.debug(traceback.format_exc())
            try:
                ro.r("rm(obj); gc()")
            except Exception:
                pass

    if results:
        log.info("rpy2 extracted DE results for: %s", list(results.keys()))
        return results

    log.info("rpy2 could not extract DE results from any Seurat object.")
    return None


# ── Step 4: Inspect organoid_regulomes for clues ─────────────────────────


def inspect_organoid_regulomes() -> None:
    """Read organoid_regulomes R scripts to document expected data format.

    This is informational only — logs what the real DE data should look
    like based on the original analysis pipeline.
    """
    regulomes_dir = Path("/home/mhough/dev/organoid_regulomes")
    if not regulomes_dir.exists():
        log.info("organoid_regulomes repo not found at %s", regulomes_dir)
        return

    log.info("Inspecting organoid_regulomes for CROP-seq data format ...")

    ko_inference = regulomes_dir / "crop_seq" / "ko_inference.R"
    if ko_inference.exists():
        log.info("  Found: %s", ko_inference)
        log.info("  The R pipeline uses:")
        log.info("    - Guide assignments stored in 'guide_assignments' assay")
        log.info("    - KO probabilities inferred via elastic-net (alpha=0.5)")
        log.info("    - Per-guide KO probability via expression error comparison")
        log.info("    - Results stored in 'perturb' assay of Seurat object")

    enrichment = regulomes_dir / "crop_seq" / "enrichment.R"
    if enrichment.exists():
        log.info("  Found: %s", enrichment)
        log.info("    - Guide enrichment tested via Fisher/CMH tests")
        log.info("    - Per-celltype enrichment of guide assignments")

    log.info("  Expected DE output format:")
    log.info("    - gene: gene symbol")
    log.info("    - log2fc: log2 fold change (KO vs control)")
    log.info("    - pval: raw p-value")
    log.info("    - padj: BH-adjusted p-value")
    log.info("  Expected KO targets: GLI3, TBR1, EOMES")


# ── Step 5: GRN-informed synthetic fallback ──────────────────────────────


def generate_synthetic_de(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Generate biologically-informed synthetic DE data from the Pando GRN.

    This is a FALLBACK. The generated data uses the GRN structure to
    produce plausible fold changes but is NOT real experimental data.
    Every output file is clearly flagged as synthetic.

    Strategy:
        - Load Pando GRN coefficients
        - For each KO target, identify its regulon (direct targets)
        - Assign negative log2FC to regulon members (downregulated when
          the TF is knocked out, assuming activator role)
        - Assign positive log2FC to targets of competing TFs
        - Add noise scaled by the GRN coefficient magnitude
        - Generate synthetic p-values correlated with effect size
    """
    log.info("Generating GRN-informed synthetic DE data (FALLBACK) ...")
    log.warning(
        "  *** This is SIMULATED data, NOT real CROP-seq results. ***"
    )
    log.warning(
        "  *** Replace with real data extracted from Seurat objects. ***"
    )

    rng = np.random.default_rng(42)

    # Try to load real Pando GRN coefficients
    grn_df = _load_pando_coefficients(data_dir)

    results = {}
    for ko in KO_TARGETS:
        de_rows = _generate_ko_de(ko, grn_df, rng)
        results[ko] = de_rows

    return results


def _load_pando_coefficients(data_dir: Path) -> pd.DataFrame | None:
    """Load Pando GRN coefficients if available."""
    for fname in ["grn_modules.tsv", "coefs.tsv"]:
        path = data_dir / "pando" / fname
        if path.exists():
            log.info("  Loading Pando GRN from %s", path)
            df = pd.read_csv(path, sep="\t")
            log.info(
                "  Loaded %d interactions (%d TFs, %d targets)",
                len(df), df["tf"].nunique(), df["target"].nunique(),
            )
            return df
    log.info("  No Pando GRN found; using hard-coded biology only.")
    return None


def _generate_ko_de(
    ko_gene: str,
    grn_df: pd.DataFrame | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate synthetic DE for a single KO, informed by GRN + known biology."""
    biology = KO_BIOLOGY.get(ko_gene, DEFAULT_BIOLOGY)
    genes = []
    log2fcs = []
    pvals = []

    # If we have real GRN data, use it
    if grn_df is not None:
        # Direct targets of the KO'd TF
        ko_targets = grn_df[grn_df["tf"] == ko_gene].copy()
        if len(ko_targets) > 0:
            log.info(
                "  %s: %d direct targets in Pando GRN", ko_gene, len(ko_targets),
            )
            for _, row in ko_targets.iterrows():
                target = row["target"]
                estimate = row.get("estimate", 0.0)
                if pd.isna(estimate):
                    estimate = 0.0

                # If TF activates target (positive coef), KO reduces expression
                # If TF represses target (negative coef), KO increases expression
                base_fc = -estimate * 1.5  # scale up for visibility
                noise = rng.normal(0, 0.2)
                fc = base_fc + noise

                genes.append(target)
                log2fcs.append(fc)
                # p-value inversely related to effect size
                pvals.append(
                    10 ** (-abs(fc) * rng.uniform(3, 8))
                )

        # Non-targets: small random effects (background)
        all_genes = set(grn_df["target"].unique()) | set(grn_df["tf"].unique())
        target_set = set(ko_targets["target"]) if len(ko_targets) > 0 else set()
        non_targets = list(all_genes - target_set - {ko_gene})
        rng.shuffle(non_targets)

        n_background = min(200, len(non_targets))
        for g in non_targets[:n_background]:
            fc = rng.normal(0, 0.15)
            genes.append(g)
            log2fcs.append(fc)
            pvals.append(rng.uniform(0.01, 1.0))

    # Layer on known biology for key genes (override if present)
    known_genes_added = set()
    for gene in biology["expected_down"]:
        if gene not in known_genes_added:
            fc = rng.uniform(-1.5, -0.3)
            if gene in genes:
                idx = genes.index(gene)
                log2fcs[idx] = fc
                pvals[idx] = 10 ** rng.uniform(-10, -3)
            else:
                genes.append(gene)
                log2fcs.append(fc)
                pvals.append(10 ** rng.uniform(-10, -3))
            known_genes_added.add(gene)

    for gene in biology["expected_up"]:
        if gene not in known_genes_added:
            fc = rng.uniform(0.3, 1.5)
            if gene in genes:
                idx = genes.index(gene)
                log2fcs[idx] = fc
                pvals[idx] = 10 ** rng.uniform(-10, -3)
            else:
                genes.append(gene)
                log2fcs.append(fc)
                pvals.append(10 ** rng.uniform(-10, -3))
            known_genes_added.add(gene)

    # If we had no GRN data, add some generic neuronal genes
    if grn_df is None:
        generic_genes = [
            "TUBB3", "MAP2", "DCX", "STMN2", "NCAM1", "VIM", "NES",
            "SOX9", "GFAP", "MKI67", "TOP2A", "PCNA", "CCND1",
            "CDH2", "GAPDH", "ACTB", "SOX2", "PAX6", "HES1",
            "NOTCH1", "SHH", "BMP4", "WNT3A", "OTX2", "FOXG1",
        ]
        for g in generic_genes:
            if g not in genes:
                fc = rng.normal(0, 0.3)
                genes.append(g)
                log2fcs.append(fc)
                pvals.append(rng.uniform(0.05, 1.0))

    # Add the KO'd gene itself (should be strongly downregulated)
    if ko_gene not in genes:
        genes.append(ko_gene)
        log2fcs.append(rng.uniform(-3.0, -1.5))
        pvals.append(10 ** rng.uniform(-15, -8))
    else:
        idx = genes.index(ko_gene)
        log2fcs[idx] = rng.uniform(-3.0, -1.5)
        pvals[idx] = 10 ** rng.uniform(-15, -8)

    # Build DataFrame
    pvals_arr = np.array(pvals)
    # BH correction (simplified: sort p-values, adjust)
    n = len(pvals_arr)
    sorted_idx = np.argsort(pvals_arr)
    padj = np.empty(n)
    for rank, idx in enumerate(sorted_idx, 1):
        padj[idx] = min(pvals_arr[idx] * n / rank, 1.0)
    # Ensure monotonicity
    for i in range(n - 2, -1, -1):
        orig_idx = sorted_idx[i]
        next_idx = sorted_idx[i + 1]
        padj[orig_idx] = min(padj[orig_idx], padj[next_idx])

    df = pd.DataFrame({
        "gene": genes,
        "log2fc": np.round(log2fcs, 4),
        "pval": pvals_arr,
        "padj": padj,
    })

    # Sort by absolute fold change descending
    df = df.sort_values("padj").reset_index(drop=True)

    log.info(
        "  %s KO: %d genes (%d sig at padj<0.05)",
        ko_gene,
        len(df),
        (df["padj"] < 0.05).sum(),
    )

    return df


# ── Utilities ────────────────────────────────────────────────────────────


def _normalize_de_df(df: pd.DataFrame, ko_gene: str) -> pd.DataFrame:
    """Normalize a DE results DataFrame to the standard column format.

    Seurat's FindMarkers output has columns like avg_log2FC, p_val, p_val_adj.
    We normalize to: gene, log2fc, pval, padj.
    """
    out = pd.DataFrame()

    # Gene column
    if "gene" in df.columns:
        out["gene"] = df["gene"].values
    elif df.index.name and df.index.name.lower() in ("gene", "genes"):
        out["gene"] = df.index.values
    elif "row.names" in df.columns:
        out["gene"] = df["row.names"].values
    else:
        # Assume the index is gene names
        out["gene"] = df.index.values

    # Log2 fold change
    fc_candidates = ["avg_log2FC", "avg_logFC", "log2fc", "log2FoldChange", "logFC", "estimate"]
    for col in fc_candidates:
        matches = [c for c in df.columns if c.lower() == col.lower()]
        if matches:
            out["log2fc"] = df[matches[0]].values.astype(float)
            break
    else:
        log.warning("  No fold change column found in: %s", list(df.columns))
        out["log2fc"] = 0.0

    # P-value
    pval_candidates = ["p_val", "pval", "pvalue", "p.value", "PValue"]
    for col in pval_candidates:
        matches = [c for c in df.columns if c.lower() == col.lower()]
        if matches:
            out["pval"] = df[matches[0]].values.astype(float)
            break
    else:
        out["pval"] = np.nan

    # Adjusted p-value
    padj_candidates = ["p_val_adj", "padj", "p.adj", "FDR", "qvalue", "p_val_adj"]
    for col in padj_candidates:
        matches = [c for c in df.columns if c.lower() == col.lower()]
        if matches:
            out["padj"] = df[matches[0]].values.astype(float)
            break
    else:
        # If no padj, use pval (or BH-correct it)
        if "pval" in out.columns and not out["pval"].isna().all():
            # Simple BH correction
            pvals = out["pval"].values
            n = len(pvals)
            sorted_idx = np.argsort(pvals)
            padj = np.empty(n)
            for rank, idx in enumerate(sorted_idx, 1):
                padj[idx] = min(pvals[idx] * n / rank, 1.0)
            out["padj"] = padj
        else:
            out["padj"] = np.nan

    # Sort by padj
    out = out.sort_values("padj").reset_index(drop=True)
    return out


# ── Step 6: Save results ────────────────────────────────────────────────


def save_results(
    results: dict[str, pd.DataFrame],
    output_dir: Path,
    is_synthetic: bool,
) -> None:
    """Save per-KO DE results and summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source": "synthetic_grn_informed" if is_synthetic else "fleck_et_al_2023_seurat",
        "is_synthetic": is_synthetic,
        "extraction_date": datetime.now(timezone.utc).isoformat(),
        "knockouts": {},
    }

    if is_synthetic:
        summary["warning"] = (
            "This data is SIMULATED based on the Pando GRN structure, "
            "NOT real CROP-seq experimental results. Replace with real "
            "data extracted from Fleck et al. Seurat objects."
        )

    for ko, df in results.items():
        filename = f"{ko.lower()}_ko_de.csv"
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)
        log.info("Saved %s (%d genes) -> %s", ko, len(df), filepath)

        n_sig = int((df["padj"] < 0.05).sum()) if "padj" in df.columns else 0
        n_up = int((df["log2fc"] > 0).sum()) if "log2fc" in df.columns else 0
        n_down = int((df["log2fc"] < 0).sum()) if "log2fc" in df.columns else 0

        summary["knockouts"][ko] = {
            "file": filename,
            "n_genes": len(df),
            "n_significant_padj05": n_sig,
            "n_upregulated": n_up,
            "n_downregulated": n_down,
            "description": KO_BIOLOGY.get(ko, {}).get("description", ""),
        }

    # Also create a combined file matching the format expected by
    # 05_perturbation.py's _try_load_cropseq() (gene,ko_gene,log2fc,padj)
    combined_rows = []
    for ko, df in results.items():
        ko_df = df[["gene", "log2fc", "padj"]].copy()
        ko_df.insert(1, "ko_gene", ko)
        combined_rows.append(ko_df)

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined_path = output_dir / "cropseq_de.csv"
        combined.to_csv(combined_path, index=False)
        log.info(
            "Saved combined DE table (%d rows) -> %s",
            len(combined), combined_path,
        )
        summary["combined_file"] = "cropseq_de.csv"
        summary["total_genes"] = len(combined)

    summary_path = output_dir / "cropseq_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary -> %s", summary_path)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract real CROP-seq differential expression data from "
            "Fleck et al. Seurat objects for perturbation prediction validation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Extraction strategies (tried in order):\n"
            "  1. pyreadr   — pure-Python .rds reader\n"
            "  2. rpy2      — R/Seurat via rpy2 bridge\n"
            "  3. synthetic — GRN-informed simulated data (fallback)\n"
            "\n"
            "The tarball is expected at <data-dir>/zenodo/seurat_objects.tar.gz\n"
            "Output is written to <data-dir>/cropseq/\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Path to the data directory (default: ../data relative to script)",
    )
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Skip real data extraction and generate synthetic data directly",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    data_dir = Path(args.data_dir).resolve()
    output_dir = data_dir / "cropseq"

    log.info("=" * 65)
    log.info("CROP-seq DE Extraction — Fleck et al. 2023")
    log.info("=" * 65)
    log.info("Data directory: %s", data_dir)
    log.info("Output directory: %s", output_dir)
    log.info("")

    results: dict[str, pd.DataFrame] | None = None
    is_synthetic = False

    if args.force_synthetic:
        log.info("--force-synthetic set, skipping real data extraction.")
    else:
        # ── Step 1: Extract tarball ──────────────────────────────────────
        log.info("Step 1: Extract Seurat objects tarball")
        log.info("-" * 45)
        extract_dir = extract_tarball(data_dir)

        if extract_dir is not None:
            # ── Step 2: Try pyreadr ──────────────────────────────────────
            log.info("")
            log.info("Step 2: Try pyreadr (.rds reader)")
            log.info("-" * 45)
            results = try_pyreadr(extract_dir)

            if results is None:
                # ── Step 3: Try rpy2/Seurat ──────────────────────────────
                log.info("")
                log.info("Step 3: Try rpy2/Seurat (R bridge)")
                log.info("-" * 45)
                results = try_rpy2(extract_dir)

    # ── Step 4: Inspect organoid_regulomes ────────────────────────────
    log.info("")
    log.info("Step 4: Inspect organoid_regulomes pipeline")
    log.info("-" * 45)
    inspect_organoid_regulomes()

    # ── Step 5 (if needed): Synthetic fallback ────────────────────────
    if results is None:
        log.info("")
        log.info("Step 5: Generate synthetic DE data (fallback)")
        log.info("-" * 45)
        results = generate_synthetic_de(data_dir)
        is_synthetic = True

    # Check completeness
    missing = [ko for ko in KO_TARGETS if ko not in results]
    if missing:
        log.warning("Missing KO results for: %s — generating synthetic.", missing)
        synthetic = generate_synthetic_de(data_dir)
        for ko in missing:
            if ko in synthetic:
                results[ko] = synthetic[ko]
                is_synthetic = True  # At least partially synthetic

    # ── Step 6: Save ──────────────────────────────────────────────────
    log.info("")
    log.info("Step 6: Save results")
    log.info("-" * 45)
    save_results(results, output_dir, is_synthetic)

    # ── Summary ───────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 65)
    log.info("Extraction complete!")
    log.info("  Data type: %s", "SYNTHETIC (replace with real data)" if is_synthetic else "REAL")
    log.info("  KO targets: %s", list(results.keys()))
    for ko, df in results.items():
        n_sig = (df["padj"] < 0.05).sum()
        log.info("    %s: %d genes (%d significant)", ko, len(df), n_sig)
    log.info("  Output: %s", output_dir)
    log.info("=" * 65)


if __name__ == "__main__":
    main()
