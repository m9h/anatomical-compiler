#!/usr/bin/env python3
"""01_prepare_data.py — Prepare data for organoid regulome benchmark.

Attempts to load real Pando output from ../data/pando/. If not found,
generates realistic synthetic data mimicking the Fleck et al. 2023
cerebral organoid GRN structure.

Outputs are saved to ../data/ subdirectories:
    data/pando/coefs.tsv        — TF-target regulatory coefficients
    data/pando/modules.tsv      — Gene-to-module assignments
    data/expression/expr.h5ad   — 500 cells x 200 genes with pseudotime
    data/cellrank/cellrank_probs.csv  — Fate absorption probabilities
    data/cropseq/cropseq_de.csv      — CROP-seq differential expression

Usage:
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --data-dir data --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Biological constants
# ---------------------------------------------------------------------------

# Master regulators: broad patterning of forebrain
MASTER_TFS = ["GLI3", "FOXG1"]

# Intermediate TFs: regional identity and progenitor specification
INTERMEDIATE_TFS = ["TBR1", "DLX1", "DLX2", "EMX1", "EOMES", "NEUROD6", "GAD1"]

# Additional TFs to reach ~100 total
_EXTRA_TF_PREFIXES = [
    "PAX6", "SOX2", "HES1", "HES5", "ASCL1", "NEUROG1", "NEUROG2",
    "LHX6", "NKX2-1", "ARX", "SP8", "COUP-TFI", "COUP-TFII",
    "FEZF2", "BCL11B", "SATB2", "CUX1", "CUX2", "SOX5", "TLE4",
    "BRN2", "FOXP2", "FOXP1", "OTX2", "OTX1", "LMX1A", "MSX1",
    "MSX2", "IRX3", "SIX3", "VAX1", "GSX2", "GSX1", "OLIG2",
    "NR2F1", "NR2F2", "PROX1", "MEIS2", "PBX1", "ZIC1", "ZIC2",
    "ZEB1", "ZEB2", "SOX6", "SOX9", "SOX10", "ID1", "ID2", "ID4",
    "HEY1", "NFIB", "NFIA", "NFIX", "TCF7L2", "LEF1", "CTNNB1",
    "NOTCH1", "NOTCH2", "DLL1", "JAG1", "RBPJ", "SMAD1", "SMAD5",
    "BMP4", "BMP7", "WNT3A", "WNT7B", "SHH", "FGF8", "FGF15",
    "TGFB1", "TGFB2", "GATA3", "GATA2", "ETV1", "ETV5", "RELN",
    "DACH1", "SALL1", "TBR2", "INSM1", "BHLHE22", "LMO3", "LDB1",
    "NFE2L2", "HMGA2", "REST", "MEF2C", "CREB1",
]

# Cell fates in organoid differentiation
FATES = ["ctx", "ge", "nt"]
FATE_LABELS = {
    "ctx": "Cortical excitatory neuron",
    "ge": "Ganglionic eminence interneuron",
    "nt": "Neural tube / meningeal",
}

# Module names reflecting biological regulation programmes
MODULE_NAMES = [
    "dorsal_patterning", "ventral_patterning", "telencephalon_identity",
    "cortical_neurogenesis", "cortical_deep_layer", "cortical_upper_layer",
    "GE_specification", "MGE_interneuron", "LGE_interneuron", "CGE_interneuron",
    "intermediate_progenitor", "radial_glia", "notch_signaling",
    "wnt_signaling", "shh_signaling", "bmp_signaling",
    "cell_cycle", "chromatin_remodeling", "migration", "synaptic_maturation",
]

# Module-to-fate dominant mapping (which fate each module primarily serves)
MODULE_FATE = {
    "dorsal_patterning": "ctx",
    "ventral_patterning": "ge",
    "telencephalon_identity": "ctx",
    "cortical_neurogenesis": "ctx",
    "cortical_deep_layer": "ctx",
    "cortical_upper_layer": "ctx",
    "GE_specification": "ge",
    "MGE_interneuron": "ge",
    "LGE_interneuron": "ge",
    "CGE_interneuron": "ge",
    "intermediate_progenitor": "ctx",
    "radial_glia": "nt",
    "notch_signaling": "nt",
    "wnt_signaling": "ctx",
    "shh_signaling": "ge",
    "bmp_signaling": "nt",
    "cell_cycle": "nt",
    "chromatin_remodeling": "nt",
    "migration": "ctx",
    "synaptic_maturation": "ctx",
}

# TFs associated with each module (hierarchical assignment)
MODULE_TF_SEEDS = {
    "dorsal_patterning": ["GLI3", "EMX1", "PAX6"],
    "ventral_patterning": ["GLI3", "NKX2-1", "GSX2"],
    "telencephalon_identity": ["FOXG1", "OTX2", "SIX3"],
    "cortical_neurogenesis": ["NEUROG1", "NEUROG2", "ASCL1"],
    "cortical_deep_layer": ["TBR1", "FEZF2", "BCL11B", "SOX5"],
    "cortical_upper_layer": ["SATB2", "CUX1", "CUX2", "BRN2"],
    "GE_specification": ["DLX1", "DLX2", "GSX1", "ARX"],
    "MGE_interneuron": ["LHX6", "SOX6", "GAD1"],
    "LGE_interneuron": ["SP8", "MEIS2", "FOXP1"],
    "CGE_interneuron": ["PROX1", "COUP-TFI", "COUP-TFII"],
    "intermediate_progenitor": ["EOMES", "NEUROD6", "INSM1", "TBR2"],
    "radial_glia": ["SOX2", "HES1", "HES5", "SOX9"],
    "notch_signaling": ["NOTCH1", "NOTCH2", "DLL1", "RBPJ", "HEY1"],
    "wnt_signaling": ["LEF1", "TCF7L2", "CTNNB1", "WNT3A"],
    "shh_signaling": ["SHH", "GLI3", "FOXG1"],
    "bmp_signaling": ["BMP4", "BMP7", "SMAD1", "MSX1", "MSX2"],
    "cell_cycle": ["ID1", "ID2", "HMGA2", "REST"],
    "chromatin_remodeling": ["ZEB1", "ZEB2", "NFIB", "NFIA"],
    "migration": ["RELN", "DACH1", "SALL1", "MEF2C"],
    "synaptic_maturation": ["NEUROD6", "CREB1", "NFE2L2", "MEF2C"],
}

NUM_TARGET_GENES = 2000
NUM_CELLS = 500
NUM_EXPR_GENES = 200
CROP_KO_GENES = ["GLI3", "TBR1", "EOMES"]


# ---------------------------------------------------------------------------
# Load real data (if available)
# ---------------------------------------------------------------------------


def try_load_real_data(data_dir: Path) -> bool:
    """Attempt to load real Pando output files.

    Returns True if real data was found and is usable, False otherwise.
    """
    coefs_path = data_dir / "pando" / "coefs.tsv"
    modules_path = data_dir / "pando" / "modules.tsv"

    if not coefs_path.exists():
        print(f"  Real data not found at {coefs_path}")
        return False

    try:
        coefs = pd.read_csv(coefs_path, sep="\t")
        required_cols = {"target", "tf", "estimate", "padj"}
        if not required_cols.issubset(set(coefs.columns)):
            print(f"  coefs.tsv missing required columns: {required_cols - set(coefs.columns)}")
            return False

        n_edges = len(coefs)
        n_sig = (coefs["padj"] < 0.05).sum()
        n_tfs = coefs["tf"].nunique()
        n_targets = coefs["target"].nunique()
        print(f"  Loaded real coefs.tsv: {n_edges} edges, {n_sig} significant (padj<0.05)")
        print(f"    {n_tfs} TFs, {n_targets} targets")

        if modules_path.exists():
            modules = pd.read_csv(modules_path, sep="\t")
            n_mods = modules["module"].nunique() if "module" in modules.columns else 0
            print(f"  Loaded real modules.tsv: {len(modules)} gene-module pairs, {n_mods} modules")

        print("  Real data loaded successfully.")
        return True

    except Exception as e:
        print(f"  Error loading real data: {e}")
        return False


# ---------------------------------------------------------------------------
# Generate synthetic data
# ---------------------------------------------------------------------------


def _build_tf_list(rng: np.random.RandomState) -> list[str]:
    """Assemble the full list of ~100 TFs."""
    tfs = list(MASTER_TFS) + list(INTERMEDIATE_TFS)
    seen = set(tfs)
    for name in _EXTRA_TF_PREFIXES:
        if name not in seen:
            tfs.append(name)
            seen.add(name)
        if len(tfs) >= 100:
            break
    # Pad with generic names if we somehow didn't reach 100
    while len(tfs) < 100:
        name = f"TF_{len(tfs)}"
        tfs.append(name)
    return tfs[:100]


def _build_target_genes(rng: np.random.RandomState) -> list[str]:
    """Generate ~2000 target gene names using realistic gene name patterns."""
    # Mix of real-ish gene name patterns
    prefixes = [
        "NRXN", "NLGN", "CADM", "CDH", "PCDH", "SLITRK", "CNTN",
        "NRCAM", "NCAM", "LRRC", "LRFN", "FLRT", "CLSTN", "LPHN",
        "GRIK", "GRIN", "GRIA", "GABR", "SLC", "KCNK", "KCNJ", "KCNQ",
        "SCN", "CACNA", "CACNB", "SYN", "SYT", "SNAP", "STX", "VAMP",
        "MAP", "MAPT", "TUBB", "TUBA", "NEFH", "NEFL", "INA",
        "STMN", "DCX", "DPYSL", "CRMP", "ARC", "FOS", "JUN",
        "EGR", "HOMER", "SHANK", "DLG", "PSD", "CAMK", "CALM",
        "NTRK", "BDNF", "NGF", "NT", "GDNF", "CNTF",
        "SOD", "CAT", "GPX", "PRDX", "TXN", "GLUL", "GLS",
        "ALDH", "ALDOC", "ENO", "HK", "PKM", "PFKFB",
        "COL", "LAMB", "FN1", "VIM", "GFAP", "AQP", "SLC1A",
    ]
    genes = []
    for prefix in prefixes:
        for suffix in range(1, 40):
            name = f"{prefix}{suffix}"
            genes.append(name)
            if len(genes) >= NUM_TARGET_GENES:
                break
        if len(genes) >= NUM_TARGET_GENES:
            break

    # Fill remaining with numbered genes
    while len(genes) < NUM_TARGET_GENES:
        genes.append(f"GENE_{len(genes)}")

    return genes[:NUM_TARGET_GENES]


def generate_coefs(
    tfs: list[str],
    targets: list[str],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Generate a realistic Pando-style coefficients table.

    Produces ~5000 TF-target edges with hierarchical structure:
    - Master TFs (GLI3, FOXG1) have many targets with larger effect sizes
    - Intermediate TFs have moderate targets
    - Other TFs have fewer targets
    """
    rows = []
    master_set = set(MASTER_TFS)
    intermediate_set = set(INTERMEDIATE_TFS)

    for tf in tfs:
        if tf in master_set:
            n_targets = rng.randint(80, 120)
            base_effect = 0.6
        elif tf in intermediate_set:
            n_targets = rng.randint(40, 70)
            base_effect = 0.4
        else:
            n_targets = rng.randint(15, 40)
            base_effect = 0.25

        chosen_targets = rng.choice(targets, size=min(n_targets, len(targets)), replace=False)

        for target in chosen_targets:
            # Effect size: mixture of activators and repressors
            sign = rng.choice([-1, 1], p=[0.3, 0.7])
            estimate = sign * (base_effect + rng.exponential(0.15))
            std_err = abs(estimate) * rng.uniform(0.1, 0.5)
            statistic = estimate / max(std_err, 1e-6)

            # P-values: most significant, some not
            # Stronger effects -> smaller p-values
            raw_p = np.exp(-abs(statistic) * rng.uniform(0.5, 2.0))
            raw_p = np.clip(raw_p, 1e-300, 1.0)

            rows.append({
                "target": target,
                "tf": tf,
                "peak": f"chr{rng.randint(1, 23)}:{rng.randint(1000000, 200000000)}-{rng.randint(200000000, 250000000)}",
                "estimate": round(float(estimate), 6),
                "std_err": round(float(std_err), 6),
                "statistic": round(float(statistic), 6),
                "pval": float(raw_p),
                "padj": 0.0,  # placeholder
            })

    df = pd.DataFrame(rows)

    # BH correction for padj
    pvals = df["pval"].values
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    rank = np.empty(n, dtype=float)
    rank[sorted_idx] = np.arange(1, n + 1)
    padj = np.minimum(pvals * n / rank, 1.0)
    # Enforce monotonicity (descending in sorted order)
    padj_sorted = padj[sorted_idx]
    for i in range(n - 2, -1, -1):
        padj_sorted[i] = min(padj_sorted[i], padj_sorted[i + 1])
    padj[sorted_idx] = padj_sorted
    df["padj"] = padj

    return df


def generate_modules(
    tfs: list[str],
    targets: list[str],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Generate gene-to-module assignments.

    Assigns each gene (TF or target) to one of ~20 modules based on
    biological hierarchical logic.
    """
    rows = []
    all_genes = set(tfs + targets)
    assigned = set()

    # First, assign TFs to their seed modules
    for mod_name, seed_tfs in MODULE_TF_SEEDS.items():
        for tf in seed_tfs:
            if tf in all_genes:
                rows.append({"gene": tf, "module": mod_name})
                assigned.add(tf)

    # Assign remaining TFs to related modules
    for tf in tfs:
        if tf not in assigned:
            mod = rng.choice(MODULE_NAMES)
            rows.append({"gene": tf, "module": mod})
            assigned.add(tf)

    # Distribute target genes across modules with biologically motivated sizes
    # Cortical modules get more targets (larger transcriptional programs)
    module_weights = {
        "dorsal_patterning": 1.5,
        "ventral_patterning": 1.2,
        "telencephalon_identity": 1.0,
        "cortical_neurogenesis": 1.8,
        "cortical_deep_layer": 1.5,
        "cortical_upper_layer": 1.5,
        "GE_specification": 1.2,
        "MGE_interneuron": 1.0,
        "LGE_interneuron": 0.8,
        "CGE_interneuron": 0.8,
        "intermediate_progenitor": 1.3,
        "radial_glia": 1.0,
        "notch_signaling": 0.7,
        "wnt_signaling": 0.8,
        "shh_signaling": 0.7,
        "bmp_signaling": 0.6,
        "cell_cycle": 0.9,
        "chromatin_remodeling": 0.7,
        "migration": 1.0,
        "synaptic_maturation": 1.2,
    }
    weights = np.array([module_weights[m] for m in MODULE_NAMES])
    probs = weights / weights.sum()

    unassigned_targets = [g for g in targets if g not in assigned]
    module_assignments = rng.choice(MODULE_NAMES, size=len(unassigned_targets), p=probs)

    for gene, mod in zip(unassigned_targets, module_assignments):
        rows.append({"gene": gene, "module": mod})

    return pd.DataFrame(rows)


def generate_expression(
    tfs: list[str],
    targets: list[str],
    modules_df: pd.DataFrame,
    rng: np.random.RandomState,
) -> "anndata.AnnData":
    """Generate a 500 cells x 200 genes AnnData with pseudotime and cell_type.

    Expression profiles are shaped by:
    - Cell fate (ctx / ge / nt) driving module-specific gene programs
    - Pseudotime (0-1) driving progressive differentiation
    - Biological noise
    """
    import anndata
    import scipy.sparse

    # Select 200 representative genes: all key TFs + random targets
    key_tfs = list(MASTER_TFS) + list(INTERMEDIATE_TFS)
    selected_genes = list(key_tfs)
    remaining_tfs = [tf for tf in tfs if tf not in selected_genes]
    rng.shuffle(remaining_tfs)
    selected_genes.extend(remaining_tfs[:30])

    n_targets_needed = NUM_EXPR_GENES - len(selected_genes)
    available_targets = [t for t in targets if t not in selected_genes]
    rng.shuffle(available_targets)
    selected_genes.extend(available_targets[:n_targets_needed])
    selected_genes = selected_genes[:NUM_EXPR_GENES]

    # Build gene-to-module lookup
    gene_mod = dict(zip(modules_df["gene"], modules_df["module"]))

    # Cell identities
    n_ctx = int(NUM_CELLS * 0.5)
    n_ge = int(NUM_CELLS * 0.3)
    n_nt = NUM_CELLS - n_ctx - n_ge

    cell_types = (
        ["RG_ctx"] * (n_ctx // 3) +
        ["IPC_ctx"] * (n_ctx // 3) +
        ["neuron_ctx"] * (n_ctx - 2 * (n_ctx // 3)) +
        ["RG_ge"] * (n_ge // 3) +
        ["IPC_ge"] * (n_ge // 3) +
        ["neuron_ge"] * (n_ge - 2 * (n_ge // 3)) +
        ["RG_nt"] * (n_nt // 2) +
        ["neuron_nt"] * (n_nt - n_nt // 2)
    )

    cell_fates = []
    for ct in cell_types:
        if "ctx" in ct:
            cell_fates.append("ctx")
        elif "ge" in ct:
            cell_fates.append("ge")
        else:
            cell_fates.append("nt")

    # Pseudotime: sorted within each fate lineage, with some noise
    pseudotime = np.zeros(NUM_CELLS)
    idx = 0
    for fate in FATES:
        fate_mask = [i for i, f in enumerate(cell_fates) if f == fate]
        n_fate = len(fate_mask)
        base_pt = np.linspace(0, 1, n_fate) + rng.normal(0, 0.05, n_fate)
        base_pt = np.clip(np.sort(base_pt), 0, 1)
        for i, ci in enumerate(fate_mask):
            pseudotime[ci] = base_pt[i]

    # Generate expression matrix
    X = np.zeros((NUM_CELLS, NUM_EXPR_GENES), dtype=np.float32)

    for j, gene in enumerate(selected_genes):
        mod = gene_mod.get(gene, "cell_cycle")
        mod_fate = MODULE_FATE.get(mod, "nt")

        # Base expression: log-normal
        base = rng.lognormal(mean=1.0, sigma=0.5, size=NUM_CELLS).astype(np.float32)

        for i in range(NUM_CELLS):
            fate = cell_fates[i]
            pt = pseudotime[i]

            # Fate-specific activation
            if fate == mod_fate:
                fate_boost = 2.0 + rng.normal(0, 0.3)
            else:
                fate_boost = 0.5 + rng.normal(0, 0.1)

            # Pseudotime modulation: genes activate at different stages
            if gene in MASTER_TFS:
                # Master TFs: active early, decreasing later
                pt_mod = 1.5 * np.exp(-2.0 * pt) + 0.3
            elif gene in INTERMEDIATE_TFS:
                # Intermediate TFs: peak at mid-pseudotime
                pt_mod = 2.0 * np.exp(-4.0 * (pt - 0.4) ** 2) + 0.2
            else:
                # Target genes: gradually increase
                pt_mod = 0.3 + 1.5 * pt + rng.normal(0, 0.1)

            X[i, j] = max(0, base[i] * fate_boost * pt_mod)

    # Add technical noise
    X += np.abs(rng.normal(0, 0.1, X.shape).astype(np.float32))

    adata = anndata.AnnData(
        X=scipy.sparse.csr_matrix(X),
        obs=pd.DataFrame({
            "cell_type": cell_types,
            "cell_fate": cell_fates,
            "pseudotime": pseudotime,
        }, index=[f"cell_{i}" for i in range(NUM_CELLS)]),
        var=pd.DataFrame({
            "gene_name": selected_genes,
            "module": [gene_mod.get(g, "cell_cycle") for g in selected_genes],
        }, index=selected_genes),
    )

    return adata


def generate_cellrank_probs(
    pseudotime: np.ndarray,
    cell_fates: list[str],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Generate CellRank-style fate absorption probabilities.

    Each cell gets a probability of reaching each of 3 fates (ctx, ge, nt).
    Earlier pseudotime -> more uncertain (mixed probabilities).
    Later pseudotime -> more committed (one fate dominates).
    """
    n_cells = len(cell_fates)
    probs = np.zeros((n_cells, 3), dtype=np.float64)

    for i in range(n_cells):
        pt = pseudotime[i]
        fate = cell_fates[i]

        # Concentration parameter: higher pt -> more certain
        concentration = 1.0 + 8.0 * pt

        # Base probabilities: bias toward true fate
        if fate == "ctx":
            alpha = [concentration * 0.7, concentration * 0.15, concentration * 0.15]
        elif fate == "ge":
            alpha = [concentration * 0.15, concentration * 0.7, concentration * 0.15]
        else:
            alpha = [concentration * 0.15, concentration * 0.15, concentration * 0.7]

        probs[i] = rng.dirichlet(alpha)

    df = pd.DataFrame({
        "cell": [f"cell_{i}" for i in range(n_cells)],
        "to_ctx": probs[:, 0],
        "to_ge": probs[:, 1],
        "to_nt": probs[:, 2],
        "pseudotime": pseudotime,
    })

    return df


def generate_cropseq_de(
    tfs: list[str],
    targets: list[str],
    modules_df: pd.DataFrame,
    coefs_df: pd.DataFrame,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Generate CROP-seq differential expression for TF knockouts.

    Simulates knockouts of GLI3, TBR1, and EOMES with biologically
    realistic downstream effects:
    - GLI3 KO: derepression of ventral genes (SHH target upregulation),
      loss of dorsal patterning
    - TBR1 KO: loss of deep-layer cortical neuron identity genes
    - EOMES KO: failure of intermediate progenitor differentiation
    """
    gene_mod = dict(zip(modules_df["gene"], modules_df["module"]))

    # Build TF -> target lookup from coefs
    tf_targets = {}
    for _, row in coefs_df.iterrows():
        if row["padj"] < 0.05:
            tf = row["tf"]
            tgt = row["target"]
            est = row["estimate"]
            tf_targets.setdefault(tf, []).append((tgt, est))

    all_genes = set(tfs + targets)
    rows = []

    for ko_gene in CROP_KO_GENES:
        # Direct targets: inverse of regulatory estimate
        direct_targets = tf_targets.get(ko_gene, [])
        affected_genes = set()

        for tgt, est in direct_targets:
            # Knocking out the TF reverses its regulatory effect
            log2fc = -est * rng.uniform(0.5, 1.5)
            # Direct targets have strong signal
            padj = 10 ** rng.uniform(-10, -2)
            rows.append({
                "gene": tgt,
                "ko_gene": ko_gene,
                "log2fc": round(float(log2fc), 4),
                "padj": float(padj),
            })
            affected_genes.add(tgt)

        # Indirect effects: genes in same modules are also affected
        ko_modules = set()
        for mod_name, seed_tfs in MODULE_TF_SEEDS.items():
            if ko_gene in seed_tfs:
                ko_modules.add(mod_name)

        for gene in all_genes:
            if gene in affected_genes or gene == ko_gene:
                continue
            gmod = gene_mod.get(gene)
            if gmod in ko_modules:
                # Indirect effect: weaker, with chance of being non-significant
                log2fc = rng.normal(0, 0.4)
                padj = 10 ** rng.uniform(-5, 0)
                rows.append({
                    "gene": gene,
                    "ko_gene": ko_gene,
                    "log2fc": round(float(log2fc), 4),
                    "padj": float(padj),
                })

        # The knocked-out gene itself
        rows.append({
            "gene": ko_gene,
            "ko_gene": ko_gene,
            "log2fc": round(float(rng.uniform(-5.0, -3.0)), 4),
            "padj": 1e-50,
        })

    return pd.DataFrame(rows)


def generate_synthetic_data(data_dir: Path, seed: int) -> None:
    """Generate all synthetic datasets and save to data_dir."""
    rng = np.random.RandomState(seed)

    print("\nGenerating synthetic data...")

    # Build gene lists
    tfs = _build_tf_list(rng)
    targets = _build_target_genes(rng)
    print(f"  TFs: {len(tfs)}")
    print(f"  Target genes: {len(targets)}")

    # 1. Coefficients
    print("\n  Generating coefs.tsv...")
    coefs_df = generate_coefs(tfs, targets, rng)
    coefs_path = data_dir / "pando" / "coefs.tsv"
    coefs_path.parent.mkdir(parents=True, exist_ok=True)
    coefs_df.to_csv(coefs_path, sep="\t", index=False)
    n_sig = (coefs_df["padj"] < 0.05).sum()
    print(f"    Total edges: {len(coefs_df)}")
    print(f"    Significant edges (padj < 0.05): {n_sig}")
    print(f"    TFs: {coefs_df['tf'].nunique()}")
    print(f"    Targets: {coefs_df['target'].nunique()}")
    print(f"    Saved to {coefs_path}")

    # 2. Modules
    print("\n  Generating modules.tsv...")
    modules_df = generate_modules(tfs, targets, rng)
    modules_path = data_dir / "pando" / "modules.tsv"
    modules_df.to_csv(modules_path, sep="\t", index=False)
    mod_counts = modules_df["module"].value_counts()
    print(f"    Gene-module assignments: {len(modules_df)}")
    print(f"    Modules: {modules_df['module'].nunique()}")
    print(f"    Largest module: {mod_counts.idxmax()} ({mod_counts.max()} genes)")
    print(f"    Smallest module: {mod_counts.idxmin()} ({mod_counts.min()} genes)")
    print(f"    Saved to {modules_path}")

    # 3. Expression data
    print("\n  Generating expression.h5ad...")
    adata = generate_expression(tfs, targets, modules_df, rng)
    expr_path = data_dir / "expression" / "expr.h5ad"
    expr_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(expr_path)
    print(f"    Cells: {adata.n_obs}")
    print(f"    Genes: {adata.n_vars}")
    print(f"    Cell types: {adata.obs['cell_type'].nunique()}")
    print(f"    Pseudotime range: [{adata.obs['pseudotime'].min():.3f}, {adata.obs['pseudotime'].max():.3f}]")
    print(f"    Saved to {expr_path}")

    # 4. CellRank probabilities
    print("\n  Generating cellrank_probs.csv...")
    cellrank_df = generate_cellrank_probs(
        adata.obs["pseudotime"].values,
        adata.obs["cell_fate"].tolist(),
        rng,
    )
    cellrank_path = data_dir / "cellrank" / "cellrank_probs.csv"
    cellrank_path.parent.mkdir(parents=True, exist_ok=True)
    cellrank_df.to_csv(cellrank_path, index=False)
    print(f"    Cells: {len(cellrank_df)}")
    print(f"    Mean fate probabilities:")
    for col in ["to_ctx", "to_ge", "to_nt"]:
        print(f"      {col}: {cellrank_df[col].mean():.3f}")
    print(f"    Saved to {cellrank_path}")

    # 5. CROP-seq DE
    print("\n  Generating cropseq_de.csv...")
    cropseq_df = generate_cropseq_de(tfs, targets, modules_df, coefs_df, rng)
    cropseq_path = data_dir / "cropseq" / "cropseq_de.csv"
    cropseq_path.parent.mkdir(parents=True, exist_ok=True)
    cropseq_df.to_csv(cropseq_path, index=False)
    for ko in CROP_KO_GENES:
        sub = cropseq_df[cropseq_df["ko_gene"] == ko]
        n_de = (sub["padj"] < 0.05).sum()
        print(f"    {ko} KO: {len(sub)} genes tested, {n_de} DE (padj < 0.05)")
    print(f"    Saved to {cropseq_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  Data preparation complete")
    print("=" * 60)
    print(f"  Pando coefs:    {coefs_path}")
    print(f"  Pando modules:  {modules_path}")
    print(f"  Expression:     {expr_path}")
    print(f"  CellRank:       {cellrank_path}")
    print(f"  CROP-seq DE:    {cropseq_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for the organoid-hgx-benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/01_prepare_data.py\n"
            "  python scripts/01_prepare_data.py --data-dir data --seed 123\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Output directory for generated data (default: ../data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Resolve data directory relative to script location
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / args.data_dir).resolve()

    print("=" * 60)
    print("  Organoid Regulome Benchmark — Data Preparation")
    print("=" * 60)
    print(f"  Data directory: {data_dir}")
    print(f"  Seed: {args.seed}")

    # Try loading real data first
    print("\nChecking for real Pando output...")
    if try_load_real_data(data_dir):
        print("\nReal data is available. Checking for supplementary files...")
        # Even with real Pando data, we may still need to generate
        # expression, cellrank, and cropseq data
        expr_path = data_dir / "expression" / "expr.h5ad"
        cellrank_path = data_dir / "cellrank" / "cellrank_probs.csv"
        cropseq_path = data_dir / "cropseq" / "cropseq_de.csv"

        missing = []
        if not expr_path.exists():
            missing.append("expression.h5ad")
        if not cellrank_path.exists():
            missing.append("cellrank_probs.csv")
        if not cropseq_path.exists():
            missing.append("cropseq_de.csv")

        if missing:
            print(f"  Missing supplementary files: {', '.join(missing)}")
            print("  Will generate synthetic versions of missing files.")
            generate_synthetic_data(data_dir, args.seed)
        else:
            print("  All data files present. Nothing to generate.")
    else:
        print("  Falling back to synthetic data generation.")
        generate_synthetic_data(data_dir, args.seed)


if __name__ == "__main__":
    main()
