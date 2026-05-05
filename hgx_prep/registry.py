"""Dataset registry: maps known DOIs/GEO accessions to configurations."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class DatasetConfig:
    name: str
    doi: str | None = None
    geo: str | None = None
    zenodo_doi: str | None = None
    expression_format: str = "h5ad"        # "h5ad" or "mtx_gz"
    grn_default: str = "de"                # "pando", "de", "supplied"
    guide_column: str | None = None
    control_patterns: list[str] = field(default_factory=lambda: [
        "non-targeting", "NT", "CTRL", "scramble", "NTC", "safe",
    ])
    has_pseudotime: bool = False
    pseudotime_column: str | None = None
    has_fates: bool = False
    fate_columns: list[str] = field(default_factory=list)
    lineage_column: str | None = None
    lineage_names: list[str] = field(default_factory=list)
    key_tfs: list[str] = field(default_factory=list)
    download_files: dict[str, str] = field(default_factory=dict)  # key -> filename
    notes: str = ""


# ---------------------------------------------------------------------------
# Built-in dataset configs
# ---------------------------------------------------------------------------

FLECK_2023 = DatasetConfig(
    name="Fleck et al. 2023 — Cerebral Organoid",
    doi="10.1038/s41586-022-05279-8",
    zenodo_doi="10.5281/zenodo.5242913",
    expression_format="mtx_gz",
    grn_default="pando",
    has_pseudotime=True,
    pseudotime_column="velocity_pseudotime",
    has_fates=True,
    fate_columns=["DF", "VF", "MH"],
    lineage_column="lineage",
    lineage_names=["telencephalon", "early", "nt"],
    key_tfs=["GLI3", "FOXG1", "TBR1", "DLX1", "DLX2", "EMX1", "EOMES", "NEUROD6"],
    notes="Pando GRN from pando/coefs.tsv; fate probs from RNA_all_velo.h5ad",
)

AZBUKINA_2025 = DatasetConfig(
    name="Azbukina et al. 2025 — Posterior Brain Atlas",
    doi="10.1101/2025.03.20.644368",
    zenodo_doi="10.5281/zenodo.14901345",
    geo="GSE297445",
    expression_format="h5ad",
    grn_default="de",
    has_pseudotime=True,
    lineage_column="cell_type",
    notes="Multi-omic atlas of midbrain and hindbrain human neural organoids",
)

WAHLE_2023 = DatasetConfig(
    name="Wahle et al. 2023 — Human Retinal Organoid",
    doi="10.1038/s41587-023-01747-2",
    expression_format="h5ad",
    grn_default="de",
    has_pseudotime=True,
    has_fates=False,
    notes="Multimodal spatiotemporal phenotyping of retinal development; CROP-seq available",
)

POLLEN_2026 = DatasetConfig(
    name="Ding/Pollen et al. 2026 — CRISPRi Perturb-seq",
    doi="10.1038/s41586-025-09997-7",
    geo="GSE284197",
    expression_format="h5ad",
    grn_default="de",
    guide_column=None,  # auto-detected
    has_pseudotime=False,
    has_fates=False,
    download_files={
        "screen": "GSE284197_screen.h5ad",
        "merged": "GSE284197_merged.h5ad",
        "IN": "GSE284197_IN.h5ad",
        "slice": "GSE284197_slice.h5ad",
        "clones": "GSE284197_clones.h5ad",
    },
    notes="44-TF CRISPRi screen in 2D cortical cultures",
)


# Registry lookup
_REGISTRY: dict[str, DatasetConfig] = {}

def _register(config: DatasetConfig) -> None:
    if config.doi:
        _REGISTRY[config.doi] = config
    if config.geo:
        _REGISTRY[config.geo] = config

_register(FLECK_2023)
_register(AZBUKINA_2025)
_register(WAHLE_2023)
_register(POLLEN_2026)


def lookup(identifier: str) -> DatasetConfig | None:
    """Look up a dataset by DOI or GEO accession. Returns None if unknown."""
    return _REGISTRY.get(identifier)


def list_known() -> list[DatasetConfig]:
    """List all registered datasets."""
    seen = set()
    result = []
    for config in _REGISTRY.values():
        if config.name not in seen:
            seen.add(config.name)
            result.append(config)
    return result
