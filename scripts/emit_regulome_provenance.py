"""emit_regulome_provenance — write the regulome substrate's MIRIAM-style manifest.

CURE-audit item 4 (extension beyond Lab-1 SBML). The Hypergraph Neural ODE,
the MII diagnostic, the fidelity-triple predictor, the FM-prior caches, and
the BO/EIG acquisition all operate over a *regulome substrate* — a
Pando-inferred GRN on CHOOSE brain-organoid scRNA-seq + ATAC (Fleck 2023).
That substrate has no SBML representation (50 k-gene learned graphs aren't
SBML-shaped), but it *does* deserve MIRIAM-style annotation: species,
reference genome, source citations, inference method.

This script writes ``models/regulome_provenance.json`` — a structured
manifest the downstream consumers (notebooks, scripts, the whitepaper)
can join against. It's the CURE-Annotation answer for the "data substrate"
side of the project, complementing the SBML-level MIRIAM annotations on
the Lab-1 ODE circuits.

The manifest is idempotent — re-running gives byte-identical output for a
given schema version, so the regulome substrate's provenance is checked
in as a static artifact, not a build-time output.

Usage
-----
    python scripts/emit_regulome_provenance.py [--output models/regulome_provenance.json]

See also
--------
- docs/cure-audit.md item 4 (Credible annotation)
- models/lab1_*.sbml — the closed-form Lab-1 ODE annotations (item 4 on SBML side)
- scripts/fm_embed.py / fm_edges_seq.py / fm_perturb_scgpt.py — the FM-prior
  layer this regulome substrate feeds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# The canonical manifest. Bumping `schema_version` is how downstream code
# detects breaking changes. Adding fields is a minor bump; removing or
# renaming fields is a major bump.
SCHEMA_VERSION = "1.0"

MANIFEST = {
    "schema_version": SCHEMA_VERSION,
    "substrate": {
        "name": "Pando-on-CHOOSE",
        "description": (
            "Gene regulatory network inferred via Pando (Fleck et al. 2023, Nature) on "
            "the CHOOSE single-cell brain-organoid screening dataset (Li, Fleck, "
            "Martins-Costa et al. 2023, Nature 621:373). The substrate underlying this "
            "project's Hypergraph Neural ODE (Lab 5), MII diagnostic (Lab 4), fidelity-"
            "triple predictor (Lab 3), high-leverage TF ranking (Lab 6), foundation-"
            "model prior caches (scripts/fm_*.py), and cancer-as-loss-of-MII signature "
            "(Lab 10)."
        ),
        "kind": "directed-bipartite-hypergraph",
        "scale": "genome (~30 k TF + target genes; ~10 k high-confidence regulatory edges)",
        "sbml_compatible": False,
        "sbml_compatibility_note": (
            "The regulome is a learned 50 k-parameter graph, not a symbolic kinetic "
            "model. SBML / CellML are not the right representations; the CURE-aligned "
            "response is to document the substrate at the manifest level (this file) "
            "and to emit Lab 8's *output* in a standards-compliant format — PhysiCell "
            "grammar (Johnson et al. 2025, REFERENCES.md ref. 77c) is the candidate "
            "target. See docs/cure-audit.md §'structural non-compliance' for the "
            "honest framing."
        ),
    },
    "species": {
        "scientific_name": "Homo sapiens",
        "common_name": "human",
        "ncbi_taxon_id": "9606",
        "miriam_uri": "https://identifiers.org/taxonomy/9606",
    },
    "reference_genome": {
        "assembly_accession": "GCA_000001405.15",
        "assembly_name": "GRCh38",
        "common_name": "hg38",
        "annotation_source": "GENCODE",
        "annotation_version": "v45",
        "ensembl_release": "110",
        "miriam_uri": "https://identifiers.org/grch38",
    },
    "citations": {
        "data_source": {
            "title": "Single-cell brain organoid screening identifies developmental defects in autism (CHOOSE)",
            "authors": ["Li C", "Fleck JS", "Martins-Costa C", "et al."],
            "journal": "Nature",
            "year": 2023,
            "volume_pages": "621:373-380",
            "doi": "10.1038/s41586-023-06473-y",
            "pubmed": "37468635",
            "miriam_uri": "https://identifiers.org/pubmed/37468635",
            "references_md_entry": "ref. 86",
        },
        "inference_method": {
            "title": "Inferring and perturbing cell fate regulomes in human brain organoids",
            "authors": ["Fleck JS", "Jovanovska M", "Camp JG", "Treutlein B"],
            "journal": "Nature",
            "year": 2023,
            "doi": "10.1038/s41586-022-05279-8",
            "method_name": "Pando",
            "method_url": "https://github.com/quadbio/Pando",
            "miriam_uri": "https://identifiers.org/doi/10.1038/s41586-022-05279-8",
        },
        "modelling_guidelines": {
            "title": "From FAIR to CURE: Guidelines for Computational Models of Biological Systems",
            "authors": ["Sauro HM", "Agmon E", "Blinov ML", "Gennari JH", "Hellerstein J", "et al."],
            "arxiv": "2502.15597",
            "year": 2025,
            "miriam_uri": "https://identifiers.org/arxiv:2502.15597",
            "references_md_entry": "ref. 99a",
            "applicable_pillar": "CURE-Credible (annotation), CURE-Understandable",
        },
    },
    "downstream_consumers": [
        {
            "artifact": "Hypergraph Neural ODE",
            "lab": "notebooks/05_hypergraph_neural_odes.ipynb",
            "uses_substrate_for": "node features + hyperedge topology",
        },
        {
            "artifact": "MII regulome diagnostic",
            "lab": "notebooks/04_modularity_identifiability.ipynb",
            "uses_substrate_for": "Hodge L0/L1 Laplacian spectrum",
        },
        {
            "artifact": "Fidelity-triple predictor",
            "lab": "notebooks/03_benchmarking_fidelity.ipynb",
            "uses_substrate_for": "perturbation-response training set",
        },
        {
            "artifact": "High-leverage TF ranking",
            "lab": "notebooks/06_control_theory.ipynb",
            "uses_substrate_for": "linearised plant for controllability + LQR",
        },
        {
            "artifact": "FM-prior caches",
            "lab": "scripts/fm_embed.py, fm_edges_seq.py, fm_perturb_scgpt.py",
            "uses_substrate_for": "node + edge + perturbation priors",
        },
        {
            "artifact": "Cancer-as-loss-of-MII diagnostic",
            "lab": "notebooks/10_cancer_module_identifiability.ipynb",
            "uses_substrate_for": "MII baseline vs the tumor-organoid trio",
        },
    ],
    "cure_alignment": {
        "pillars_satisfied": ["Credible (annotation)", "Understandable (standards reference)"],
        "explicitly_not_applicable": "SBML/CellML (substrate is a learned graph, not a kinetic model)",
        "see_also": [
            "docs/cure-audit.md",
            "REFERENCES.md (refs 86, 99a)",
            "docs/foundation-models.md",
            "models/README.md",
        ],
    },
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--output", default="models/regulome_provenance.json")
    p.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if the on-disk manifest differs from the in-script canonical value",
    )
    args = p.parse_args(argv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(MANIFEST, indent=2, sort_keys=False) + "\n"

    if args.check:
        if not out_path.exists():
            print(f"emit_regulome_provenance: --check FAIL — {out_path} does not exist")
            return 1
        existing = out_path.read_text(encoding="utf-8")
        if existing != canonical:
            print(
                f"emit_regulome_provenance: --check FAIL — {out_path} differs from canonical "
                "in-script manifest. Re-run without --check to regenerate."
            )
            return 1
        print(f"emit_regulome_provenance: --check OK — {out_path} matches canonical.")
        return 0

    out_path.write_text(canonical, encoding="utf-8")
    print(f"emit_regulome_provenance: wrote {out_path} ({len(canonical)} bytes, schema_version={SCHEMA_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
