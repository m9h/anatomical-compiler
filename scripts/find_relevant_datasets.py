#!/usr/bin/env python3
"""Find relevant perturbation datasets from scPerturb and PerturBase.

Downloads the scPerturb dataset index, filters for human neural/brain datasets,
computes overlap with Fleck's 720 Pando TFs, and ranks by relevance.

Usage:
    .venv/bin/python scripts/find_relevant_datasets.py
"""
from __future__ import annotations
import json, sys, csv, io
from pathlib import Path
from urllib.request import urlretrieve, urlopen

import numpy as np
import pandas as pd


def load_fleck_tfs(data_dir: Path) -> set[str]:
    with open(data_dir / "tf_names.json") as f:
        return set(json.load(f))


def fetch_scperturb_index() -> pd.DataFrame | None:
    """Fetch scPerturb dataset metadata from their supplementary table."""
    # scPerturb publishes metadata on their site / zenodo
    # Try the Figshare/Zenodo metadata
    urls = [
        "https://zenodo.org/api/records/10044268",  # scPerturb zenodo record
    ]
    for url in urls:
        try:
            print(f"  Fetching scPerturb metadata from {url}...")
            resp = urlopen(url, timeout=30)
            data = json.loads(resp.read())
            files = data.get("files", [])
            print(f"  Found {len(files)} files in record")
            for f in files:
                print(f"    {f['key']} ({f['size']/1e6:.1f} MB)")
            return files
        except Exception as e:
            print(f"  Failed: {e}")
    return None


def search_geo_neural_perturbation():
    """Search GEO for neural perturbation datasets via Entrez."""
    try:
        from urllib.request import urlopen
        from urllib.parse import quote
        query = quote(
            '("CRISPRi"[All Fields] OR "CRISPR"[All Fields] OR "perturbation"[All Fields]) '
            'AND ("organoid"[All Fields] OR "cortical"[All Fields] OR "neural"[All Fields] '
            'OR "brain"[All Fields]) '
            'AND "Homo sapiens"[Organism] '
            'AND "Expression profiling by high throughput sequencing"[DataSet Type] '
            'AND ("2022"[PDAT] : "2026"[PDAT])'
        )
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&term={query}&retmax=50&retmode=json"
        print(f"\n  Searching GEO (Entrez)...")
        resp = urlopen(url, timeout=30)
        data = json.loads(resp.read())
        result = data.get("esearchresult", {})
        ids = result.get("idlist", [])
        count = result.get("count", "0")
        print(f"  Found {count} total results, fetching top {len(ids)}")

        if not ids:
            return []

        # Fetch summaries
        id_str = ",".join(ids)
        sum_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gds&id={id_str}&retmode=json"
        resp2 = urlopen(sum_url, timeout=30)
        summaries = json.loads(resp2.read())

        datasets = []
        for uid in ids:
            doc = summaries.get("result", {}).get(uid, {})
            if not doc:
                continue
            title = doc.get("title", "")
            accession = doc.get("accession", "")
            gpl = doc.get("gpl", "")
            gds_type = doc.get("gdstype", "")
            n_samples = doc.get("n_samples", 0)
            summary = doc.get("summary", "")[:200]

            datasets.append({
                "uid": uid,
                "accession": accession,
                "title": title,
                "n_samples": n_samples,
                "type": gds_type,
                "summary": summary,
            })

        return datasets
    except Exception as e:
        print(f"  GEO search failed: {e}")
        return []


def known_neural_perturbation_datasets() -> list[dict]:
    """Curated list of known neural/cortical perturbation datasets."""
    return [
        {
            "name": "Pollen/Ding 2D Screen",
            "paper": "Ding, Pollen et al. Nature 2026",
            "system": "Primary cortex, 2D CRISPRi",
            "cells": 137317,
            "n_perturbed_genes": 44,
            "accession": "GSE284197 (screen.h5ad)",
            "status": "PROCESSED",
            "perturbed_genes": "ARX, ASCL1, ATF7, BHLHE22, CTCF, DEAF1, E2F1, EMX1, FOS, JUN, KAT7, KLF10, KLF11, KLF3, MEF2C, MEIS2, NEUROD2, NEUROD6, NFIA, NFIB, NR2E1, NR4A2, PHF21A, POU2F1, POU3F1, POU3F2, RFX2, SATB2, SOX2, SOX5, SOX6, SOX9, TBR1, TCF12, TCF3, TCF7L1, TFAP2C, VEZF1, ZBTB18, ZBTB20, ZNF148, ZNF219, ZNF281, ZNF441",
        },
        {
            "name": "Pollen/Ding Slice",
            "paper": "Ding, Pollen et al. Nature 2026",
            "system": "Primary cortex, 3D slice CRISPRi",
            "cells": 18082,
            "n_perturbed_genes": 5,
            "accession": "GSE284197 (slice.h5ad)",
            "status": "PROCESSED",
            "perturbed_genes": "ARX, NEUROD2, NR2E1, SOX2, ZNF219",
        },
        {
            "name": "Pollen/Ding IN",
            "paper": "Ding, Pollen et al. Nature 2026",
            "system": "Interneurons, CRISPRi",
            "cells": 31584,
            "n_perturbed_genes": 5,
            "accession": "GSE284197 (IN.h5ad)",
            "status": "PROCESSED",
            "perturbed_genes": "ARX, NEUROD2, NR2E1, SOX2, ZNF219",
        },
        {
            "name": "Sharf/Li CHOOSE",
            "paper": "Li, Sharf et al. Nature 2023",
            "system": "Telencephalic organoid, CRISPR KO",
            "cells": 12128,
            "n_perturbed_genes": 36,
            "accession": "Zenodo 7083558",
            "status": "PROCESSED",
            "perturbed_genes": "ADNP, ARID1B, ASH1L, ASXL3, BAZ2B, BCL11A, CHD2, CHD8, CIC, DDX3X, DEAF1, FOXP1, ILF2, IRF2BPL, KAT2B, KDM5B, KDM6A, KDM6B, KMT2A, KMT2C, KMT5B, LEO1, MECP2, MED13, MED13L, MYT1L, PHF3, POGZ, SETD5, SMARCC2, SRCAP, SRSF11, TBL1XR1, TBR1, TCF20, WAC",
        },
        {
            "name": "Paulsen Assembloid",
            "paper": "Paulsen et al. Nature 2023",
            "system": "Forebrain assembloid, CRISPR (guide enrichment)",
            "cells": "~1000 assembloids (pooled screen)",
            "n_perturbed_genes": 425,
            "accession": "Unknown — check paper",
            "status": "NOT PROCESSED — readout may be guide enrichment, not scRNA-seq per perturbation",
            "perturbed_genes": "425 NDD genes (CSDE1, SMAD4, LNPK, + 422 others)",
            "note": "Largest neural CRISPR screen but readout is FACS-based guide enrichment, not full transcriptomes per perturbation",
        },
        {
            "name": "Fleck organoid GRN",
            "paper": "Fleck et al. Nature 2023",
            "system": "Cerebral organoid, Pando multiome GRN",
            "cells": 49718,
            "n_perturbed_genes": 8,
            "accession": "Zenodo 5242913",
            "status": "PROCESSED (reference GRN)",
            "perturbed_genes": "GLI3, FOXG1, TBR1, DLX1, DLX2, EMX1, EOMES, NEUROD6 (simulated KO)",
        },
        {
            "name": "Zenk/Fleck EpiScape",
            "paper": "Zenk, Fleck et al. Nat Neurosci 2024",
            "system": "Brain organoid, scCUT&Tag + scRNA-seq",
            "cells": 150000,
            "n_perturbed_genes": 0,
            "accession": "episcape.ethz.ch (browser only)",
            "status": "NOT PROCESSED — no downloadable GRN, Pando GRN not exported",
            "perturbed_genes": "EED inhibitor (global epigenomic, not TF-specific)",
            "note": "Valuable for epigenomic validation of GRN edges but requires PI contact for data",
        },
    ]


def compute_overlap(perturbed_genes_str: str, fleck_tfs: set[str]) -> dict:
    """Compute overlap between a dataset's perturbed genes and Fleck TFs."""
    genes = [g.strip() for g in perturbed_genes_str.split(",")]
    overlap = set(genes) & fleck_tfs
    return {
        "n_perturbed": len(genes),
        "n_overlap": len(overlap),
        "overlap_frac": len(overlap) / len(genes) if genes else 0,
        "overlap_genes": sorted(overlap),
    }


def main():
    data_dir = Path("data/processed")
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    print("=" * 64)
    print("  Dataset Finder: Neural Perturbation Datasets for GRN Validation")
    print("=" * 64)

    # Load Fleck TFs
    fleck_tfs = load_fleck_tfs(data_dir)
    print(f"\n  Fleck Pando TFs: {len(fleck_tfs)}")

    # 1. Known datasets with overlap analysis
    print("\n" + "=" * 64)
    print("  1. KNOWN DATASETS")
    print("=" * 64)

    known = known_neural_perturbation_datasets()
    for ds in known:
        if "perturbed_genes" in ds and ds["perturbed_genes"]:
            ov = compute_overlap(ds["perturbed_genes"], fleck_tfs)
            ds["overlap"] = ov

    print(f"\n  {'Dataset':<25} {'System':<30} {'Pert':>5} {'Overlap':>8} {'Status'}")
    print("  " + "-" * 85)
    for ds in known:
        ov = ds.get("overlap", {})
        n_ov = ov.get("n_overlap", "?")
        n_p = ov.get("n_perturbed", ds.get("n_perturbed_genes", "?"))
        print(f"  {ds['name']:<25} {ds['system']:<30} {str(n_p):>5} "
              f"{str(n_ov):>8} {ds['status'][:30]}")
        if ov.get("overlap_genes"):
            print(f"  {'':25} Overlap: {', '.join(ov['overlap_genes'][:15])}")

    # 2. GEO search for additional datasets
    print("\n" + "=" * 64)
    print("  2. GEO SEARCH (automated)")
    print("=" * 64)

    geo_results = search_geo_neural_perturbation()
    if geo_results:
        print(f"\n  Found {len(geo_results)} GEO datasets:")
        for ds in geo_results[:20]:
            print(f"    {ds['accession']}: {ds['title'][:70]}... ({ds['n_samples']} samples)")
    else:
        print("  No results or search failed")

    # 3. scPerturb index
    print("\n" + "=" * 64)
    print("  3. scPerturb DATASETS")
    print("=" * 64)

    scperturb_files = fetch_scperturb_index()

    # 4. Recommendations
    print("\n" + "=" * 64)
    print("  4. RECOMMENDATIONS")
    print("=" * 64)

    ranked = sorted(
        [ds for ds in known if ds.get("overlap", {}).get("n_overlap", 0) > 0],
        key=lambda x: -x["overlap"]["n_overlap"]
    )
    print(f"\n  Ranked by Fleck TF overlap:")
    for i, ds in enumerate(ranked, 1):
        ov = ds["overlap"]
        status = "DONE" if "PROCESSED" in ds["status"] else "TODO"
        print(f"  {i}. {ds['name']:<25} {ov['n_overlap']:>3} overlap TFs  [{status}]")

    print("\n  Datasets to pursue next:")
    for ds in known:
        if "NOT PROCESSED" in ds.get("status", "") and ds.get("overlap", {}).get("n_overlap", 0) == 0:
            continue
        if "NOT PROCESSED" in ds.get("status", ""):
            print(f"    - {ds['name']}: {ds.get('note', '')[:80]}")

    # Save inventory
    inventory = {
        "fleck_n_tfs": len(fleck_tfs),
        "known_datasets": [{k: v for k, v in ds.items() if k != "overlap"}
                           for ds in known],
        "overlap_summary": [{
            "name": ds["name"],
            "n_overlap": ds.get("overlap", {}).get("n_overlap", 0),
            "overlap_genes": ds.get("overlap", {}).get("overlap_genes", []),
        } for ds in known],
        "geo_results": geo_results[:20] if geo_results else [],
    }
    inv_path = fig_dir / "dataset_inventory.json"
    with open(inv_path, "w") as f:
        json.dump(inventory, f, indent=2, default=str)
    print(f"\n  Inventory saved: {inv_path}")


if __name__ == "__main__":
    main()
