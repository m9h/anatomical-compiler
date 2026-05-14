"""annotate_h5ad_for_geneformer — add ensembl_id + n_counts to an h5ad in place.

Geneformer's TranscriptomeTokenizer requires:
  - adata.var['ensembl_id']  Ensembl gene ID per row in adata.var
  - adata.obs['n_counts']     total counts per cell in adata.obs

Most Perturb-seq h5ads ship with gene SYMBOLS in var_names and no n_counts.
This script maps symbols → Ensembl IDs using a GENCODE GTF (any release),
computes n_counts from the X matrix, and writes a new h5ad.

Genes with no mapping in the GTF are dropped (configurable).

Usage:
    python scripts/annotate_h5ad_for_geneformer.py \
        --input  data/pollen_slice.h5ad \
        --gtf    /data/mhough/refs/gencode/gencode.v47.basic.annotation.gtf \
        --output data/pollen_slice_geneformer.h5ad
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_NAME_RE = re.compile(r'gene_name "([^"]+)"')
_ID_RE = re.compile(r'gene_id "([^"]+)"')


def build_symbol_to_ensembl(gtf_path: Path) -> dict[str, str]:
    """Parse GENCODE GTF; return {gene_symbol → ensembl_id} (no version suffix)."""
    mapping: dict[str, str] = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "gene":
                continue
            m_name = _NAME_RE.search(cols[8])
            m_id = _ID_RE.search(cols[8])
            if not (m_name and m_id):
                continue
            sym = m_name.group(1)
            ensg = m_id.group(1).split(".")[0]  # strip version
            mapping.setdefault(sym, ensg)
            mapping.setdefault(ensg, ensg)  # passthrough Ensembl-keyed var_names
    return mapping


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--input", required=True, help="input h5ad")
    p.add_argument("--gtf", required=True, help="GENCODE GTF for symbol → Ensembl mapping")
    p.add_argument("--output", required=True, help="output h5ad")
    p.add_argument("--keep-unmapped", action="store_true",
                   help="keep genes with no Ensembl mapping (with empty ensembl_id); default drops them")
    args = p.parse_args()

    import anndata as ad
    import scipy.sparse as sp

    print(f"annotate_h5ad: reading {args.input}", file=sys.stderr)
    adata = ad.read_h5ad(args.input)
    print(f"  shape={adata.shape}", file=sys.stderr)

    print(f"annotate_h5ad: building symbol → Ensembl map from {args.gtf}", file=sys.stderr)
    mapping = build_symbol_to_ensembl(Path(args.gtf))
    print(f"  {len(mapping)} symbols (incl. Ensembl-passthrough)", file=sys.stderr)

    ensembl_ids = [mapping.get(name, "") for name in adata.var_names]
    n_mapped = sum(1 for x in ensembl_ids if x)
    n_unmapped = adata.n_vars - n_mapped
    print(
        f"annotate_h5ad: mapped {n_mapped}/{adata.n_vars} genes "
        f"({n_unmapped} unmapped)",
        file=sys.stderr,
    )
    adata.var["ensembl_id"] = ensembl_ids

    if not args.keep_unmapped and n_unmapped:
        keep_mask = np.array([bool(x) for x in ensembl_ids])
        adata = adata[:, keep_mask].copy()
        print(f"  dropped {n_unmapped} unmapped genes; new shape={adata.shape}",
              file=sys.stderr)

    if "n_counts" not in adata.obs.columns:
        X = adata.X
        if sp.issparse(X):
            n_counts = np.asarray(X.sum(axis=1)).flatten()
        else:
            n_counts = np.asarray(X).sum(axis=1)
        adata.obs["n_counts"] = n_counts.astype(np.float32)
        print(f"annotate_h5ad: computed n_counts (mean={n_counts.mean():.0f})",
              file=sys.stderr)
    else:
        print("annotate_h5ad: n_counts already present", file=sys.stderr)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    print(f"annotate_h5ad: wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
