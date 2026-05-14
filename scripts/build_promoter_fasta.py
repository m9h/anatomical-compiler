"""build_promoter_fasta — extract upstream-of-TSS promoter sequences from a GTF + FASTA.

Reproduces the awk + bedtools one-liner in docs/dgx-spark-setup.md §3 but
without bedtools, using pyfaidx for random-access reads from a large
reference FASTA. For each "gene" record in the GTF, takes the configured
window of bases immediately upstream of the TSS (strand-aware), reverse-
complements minus-strand entries, and writes one record per gene to FASTA
keyed by `gene_name`.

The output matches what `bedtools getfasta -s -nameOnly` would produce on
the same BED window — 2000 bp ending at (and including) the TSS in
transcript orientation, by default.

Usage
-----
    python scripts/build_promoter_fasta.py \
        --fasta /data/mhough/refs/hg38/hg38.fa \
        --gtf   /data/mhough/refs/gencode/gencode.v47.basic.annotation.gtf \
        --output /data/mhough/refs/promoters_hg38_2kb.fa \
        --window 2000
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_NAME_RE = re.compile(r'gene_name "([^"]+)"')


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgtNn", "TGCAtgcaNn"))[::-1]


def iter_gtf_genes(gtf_path: Path):
    """Yield (chrom, start, end, strand, gene_name) for each `gene` record."""
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "gene":
                continue
            m = _NAME_RE.search(cols[8])
            if not m:
                continue
            yield cols[0], int(cols[3]), int(cols[4]), cols[6], m.group(1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--fasta", required=True, help="path to reference FASTA (.fai built on first run)")
    p.add_argument("--gtf", required=True, help="path to GENCODE-format GTF")
    p.add_argument("--output", required=True, help="output FASTA path")
    p.add_argument("--window", type=int, default=2000, help="bp upstream of TSS (default 2000)")
    args = p.parse_args()

    try:
        from pyfaidx import Fasta
    except ImportError:
        print("pyfaidx required: pip install pyfaidx", file=sys.stderr)
        return 2

    fa_path = Path(args.fasta)
    gtf_path = Path(args.gtf)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"build_promoter_fasta: indexing {fa_path} (first run builds .fai)…", file=sys.stderr)
    fa = Fasta(str(fa_path))
    print(f"  {len(fa.keys())} contigs", file=sys.stderr)

    n_written = 0
    n_skipped_contig = 0
    n_skipped_short = 0
    seen_names: set[str] = set()

    with open(out_path, "w") as out:
        for chrom, start, end, strand, gene_name in iter_gtf_genes(gtf_path):
            if chrom not in fa:
                n_skipped_contig += 1
                continue
            # 0-based half-open slice covering the window upstream of TSS in transcript orientation.
            # + strand: window = [start-w, start) genomic; TSS (1-based start) is last base.
            # - strand: window = [end-1, end-1+w) genomic; after revcomp, TSS (1-based end) is last base.
            if strand == "+":
                a = max(0, start - args.window)
                b = start
            elif strand == "-":
                a = end - 1
                b = a + args.window
            else:
                continue

            seq = str(fa[chrom][a:b])
            if not seq:
                n_skipped_short += 1
                continue
            if strand == "-":
                seq = revcomp(seq)

            # Disambiguate duplicate gene_name entries (PAR genes, readthroughs, etc.)
            label = gene_name
            if gene_name in seen_names:
                label = f"{gene_name}|{chrom}:{start}-{end}({strand})"
            else:
                seen_names.add(gene_name)
            out.write(f">{label}\n{seq}\n")
            n_written += 1

    print(
        f"build_promoter_fasta: wrote {n_written} records to {out_path} "
        f"(skipped {n_skipped_contig} missing contigs, {n_skipped_short} empty windows)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
