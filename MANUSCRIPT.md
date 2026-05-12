# Manuscript — moved

The manuscript is now a single knitr/Sweave source: **[`publication/paper.Rnw`](publication/paper.Rnw)** (canonical), which knits to `publication/paper.tex` → `publication/paper.pdf`.

```bash
Rscript -e 'knitr::knit("publication/paper.Rnw", output="publication/paper.tex")'
tectonic publication/paper.tex
```

It consolidates what used to live in `MANUSCRIPT.md`, `FOUNDATIONS.md` (→ §1 Introduction / background), `SYNTHETIC_MORPHOGENESIS.md`, `STATUS.md`, and `ROADMAP.md` Phase 3G (→ §3 Results), and pulls result tables / inline values live from `figures/*_results.json` (Python/JAX analyses) and `data/cropseq/*.csv` (R/Seurat DE).

- **Bibliography:** [`REFERENCES.md`](REFERENCES.md) (human-readable master list); the same entries are in the `thebibliography` block of `paper.Rnw`.
- **Roadmap / status:** [`ROADMAP.md`](ROADMAP.md), [`STATUS.md`](STATUS.md).

Do not edit `publication/paper.tex` by hand — it is generated. Edit `paper.Rnw` and re-knit.
