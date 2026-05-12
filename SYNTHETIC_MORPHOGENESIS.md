# Synthetic morphogenesis primer — moved

The conceptual framing (software = GRN / hardware = cell mechanics; Davies' Regulatory /
Effector / Sensor modules; 4D bioprinting and active mechanics; the identifiability challenge;
fidelity benchmarking against primary-tissue blueprints) is now woven through the merged
manuscript — primarily **§1 Introduction** and **§4 Conclusions and Outlook** of
**[`publication/paper.Rnw`](publication/paper.Rnw)** → `publication/paper.pdf`.

The forward experimental programme (model-in-the-loop 4D bioprinting; hybrid programmed-plus-
printed tissues; optogenetic morphogenesis; microglia/vasculature titration; bioelectric
patterning; a cancer-as-loss-of-module-identifiability assay) is §4.3 of that document.

```bash
Rscript -e 'knitr::knit("publication/paper.Rnw", output="publication/paper.tex")'
tectonic publication/paper.tex
```

Bibliography: **[`REFERENCES.md`](REFERENCES.md)**. Roadmap: [`ROADMAP.md`](ROADMAP.md).
