# Soutenance (defense slides)

`AML-GraphMiner_soutenance.pptx` (16 slides, in French) is generated from the measured results of
the repository — nothing is typed by hand — and can be edited freely in PowerPoint or
LibreOffice afterwards:

```bash
pip install python-pptx
python paper/soutenance/make_slides.py   # also writes resultats_ibm.png and resultats_elliptic.png
```

Sources: `benchmarks/results_parts/*/*/results.csv` (3-seed benchmark),
`benchmarks/results_parts/hybrid/hybrid.json` (hybrid detector), `results/eda-*/summary.json`
(data), `docs/explainability.md` (measured neighbourhood effect), `paper/figures/*.png`
(`python paper/figures/make_figures.py`). Fill in the student's name on the title slide.
