# Manuscript (Software Impacts)

`main.tex` is a draft of an *Original Software Publication* for **Software Impacts**
(Elsevier), structured as in the project specification (§7.2): abstract of 150–200 words (199),
five keywords, the **Code metadata** table (C1–C9), 1 Motivation and significance, 2 Software
description (architecture, detectors, YAML configuration, Hub model format, a load → detect →
explain listing), 3 Illustrative examples (the three notebooks, with the measured results and the
benchmark table), 4 Impact (academic, teaching, industrial, regulatory FATF / UEMOA-BCEAO,
geographic), 5 Conclusions. Every number in it was measured with the repository. The software
must be publicly shared under an approved licence (MIT is) before submission; always check the
current *Guide for Authors*.

`memoire_plan.md` is the detailed plan of the thesis (in French) with the measured results, and
`soutenance/` holds the defence slides generated from the results (`soutenance/README.md`).

## Build

```bash
python figures/make_figures.py            # architecture.pdf and example_explanation.pdf (synthetic data)
latexmk -pdf main.tex                     # uses Elsevier's elsarticle if installed, otherwise a preview stub
```

Without `elsarticle.cls` (see `preview_stub.tex`) you get a layout *preview*, not the journal
layout; on Overleaf the real class is used automatically.

## To do before submission

- [ ] Author names, affiliations, e-mail (`main.tex` front matter, code-metadata C9).
- [ ] Publish the repository, archive the release on Zenodo, fill C2 (permanent link / DOI), C8 (docs).
- [x] Run `benchmarks/` on the real IBM AML and Elliptic data and regenerate `benchmarks/results/results.tex`
      (included automatically).
- [ ] Add a sourced figure for the size of the AML software market (the specification quotes
      ≈ 3.5 bn USD in 2024 without a source; it is not in the text).
- [ ] Update the "Impact" section with real uses (course, research project, publications).
- [ ] Acknowledgements, funding, competing-interest and AI-use statements (journal policy).
- [ ] Check every reference against the source; add the software's own DOI reference.
- [ ] Respect the three-page body limit in the journal's own template.

## References checked

Added and checked online: UNODC (2011) illicit financial flows report (1.6 trillion USD laundered
in 2009, 2.7 % of world GDP, consistent with the IMF 2–5 % range); FATF Recommendations; FATF
(July 2021) *Opportunities and Challenges of New Technologies for AML/CFT*; Directive
n° 02/2015/CM/UEMOA (published by the BCEAO); Weber et al. 2018 (arXiv:1812.00076, AMLSim); van den
Beukel, Rožanec & Varbanescu, *Tide* (arXiv:2603.01863, 2026 — the specification dates it 2024,
the arXiv record is from 2026).

`references.bib` entries for Altman et al. (NeurIPS 2023), Weber et al. (arXiv:1908.02591),
Cardoso, Saleiro & Bizarro (LaundroGraph, arXiv:2210.14360) and Egressy et al. (AAAI 2024) were
checked against the publishers' / arXiv pages (authors, title, venue, year). Note that LaundroGraph
is by **Cardoso et al.**, not "Oliveira et al.". The classic references (GCN, GraphSAGE, GAT,
GNNExplainer, PyG, scikit-learn, NetworkX) are standard citations; verify page numbers if the journal
requires them.
