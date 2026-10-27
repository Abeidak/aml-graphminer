# Manuscript (Software Impacts)

`main.tex` is a draft of an *Original Software Publication* for **Software Impacts**
(Elsevier). The journal's template asks for: title, authors, a ≈100-word abstract, up to six
keywords, the **Code metadata** table (C1–C9), a body of up to three pages (purpose and
functionality, impact, limitations), optional illustrative example, acknowledgements and
references, and requires the software to be publicly shared under one of its approved licences
(MIT is). Always check the current *Guide for Authors* before submitting.

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
- [ ] Run `benchmarks/` on the real IBM AML and Elliptic data and regenerate `benchmarks/results/results.tex`
      (the table is included automatically from there; delete synthetic rows).
- [ ] Update the "Impact" section with real uses (course, research project, publications).
- [ ] Acknowledgements, funding, competing-interest and AI-use statements (journal policy).
- [ ] Check every reference against the source; add the software's own DOI reference.
- [ ] Respect the three-page body limit in the journal's own template.

## References checked

`references.bib` entries for Altman et al. (NeurIPS 2023), Weber et al. (arXiv:1908.02591),
Cardoso, Saleiro & Bizarro (LaundroGraph, arXiv:2210.14360) and Egressy et al. (AAAI 2024) were
checked against the publishers' / arXiv pages (authors, title, venue, year). Note that LaundroGraph
is by **Cardoso et al.**, not "Oliveira et al.". The classic references (GCN, GraphSAGE, GAT,
GNNExplainer, PyG, scikit-learn, NetworkX) are standard citations; verify page numbers if the journal
requires them.
