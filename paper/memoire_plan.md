# Plan of the end-of-studies report (mémoire PFE)

Suggested structure, mapped to what the repository provides. Chapters 4 and 5 need the benchmark
runs on the real data (`benchmarks/`); nothing in this plan is a result.

1. **Introduction** — money laundering, scale of the problem, why graphs, objectives of the project
   (from the specification), contributions, organisation of the report.
2. **State of the art** — AML typologies (fan-in/out, smurfing, cycles, layering); rule-based systems;
   machine learning on tabular features; GNNs (GCN, GraphSAGE, GAT) for AML (Weber et al. 2019;
   Altman et al. 2023; Egressy et al. 2024; Cardoso et al. 2022); explainability (GNNExplainer,
   attention, perturbation); datasets and evaluation pitfalls (temporal leakage, imbalance).
3. **Analysis and design** — requirements from the specification; analysis of the original
   Colab notebooks and their gaps (`notebooks/colab_legacy/`); six-layer architecture
   (`docs/architecture.md`); data model; evaluation protocol (`docs/concepts.md`).
4. **Implementation** — ingestion and adapters, leak-free features, motif detectors and risk
   communities (definitions and parameters, `docs/patterns.md`), supervised GNNs and the
   unsupervised graph auto-encoder (`docs/models.md`), explainers (GNNExplainer, PGExplainer,
   Captum, occlusion), the web investigation interface (`docs/web_interface.md`, screenshots),
   CLI / API / tracking / Hub / Neo4j, quality (tests, typing, CI, Docker, documentation).
5. **Experiments and results** — datasets (IBM HI/LI-Small, Elliptic) and their EDA
   (`amlgm eda`), protocol, results tables from `benchmarks/make_report.py` (mean ± std, confidence
   intervals), per-motif recall, supervised vs. unsupervised (`gae`) detection, ablations
   (with/without node features, windows), explanation case studies (`notebooks/03_explainability`,
   web interface), discussion and threats to validity (`docs/limitations.md`).
6. **Conclusion and perspectives** — what was achieved against the specification, limits, future
   work (temporal GNNs, multigraph GNNs, fairness audits, real-data validation).
7. **Annexes** — CLI reference, configuration reference, REST API, installation, class diagram,
   code listings, the Software Impacts manuscript (`paper/`).
