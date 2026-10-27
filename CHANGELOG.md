# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First public version, built from the two Colab notebooks of the project (GCN / GraphSAGE / GAT on
IBM AML *LI-Small* and *HI-Small*).

### Added

- **Ingestion**: adapters for the IBM AML files (position-based mapping of the duplicated
  `Account` column), canonical CSV/JSON/Parquet/DataFrame/records, explicit column mappings;
  validation with readable errors; parser of `*_Patterns.txt` giving the `pattern` column;
  Elliptic Bitcoin reader; deterministic synthetic generators (IBM-like and Elliptic-like).
- **Features**: causal trailing-window statistics, account-level rates, PageRank, optional
  betweenness; `FeaturePipeline` fitted on train, serialised to JSON, applied per split snapshot.
- **Graphs**: NetworkX builders, array / PyG converters, heterogeneous account–transaction graph.
- **Structural-motif detectors**: fan-in, fan-out, smurfing (structuring and split), time-respecting
  cycles, scatter-gather, gather-scatter; scanner with registry, YAML configuration, per-motif
  precision calibration and evaluation helpers.
- **Models**: tabular baselines (logistic regression, random forest, HistGradientBoosting,
  XGBoost); edge-classification GCN / GraphSAGE / GAT with neighbour sampling, weighted BCE or
  focal loss, validation-based model selection, PyTorch or Lightning training loops, checkpoints
  carrying the feature statistics; node-classification GNNs for Elliptic.
- **Explainers**: pattern evidence, model-agnostic occlusion (edges, accounts, raw attributes),
  GNNExplainer adapter, GAT attention, JSON / NetworkX / Markdown / matplotlib views.
- **Evaluation**: temporal split, PR-AUC / F1 / precision / recall / recall@k metrics, bootstrap
  confidence intervals, multi-seed benchmark runner, Elliptic benchmark.
- **Interfaces**: `amlgm` CLI (`generate, inspect, init-config, scan, train, score, explain,
  benchmark, elliptic, serve, export-hub`), REST API (`/health, /score, /explain, /scan`),
  MLflow tracker, Hugging Face Hub export with model card, YAML configuration.
- **Project**: tests (pytest, hypothesis), notebooks `01`–`04`, benchmark scripts and report
  generator, Sphinx documentation, Dockerfile and compose file, GitHub Actions (CI, release),
  Software Impacts manuscript draft.
- **Risk communities**: `CommunityDetector` (`community` section of the pattern configuration) —
  groups organised around relay accounts, connected components split with Louvain, scored on
  internal share, relays, speed and cycles; enabled in the scanner by default.
- **Unsupervised GNN**: `AutoEncoderDetector` (`amlgm train --model gae`, benchmark model `gae`) —
  graph auto-encoder with structure and attribute decoders, trimmed (contamination-robust) loss,
  label-free model selection, percentile scores; trains on unlabelled data.
- **Explainers**: `PGExplainerAdapter` (PyG PGExplainer on the transaction's neighbourhood),
  `CaptumExplainerAdapter` (Integrated Gradients on neighbours, account and transaction
  features), one dispatcher `explain_transaction` / `available_methods` used by every interface.
- **Web investigation interface** served by `amlgm serve` at `/ui/` (dashboard, alerts with their
  sub-graph, explained transactions, account profiles, CSV exports) and its JSON API under
  `/api/...` (`InvestigationService`); `amlgm serve --demo` needs no preparation; file upload.
- **EDA**: `aml_graphminer.eda` (`eda_summary`, `eda_figures`, `eda_report`) and `amlgm eda`.
- **Neo4j export**: `aml_graphminer.graph.neo4j` (CSV + `import.cypher` + investigation
  `queries.cypher`, optional push with the official driver) and `amlgm export-neo4j`.
- **Datasets**: `amlgm download ibm|elliptic` (Kaggle API, `data` extra).
- Ingestion modules named after the specification: `csv_adapter`, `ibm_adapter`,
  `elliptic_adapter`; notebook `02` renamed `02_ibm_aml_benchmark`.
- `GNNConfig.inference` (`sampled` | `full`); `GNNDetector.describe()` and `.metrics`.
- Docker: the image serves the web demo by default, `--build-arg WITH_GNN=1` adds PyTorch/PyG,
  `docker-compose.demo.yml` and `docker-compose.neo4j.yml`.

### Fixed

- GNN scores were not reproducible: neighbour sampling at inference drew from the global random
  stream, so the same transaction got a different score at each call. Sampling is now seeded and
  isolated (`fork_rng`), and same-timestamp transactions are ordered canonically, so scores no
  longer depend on calls or row order.
- `GNNDetector` had no `describe()`: `amlgm train --model gcn|sage|gat` crashed after saving and
  `/health` failed for GNN models.
- MLflow 3 refuses the default `./mlruns` directory store: tracking now defaults to
  `sqlite:///mlflow.db` (and still accepts directory stores explicitly).
- Notebooks `02` / `04` read the IBM files without their `*_Patterns.txt`, leaving the per-motif
  recall empty on real data; they now use `load_dataset`.
- The CI core job referenced a missing `.coveragerc-core`; mypy errors with recent pandas stubs;
  non-writable NumPy arrays passed to PyTorch; deprecated licence metadata; Sphinx warnings.

- Measured IBM AML HI-Small / LI-Small results of the CPU models (`benchmarks/results/`), EDA of
  both files (`results/eda-*`), motif thresholds tuned on the validation splits
  (`configs/patterns.yaml`).
- `make_report.py` merges several result folders (CPU runs + GPU runs of the same dataset).
- XGBoost baselines are saved in XGBoost's portable JSON format (`baseline.joblib.xgb.json`):
  its pickle failed to load across platforms ("input stream corrupted", Linux -> Windows).
- Notebook `04` runs in the browser on Kaggle or Colab: installs the package from an uploaded
  wheel, finds the IBM files of the Kaggle dataset, writes `results/<dataset>` for `make_report.py`.

### Known limitations

- Even tuned, the motif scanner stays far below the learned models on IBM AML (test PR-AUC 0.005
  on HI-Small; GraphSAGE 0.531, random forest 0.499); the unsupervised GAE is at chance level;
  on Elliptic the GNNs (best: GraphSAGE 0.497) stay well below the tree models (about 0.79).
- GNN training needs `pyg-lib` or `torch-sparse` in addition to `pip install aml-graphminer[gnn]`.
- The REST API has no authentication.
