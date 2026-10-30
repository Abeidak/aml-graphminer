# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Optional API key**: `amlgm serve` with `$AMLGM_API_KEY` (or `--api-key`) answers 401 on the
  scoring and investigation routes unless the `X-API-Key` header carries the key (constant-time
  comparison); health check, documentation and web assets stay open. The web interface asks for
  the key once per browser tab and downloads its CSV exports through the same authenticated calls;
  `scripts/check_service.py --api-key` checks a protected service.
- `benchmarks/io_engines.py`: time and peak memory of `load_dataset` with pandas or Polars, each in
  a fresh process (`benchmarks/results_parts/io/`).

### Changed

- PyTorch Lightning training records every epoch (train loss, validation PR-AUC) in the history
  and the tracker, like the PyTorch loop, and reports the epoch of the best checkpoint.
- `amlgm train --mlflow` logs the GNN configuration as run parameters and the test metrics; runs
  are named after the model and the training loop (`train-sage-lightning`).

### Fixed

- The Polars reading engine was documented as lighter on memory; measured on HI-Small it is
  faster (12.8 s instead of 17.3 s for `load_dataset`) but peaks higher (4.2 GB instead of 2.6 GB).

## [0.1.1] - 2026-10-30

### Added

- Web interface: the dashboard opens on the **graph of the five strongest alerts** (accounts
  and transactions, click a node for its profile), and the *Alertes* tab shows the strongest
  alert with its sub-graph at once.
- README and `CITATION.cff`: the two authors and the two supervisors.

### Fixed

- PyPI showed the first author only: the names are now in `Author`, the addresses in
  `Maintainer-email`.
- Graphs made of several separate groups (several alerts) were drawn as tiny dots: every
  connected component is now laid out on its own and the components are packed in rows.

## [0.1.0] - 2026-10-29

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
- **Jupyter plugin** (`aml_graphminer.jupyter`, `jupyter` extra): scan results, alerts,
  explanations and benchmark reports display as HTML in notebooks (tables, interactive
  explanation graph, values from the data escaped, one-line plain-text forms);
  `%load_ext aml_graphminer` adds `%amlgm` (the command line inside the kernel), `%aml_scan` and
  `%aml_explain`; notebook `05_jupyter_plugin`; `scripts/py2nb.py` runs the magics when it
  executes a notebook.
- **Acceptance checks** of a running service: `scripts/check_service.py` (every REST route,
  uploads, exports, error codes) and `tests/e2e/ui_check.js` (the web interface driven in a
  headless browser).
- **Web interface in French throughout**: `api/static/fr.js` words in French what the API sends
  in English (motif descriptions and evidence, alert and explanation sentences, attribute
  names, payment formats, currencies, error messages, demo dataset name), numbers in the
  French format. The API now gives the figures behind its English sentences: `details` of an
  explanation (neighbours, measured neighbourhood effect, PGExplainer training set), `evidence`
  of the listed alerts, `origin` of a dataset. `tests/test_web_texts.py` runs `fr.js` with
  Node.js on real answers of the API and `ui_check.js` fails on any English word in the page.

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
- **Train/serve skew on minute-resolution data**: `score()` (CLI `score`, REST API, web
  interface, explanations) re-ordered transactions of the same timestamp by content, while the
  training features were computed in file order; on IBM AML (99.7 % of the rows move) the saved
  HistGradientBoosting model fell from a test PR-AUC of 0.484 to 0.317 when scored that way.
  Ties are now ordered by `source_row` everywhere (`validate_transactions` and the context
  stacking); scores still do not depend on the order rows are supplied, and the training
  features of IBM files are unchanged (the benchmark numbers stay valid).
- `PatternScanner.fit_calibration` counted the last transaction of the slice for alert
  transactions absent from it (index `-1`).
- `benchmarks/run_elliptic.py` printed a traceback when the data folder was missing; it now
  prints a one-line error. Obsolete comments (`import.cypher`, `.coveragerc-core`), README
  `--tx` example, `.gitignore` (`hub/`, `why.png`, `why.json`, `amlgm.yaml`); `/docs` needs
  Internet (Swagger UI from a CDN) and `serve` takes a pre-cut slice file: documented.

- Measured IBM AML HI-Small / LI-Small results of the CPU models (`benchmarks/results/`), EDA of
  both files (`results/eda-*`), motif thresholds tuned on the validation splits
  (`configs/patterns.yaml`).
- `make_report.py` merges several result folders (CPU runs + GPU runs of the same dataset).
- Every GNN explanation reports the measured neighbourhood effect (score with the local sub-graph
  vs. the transaction alone, `neighbourhood_effect`). On the IBM AML GraphSAGE model it is small,
  which is why PGExplainer returns all-zero masks: a faithful answer, documented in
  `docs/explainability.md`, not a bug.
- XGBoost baselines are saved in XGBoost's portable JSON format (`baseline.joblib.xgb.json`):
  its pickle failed to load across platforms ("input stream corrupted", Linux -> Windows).
- Notebook `04` runs in the browser on Kaggle or Colab: installs the package from an uploaded
  wheel, finds the IBM files of the Kaggle dataset, writes `results/<dataset>` for `make_report.py`.
- **Currency conversion in the motif detectors** (`currency` section): amounts are compared in a
  reference currency with exchange rates inferred from the cross-currency transactions
  (`exchange_rates`, least squares over the exchanged pairs); IBM laundering chains change
  currency between hops, which the amount checks could not follow.
- **Unlabelled history for scanned slices** (`lookback`, `PatternScanner.scan(df, history=...)`):
  the benchmark, `MotifDetector.fit` and `score` give the scanner the days before a slice, so a
  motif straddling its start is seen whole; only the slice is scored.
- `transaction_types` restricts the motif search to some payment formats (`[ACH]` in the tuned
  IBM configuration, chosen on validation).
- **Motif scanner for J2** (test PR-AUC HI-Small 0.399 → 0.503, LI-Small 0.145 → 0.189, every
  setting chosen on validation): `relay` detector (`A -> B -> C`, the account forwards 0.9–1.2x
  what it received in another currency: the IBM *stack* motif, opt-in), precision calibrated per
  band of alert score (`calibration_bins`), fan `min_span` / `min_mean_amount`, gather-scatter
  `scatter_lead`, scanner `exclude` (self-transfers, same bank, cross-currency) and `combine`
  (`max` / `noisy_or`), `PatternScanner.calibrate` on an existing scan; per-motif recall maps
  *stack* to `relay`.
- **Free cloud deployment of the API** (deliverable L3): `scripts/deploy_space.py` publishes the
  REST API + web interface demo as a Hugging Face Docker Space (port 7860, user 1000);
  `amlgm serve --max-upload-mb` caps uploads on a public demo.
- **Hybrid detector** (`HybridDetector`, `amlgm train --model hybrid --base DIR`, benchmark model
  `hybrid-<model>`): a saved model stacked with the motif scanner by a logistic regression fitted
  on the validation slice, with the test metrics of the model alone and of the motifs alone.
- **Batch explanation of alerts**: `explain_alerts` and `amlgm explain --top K --out-dir DIR`
  write a case file (one JSON + Markdown per alert, `index.csv`, `README.md`).
- Pydantic validation wired in: every REST API record goes through the `Transaction` model, and
  `load_dataset` spot-checks 1 000 rows (`check_sample`); errors name the transaction.
- Property-based tests (Hypothesis) of the scanner, the cycle detector, the exchange-rate solver
  and the graph converters; tests of `to_pyg_data`, `to_pyg_hetero` and the Polars reader (their
  `pragma: no cover` exclusions are removed); the CI executes notebooks `01`–`03` with nbval.
- GNNExplainer explanations listed no account attribute, although the specification asks for
  edges *and* features: the feature masks of the sender and the receiver are now reported.
- Web interface: an answer arriving after the user had switched to (or closed) another dataset
  could fill the tables with the previous dataset's rows, and a double click on *close* could
  close two datasets.
- CLI on Windows: an output redirected to a file or a pipe (legacy code page) failed with
  `UnicodeEncodeError` on the arrow of the explanations and wrote nothing.
- The *relay* motif had no description, the *smurfing* one ignored the *split* mode, and
  `/api/model` listed motifs the service does not look for (relay is opt-in).

### Known limitations

- The motif scanner reaches the 0.5 objective on HI-Small only narrowly (test PR-AUC 0.503, 95 %
  bootstrap interval 0.479–0.526; LI-Small 0.189), with settings that are partly properties of the
  IBM simulator (ACH only, no cross-currency payment) and a calibration that uses the validation
  labels; 36 % of the HI-Small laundering belongs to no motif; the unsupervised GAE is at chance
  level;
  on Elliptic the GNNs (best: GraphSAGE 0.497) stay well below the tree models (about 0.79).
- GNN training needs `pyg-lib` or `torch-sparse` in addition to `pip install aml-graphminer[gnn]`.
- The REST API has no authentication.
