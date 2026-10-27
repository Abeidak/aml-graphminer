# Architecture

AML-GraphMiner is organised in six layers. Each layer depends only on the ones below it, the
heavy dependencies (PyTorch, PyG, FastAPI, MLflow) are imported lazily, and every interface
(CLI, REST API, notebooks) is a thin wrapper around the same Python SDK.

```text
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 6  Interfaces     cli.py (amlgm) · api/ (ScoringService, InvestigationService, │
 │                   FastAPI, web UI in api/static) · hub.py · notebooks/         │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ 5  Explainers     patterns · occlusion · GNNExplainer · PGExplainer · Captum   │
 │                   attention · subgraph extraction · dispatch · plot            │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ 4  Detectors      patterns/ (6 motif detectors + risk communities + scanner)   │
 │                   baselines.py (rf/hgb/xgb/logreg)                             │
 │                   gnn/ (GCN/SAGE/GAT supervised + graph auto-encoder)          │
 │                   bundle.py (MotifDetector, TabularDetector, load_detector)    │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ 3  Graph          networkx_builder (+ Louvain) · pyg_builder · convert         │
 │                   (nx ⇄ arrays ⇄ PyG) · hetero (typed graphs) · neo4j export   │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ 2  Features       temporal (causal windows) · network (rates, entropy,         │
 │                   PageRank, betweenness) · pipeline (fit/transform, JSON stats)│
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ 1  Ingestion      csv_adapter · ibm_adapter · elliptic_adapter · schema        │
 │                   (Pydantic) · readers (CSV/JSON/Parquet) · synthetic          │
 └──────────────────────────────────────────────────────────────────────────────┘
   cross-cutting: config.py · tracking.py (MLflow) · eda.py · datasets.py (Kaggle) ·
                  evaluation/ (temporal split, metrics, benchmark, Elliptic)
```

## Data flow

```text
raw file ──adapter──▶ canonical DataFrame ──temporal_split──▶ train / val / test masks
                              │
                              ├──▶ PatternScanner ──▶ alerts (accounts, tx, evidence)
                              │
                              └──▶ FeaturePipeline.fit(train) ──▶ FeatureSet per split
                                         │                              │
                                         │                              ├──▶ TabularBaseline
                                         │                              └──▶ EdgeGNN (neighbour sampling)
                                         ▼
                              stats saved in config.json ──▶ identical transform at inference
```

## Contracts between layers

* **Table contract** — everything downstream of ingestion receives the canonical table described
  in {doc}`concepts`: sorted by time, integer `tx_id` index, `label` possibly `NaN`.
* **Detector contract** — `score(df, context=None) -> Series` indexed like `df`, `predict`,
  `save(dir)`, `describe()`, a `threshold`; `load_detector(dir)` dispatches on the `model_type`
  stored in `config.json` (`patterns`, `tabular`, `gnn`, `gnn_unsupervised`). Scores are
  reproducible and do not depend on the order of the rows.
* **Explanation contract** — a *score function* `fn(target_df, context_df) -> ndarray` is all an
  explainer needs from a detector; `detector_score_fn` adapts any detector.
* **Persistence contract** — a model is a directory: `config.json` (+ weights). It carries the
  feature statistics so that inference needs no training data.

## Dependency policy

Core installs need only NumPy, pandas, SciPy, scikit-learn, NetworkX, Pydantic, PyYAML, click and
joblib. Everything else is an *extra* (`gnn`, `explain`, `api`, `parquet`, `scale`, `xgboost`,
`tracking`, `hub`, `neo4j`, `data`, `viz`). A missing extra raises `OptionalDependencyError` naming
the extra to install.

## Repository layout

```text
src/aml_graphminer/   the library (typed, py.typed)
tests/                pytest suite (hypothesis property tests, torch tests skipped without torch)
notebooks/            01–04 notebooks; src/ holds the .py sources; colab_legacy/ the original notebooks
benchmarks/           IBM / Elliptic runners and the report generator
configs/              default.yaml, patterns.yaml
docs/                 this documentation (Sphinx + MyST)
docker/               Dockerfile, docker-compose.yml (model), .demo.yml (web demo), .neo4j.yml
paper/                Software Impacts manuscript (LaTeX)
scripts/              py2nb.py (notebook builder)
```

## Mapping to the project specification (CDC §5.3)

| CDC path | implementation |
|---|---|
| `ingestion/csv_adapter.py`, `ibm_adapter.py`, `elliptic_adapter.py` | same names (public entry points); code in `adapters.py`, `readers.py`, `schema.py`, `ibm_patterns.py`, `dataset.py`, `elliptic.py` |
| `features/` | `temporal.py`, `network.py`, `pipeline.py` |
| `graph/networkx_builder.py`, `graph/pyg_builder.py` | same names, plus `convert.py`, `hetero.py`, `neo4j.py` |
| `detectors/patterns/` | `fan.py`, `smurfing.py`, `cycles.py`, `layering.py`, `community.py`, `scanner.py` |
| `detectors/gnn/` | `models.py` (GCN/SAGE/GAT), `trainer.py`, `lightning_trainer.py`, `unsupervised.py` (auto-encoder) |
| `explainers/` | `patterns.py`, `occlusion.py`, `gnn_explainers.py`, `subgraph.py`, `dispatch.py` |
| `api/`, `cli.py` | `api/app.py`, `api/service.py`, `api/investigation.py`, `api/static/` (web UI), `cli.py` |
| `notebooks/01_quickstart`, `02_ibm_aml_benchmark`, `03_explainability` | same names (+ `04_gnn_training_colab`) |
