# AML-GraphMiner

**Explainable money-laundering detection on transaction graphs.**
AML-GraphMiner is an open-source Python toolkit that takes raw financial transactions and
returns *scored, explained* alerts. It combines

* **structural-motif detectors** — fan-in, fan-out, smurfing, cycles, scatter-gather and
  gather-scatter — and a **risk-community** detector (Louvain), that need no training and carry
  their own evidence,
* **tabular baselines** (logistic regression, random forest, gradient boosting, XGBoost),
  supervised **Graph Neural Networks** (GCN, GraphSAGE, GAT) and an **unsupervised graph
  auto-encoder**, trained on leak-free, time-aware features,
* **explanations** an investigator can audit (pattern evidence, occlusion, GNNExplainer,
  PGExplainer, Captum attributions, attention weights),
* a **web investigation interface**, a **command line** (`amlgm`), a **REST API**, a
  **Jupyter plugin**, **MLflow** tracking, **Hugging Face Hub** export and a **Neo4j** export,
  all on top of the same Python SDK.

```bash
pip install "aml-graphminer[viz,api]"
amlgm serve --demo                             # web interface on a synthetic demo: http://127.0.0.1:8000/ui/
amlgm generate data/ --accounts 1500           # synthetic IBM-AML-like data (no download needed)
amlgm scan data/Synth-Small_Trans.csv --out alerts/
amlgm train data/Synth-Small_Trans.csv --model hgb --out runs/hgb
amlgm explain runs/hgb data/Synth-Small_Trans.csv --tx 1234 --plot why.png    # any tx_id, e.g. from alerts/alerts.csv
```

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
concepts
```

```{toctree}
:maxdepth: 2
:caption: User guide

data
features
patterns
models
explainability
evaluation
```

```{toctree}
:maxdepth: 2
:caption: Interfaces and operations

web_interface
cli
jupyter
rest_api
configuration
deployment
```

```{toctree}
:maxdepth: 2
:caption: Reference

architecture
extending
limitations
reference
changelog
```
