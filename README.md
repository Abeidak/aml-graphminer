# AML-GraphMiner

**Explainable money-laundering detection on transaction graphs** — structural motifs, risk
communities, tabular baselines, supervised Graph Neural Networks (GCN, GraphSAGE, GAT) and an
unsupervised graph auto-encoder behind one Python SDK, a web investigation interface, a command
line and a REST API.

[![CI](https://github.com/Abeidak/aml-graphminer/actions/workflows/ci.yml/badge.svg)](https://github.com/Abeidak/aml-graphminer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> Research software developed at the Al-Khawarizmi research unit, Faculty of Sciences and
> Techniques, University of Nouakchott. **Not a compliance product** — see
> [Limitations](docs/limitations.md).

## Why

Anti-money-laundering systems flag very few transactions correctly (≈ 0.1 % of transactions are
illicit), and an alert nobody can explain cannot be acted on. Published GNN experiments are often
hard to reproduce: random splits that leak the future, features fitted on test data, no
confidence intervals, no explanation of the alerts. AML-GraphMiner packages the whole pipeline
with the **evaluation protocol built in**:

* a **temporal** train / validation / test split and **leak-free** features (fitted on train, causal
  windows, graph snapshots per split);
* six **structural-motif detectors** (fan-in, fan-out, smurfing, cycles, scatter-gather,
  gather-scatter) and a **risk-community** detector (relay accounts, Louvain), whose alerts carry
  their own evidence, plus **per-motif recall** on the IBM data;
* **tabular baselines** (logistic regression, random forest, HistGradientBoosting, XGBoost),
  supervised **GCN / GraphSAGE / GAT** edge classifiers and an **unsupervised graph auto-encoder**
  (no label needed) on the same features and split, with reproducible scores;
* **explanations** for every alert: exact pattern evidence, model-agnostic occlusion,
  GNNExplainer, PGExplainer, Captum attributions and attention weights;
* a **web investigation interface** (dashboard, alerts with their sub-graph, explained
  transactions, account profiles), a **REST API** (OpenAPI 3.1), the `amlgm` **command line**,
  **MLflow** tracking, **Hugging Face Hub** export and a **Neo4j** export;
* a `benchmark` command that compares everything over several seeds with standard deviations
  and confidence intervals, and an **EDA** report of any dataset.

## Install

```bash
pip install "aml-graphminer[viz,api]"    # core + figures + REST API / web interface (no PyTorch needed)
pip install "aml-graphminer[gnn]"        # + PyTorch, PyG, Lightning (see docs for the pyg-lib wheel)
pip install "aml-graphminer[all]"        # everything (Captum, XGBoost, MLflow, Hub, Neo4j, Kaggle ...)
```

Python 3.11 or newer. From source: `pip install -e ".[dev,viz,api]"` — step-by-step instructions
for Windows (virtual environment, PyTorch CPU and `pyg-lib`) are in
[docs/installation.md](docs/installation.md); a French launch-and-test guide is in
[GUIDE_FR.md](GUIDE_FR.md).

## Try it in two minutes (synthetic data, nothing to download)

```bash
amlgm serve --demo --open-browser                         # web interface: http://127.0.0.1:8000/ui/
```

```bash
amlgm generate data/ --accounts 1500                       # IBM-AML-like table + patterns file
amlgm eda    data/Synth-Small_Trans.csv --out eda/         # exploratory data analysis report
amlgm scan   data/Synth-Small_Trans.csv --out alerts/      # motif and community alerts with evidence
amlgm train  data/Synth-Small_Trans.csv --model hgb --out runs/hgb
amlgm train  data/Synth-Small_Trans.csv --model gae --out runs/gae --epochs 3   # unsupervised (gnn extra)
amlgm explain runs/hgb data/Synth-Small_Trans.csv --tx 1234 --plot why.png
amlgm serve  runs/hgb --data data/Synth-Small_Trans.csv    # REST API + web interface on that file
```

```python
import aml_graphminer as aml
from aml_graphminer.detectors.bundle import TabularDetector

df = aml.generate_transactions(n_accounts=900, n_background=7000, illicit_ratio=0.02)
alerts = aml.PatternScanner().scan(df).to_frame()  # structural motifs, no training
detector = TabularDetector.fit(df, "hgb")  # temporal split, leak-free features
print(detector.metrics)  # test metrics, validation threshold
```

Docker, nothing else installed: `docker compose -f docker/docker-compose.demo.yml up --build`,
then open <http://127.0.0.1:8000/ui/>.

Real data: `amlgm download ibm --out data` (Kaggle API token) or download the IBM *Transactions
for Anti-Money Laundering* files from
[Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
yourself, then `load_dataset("data/HI-Small_Trans.csv")` (attaches the `*_Patterns.txt` ground
truth). The toolkit never downloads data unless asked.

## Notebooks

| notebook | content | state |
|---|---|---|
| [`01_quickstart`](notebooks/01_quickstart.ipynb) | the whole toolkit on synthetic data: motifs, communities, tabular model, explanations, unsupervised GAE | executed |
| [`02_ibm_aml_benchmark`](notebooks/02_ibm_aml_benchmark.ipynb) | EDA + motifs vs. tabular vs. GNNs vs. GAE, seeds, CI, per-motif recall (set `AML_DATA` for IBM) | executed on synthetic data |
| [`03_explainability`](notebooks/03_explainability.ipynb) | pattern evidence, occlusion, GNNExplainer, PGExplainer, Captum, attention | executed |
| [`04_gnn_training_colab`](notebooks/04_gnn_training_colab.ipynb) | GNN rows of the IBM AML benchmark (GCN/SAGE/GAT/GAE, 3 seeds) on a free GPU in the browser (Kaggle or Colab), Elliptic, model export | not executed in the repository (needs GPU + data) |

The original Colab notebooks this project grew from are kept in `notebooks/colab_legacy/`.

## Benchmarks

`benchmarks/run_ibm.py`, `benchmarks/run_elliptic.py` and `benchmarks/make_report.py` reproduce the
comparison of the project specification and write `REPORT.md` plus a LaTeX table
([`benchmarks/results/REPORT.md`](benchmarks/results/REPORT.md)). Measured on the IBM AML test
splits (temporal 60/20/20, mean ± std over 3 seeds; tabular models and motifs on a laptop CPU,
GNNs on a Kaggle T4 GPU with notebook `04`; motif thresholds of
[`configs/patterns.yaml`](configs/patterns.yaml) tuned on validation only):

| PR-AUC (test) | HI-Small (prevalence 0.177 %) | LI-Small (prevalence 0.067 %) |
|---|---|---|
| GraphSAGE | **0.531 ± 0.022** | 0.176 ± 0.018 |
| random forest | 0.499 ± 0.017 | **0.191 ± 0.007** |
| HistGradientBoosting | 0.476 ± 0.011 | 0.106 ± 0.016 |
| XGBoost | 0.474 ± 0.005 | 0.078 ± 0.007 |
| GAT | 0.447 ± 0.007 | 0.110 ± 0.026 |
| GCN | 0.374 ± 0.031 | 0.074 ± 0.024 |
| logistic regression | 0.294 | 0.032 |
| structural motifs | 0.005 | 0.001 |
| graph auto-encoder (no labels) | 0.003 ± 0.001 | 0.001 ± 0.000 |

GPU results are merged with `python benchmarks/make_report.py benchmarks/results <part folders>`
(see [`benchmarks/README.md`](benchmarks/README.md)).

Elliptic Bitcoin (node classification, time steps 35–49 as test, prevalence 6.5 %, 3 seeds, Kaggle
T4): HistGradientBoosting 0.790 ± 0.004, XGBoost 0.788 ± 0.002, random forest 0.786 ± 0.001,
GraphSAGE 0.497 ± 0.056, GAT 0.351 ± 0.042, GCN 0.300 ± 0.055, logistic regression 0.209 (test
PR-AUC).

Trained models of every kind for both IBM files (seed 0) are produced by one private Kaggle
kernel per model; the GNN and motif detectors load on any PC, while the scikit-learn / XGBoost
ones must be re-trained (or loaded) with the same library versions (`amlgm train ... --out runs/...`).

## Documentation

`docs/` (Sphinx): installation, concepts, data (download, EDA), features, patterns and risk
communities, models (supervised and unsupervised), explainability, evaluation, web interface, CLI,
REST API, configuration, deployment (Docker, Neo4j, Hub), architecture, extending, limitations and
the API reference. Build with `pip install -e ".[docs]" && sphinx-build -n docs docs/_build/html`.

## Project layout

```text
src/aml_graphminer/   ingestion · features · graph · detectors · explainers · evaluation · api (+ web UI) · cli
tests/ notebooks/ benchmarks/ docs/ docker/ configs/ paper/ scripts/
```

## Quality

The test suite (pytest + hypothesis, 440+ tests) covers 98 % of the code with every extra
installed and 97 % of the torch-free core; ruff, mypy, the Sphinx build (nitpicky), the notebooks,
the wheel and the Docker image are checked by GitHub Actions (`.github/workflows/ci.yml`).

## Contributing, citation, licence

See [`CONTRIBUTING.md`](CONTRIBUTING.md). If you use the software, cite it with
[`CITATION.cff`](CITATION.cff) (and the paper once published). MIT licence — see
[`LICENSE`](LICENSE). The IBM AML and Elliptic datasets have their own licences; check them
before redistributing anything derived from them.

---

## Résumé (français)

**AML-GraphMiner** est une boîte à outils Python open source pour la détection explicable du
blanchiment d'argent sur des graphes de transactions : détecteurs de motifs structurels
(fan-in, fan-out, smurfing, cycles, scatter-gather, gather-scatter) et de communautés à risque,
modèles tabulaires de référence, réseaux de neurones sur graphes supervisés (GCN, GraphSAGE, GAT)
et auto-encodeur de graphe non supervisé, avec un protocole d'évaluation temporel sans fuite de
données, des explications pour chaque alerte (preuves des motifs, occlusion, GNNExplainer,
PGExplainer, Captum, attention), une **interface web d'investigation**, une ligne de commande
(`amlgm`), une API REST et des exports MLflow, Hugging Face Hub et Neo4j. Le projet est développé
à l'Unité de recherche Al-Khawarizmi (Université de Nouakchott). Guide de lancement et de test en
français : [GUIDE_FR.md](GUIDE_FR.md).
