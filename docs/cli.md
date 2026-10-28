# Command line

The `amlgm` command wraps the SDK. `-v` / `-vv` raise the log level; every command that reads
transactions accepts `--schema [auto|ibm|canonical]`, `--nrows N` and `--patterns-file FILE`.

| command | purpose | main options |
|---|---|---|
| `amlgm generate OUT_DIR` | write a synthetic IBM-like dataset | `--accounts`, `--background`, `--days`, `--illicit-ratio`, `--seed` |
| `amlgm download ibm\|elliptic` | fetch the reference datasets from Kaggle (`data` extra + Kaggle token) | `--out`, `--size small\|medium\|large`, `--group HI\|LI`, `--force` |
| `amlgm inspect DATA` | validate a dataset and print its summary | |
| `amlgm eda DATA --out DIR` | exploratory data analysis: `EDA.md`, `summary.json`, figures | `--no-figures` |
| `amlgm init-config [PATH]` | write the default YAML configuration | `--force` |
| `amlgm scan DATA` | structural-motif and risk-community alerts (`alerts.csv`, `alerts.json`) | `--config`, `--calibrate`, `--out`, `--top` |
| `amlgm train DATA --out DIR` | train and save a detector | `--model patterns\|rf\|hgb\|xgb\|logreg\|gcn\|sage\|gat\|gae\|hybrid`, `--base MODEL_DIR` (hybrid), `--epochs`, `--trainer torch\|lightning`, `--mlflow`, `--seed` |
| `amlgm score MODEL_DIR DATA` | score transactions with a saved detector | `--context FILE`, `--out scores.csv`, `--top` |
| `amlgm explain MODEL_DIR DATA --tx ID` | explain one transaction | `--method auto\|patterns\|occlusion\|gnnexplainer\|pgexplainer\|captum\|attention`, `--hops`, `--max-edges`, `--out`, `--plot` |
| `amlgm explain MODEL_DIR DATA --top K --out-dir DIR` | explain the K highest-scored alerts in one go (case file: one JSON + Markdown per alert, `index.csv`, `README.md`) | same `--method` / `--hops` / `--max-edges` |
| `amlgm benchmark DATA` | motifs vs. tabular vs. GNNs vs. auto-encoder on a temporal split | `--models` (repeat), `--seeds` (repeat), `--out`, `--dataset-name`, `--mlflow` |
| `amlgm elliptic DIR` | Elliptic node-classification benchmark | `--models`, `--seeds`, `--out` |
| `amlgm serve [MODEL_DIR]` | REST API + web investigation interface (`/ui/`) | `--demo`, `--data FILE` (repeat), `--host`, `--port`, `--open-browser` |
| `amlgm export-neo4j DATA OUT_DIR` | CSV + Cypher scripts for Neo4j, optionally loaded into a database | `--model`, `--scan/--no-scan`, `--push`, `--uri`, `--user`, `--password`, `--database` |
| `amlgm export-hub MODEL_DIR OUT_DIR` | model repository with a generated model card | `--repo-id`, `--push`, `--token` |

`--model gae` trains the **unsupervised** graph auto-encoder: it accepts unlabelled files (labels,
when present, only choose the alert threshold). Every other learned model needs fully labelled
data.

Errors raised on purpose by the library (bad configuration, missing optional dependency, invalid
data) are printed as one line (`error: ...`) on stderr and make the command exit with status 2; usage
errors reported by the option parser keep click's own exit code. `-vv` raises the log level
(the library logs what it is doing: files read, patterns attached, epochs ...).

## Example session

```bash
amlgm download ibm --out data --size small            # HI-Small / LI-Small + *_Patterns.txt (Kaggle token)
amlgm eda data/HI-Small_Trans.csv --out eda/hi-small
amlgm init-config amlgm.yaml                          # edit thresholds, windows, GNN hyper-parameters
amlgm scan data/HI-Small_Trans.csv --config amlgm.yaml --calibrate --out alerts/
amlgm train data/HI-Small_Trans.csv --model sage --config amlgm.yaml --out runs/sage --epochs 20
amlgm train data/HI-Small_Trans.csv --model gae --out runs/gae --epochs 3
amlgm benchmark data/HI-Small_Trans.csv --config amlgm.yaml --out results/ --models patterns --models rf --models sage --models gae --seeds 0 --seeds 1 --seeds 2
amlgm explain runs/sage data/HI-Small_Trans.csv --tx 123456 --method captum --plot why.png
amlgm export-hub runs/sage hub/aml-sage --repo-id my-org/aml-sage
amlgm export-neo4j data/HI-Small_Trans.csv neo4j-import --model runs/sage
amlgm serve runs/sage --open-browser                  # http://127.0.0.1:8000/ui/
```

`amlgm serve --demo` needs nothing at all: it trains a small demo detector on synthetic data
(once, cached in the temporary directory), analyses a synthetic dataset and opens the web
interface.
