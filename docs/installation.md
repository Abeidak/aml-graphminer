# Installation

AML-GraphMiner needs **Python 3.11 or newer**. The core installation has no deep-learning
dependency, so ingestion, features, structural motifs, tabular baselines, metrics and the CLI
work everywhere, including on a laptop without a GPU.

```bash
pip install aml-graphminer                 # core
pip install "aml-graphminer[viz]"          # + matplotlib (explanation and EDA figures)
pip install "aml-graphminer[gnn]"          # + PyTorch, PyTorch Geometric, Lightning
pip install "aml-graphminer[api]"          # + FastAPI / uvicorn (REST API + web interface)
pip install "aml-graphminer[all]"          # everything (+ Captum, Parquet, Polars, XGBoost, MLflow, Hub, Neo4j, Kaggle)
```

| extra | adds | needed for |
|---|---|---|
| `gnn` | `torch`, `torch_geometric`, `lightning` | GCN / SAGE / GAT and the graph auto-encoder, GNN explainers |
| `explain` | `captum` | the Captum (Integrated Gradients) explainer |
| `api` | `fastapi`, `uvicorn`, `python-multipart` | `amlgm serve`: REST API, web interface, file upload |
| `parquet` | `pyarrow` | Parquet input |
| `scale` | `polars`, `pyarrow` | faster CSV reading (`engine="polars"`: −26 % time, more memory; see {doc}`data`) |
| `xgboost` | `xgboost` | the `xgb` baseline |
| `tracking` | `mlflow` | `--mlflow` |
| `hub` | `huggingface_hub` | `amlgm export-hub --push` |
| `neo4j` | `neo4j` (official driver) | `amlgm export-neo4j --push` (the CSV export needs nothing) |
| `data` | `kaggle` | `amlgm download ibm\|elliptic` |
| `viz` | `matplotlib` | explanation plots, EDA figures |
| `dev`, `docs` | test / documentation tooling | contributing |

The toolkit was validated with Python 3.12, PyTorch 2.10 (CPU), PyTorch Geometric 2.8,
`pyg-lib` 0.7, Lightning 2.6, Captum 0.9, scikit-learn 1.9, pandas 3.0, FastAPI 0.14x and MLflow 3,
on Windows 11 and Linux.

Asking for a feature whose extra is missing raises an error that names the extra to install,
e.g. `pip install 'aml-graphminer[gnn]'`.

## PyTorch Geometric and the neighbour sampler

`pip install "aml-graphminer[gnn]"` installs PyTorch, `torch_geometric` and Lightning from PyPI.
Training on large graphs uses PyG's **neighbour sampler**, which additionally needs the compiled
`pyg-lib` (recommended) or `torch-sparse` package. These are *not* on PyPI: they are wheels
published per PyTorch/CUDA build.

```bash
python - <<'PY'
import torch
v = torch.__version__.split("+")[0]
cuda = "cu" + torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
print(f"pip install pyg-lib -f https://data.pyg.org/whl/torch-{v}+{cuda}.html")
PY
```

Run the command it prints. If the sampler is missing, AML-GraphMiner stops *before* training with
a message that repeats this instruction (instead of PyG's late `ImportError`). On Google Colab,
PyTorch is preinstalled and the notebook `04_gnn_training_colab` contains the cell. Everything
that does not train a GNN (motifs, tabular models, explanations of tabular models, the CLI)
works without any of this.

### Windows (CPU) in a virtual environment

`pyg-lib` publishes Windows wheels only for some PyTorch versions; pin a version that has one,
e.g. PyTorch 2.10 with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install pyg-lib -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
.\.venv\Scripts\python.exe -m pip install -e ".[all,dev,docs]"
```

Activate it (`.\.venv\Scripts\Activate.ps1`) before running `amlgm`, `pytest` or `jupyter`, and
select the same interpreter in your editor.

## From source

```bash
git clone https://github.com/Abeidak/aml-graphminer
cd aml-graphminer
pip install -e ".[dev,viz]"
pytest
```

## Docker

```bash
docker build -f docker/Dockerfile -t aml-graphminer .
docker run --rm aml-graphminer amlgm --version
docker run --rm -p 127.0.0.1:8000:8000 aml-graphminer      # web interface demo on http://127.0.0.1:8000/ui/
```

See {doc}`deployment` for the compose files (demo, trained model, Neo4j) and the GNN image.
