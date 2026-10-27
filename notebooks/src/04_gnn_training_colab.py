# %% [markdown]
# # GNN benchmark on the IBM AML data on a free GPU (Kaggle or Google Colab)
#
# The GNN part of the benchmark of the project specification (milestone J3): GCN, GraphSAGE and
# GAT edge classifiers and the label-free graph auto-encoder (GAE), three seeds, on IBM AML
# *HI-Small* and *LI-Small*, with **exactly the settings of `benchmarks/run_ibm.py`** (package
# defaults, temporal 60/20/20 split, best epoch chosen on validation PR-AUC, test never used for
# any choice). The CPU models (motifs, logistic regression, random forest, gradient boosting) are
# run on the laptop; this notebook only adds the GNN rows, which need a GPU.
#
# Everything runs **in the browser** — no VS Code connection is needed.
#
# **Kaggle (recommended: the IBM files are already hosted there, no upload)**
#
# 1. On the laptop, build the package: `python -m build` → `dist/aml_graphminer-0.1.0-py3-none-any.whl`.
# 2. kaggle.com → *Create* → *New Notebook*; *File* → *Import Notebook* → this `.ipynb`.
# 3. *Add Input* → search **ibm-transactions-for-anti-money-laundering-aml** (ealtman2019) → *Add*.
# 4. *Add Input* → *Upload* → *New Dataset*: drop the `.whl` file, name it `aml-graphminer-wheel`
#    (private).
# 5. Right panel → *Session options*: *Accelerator* **GPU T4 x2** (or P100) and *Internet* **On**
#    (both need a phone-verified Kaggle account).
# 6. Run all cells once with `QUICK = True` (a few minutes, checks the installation). Then set
#    `QUICK = False` and click *Save Version* → **Save & Run All (Commit)**: the run continues in
#    the background (browser can be closed); the results appear in the *Output* tab of the version
#    as `aml-gnn-results.zip`.
#
# **Google Colab (browser)** — colab.research.google.com → *File* → *Upload notebook*; *Runtime*
# → *Change runtime type* → **T4 GPU**. Put the `.whl` and the four IBM files (`HI-Small_Trans.csv`,
# `HI-Small_Patterns.txt`, `LI-Small_Trans.csv`, `LI-Small_Patterns.txt`) in Google Drive under
# `MyDrive/aml/`. Keep the tab open: free Colab sessions stop when idle, and *LI-Small* may not fit
# in the RAM of a free session — prefer Kaggle for the full run.
#
# **Back on the laptop** — unzip `aml-gnn-results.zip` (for example into `gpu-results/`) and merge
# the GNN rows into the CPU tables:
#
# ```
# python benchmarks/make_report.py benchmarks/results gpu-results/results
# ```

# %%
QUICK = True  # True: first 300 000 rows, 1 seed, 2 epochs (setup check); False: the full benchmark
DATASETS = ["HI-Small", "LI-Small"]
MODELS = ["gcn", "sage", "gat", "gae"]
SEEDS = [0] if QUICK else [0, 1, 2]
NROWS = 300_000 if QUICK else None
TRAIN_ONE_MODEL = True  # also save one GraphSAGE model (web interface, Hugging Face Hub)

# %% [markdown]
# ## Environment and installation
#
# The package is installed from the uploaded wheel (it is not on PyPI yet). PyTorch is the one of
# the GPU image; PyTorch Geometric comes with the `gnn` extra; the neighbour sampler needs the
# compiled `pyg-lib` wheel matching that PyTorch / CUDA build, published by PyG per
# `torch-<major>.<minor>.0`.

# %%
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
# the Kaggle image also ships the google.colab package: detect Colab by its runtime file
ON_COLAB = not ON_KAGGLE and Path("/var/colab/hostname").exists()
if ON_COLAB:
    from google.colab import drive

    drive.mount("/content/drive")
print("Kaggle:", ON_KAGGLE, "| Colab:", ON_COLAB, "| Python", platform.python_version())


def pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


wheels = sorted(
    {
        w
        for pattern in (
            "/kaggle/input/**/aml_graphminer-*.whl",
            "/content/drive/MyDrive/aml/aml_graphminer-*.whl",
            "/content/aml_graphminer-*.whl",
            "dist/aml_graphminer-*.whl",
        )
        for w in glob.glob(pattern, recursive=True)
    }
)
if not wheels:
    raise SystemExit(
        "aml_graphminer-*.whl not found: build it with `python -m build` and add it as an input "
        "(Kaggle) or put it in MyDrive/aml/ (Colab) - see the first cell"
    )
pip(f"{wheels[-1]}[gnn,viz]")
print("installed", Path(wheels[-1]).name)

# %%
import torch

torch_minor = ".".join(torch.__version__.split("+")[0].split(".")[:2]) + ".0"
cuda = "cu" + torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
pyg_wheels = f"https://data.pyg.org/whl/torch-{torch_minor}+{cuda}.html"
try:
    pip("pyg-lib", "-f", pyg_wheels)
except subprocess.CalledProcessError:
    print(
        f"no pyg-lib wheel for torch {torch_minor}+{cuda} at {pyg_wheels}: training will stop "
        "with an explicit message; pick an older GPU image (Kaggle: Session options > Environment)"
    )
print(
    "torch",
    torch.__version__,
    "| CUDA",
    torch.version.cuda,
    "| GPU:",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
)

# %%
import warnings

import pandas as pd

import aml_graphminer as aml
from aml_graphminer.config import load_config
from aml_graphminer.detectors.gnn import GNNConfig
from aml_graphminer.evaluation import run_benchmark
from aml_graphminer.ingestion import load_dataset

warnings.filterwarnings("ignore")
cfg = load_config(None)  # the package defaults: the settings used by benchmarks/run_ibm.py
gnn_config = cfg.gnn_config()
if QUICK:
    gnn_config = GNNConfig.from_dict({**gnn_config.to_dict(), "epochs": 2})
print(gnn_config)

OUT = (
    Path("/kaggle/working/aml")
    if ON_KAGGLE
    else Path("/content/drive/MyDrive/aml/gpu-output")
    if ON_COLAB
    else Path("gpu-output")
)
OUT.mkdir(parents=True, exist_ok=True)


def find(name: str) -> Path:
    """Locate ``<name>_Trans.csv`` (its ``*_Patterns.txt`` must sit next to it)."""
    roots = [os.environ.get("AML_DATA_DIR"), "/kaggle/input", "/content/drive/MyDrive/aml", "data"]
    for root in filter(None, roots):
        hits = sorted(glob.glob(f"{root}/**/{name}_Trans.csv", recursive=True))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(
        f"{name}_Trans.csv not found: add the Kaggle dataset or set AML_DATA_DIR"
    )


def environment() -> dict[str, str]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "aml_graphminer": aml.__version__,
        "torch": torch.__version__,
        "cuda": str(torch.cuda.is_available()),
        "runtime": "kaggle" if ON_KAGGLE else "colab" if ON_COLAB else "local",
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    import torch_geometric

    info["torch_geometric"] = torch_geometric.__version__
    return info


# %% [markdown]
# ## The benchmark (GNN rows of the report)
#
# `QUICK = True` writes into `quick/` (never merged into the report); `QUICK = False` writes
# `results/<dataset>/results.csv`, the files read by `benchmarks/make_report.py`.

# %%
for name in DATASETS:
    path = find(name)
    df = load_dataset(path, schema="ibm", nrows=NROWS)  # + the motif names of *_Patterns.txt
    print(f"\n=== {name}: {len(df):,} transactions, {int(df['label'].sum())} laundering ===")
    started = time.perf_counter()
    report = run_benchmark(
        df,
        models=MODELS,
        seeds=SEEDS,
        dataset=name,
        fractions=cfg.benchmark.fractions,
        feature_config=cfg.feature_config(),
        gnn_config=gnn_config,
    )
    target = report.save(OUT / ("quick" if QUICK else "results") / name)
    (target / "environment.json").write_text(
        json.dumps(
            {**environment(), "wall_seconds": round(time.perf_counter() - started, 1)}, indent=2
        ),
        encoding="utf-8",
    )
    print(report.to_markdown())
    del df

# %% [markdown]
# ## One trained model (web interface, explanation, Hugging Face Hub — deliverable L6)

# %%
if TRAIN_ONE_MODEL:
    from aml_graphminer.detectors.gnn import GNNDetector
    from aml_graphminer.explainers import explain_transaction, plot_explanation

    name = DATASETS[0]
    df = load_dataset(find(name), schema="ibm", nrows=NROWS)
    detector = GNNDetector.fit(
        df, gnn_config.with_kind("sage"), feature_config=cfg.feature_config(), seed=0
    )
    model_dir = detector.save(OUT / "runs" / f"sage-{name}")
    print(pd.Series(detector.metrics).round(4))
    masks = aml.temporal_split(df)
    scores = detector.score(df[masks["test"]], context=df[~masks["test"]])
    tx = int(scores.idxmax())
    explanation = explain_transaction(detector, df, tx, "gnnexplainer")
    print(explanation.to_markdown())
    plot_explanation(explanation)

# %% [markdown]
# The model folder (`runs/sage-HI-Small`) works with `amlgm serve`, `amlgm score` and
# `amlgm export-hub runs/sage-HI-Small hub/sage --repo-id <your-hf-user>/<name> --push`
# (Hugging Face write token required) once copied to the laptop.

# %% [markdown]
# ## Cross-domain validation on Elliptic Bitcoin (optional)
#
# Kaggle: *Add Input* → **elliptic-data-set** (ellipticco). Colab: the three CSV files in
# `MyDrive/aml/elliptic/`. Skipped when the files are absent.

# %%
from aml_graphminer.evaluation.elliptic import run_elliptic_benchmark
from aml_graphminer.ingestion import read_elliptic

hits = sorted(
    glob.glob("/kaggle/input/**/elliptic_txs_features.csv", recursive=True)
    + glob.glob("/content/drive/MyDrive/aml/elliptic/**/elliptic_txs_features.csv", recursive=True)
)
if hits:
    elliptic = read_elliptic(Path(hits[0]).parent)
    print(elliptic.summary())
    ell_config = GNNConfig(kind="gcn", hidden=64, epochs=20 if QUICK else 200, patience=20, lr=1e-2)
    ell_report = run_elliptic_benchmark(
        elliptic, ["rf", "hgb", "gcn", "sage", "gat"], seeds=SEEDS, gnn_config=ell_config
    )
    ell_target = ell_report.save(OUT / ("quick" if QUICK else "results") / "elliptic")
    (ell_target / "environment.json").write_text(json.dumps(environment(), indent=2))
    print(ell_report.to_markdown())
else:
    print("Elliptic files not found: section skipped.")

# %% [markdown]
# ## Download the results

# %%
archive = shutil.make_archive(str(OUT.parent / "aml-gnn-results"), "zip", OUT)
print("results archive:", archive)
if ON_COLAB:
    from google.colab import files

    files.download(archive)
