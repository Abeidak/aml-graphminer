# %% [markdown]
# # The Jupyter plugin
#
# AML-GraphMiner plugs into notebooks in two ways:
#
# * its results **display themselves**: a scan shows its alerts per motif, an alert its
#   evidence, an explanation its verdict and an interactive graph (hover the arrows and the
#   circles), a benchmark its table of metrics;
# * the extension `%load_ext aml_graphminer` adds three **magics**: `%amlgm` runs any command of
#   the command line inside the kernel, `%aml_scan` scans a DataFrame of the notebook and
#   `%aml_explain` explains a transaction.
#
# Everything below runs on synthetic data in about a minute, without any download. The files it
# writes go to `data/` and `runs/` next to the notebook.

# %%
# %pip install -q "aml-graphminer[viz]"

# %% [markdown]
# ## 1. The command line inside the notebook
#
# `%amlgm` takes exactly the commands and options of the `amlgm` command line. It runs in the
# interpreter of the kernel, so there is no PATH to set up and the version of the package is the
# one installed in the kernel. Here it writes a synthetic IBM-like dataset and scans it.

# %%
# %load_ext aml_graphminer
# %amlgm generate data --accounts 600 --background 5000 --seed 5

# %%
# %amlgm scan data/Synth-Small_Trans.csv --top 5

# %% [markdown]
# ## 2. Results that display themselves
#
# `%aml_scan` scans a DataFrame of the notebook (or a file) and returns the `ScanResult`, which is
# shown as a summary per motif and a table of the strongest alerts. The same display appears for
# `aml.PatternScanner().scan(df)` left at the end of a cell: no extension is needed for it.

# %%
import aml_graphminer as aml
from aml_graphminer.detectors.bundle import MotifDetector
from aml_graphminer.ingestion import load_dataset

df = load_dataset("data/Synth-Small_Trans.csv")
print(f"{len(df):,} transactions, {int(df['label'].sum())} of them laundering")

# %%
# %aml_scan df

# %% [markdown]
# Every alert carries its evidence — the facts a compliance officer writes in a suspicious
# activity report:

# %%
result = aml.PatternScanner().scan(df)
max(result.alerts, key=result.weight)

# %% [markdown]
# ## 3. Explaining a transaction
#
# `%aml_explain MODEL DATA TX` takes a detector of the notebook or a model folder written by
# `amlgm train`. First the motif detector, calibrated on the validation period: its explanation
# is exact (the alerts that contain the transaction). In the graph the explained transaction is
# red; hover an arrow for the transaction, a circle for the account.

# %%
motifs = MotifDetector.fit(df, fractions=(0.6, 0.2, 0.2))
tx = int(motifs.score(df).idxmax())
print("transaction with the highest score:", tx)

# %%
# %aml_explain motifs df {tx}

# %% [markdown]
# Now a detector trained by the command line and read back from its folder, on the strongest
# **cycle** of the scan. The ring of accounts is plain in the graph, yet the verdict can be *no
# alert*: the calibration measured on the validation period how often each motif was really
# laundering, and cycles were rarely so in these data. An explanation always gives both the
# evidence and the verdict.

# %%
# %amlgm train data/Synth-Small_Trans.csv --model patterns --out runs/motifs
cycle = max((a for a in result.alerts if a.pattern == "cycle"), key=result.weight)
cycle_tx = cycle.tx_ids[0]
print(f"cycle of {len(cycle.accounts)} accounts, raw alert score {cycle.score:.3f}")
print(
    f"precision of cycle alerts on the validation period: {motifs.scanner.calibration['cycle']:.2f}"
)

# %%
# %aml_explain runs/motifs df {cycle_tx}

# %% [markdown]
# ## 4. Benchmarks
#
# A `BenchmarkReport` shows the mean (and, with several seeds, the standard deviation) of the
# test metrics of every model. On a sample this small (a handful of laundering transactions in
# the test period) the numbers only illustrate the display: notebook 02 runs the real benchmark.

# %%
from aml_graphminer.evaluation.benchmark import run_benchmark

run_benchmark(df, models=("patterns", "hgb"), seeds=(0,))

# %% [markdown]
# ## 5. The same views outside a notebook
#
# The HTML comes from `aml_graphminer.jupyter`: it can be saved, for instance to attach an
# explanation to a case file.

# %%
from pathlib import Path

from aml_graphminer.explainers import explain_transaction
from aml_graphminer.jupyter import explanation_html

page = Path("runs/why.html")
page.write_text(explanation_html(explain_transaction(motifs, df, tx)), encoding="utf-8")
print(f"{page} written ({page.stat().st_size:,} bytes)")
