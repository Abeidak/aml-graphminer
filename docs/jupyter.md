# Jupyter plugin

Notebooks are where investigators and researchers explore alerts. AML-GraphMiner plugs into
Jupyter in two ways: its results **display themselves** as tables and an interactive graph, and
an IPython extension adds **magics** that run the toolkit from a cell. Both work in Jupyter
Notebook, JupyterLab, VS Code and Google Colab; the notebook
[`05_jupyter_plugin`](https://github.com/Abeidak/aml-graphminer/blob/main/notebooks/05_jupyter_plugin.ipynb)
shows every feature on synthetic data.

## Rich display (nothing to load)

An object left at the end of a cell is shown as HTML:

| object | how you get it | what the notebook shows |
|---|---|---|
| `ScanResult` | `aml.PatternScanner().scan(df)` | number of alerts and flagged transactions, alerts per motif, the 10 strongest alerts (motif, score, size, amount, period, accounts) |
| `Alert` | `result.alerts[0]` | the alert with its evidence (hub account, time span, conservation ratio, ...) |
| `Explanation` | `explain_transaction(detector, df, tx)` | verdict (score, threshold), summary, **explanation graph**, the transactions and attributes that matter most, the motifs involved |
| `BenchmarkReport` | `run_benchmark(df, models=...)` | mean ± standard deviation of the test metrics per model |

In the explanation graph, accounts are circles (the two accounts of the explained transaction in
red, size ∝ importance) and transactions are arrows: red for the explained one, blue when they
raise the score, grey when they lower it or do not matter; the width follows the importance.
Hovering an arrow shows the transaction (id, amount, time, importance), hovering a circle shows
the account. Text follows the colours of the notebook theme (light or dark).

Values coming from the data are escaped: an account named `<script>` is displayed, never run.
The same HTML is available outside notebooks, for instance to attach an explanation to a case
file:

```python
from pathlib import Path

from aml_graphminer.jupyter import explanation_html

Path("why.html").write_text(explanation_html(explanation), encoding="utf-8")
```

## Magics

Load the extension once per notebook (IPython is always present in a notebook; elsewhere
`pip install "aml-graphminer[jupyter]"`):

```text
%load_ext aml_graphminer
```

### `%amlgm` — the command line inside the kernel

Every command of the command line ({doc}`cli`) with the same options, run by the interpreter of
the notebook: no PATH to configure, and the version of the package installed in the kernel is the
one used. IPython replaces `{variable}` by the value of a notebook variable first.

```text
%amlgm --help
%amlgm generate data --accounts 800
%amlgm train data/Synth-Small_Trans.csv --model hgb --out runs/hgb
%amlgm explain runs/hgb data/Synth-Small_Trans.csv --top 5 --out-dir case
%amlgm scan {path} --top 5
```

### `%aml_scan` — motif alerts of a DataFrame

```text
%aml_scan df
result = %aml_scan data/Synth-Small_Trans.csv --config amlgm.yaml
```

`DATA` is the name of a DataFrame of the notebook or a transaction file (any format accepted by
`amlgm`). `--config` takes a YAML configuration (`amlgm init-config`). The
`ScanResult` is returned, so it is displayed, and can be kept in a variable.

### `%aml_explain` — why was this transaction flagged?

```text
%aml_explain motifs df 1234
%aml_explain runs/hgb df 1234 --method occlusion --hops 2
exp = %aml_explain runs/sage df 1234 --method gnnexplainer
```

`MODEL` is a detector of the notebook (for example the result of `MotifDetector.fit`) or a model
folder written by `amlgm train --out`; `DATA` as for `%aml_scan`; `TX` the id of the transaction.
Options: `--method auto|patterns|occlusion|gnnexplainer|pgexplainer|captum|attention` (see
{doc}`explainability`), `--hops`, `--max-edges`. The `Explanation` is returned and displayed.

Errors (unknown file, transaction not found, a method that needs a GNN detector, ...) are shown
as one line, like in a terminal. `%aml_scan?` and `%aml_explain?` print the full help.
