# Data

## IBM Transactions for Anti-Money Laundering

The synthetic IBM datasets (Altman et al., NeurIPS 2023) come as `HI|LI-Small|Medium|Large`:
`*_Trans.csv` with 11 columns and, for the labelled files, `*_Patterns.txt` listing the
injected laundering attempts. They are on Kaggle (account required); the toolkit downloads them
only when asked, with your own Kaggle API token (`data` extra):

```bash
amlgm download ibm --out data --size small --group HI --group LI   # HI/LI-Small + *_Patterns.txt
amlgm download elliptic --out data/elliptic
```

The token is created on kaggle.com (Settings → API → *Create New Token*) and saved as
`~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`), or exported as
`KAGGLE_USERNAME` / `KAGGLE_KEY`. Downloading by hand from the Kaggle page works just as well.

```python
from aml_graphminer.ingestion import load_dataset

df = load_dataset(
    "data/HI-Small_Trans.csv"
)  # schema auto-detected + sibling *_Patterns.txt attached
df = aml.read_transactions("data/HI-Small_Trans.csv", nrows=500_000, engine="polars")  # table only
```

`load_dataset` is what the CLI uses: it reads the table **and** fills the `pattern` column from
the sibling `*_Patterns.txt`. `read_transactions` reads the table only (`pattern` stays empty).

Pandas or Polars? Measured with `benchmarks/io_engines.py` (full `load_dataset`, fresh process,
best of 2 runs; [results](https://github.com/Abeidak/aml-graphminer/tree/main/benchmarks/results_parts/io)):

| file | rows | pandas | polars |
|---|---:|---|---|
| HI-Small | 5,078,345 | 17.3 s, peak 2.6 GB | 12.8 s, peak 4.2 GB |
| LI-Small | 6,924,049 | 23.0 s, peak 3.5 GB | 18.0 s, peak 4.8 GB |

Polars parses faster but the conversion to the canonical schema then works on Arrow-backed
strings and needs more memory: keep the default (`pandas`) when memory is tight.

Notes on the format, all handled by the `IBMAdapter`:

* two columns are named `Account`; the mapping is done **by position**, like the original
  notebooks, so it works with pandas and polars alike;
* accounts are identified as `<bank>_<account>`;
* the file is not sorted; the table is stably sorted by time and `source_row` keeps the
  original line number;
* `*_Patterns.txt` contains `BEGIN LAUNDERING ATTEMPT - <TYPE>` / `END LAUNDERING ATTEMPT`
  blocks. They are matched to the transactions on
  `(timestamp, sender, receiver, amount, transaction_type)` and fill the `pattern` column,
  which enables the **per-motif recall** of the benchmark.

```{note}
The patterns parser follows the published description of the file format and is tolerant
(case, hyphens, 10 or 11 fields). Check it on your copy with
`amlgm -v inspect HI-Small_Trans.csv`: the log line `attached N/M laundering-pattern
transactions` tells how many pattern transactions were found in the CSV, and the summary lists
the count per motif.
```

## Elliptic Bitcoin

Weber et al. (2019): 203 769 transactions (nodes) with 166 anonymised features, 49 time steps,
classes *illicit / licit / unknown*. It is a **node-classification** task and is handled by
`read_elliptic` and `run_elliptic_benchmark` (training on steps ≤ 29, validation up to 34,
test afterwards; unknown nodes are ignored in the metrics).

```bash
amlgm elliptic /data/elliptic --models rf --models hgb --models sage --seeds 0 --seeds 1 --seeds 2
```

## Your own data

Any CSV / JSON / Parquet / DataFrame / list of dicts with the canonical column names works.
Other column names can be mapped explicitly:

```python
df = aml.read_transactions(
    "bank_export.csv",
    schema={
        "timestamp": "booking_date",
        "sender": "debtor_iban",
        "receiver": "creditor_iban",
        "amount": "amt_eur",
    },
)
```

Custom formats can subclass `aml_graphminer.ingestion.adapters.Adapter` (see {doc}`extending`).
Validation errors list the offending rows; `strict=False` drops them and records the count in
`df.attrs["dropped_rows"]`.

## Exploratory data analysis

`amlgm eda data/HI-Small_Trans.csv --out eda/hi-small` writes `EDA.md`, `summary.json` and the
figures (amount distribution by class, transactions and laundering share per day, activity per
hour, payment formats with their laundering rate, degree distributions, ground-truth motifs).
In Python:

```python
from aml_graphminer.eda import eda_figures, eda_report, eda_summary

stats = eda_summary(df)  # JSON-ready dict (daily, hourly, amounts, formats, degree, structure ...)
eda_figures(df, "eda/")  # PNG files (viz extra)
eda_report(df, "eda/", name="HI-Small")  # both + EDA.md
```

The same statistics feed the dashboard of the web interface ({doc}`web_interface`).

## Synthetic data

`generate_transactions` (and `amlgm generate`) produce an IBM-like table with injected motifs
**and** look-alike legitimate behaviour (payroll bursts, merchant bursts, unrelated triangles,
refunds). It exists for tests, demos and CI — *numbers measured on it say nothing about real
data*.
