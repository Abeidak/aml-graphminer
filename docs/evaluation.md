# Evaluation and benchmarking

## Protocol

`run_benchmark(df, models, seeds)` implements the comparison required by the project
specification:

1. temporal split 60 / 20 / 20 (`fractions` is configurable);
2. one `FeaturePipeline` fitted on the training split; all models share its features;
3. for every model and seed: train, score validation and test, choose the **threshold on
   validation**, report metrics on **test**;
4. deterministic models (the motif scanner) run once, stochastic ones once per seed;
5. results are aggregated as mean ± standard deviation over seeds.

```python
from aml_graphminer.evaluation import run_benchmark

report = run_benchmark(
    df, models=["patterns", "rf", "hgb", "gcn", "sage", "gat"], seeds=(0, 1, 2), dataset="HI-Small"
)
report.summary()  # mean / std per model
report.per_pattern  # recall of the motif scanner per laundering pattern
report.save("results/HI-Small")  # results.csv, summary.csv/.md, config.json, per_pattern_recall.csv
```

`amlgm benchmark` does the same from the command line; `benchmarks/run_ibm.py` loops over the four
IBM files and `benchmarks/make_report.py` turns the result folders into `REPORT.md` and a LaTeX
table (see `benchmarks/README.md`).

## Metrics

| metric | definition | when to use |
|---|---|---|
| `pr_auc` | average precision | headline metric under extreme imbalance |
| `roc_auc` | area under the ROC curve | secondary; optimistic when positives are rare |
| `precision`, `recall`, `f1` | at the validation threshold | operating point |
| `recall_at_k` | precision among the top-*P* scores (*P* = number of positives) | the metric of the original notebooks |
| `recall_at_100`, `recall_at_1000` | share of all positives found among the top-*k* alerts | fixed investigation budget |
| `recall_at_0.1pct`, `recall_at_1pct` | share of all positives found among the top 0.1 % / 1 % scores | alert-rate budgets |

`bootstrap_ci(y, p)` gives a percentile confidence interval by resampling test transactions;
`aggregate_runs` gives the seed variability. Report both: the first measures test-set noise,
the second training noise.

## Elliptic

`run_elliptic_benchmark` follows Weber et al.: labelled nodes only, time steps ≤ 29 train, 30–34
validate, 35–49 test. The `unknown` class is ignored in training and metrics.

## Reporting checklist

* State the dataset variant (HI/LI, Small/Medium/Large) and the split sizes — they are in
  `config.json` (`split_sizes`, `illicit_in_split`).
* Give PR-AUC with standard deviation over **at least three seeds** for stochastic models.
* Say how the threshold was chosen (validation) and how many alerts that means per day.
* Report per-motif recall, not only the aggregate: a model can ace fan-outs and miss cycles.
* Keep `environment.json` (library versions, GPU) and the exact configuration.
* Never present synthetic-generator numbers as results on IBM or Elliptic; `make_report.py`
  labels such runs as synthetic.

## What a benchmark number does *not* tell you

The IBM data are synthetic and their positives are injected motifs, so high scores partly measure
how well a model recovers a generator. Treat results as a controlled comparison of methods, not as
the performance you will see in a bank. See {doc}`limitations`.
