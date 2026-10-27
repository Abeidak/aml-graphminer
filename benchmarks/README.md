# Benchmarks

Reproduce the comparison of the project specification — **structural motifs vs. tabular
baselines vs. GCN / GraphSAGE / GAT vs. the unsupervised graph auto-encoder (`gae`)** — on the IBM
AML and Elliptic datasets.

> `benchmarks/results/REPORT.md` holds the measured IBM AML HI-Small / LI-Small results of the CPU
> models (motifs, logistic regression, random forest, HistGradientBoosting; 3 seeds), produced
> from `benchmarks/results_parts/<model>/<dataset>` (one process per model to fit in 16 GB RAM)
> with `--config configs/patterns.yaml`, and the GNN rows (GCN, GraphSAGE, GAT, GAE; 3 seeds)
> produced on a Kaggle T4 GPU by notebook `04` (`benchmarks/results_parts/gnn`, about 1 h 25 for
> HI-Small and 2 h 15 for LI-Small). Report: `python benchmarks/make_report.py benchmarks/results
> benchmarks/results_parts/patterns benchmarks/results_parts/logreg benchmarks/results_parts/rf
> benchmarks/results_parts/hgb benchmarks/results_parts/xgb benchmarks/results_parts/gnn
> benchmarks/results_parts/elliptic-extra`. Elliptic (rf, hgb, GCN, GraphSAGE, GAT; 3 seeds) was
> run on Kaggle too, with notebook `04` restricted to its Elliptic section
> (`benchmarks/results_parts/gnn/elliptic`, 18 min); XGBoost (IBM, Elliptic) and logistic
> regression (Elliptic) come from Kaggle CPU kernels (`results_parts/xgb`, `elliptic-extra`). Runs on
> synthetic data (`--synthetic`) only check that the pipeline works, and
> `make_report.py` labels them as such (folders whose name contains `smoke` are git-ignored).
>
> Measured costs on a 12-thread laptop CPU, 16 GB RAM (HI-Small, 3 seeds): motifs 4 min,
> HistGradientBoosting 5 min, random forest 99 min (peak 7.3 GiB), logistic regression 23 min
> (lbfgs stops at `max_iter=500` without converging). `--nrows` takes the first lines of the
> file, which is **not** sorted by time: on 500 000 lines the training split holds only 15
> laundering transactions — use it to test the pipeline, not to compare models.

## 1. Get the data (not downloaded automatically)

With a Kaggle API token (`data` extra): `amlgm download ibm --out ~/data/ibm-aml` and
`amlgm download elliptic --out ~/data/elliptic`. Or by hand:

| dataset | files | source |
|---|---|---|
| IBM AML (Altman et al., NeurIPS 2023) | `HI-Small_Trans.csv`, `LI-Small_Trans.csv` (+ `*_Patterns.txt`; *Medium* and *Large* optional) | [Kaggle: IBM Transactions for Anti-Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) — a Kaggle account is required |
| Elliptic Bitcoin (Weber et al., 2019) | `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv` | [Kaggle: Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) |

Check each dataset's licence before redistributing anything derived from it. Put the IBM files in
one folder (`~/data/ibm-aml`) and the Elliptic files in another (`~/data/elliptic`).

## 2. Run

```bash
pip install -e ".[gnn,xgboost]"         # + the pyg-lib wheel for your torch build (docs/installation.md)

# IBM AML: 4 files x 6 models x 3 seeds (the motif scanner is deterministic and runs once)
python benchmarks/run_ibm.py --data-dir ~/data/ibm-aml --datasets HI-Small LI-Small \
       --models patterns rf hgb gcn sage gat --seeds 0 1 2 --out benchmarks/results

# Elliptic (node classification)
python benchmarks/run_elliptic.py --data-dir ~/data/elliptic --models rf hgb gcn sage gat \
       --seeds 0 1 2 --out benchmarks/results

# tables
python benchmarks/make_report.py benchmarks/results      # REPORT.md + results.tex
```

Use `--config my.yaml` to change features, motif thresholds or GNN hyper-parameters (the full
configuration is stored in each `config.json`), and `--nrows 500000` for a quick partial run.
Quick pipeline check without any data: `python benchmarks/run_ibm.py --synthetic --models patterns logreg rf --seeds 0 1 --out /tmp/smoke`.

## 3. What is written

```text
benchmarks/results/<dataset>/
    results.csv              one row per model x seed, all metrics
    summary.csv / summary.md mean and std over seeds
    config.json              configuration, split sizes, positives per split
    per_pattern_recall.csv   recall of the motif scanner per laundering pattern (IBM)
    environment.json         library versions, GPU, wall-clock time
benchmarks/results/REPORT.md      tables for all datasets
benchmarks/results/results.tex    booktabs table for the paper (paper/main.tex includes it when present)
```

## 4. Protocol (so that results are comparable)

* Temporal split 60/20/20 on the time-sorted transactions; Elliptic: steps 1–29 / 30–34 / 35–49.
* Features fitted on the training split only; thresholds selected on validation; test scored once.
* Stochastic models: at least 3 seeds, report mean ± std. PR-AUC is the headline metric;
  `recall_at_k` reproduces the metric of the original notebooks.
* GNN hyper-parameters are the defaults of `GNNConfig` unless a configuration file says otherwise;
  if you tune, tune on validation only and report the search space.

## 5. Practical notes

* GNN runs need a GPU for *Medium* and *Large*. Expect several minutes per GNN per seed on
  *Small* with a T4; time is recorded in `results.csv` (`run_seconds`).
* The cycle detector is bounded by search budgets; a warning in the log means the result is
  incomplete on that dataset — tighten `cycle.max_span` rather than ignoring it.
* The IBM *Large* files are several GB: use `--nrows` while testing, and the `scale` extra
  (Polars) for reading.
