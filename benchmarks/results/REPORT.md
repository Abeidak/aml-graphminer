# Benchmark report

## HI-Small

Merged from: `benchmarks\results_parts\patterns\HI-Small`, `benchmarks\results_parts\logreg\HI-Small`, `benchmarks\results_parts\rf\HI-Small`, `benchmarks\results_parts\hgb\HI-Small`, `benchmarks\results_parts\xgb\HI-Small`, `benchmarks\results_parts\gnn\HI-Small`.

Temporal split sizes: {'train': 3047007, 'val': 1015669, 'test': 1015669}; positives per split: {'train': 2298, 'val': 1082, 'test': 1797}.

| model | runs | PR-AUC | F1 | Precision | Recall | Recall@k |
|---|---|---|---|---|---|---|
| sage | 3 | 0.531 ± 0.022 | 0.510 ± 0.035 | 0.834 ± 0.012 | 0.368 ± 0.035 | 0.513 ± 0.021 |
| rf | 3 | 0.499 ± 0.017 | 0.563 ± 0.006 | 0.839 ± 0.015 | 0.424 ± 0.010 | 0.492 ± 0.018 |
| hgb | 3 | 0.476 ± 0.011 | 0.507 ± 0.005 | 0.740 ± 0.019 | 0.386 ± 0.001 | 0.470 ± 0.013 |
| xgb | 3 | 0.474 ± 0.005 | 0.441 ± 0.019 | 0.742 ± 0.054 | 0.315 ± 0.029 | 0.459 ± 0.003 |
| gat | 3 | 0.447 ± 0.007 | 0.387 ± 0.028 | 0.872 ± 0.035 | 0.250 ± 0.025 | 0.452 ± 0.007 |
| gcn | 3 | 0.374 ± 0.031 | 0.363 ± 0.029 | 0.785 ± 0.036 | 0.236 ± 0.022 | 0.389 ± 0.030 |
| logreg | 3 | 0.294 ± 0.000 | 0.352 ± 0.000 | 0.334 ± 0.000 | 0.373 ± 0.000 | 0.349 ± 0.000 |
| patterns | 1 | 0.005 | 0.004 | 0.002 | 0.233 | 0.008 |
| gae | 3 | 0.003 ± 0.001 | 0.006 ± 0.001 | 0.003 ± 0.001 | 0.480 ± 0.047 | 0.002 ± 0.001 |

Mean ± standard deviation over seeds; deterministic models are run once.

### Per-motif recall of the structural-motif scanner (test split)

| pattern | n_tx | recall_any | recall_same |
|---|---|---|---|
| bipartite | 74 | 0.608 | 0.0 |
| cycle | 109 | 0.771 | 0.0 |
| fan_in | 137 | 0.883 | 0.869 |
| fan_out | 140 | 0.707 | 0.55 |
| gather_scatter | 401 | 0.499 | 0.102 |
| random | 86 | 0.814 | 0.0 |
| scatter_gather | 252 | 0.921 | 0.024 |
| stack | 140 | 0.636 | 0.0 |

<details><summary>Environment of `benchmarks\results_parts\patterns\HI-Small`</summary>

```json
{
  "python": "3.12.10",
  "platform": "Windows-11-10.0.26200-SP0",
  "aml_graphminer": "0.1.0",
  "numpy": "2.5.4",
  "pandas": "3.0.6",
  "sklearn": "1.9.1",
  "torch": "2.10.0+cpu",
  "torch_geometric": "2.8.1",
  "cuda": "False",
  "wall_seconds": 190.5
}
```

</details>

<details><summary>Environment of `benchmarks\results_parts\logreg\HI-Small`</summary>

```json
{
  "python": "3.12.10",
  "platform": "Windows-11-10.0.26200-SP0",
  "aml_graphminer": "0.1.0",
  "numpy": "2.5.4",
  "pandas": "3.0.6",
  "sklearn": "1.9.1",
  "torch": "2.10.0+cpu",
  "torch_geometric": "2.8.1",
  "cuda": "False",
  "wall_seconds": 1328.6
}
```

</details>

<details><summary>Environment of `benchmarks\results_parts\rf\HI-Small`</summary>

```json
{
  "python": "3.12.10",
  "platform": "Windows-11-10.0.26200-SP0",
  "aml_graphminer": "0.1.0",
  "numpy": "2.5.4",
  "pandas": "3.0.6",
  "sklearn": "1.9.1",
  "torch": "2.10.0+cpu",
  "torch_geometric": "2.8.1",
  "cuda": "False",
  "wall_seconds": 5891.9
}
```

</details>

<details><summary>Environment of `benchmarks\results_parts\hgb\HI-Small`</summary>

```json
{
  "python": "3.12.10",
  "platform": "Windows-11-10.0.26200-SP0",
  "aml_graphminer": "0.1.0",
  "numpy": "2.5.4",
  "pandas": "3.0.6",
  "sklearn": "1.9.1",
  "torch": "2.10.0+cpu",
  "torch_geometric": "2.8.1",
  "cuda": "False",
  "wall_seconds": 255.8
}
```

</details>

<details><summary>Environment of `benchmarks\results_parts\xgb\HI-Small`</summary>

```json
{
  "python": "3.13.15",
  "aml_graphminer": "0.1.0",
  "torch": "2.11.0+cpu",
  "runtime": "kaggle",
  "model": "xgb",
  "wall_seconds": 699.4
}
```

</details>

<details><summary>Environment of `benchmarks\results_parts\gnn\HI-Small`</summary>

```json
{
  "python": "3.13.15",
  "platform": "Linux-6.18.55+-x86_64-with-glibc2.39",
  "aml_graphminer": "0.1.0",
  "torch": "2.11.0+cu128",
  "cuda": "True",
  "runtime": "kaggle",
  "gpu": "Tesla T4",
  "torch_geometric": "2.8.1",
  "wall_seconds": 5002.7
}
```

</details>

## LI-Small

Merged from: `benchmarks\results_parts\patterns\LI-Small`, `benchmarks\results_parts\logreg\LI-Small`, `benchmarks\results_parts\rf\LI-Small`, `benchmarks\results_parts\hgb\LI-Small`, `benchmarks\results_parts\xgb\LI-Small`, `benchmarks\results_parts\gnn\LI-Small`.

Temporal split sizes: {'train': 4154429, 'val': 1384810, 'test': 1384810}; positives per split: {'train': 1813, 'val': 827, 'test': 925}.

| model | runs | PR-AUC | F1 | Precision | Recall | Recall@k |
|---|---|---|---|---|---|---|
| rf | 3 | 0.191 ± 0.007 | 0.260 ± 0.005 | 0.561 ± 0.084 | 0.170 ± 0.012 | 0.231 ± 0.007 |
| sage | 3 | 0.176 ± 0.018 | 0.219 ± 0.023 | 0.646 ± 0.156 | 0.135 ± 0.026 | 0.200 ± 0.018 |
| gat | 3 | 0.110 ± 0.026 | 0.135 ± 0.026 | 0.468 ± 0.168 | 0.080 ± 0.013 | 0.149 ± 0.021 |
| hgb | 3 | 0.106 ± 0.016 | 0.157 ± 0.006 | 0.172 ± 0.060 | 0.160 ± 0.040 | 0.154 ± 0.007 |
| xgb | 3 | 0.078 ± 0.007 | 0.150 ± 0.015 | 0.153 ± 0.007 | 0.149 ± 0.023 | 0.150 ± 0.008 |
| gcn | 3 | 0.074 ± 0.024 | 0.108 ± 0.037 | 0.234 ± 0.100 | 0.078 ± 0.033 | 0.124 ± 0.016 |
| logreg | 3 | 0.032 ± 0.000 | 0.082 ± 0.000 | 0.063 ± 0.000 | 0.117 ± 0.000 | 0.076 ± 0.000 |
| gae | 3 | 0.001 ± 0.000 | 0.003 ± 0.001 | 0.001 ± 0.000 | 0.471 ± 0.079 | 0.001 ± 0.001 |
| patterns | 1 | 0.001 | 0.002 | 0.001 | 0.431 | 0.000 |

Mean ± standard deviation over seeds; deterministic models are run once.

### Per-motif recall of the structural-motif scanner (test split)

| pattern | n_tx | recall_any | recall_same |
|---|---|---|---|
| bipartite | 45 | 0.4 | 0.0 |
| cycle | 43 | 0.721 | 0.0 |
| fan_in | 15 | 0.867 | 0.8 |
| fan_out | 50 | 0.66 | 0.42 |
| gather_scatter | 76 | 0.408 | 0.026 |
| random | 24 | 0.792 | 0.0 |
| scatter_gather | 64 | 0.969 | 0.0 |
| stack | 26 | 0.538 | 0.0 |

<details><summary>Environment of `benchmarks\results_parts\patterns\LI-Small`</summary>

```json
{
  "python": "3.12.10",
  "platform": "Windows-11-10.0.26200-SP0",
  "aml_graphminer": "0.1.0",
  "numpy": "2.5.4",
  "pandas": "3.0.6",
  "sklearn": "1.9.1",
  "torch": "2.10.0+cpu",
  "torch_geometric": "2.8.1",
  "cuda": "False",
  "wall_seconds": 410.2
}
```

</details>

<details><summary>Environment of `benchmarks\results_parts\logreg\LI-Small`</summary>

```json
{
  "python": "3.12.10",
  "platform": "Windows-11-10.0.26200-SP0",
  "aml_graphminer": "0.1.0",
  "numpy": "2.5.4",
  "pandas": "3.0.6",
  "sklearn": "1.9.1",
  "torch": "2.10.0+cpu",
  "torch_geometric": "2.8.1",
  "cuda": "False",
  "wall_seconds": 938.2
}
```

</details>

<details><summary>Environment of `benchmarks\results_parts\rf\LI-Small`</summary>

```json
{
  "python": "3.12.10",
  "platform": "Windows-11-10.0.26200-SP0",
  "aml_graphminer": "0.1.0",
  "numpy": "2.5.4",
  "pandas": "3.0.6",
  "sklearn": "1.9.1",
  "torch": "2.10.0+cpu",
  "torch_geometric": "2.8.1",
  "cuda": "False",
  "wall_seconds": 5693.5
}
```

</details>

<details><summary>Environment of `benchmarks\results_parts\hgb\LI-Small`</summary>

```json
{
  "python": "3.12.10",
  "platform": "Windows-11-10.0.26200-SP0",
  "aml_graphminer": "0.1.0",
  "numpy": "2.5.4",
  "pandas": "3.0.6",
  "sklearn": "1.9.1",
  "torch": "2.10.0+cpu",
  "torch_geometric": "2.8.1",
  "cuda": "False",
  "wall_seconds": 381.2
}
```

</details>

<details><summary>Environment of `benchmarks\results_parts\xgb\LI-Small`</summary>

```json
{
  "python": "3.13.15",
  "aml_graphminer": "0.1.0",
  "torch": "2.11.0+cpu",
  "runtime": "kaggle",
  "model": "xgb",
  "wall_seconds": 942.3
}
```

</details>

<details><summary>Environment of `benchmarks\results_parts\gnn\LI-Small`</summary>

```json
{
  "python": "3.13.15",
  "platform": "Linux-6.18.55+-x86_64-with-glibc2.39",
  "aml_graphminer": "0.1.0",
  "torch": "2.11.0+cu128",
  "cuda": "True",
  "runtime": "kaggle",
  "gpu": "Tesla T4",
  "torch_geometric": "2.8.1",
  "wall_seconds": 8012.3
}
```

</details>

## elliptic

Merged from: `benchmarks\results_parts\gnn\elliptic`, `benchmarks\results_parts\elliptic-extra\elliptic`.

Temporal split sizes: {'train': 26381, 'val': 3513, 'test': 16670}; positives per split: n/a.

| model | runs | PR-AUC | F1 | Precision | Recall | Recall@k |
|---|---|---|---|---|---|---|
| hgb | 3 | 0.790 ± 0.004 | 0.756 ± 0.037 | 0.798 ± 0.083 | 0.721 ± 0.003 | 0.729 ± 0.005 |
| xgb | 3 | 0.788 ± 0.002 | 0.762 ± 0.021 | 0.792 ± 0.050 | 0.735 ± 0.004 | 0.739 ± 0.001 |
| rf | 3 | 0.786 ± 0.001 | 0.791 ± 0.007 | 0.877 ± 0.019 | 0.721 ± 0.002 | 0.733 ± 0.002 |
| sage | 3 | 0.497 ± 0.056 | 0.566 ± 0.021 | 0.570 ± 0.017 | 0.563 ± 0.026 | 0.566 ± 0.023 |
| gat | 3 | 0.351 ± 0.042 | 0.398 ± 0.023 | 0.336 ± 0.049 | 0.496 ± 0.032 | 0.407 ± 0.010 |
| gcn | 3 | 0.300 ± 0.055 | 0.382 ± 0.040 | 0.387 ± 0.059 | 0.378 ± 0.024 | 0.381 ± 0.058 |
| logreg | 3 | 0.209 ± 0.000 | 0.324 ± 0.000 | 0.202 ± 0.000 | 0.812 ± 0.000 | 0.241 ± 0.000 |

Mean ± standard deviation over seeds; deterministic models are run once.

<details><summary>Environment of `benchmarks\results_parts\elliptic-extra\elliptic`</summary>

```json
{
  "python": "3.13.15",
  "aml_graphminer": "0.1.0",
  "torch": "2.11.0+cpu",
  "runtime": "kaggle",
  "model": "elliptic-extra"
}
```

</details>
