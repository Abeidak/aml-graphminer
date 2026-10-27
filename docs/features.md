# Features

`FeaturePipeline` turns a transaction table into numeric arrays for models. It is fitted once
(on training transactions), serialised as JSON inside every saved model and re-applied
unchanged at inference time.

```python
pipeline = aml.FeaturePipeline().fit(df, masks["train"])
features = pipeline.transform_splits(df, masks)  # {"train": FeatureSet, "val": ..., "test": ...}
features["train"].edge.shape, features["train"].node.shape
```

## Transaction (edge) features

| group | columns | notes |
|---|---|---|
| amount | `log_paid`, `log_recv` | `log1p`, standardised with training statistics |
| currency / bank | `same_cur`, `same_bank` | sender and receiver share the currency / bank |
| time of day | `h_sin`, `h_cos` | cyclic encoding of the hour |
| payment format | `fmt_ACH`, `fmt_Wire`, ... | one-hot, vocabulary fixed at fit time |
| sender history | `s_cnt_*`, `s_sum_*`, `s_mean_*`, `s_std_*` | trailing windows `1h`, `24h`, `7d`, `all`; counts and sums are `log1p`-compressed, mean / std are taken on `log1p(amount)` |
| sender gap | `s_gap` | seconds since the sender's previous transaction |
| receiver history | `r_cnt_*`, `r_sum_*`, `r_mean_*`, `r_std_*`, `r_gap` | same, for the receiving account (amounts received) |

Windows are *causal*: the statistics of a transaction cover only earlier transactions of the
same account, so training features never contain future information. Windows, and whether the
temporal block is built at all, are set by `FeatureConfig(windows=("1h", "24h", "7d", "all"))`.

## Account (node) features

Rates per day computed on the graph that was observable at the split, so that a short training
period and a long test period are comparable: `out_rate`, `in_rate`, `out_amt_mean`,
`in_amt_mean`, `out_vol_rate`, `in_vol_rate`, `uniq_out_rate`, `uniq_in_rate`, `io_ratio`
(in/out volume), `ent_out`, `ent_in` (entropy of counterparties), `velocity` and `pagerank`
(power iteration, damping 0.85). Betweenness centrality on a sample of `betweenness_k` pivots
is optional (`betweenness: true`) because it is expensive.

Tabular baselines receive the edge features plus the node features of the sender and of the
receiver (`with_nodes=True`); GNNs receive the node features as node attributes and the edge
features on the classified edge.

## Why this matters

Fitting normalisation statistics on all data, computing account aggregates over the whole
period, or using future transactions in windows are the most common leaks in AML papers.
They inflate PR-AUC and make a model look deployable when it is not. The test-suite checks
the two properties that matter: windows are causal (appending later transactions never
changes the features of earlier ones) and the training features do not move when the
validation and test transactions are altered (while the test snapshot, which is allowed to
see the past, does change).
