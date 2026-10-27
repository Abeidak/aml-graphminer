# Concepts

## The canonical transaction table

Every input format (CSV, JSON, Parquet, IBM AML, a pandas DataFrame, a list of dicts) is
converted by an *adapter* into one table, validated by `validate_transactions`:

| column | type | meaning |
|---|---|---|
| `timestamp` | datetime | when the transaction happened (the table is sorted by it) |
| `sender`, `receiver` | str | account identifiers (IBM: `<bank>_<account>`) |
| `amount` | float ≥ 0 | amount paid, in `currency` |
| `currency`, `currency_received` | str | payment / reception currency |
| `amount_received` | float | amount received (defaults to `amount`) |
| `transaction_type` | str | payment format (ACH, Wire, Cash, ...) |
| `sender_bank`, `receiver_bank` | str | institution identifiers |
| `label` | 0 / 1 / NaN | laundering flag; `NaN` = unlabelled |
| `pattern` | str | motif type when known (IBM `*_Patterns.txt`), else empty |
| `source_row` | int | position in the original file (traceability) |

The **index** is `tx_id`, the position after the stable sort by time. Explanations, alerts and
API answers all refer to it. `df.attrs["dropped_rows"]` records rows removed in lenient mode.

## Time is the only honest split

Random train/test splits on transactions let a model see the *future* of an account while
being evaluated on its past. Everything in AML-GraphMiner follows the temporal protocol of the
original project notebooks:

1. sort by time; the first 60 % of transactions train, the next 20 % validate, the last 20 % test
   (`temporal_split`, configurable);
2. feature statistics (means, standard deviations, categorical vocabularies) are fitted on the
   **training** transactions only;
3. edge features use *trailing* windows (1 h, 24 h, 7 d, all-time) that only look at the past;
4. node features are rates computed on the graph that was observable at that split —
   training graph ⊂ training + validation graph ⊂ everything;
5. decision thresholds are chosen on the **validation** split and then frozen for the test.

Labels are never used to build features, and a transaction never receives messages from the
future in a GNN.

## Two families of detectors

| | structural motifs | learned models |
|---|---|---|
| training | none (optional per-motif calibration on validation) | tabular baselines, GCN / GraphSAGE / GAT |
| output | alerts with accounts, transactions and numeric evidence | a score per transaction |
| strength | exact explanations, no labels needed | recall on patterns nobody wrote a rule for |
| weakness | only finds what its definition covers; sensitive to thresholds | needs labels, explanations are approximate |

They share the same interface (`score`, `predict`, `save`, `load`, `describe`) and are
compared on the same split by `run_benchmark`.

## Metrics

Money laundering is rare (≈ 0.1 % of IBM transactions): accuracy and ROC-AUC look excellent for
a model that flags nothing. The toolkit reports **PR-AUC**, **F1 / precision / recall** at the
validation threshold, and rank metrics: `recall_at_k` (k = number of positives, i.e.
R-precision, as in the original notebooks), `recall_at_<n>` for a fixed alert budget and
`recall_at_<rate>pct` for a fixed share of flagged transactions. See {doc}`evaluation`.
