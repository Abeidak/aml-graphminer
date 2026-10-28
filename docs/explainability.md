# Explainability

A score is not an investigation. Every method below returns the same `Explanation` object:
the transaction, its score and threshold, the neighbouring **transactions** and **accounts**
with a signed importance (positive = pushes the score up), optional raw-**attribute**
sensitivities, the structural **patterns** involved and a one-sentence summary. It can be
printed (`to_markdown()`), stored (`to_json()`), turned into a NetworkX graph
(`to_networkx()`) and drawn (`plot_explanation`).

| method | detectors | what it tells you | cost |
|---|---|---|---|
| `patterns` | motif scanner | the alert(s) that contain the transaction, with accounts, transactions and numbers — exact | negligible |
| `occlusion` | any detector (motifs, tabular, GNN, auto-encoder) | importance of each neighbouring transaction / account (score drop when removed) and of each raw attribute | `O(max_occlusions)` model calls |
| `gnnexplainer` | supervised GNN | PyG `GNNExplainer` (Ying et al., 2019) soft mask over the transactions and account features of the neighbourhood | optimisation per explanation |
| `pgexplainer` | supervised GNN | PyG `PGExplainer` (Luo et al., 2020): an explainer network trained once on many alerts predicts the important transactions of any alert | training once, then one forward pass |
| `captum` | supervised GNN | Captum Integrated Gradients: signed attributions of the neighbouring transactions, of the sender / receiver features and of the transaction's own features (`explain` extra) | ~50 forward passes |
| `attention` | GAT | attention weights on the neighbourhood (descriptive, not causal) | one forward pass |

`explain_transaction(detector, df, tx_id, method="auto")` is the single entry point used by the
CLI, the REST API and the web interface (`available_methods(detector)` lists what applies;
`"auto"` means `patterns` for the motif scanner and `occlusion` otherwise).

```python
from aml_graphminer.explainers import (
    OcclusionExplainer,
    TransactionIndex,
    detector_score_fn,
    explain_with_patterns,
    plot_explanation,
)

index = TransactionIndex(df)  # build once, reuse
exp = explain_with_patterns(df, tx_id, scanner.scan(df))  # exact, motif-based
exp = OcclusionExplainer(detector_score_fn(detector), hops=2, max_edges=60).explain(
    df, tx_id, index
)
print(exp.to_markdown())
plot_explanation(exp, path="why.png")
```

```bash
amlgm explain runs/hgb data/Synth-Small_Trans.csv --tx 1234 --method occlusion --plot why.png --out why.json
```

## Reading an occlusion explanation

* **Edges** — each neighbouring transaction is removed in turn from the context and the score is
  recomputed: `importance = score − score_without`. Positive edges *support* the alert, negative
  ones argue against it. Transactions closest in time to the explained one are tried first, up to
  `max_occlusions`.
* **Nodes** — the same with all transactions of an account removed (the hub of a fan-in scores
  high).
* **Attributes** — amount, received amount, payment format, currencies and banks of the
  explained transaction are replaced one at a time by the typical neighbourhood value (median /
  mode). Other columns stay untouched, so derived features (paid / received ratio) move too: this
  is a *sensitivity*, not a causal effect.

```{important}
Explanations are **local**. The detector only sees the sampled neighbourhood
(`hops`, `max_edges`), so `hops` must reach the structure the model reacts to — a 2-layer GNN
needs 2, a 6-account cycle about 3. With too small a radius the explanation honestly reports a
score of 0 for a structure it cannot see. Prefer `patterns` for the motif scanner: it is exact.
```

## Checking an explanation

Before showing an explanation to an investigator, verify it is *faithful*: remove the top-ranked
neighbours and check that the score really drops (notebook `03_explainability`, section 3).
Explanations of two seeds of the same model should agree on the top neighbours; if they do not,
say so in the case file.

## GNN explainers

`GNNExplainerAdapter(trained, hops=2, max_edges=60)`, `PGExplainerAdapter(trained)`,
`CaptumExplainerAdapter(trained)` and `AttentionExplainer(trained)` work on a re-computed
sub-graph around the transaction, with the feature statistics saved in the model. Features of that
sub-graph are an approximation of the full-graph features, which keeps explanations fast but means
their scores can differ slightly from the batch score. They need the `gnn` extra (and `explain`
for Captum).

```python
from aml_graphminer.explainers import CaptumExplainerAdapter, PGExplainerAdapter

# PGExplainer: learn once from the neighbourhoods of the strongest alerts, then explain any alert
pg = PGExplainerAdapter(gnn.trained, hops=2, epochs=30).fit(df, top_alert_ids, index)
print(pg.explain(df, tx_id, index).to_markdown())

# Captum: which neighbours, which account features and which transaction features matter
exp = CaptumExplainerAdapter(gnn.trained, attribution="IntegratedGradients").explain(
    df, tx_id, index
)
for feature in exp.features[:5]:
    print(feature.name, round(feature.importance, 4))  # e.g. tx.log_paid, sender.out_rate
```

PyG's `PGExplainer` supports node- and graph-level tasks only; the adapter presents the
neighbourhood of the explained transaction as a "graph" whose single output is the logit of that
transaction, which is exactly the edge-level question. Captum's binary convention needs
probabilities, so the adapter wraps the model with a sigmoid; the baseline of Integrated
Gradients is the average (standardised) transaction.

## Is the neighbourhood really what the model uses?

An edge mask only explains a decision if removing the masked transactions changes the score.
Every GNN explanation therefore ends with the **measured neighbourhood effect**: the score of the
explained transaction with its local sub-graph and with the transaction alone
(`aml_graphminer.explainers.gnn_explainers.neighbourhood_effect`).

On the GraphSAGE model trained on IBM AML HI-Small, for the 8 highest-scored transactions of the
test period, the score with the whole neighbourhood (80 transactions) was 0.988–1.000 and
0.960–1.000 with the transaction alone; removing the 5 neighbours ranked first by GNNExplainer left
it unchanged (for example 0.988 → 0.989). The decision rests on the transaction and the profiles
of its two accounts, not on the neighbouring transactions. Consequences:

* **PGExplainer** returns all-zero masks: with its size penalty this is the faithful answer, not
  a failure (without the penalty its masks become arbitrary).
* **GNNExplainer** masks around 0.8 must not be read as "these neighbours caused the alert" when
  the measured effect is small; check with occlusion or the neighbourhood effect.
* **Captum** (feature attributions) and **occlusion** are then the informative methods: they show
  which transaction and account features drive the score.

## REST API and web interface

`POST /explain` and `GET /api/datasets/{id}/transactions/{tx}/explain` accept every method that
applies to the served detector (`GET /health` lists them); the web interface draws the
explanation as a graph — see {doc}`rest_api` and {doc}`web_interface`.
