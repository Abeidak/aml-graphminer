# Learned models

## Tabular baselines

`TabularBaseline(kind)` with `kind` in `rf` (random forest), `hgb` (scikit-learn
HistGradientBoosting), `xgb` (XGBoost, `xgboost` extra) and `logreg`. Class imbalance is handled
by class weights (`xgb`: `scale_pos_weight`). Extra parameters (`params={...}` in `TabularDetector.fit`, keyword
arguments in `TabularBaseline`) go to the underlying estimator.

```python
from aml_graphminer.detectors.bundle import TabularDetector

detector = TabularDetector.fit(df, "rf", seed=0, params={"n_estimators": 500})
detector.save("runs/rf")
```

`TabularDetector.fit` performs the temporal split, fits the feature pipeline on the training
part, trains, selects the threshold on validation and stores the test metrics. These baselines
are the honest reference: a GNN is only interesting if it beats them *on the same features and
the same split*.

## Graph Neural Networks (`gnn` extra)

Transactions are classified as **edges**. `EdgeGNN` encodes accounts with message-passing
layers (2 by default) (`GCNConv`, `SAGEConv` or `GATConv`) and classifies each transaction with an MLP over
`[h_sender, h_receiver, transaction features]` — the architecture of the original project
notebooks, now packaged:

```python
from aml_graphminer.detectors.gnn import GNNConfig, GNNDetector

config = GNNConfig(
    kind="gat",
    hidden=64,
    layers=2,
    epochs=20,
    num_neighbors=[10, 5],
    loss="bce",
    pos_weight=30.0,
    patience=5,
)
detector = GNNDetector.fit(df, config, seed=0)
detector.save("runs/gat")
alerts = GNNDetector.load("runs/gat").predict(new_df, context=history_df)
```

Design choices worth knowing:

* **Mini-batch neighbour sampling** (`LinkNeighborLoader`) so that IBM *Medium* and *Large* fit in
  memory. Messages flow in both directions, but only along transactions that existed at the
  time of the split (train graph ⊂ validation graph ⊂ test graph). This needs the compiled
  `pyg-lib` or `torch-sparse` package; see {doc}`installation`.
* **Loss**: class-weighted binary cross-entropy (`pos_weight`) or focal loss (`loss="focal"`,
  `focal_gamma`, `focal_alpha`).
* **Model selection**: the epoch with the best validation PR-AUC is restored (never the test
  split); `patience` enables early stopping.
* **Trainers**: a plain PyTorch loop (default) or PyTorch Lightning (`trainer="lightning"`,
  `amlgm train --trainer lightning`).
* **Reproducibility**: `seed` seeds Python, NumPy and PyTorch; GPU kernels may still be
  non-deterministic, so report several seeds.
* **Checkpoints** are directories: `model.pt` (weights) and `config.json` (hyper-parameters,
  feature statistics, threshold, history, metrics). `GNNDetector.load` rebuilds the model and the
  feature pipeline from them; `load_detector(dir)` dispatches on `model_type`.
* **Inference on new data**: `score(df, context=history)` — `context` transactions belong to the
  graph (so neighbourhoods are complete) but are not scored.
* **Reproducible scores**: an alert must be reproducible to be auditable. `GNNConfig.inference`
  chooses how validation, test and new transactions are scored:
  * `"sampled"` (default) — the neighbour sampling used for training, with a **fixed seed**
    isolated from the global random stream: scoring the same data twice gives the same scores,
    whatever the order of the rows, and it scales to the largest IBM files;
  * `"full"` — one message-passing pass over *all* neighbours, then the transactions are decoded in
    chunks: exact and independent of how transactions are batched, with a memory cost
    proportional to the number of edges (fine for the API, IBM *Small* on a 16 GB machine).

### Node classification (Elliptic)

`NodeGNN` applies the same layers to node features and is used by
`run_elliptic_benchmark(..., models=["gcn", "sage", "gat"])`.

## Unsupervised graph auto-encoder (`gae`, `gnn` extra)

Confirmed laundering cases are rare and arrive late, so the toolkit also learns **without
labels**. `AutoEncoderDetector` is a graph auto-encoder in the spirit of GAE (Kipf & Welling,
2016) and DOMINANT (Ding et al., 2019), adapted to transactions:

* a GCN / GraphSAGE / GAT **encoder** (`GNNConfig.kind`) embeds the accounts;
* a **structure decoder** scores how plausible the link `sender → receiver` is (inner product,
  trained against randomly re-wired pairs);
* an **attribute decoder** rebuilds the features of the transaction (amount, format, time,
  recent activity of both accounts) from the two embeddings.

The anomaly score mixes the two standardised reconstruction errors (`alpha`, default 0.7 for the
attribute error) and is reported as a **percentile of the training distribution** (0.99 = more
unusual than 99 % of the training transactions). Training is robust to contamination — in every
mini-batch the `trim` fraction (5 %) of worst-reconstructed transactions is left out of the
gradient, so hidden laundering is not learned as "normal" — and the epoch is selected on the
validation **reconstruction loss**: no label is used at any point of training.

```python
from aml_graphminer.detectors.gnn import AutoEncoderDetector, GNNConfig

gae = AutoEncoderDetector.fit(df, GNNConfig(kind="sage", hidden=32, epochs=3), contamination=0.01)
gae.predict(new_df)  # score = anomaly percentile, alert = score >= threshold
```

With labelled validation data the alert threshold is F1-optimal on validation (and test metrics
are reported); without labels it is the `1 − contamination` percentile. The same model is
available as `amlgm train --model gae` (works on unlabelled files), in `run_benchmark(...,
models=["gae"])` and in the REST API / web interface; any explanation method that works with any
detector (`occlusion`) explains its alerts.

```{note}
Auto-encoders slowly learn to rebuild the anomalies hidden in their training data. Keep the
number of epochs small (2-5) and compare with the supervised models on the same split before
drawing conclusions.
```

## Choosing between them

| situation | start with |
|---|---|
| no labels, need explainable alerts quickly | `PatternScanner` |
| no labels, want a learned anomaly score | `gae` (graph auto-encoder) |
| labels, tabular data, want a strong simple model | `hgb` / `rf` |
| labels, relational signal (mules, layering) matters | `sage` / `gat` |
| must justify each decision to an auditor | motifs + occlusion explanations; use GNNs as a second opinion |
