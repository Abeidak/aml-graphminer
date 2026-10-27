# Configuration

One YAML file configures features, motif detectors, GNN hyper-parameters, the benchmark and
experiment tracking. Every key is optional.

```bash
amlgm init-config amlgm.yaml          # fully populated defaults (also in configs/default.yaml)
amlgm scan data.csv --config amlgm.yaml
```

```yaml
features:                 # FeatureConfig
  windows: [1h, 24h, 7d, all]
  temporal: true
  pagerank: true
  betweenness: false
patterns:                 # PatternConfig, one section per detector (see "Structural-motif detectors")
  fan_in: {min_counterparties: 5, window: 24h}
  community: {min_flow_through: 0.75, max_span: 3d, max_member_degree: 40}
gnn:                      # GNNConfig (supervised GNNs and the encoder of the auto-encoder)
  kind: sage              # gcn | sage | gat
  hidden: 64
  layers: 2
  epochs: 10
  num_neighbors: [10, 5]
  loss: bce               # bce | focal
  pos_weight: 30.0
  inference: sampled      # sampled (seeded, scalable) | full (exact full neighbourhoods)
benchmark:
  models: [patterns, rf, hgb]
  seeds: [0, 1, 2]
  fractions: [0.6, 0.2, 0.2]
tracking:
  backend: none           # none | mlflow
  experiment: aml-graphminer
  uri: null
```

Configuration is validated when it is loaded: unknown keys, out-of-range values and malformed
durations raise `ConfigError` with the offending key. `configs/patterns.yaml` is a ready-made
file focused on the motif thresholds: a file that contains nothing but detector sections is read as
the `patterns` section, so `amlgm scan data.csv --config configs/patterns.yaml` works.

In Python:

```python
from aml_graphminer.config import load_config

cfg = load_config("amlgm.yaml")
cfg.feature_config(), cfg.pattern_config(), cfg.gnn_config(), cfg.tracker("run-1")
```

## Experiment tracking

`tracking.backend: mlflow` (or `--mlflow`) logs the run parameters and metrics (per epoch with the
plain PyTorch trainer) and, for `amlgm benchmark`, the summary table to MLflow (`tracking` extra).
`uri` selects the tracking store or server; without it (and without `MLFLOW_TRACKING_URI`) runs
go to the SQLite database `mlflow.db` of the working directory — the backend recommended by
MLflow 3, which refuses plain `./mlruns` folders by default. Browse the runs with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`. A directory store (`uri: file:///...`) is
still accepted. Without the backend a no-op tracker is used, so nothing changes in the code paths.
