# API reference

Generated from the docstrings. Optional heavy dependencies are mocked while building the
documentation, so every module is listed even if PyTorch is not installed.

## Top-level helpers

```{eval-rst}
.. automodule:: aml_graphminer
   :members: __version__
```

## Ingestion

```{eval-rst}
.. automodule:: aml_graphminer.ingestion.readers
   :members:
.. automodule:: aml_graphminer.ingestion.adapters
   :members:
.. automodule:: aml_graphminer.ingestion.schema
   :members:
.. automodule:: aml_graphminer.ingestion.dataset
   :members:
.. automodule:: aml_graphminer.ingestion.ibm_patterns
   :members:
.. automodule:: aml_graphminer.ingestion.elliptic
   :members:
.. automodule:: aml_graphminer.synthetic.generator
   :members: generate_transactions
.. automodule:: aml_graphminer.datasets
   :members:
.. automodule:: aml_graphminer.eda
   :members:
```

The modules named after the project specification (`csv_adapter`, `ibm_adapter`,
`elliptic_adapter`) re-export the objects documented above:

```{eval-rst}
.. automodule:: aml_graphminer.ingestion.csv_adapter
.. automodule:: aml_graphminer.ingestion.ibm_adapter
.. automodule:: aml_graphminer.ingestion.elliptic_adapter
```

## Features and graphs

```{eval-rst}
.. automodule:: aml_graphminer.features.pipeline
   :members:
.. automodule:: aml_graphminer.features.temporal
   :members:
.. automodule:: aml_graphminer.features.network
   :members:
.. automodule:: aml_graphminer.graph.networkx_builder
   :members:
.. automodule:: aml_graphminer.graph.convert
   :members:
.. automodule:: aml_graphminer.graph.hetero
   :members:
.. automodule:: aml_graphminer.graph.pyg_builder
   :members:
.. automodule:: aml_graphminer.graph.neo4j
   :members:
```

## Detectors

```{eval-rst}
.. automodule:: aml_graphminer.detectors.patterns.scanner
   :members:
.. automodule:: aml_graphminer.detectors.patterns.base
   :members:
.. automodule:: aml_graphminer.detectors.patterns.community
   :members:
.. automodule:: aml_graphminer.detectors.patterns.config
   :members:
.. automodule:: aml_graphminer.detectors.patterns.evaluation
   :members:
.. automodule:: aml_graphminer.detectors.baselines
   :members:
.. automodule:: aml_graphminer.detectors.bundle
   :members:
.. automodule:: aml_graphminer.detectors.hybrid
   :members:
.. automodule:: aml_graphminer.detectors.gnn.config
   :members:
.. automodule:: aml_graphminer.detectors.gnn.detector
   :members:
.. automodule:: aml_graphminer.detectors.gnn.models
   :members:
.. automodule:: aml_graphminer.detectors.gnn.trainer
   :members:
.. automodule:: aml_graphminer.detectors.gnn.node_classifier
   :members:
.. automodule:: aml_graphminer.detectors.gnn.unsupervised
   :members:
```

## Explainers

```{eval-rst}
.. automodule:: aml_graphminer.explainers.base
   :members:
.. automodule:: aml_graphminer.explainers.patterns
   :members:
.. automodule:: aml_graphminer.explainers.occlusion
   :members:
.. automodule:: aml_graphminer.explainers.dispatch
   :members:
.. automodule:: aml_graphminer.explainers.scorers
   :members:
.. automodule:: aml_graphminer.explainers.subgraph
   :members:
.. automodule:: aml_graphminer.explainers.gnn_explainers
   :members:
.. automodule:: aml_graphminer.explainers.plot
   :members:
```

## Evaluation

```{eval-rst}
.. automodule:: aml_graphminer.evaluation.splits
   :members:
.. automodule:: aml_graphminer.evaluation.metrics
   :members:
.. automodule:: aml_graphminer.evaluation.benchmark
   :members:
.. automodule:: aml_graphminer.evaluation.elliptic
   :members:
```

## Interfaces

```{eval-rst}
.. automodule:: aml_graphminer.config
   :members:
.. automodule:: aml_graphminer.tracking
   :members:
.. automodule:: aml_graphminer.hub
   :members:
.. automodule:: aml_graphminer.api.service
   :members:
.. automodule:: aml_graphminer.api.investigation
   :members:
.. automodule:: aml_graphminer.api.schemas
   :members:
.. automodule:: aml_graphminer.exceptions
   :members:
```
