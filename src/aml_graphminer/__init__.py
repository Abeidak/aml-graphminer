"""AML-GraphMiner — explainable money-laundering detection on transaction graphs.

The package is organised in six layers (see ``docs/architecture.md``):

1. :mod:`aml_graphminer.ingestion`   — adapters (CSV / JSON / Parquet / IBM AML / Elliptic)
2. :mod:`aml_graphminer.features`    — temporal, network and categorical features
3. :mod:`aml_graphminer.graph`       — NetworkX and PyTorch-Geometric representations
4. :mod:`aml_graphminer.detectors`   — structural-motif detectors and GNN / tabular models
5. :mod:`aml_graphminer.explainers`  — auditable explanations for every alert
6. :mod:`aml_graphminer.api` / :mod:`aml_graphminer.cli` — interfaces on top of the same SDK

Importing the package is cheap: heavy dependencies (PyTorch, PyG) are only imported
when the GNN sub-packages are used, and the public helpers below are loaded lazily.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

# public name -> (module, attribute)
_LAZY: dict[str, tuple[str, str]] = {
    "read_transactions": ("aml_graphminer.ingestion", "read_transactions"),
    "validate_transactions": ("aml_graphminer.ingestion", "validate_transactions"),
    "generate_transactions": ("aml_graphminer.synthetic", "generate_transactions"),
    "FeaturePipeline": ("aml_graphminer.features", "FeaturePipeline"),
    "PatternScanner": ("aml_graphminer.detectors.patterns", "PatternScanner"),
    "PatternConfig": ("aml_graphminer.detectors.patterns", "PatternConfig"),
    "TabularBaseline": ("aml_graphminer.detectors.baselines", "TabularBaseline"),
    "temporal_split": ("aml_graphminer.evaluation", "temporal_split"),
    "compute_metrics": ("aml_graphminer.evaluation", "compute_metrics"),
    "build_digraph": ("aml_graphminer.graph", "build_digraph"),
    "build_multidigraph": ("aml_graphminer.graph", "build_multidigraph"),
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:  # PEP 562
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'aml_graphminer' has no attribute {name!r}") from None
    return getattr(importlib.import_module(module), attr)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
