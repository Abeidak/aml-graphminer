"""Evaluation utilities: temporal split, metrics, benchmark runner.

The benchmark runner imports every detector family, so it is loaded lazily (PEP 562) to keep
``aml_graphminer.evaluation.metrics`` importable from the detectors without a circular import.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .metrics import (
    aggregate_runs,
    best_threshold,
    bootstrap_ci,
    compute_metrics,
    precision_at_k,
    recall_at_k,
    to_markdown,
)
from .splits import temporal_split

if TYPE_CHECKING:  # pragma: no cover
    from .benchmark import (
        ALL_MODELS,
        GNN_KINDS,
        UNSUPERVISED_KINDS,
        BenchmarkReport,
        run_benchmark,
    )

_LAZY = {
    "ALL_MODELS": "benchmark",
    "GNN_KINDS": "benchmark",
    "UNSUPERVISED_KINDS": "benchmark",
    "BenchmarkReport": "benchmark",
    "run_benchmark": "benchmark",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = [
    "ALL_MODELS",
    "GNN_KINDS",
    "UNSUPERVISED_KINDS",
    "BenchmarkReport",
    "aggregate_runs",
    "best_threshold",
    "bootstrap_ci",
    "compute_metrics",
    "precision_at_k",
    "recall_at_k",
    "run_benchmark",
    "temporal_split",
    "to_markdown",
]
