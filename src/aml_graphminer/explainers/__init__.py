"""Layer 5 — explainability: pattern evidence, occlusion, GNNExplainer, PGExplainer, Captum, attention."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .base import EdgeImportance, Explanation, FeatureImportance, NodeImportance
from .dispatch import (
    EXPLAIN_METHODS,
    available_methods,
    default_method,
    explain_alerts,
    explain_transaction,
)
from .occlusion import OcclusionExplainer
from .patterns import PATTERN_DESCRIPTIONS, describe_alert, explain_with_patterns
from .scorers import ScoreFn, detector_score_fn, pattern_score_fn, tabular_score_fn
from .subgraph import TransactionIndex, subgraph_to_networkx

if TYPE_CHECKING:  # pragma: no cover
    from .gnn_explainers import (
        AttentionExplainer,
        CaptumExplainerAdapter,
        GNNExplainerAdapter,
        PGExplainerAdapter,
    )
    from .plot import plot_explanation

_LAZY = {
    "AttentionExplainer": "gnn_explainers",
    "CaptumExplainerAdapter": "gnn_explainers",
    "GNNExplainerAdapter": "gnn_explainers",
    "PGExplainerAdapter": "gnn_explainers",
    "plot_explanation": "plot",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = [
    "EXPLAIN_METHODS",
    "PATTERN_DESCRIPTIONS",
    "AttentionExplainer",
    "CaptumExplainerAdapter",
    "EdgeImportance",
    "Explanation",
    "FeatureImportance",
    "GNNExplainerAdapter",
    "NodeImportance",
    "OcclusionExplainer",
    "PGExplainerAdapter",
    "ScoreFn",
    "TransactionIndex",
    "available_methods",
    "default_method",
    "describe_alert",
    "detector_score_fn",
    "explain_alerts",
    "explain_transaction",
    "explain_with_patterns",
    "pattern_score_fn",
    "plot_explanation",
    "subgraph_to_networkx",
    "tabular_score_fn",
]
