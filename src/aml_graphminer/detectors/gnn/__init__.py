"""GNN detectors — supervised GCN / GraphSAGE / GAT and an unsupervised graph auto-encoder.

Everything but the configuration requires the ``gnn`` extra. Heavy modules (PyTorch, PyTorch
Geometric, Lightning) are imported lazily, so ``from aml_graphminer.detectors.gnn import
GNNConfig`` works on a minimal installation.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ...exceptions import OptionalDependencyError
from .config import GNN_KINDS, INFERENCE_MODES, GNNConfig

if TYPE_CHECKING:  # pragma: no cover
    from .checkpoint import load_checkpoint, save_checkpoint
    from .detector import GNNDetector
    from .experiment import run_gnn_experiment
    from .inference import score_transactions
    from .models import EdgeGNN, NodeGNN
    from .node_classifier import run_node_experiment
    from .trainer import TrainedGNN, fit_edge_gnn
    from .unsupervised import (
        AutoEncoderDetector,
        EdgeAutoEncoder,
        TrainedAutoEncoder,
        run_gae_experiment,
    )

_LAZY = {
    "EdgeGNN": "models",
    "NodeGNN": "models",
    "GNNDetector": "detector",
    "TrainedGNN": "trainer",
    "fit_edge_gnn": "trainer",
    "run_gnn_experiment": "experiment",
    "run_node_experiment": "node_classifier",
    "score_transactions": "inference",
    "save_checkpoint": "checkpoint",
    "load_checkpoint": "checkpoint",
    "AutoEncoderDetector": "unsupervised",
    "EdgeAutoEncoder": "unsupervised",
    "TrainedAutoEncoder": "unsupervised",
    "run_gae_experiment": "unsupervised",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        try:
            module = importlib.import_module(f".{_LAZY[name]}", __name__)
        except ImportError as exc:
            missing = (getattr(exc, "name", None) or "torch").split(".")[0]
            raise OptionalDependencyError(
                missing, "gnn", f"{name} (graph neural networks)"
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = [
    "GNN_KINDS",
    "INFERENCE_MODES",
    "AutoEncoderDetector",
    "EdgeAutoEncoder",
    "EdgeGNN",
    "GNNConfig",
    "GNNDetector",
    "NodeGNN",
    "TrainedAutoEncoder",
    "TrainedGNN",
    "fit_edge_gnn",
    "load_checkpoint",
    "run_gae_experiment",
    "run_gnn_experiment",
    "run_node_experiment",
    "save_checkpoint",
    "score_transactions",
]
