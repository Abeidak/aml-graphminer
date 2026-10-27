"""Train + evaluate one GNN on a transaction table (called by the benchmark runner)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...evaluation.metrics import compute_metrics
from ...features.pipeline import FeatureSet
from ...tracking import Tracker
from .config import GNNConfig
from .trainer import TrainedGNN, fit_edge_gnn, score_splits, threshold_from_validation


def run_gnn_experiment(
    kind: str,
    df: pd.DataFrame,
    features: dict[str, FeatureSet],
    masks: dict[str, np.ndarray],
    config: GNNConfig | None = None,
    seed: int = 0,
    *,
    tracker: Tracker | None = None,
    pipeline_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train ``kind`` on the train split, select the threshold on validation, score validation/test.

    Returns a dictionary with ``scores_val``, ``scores_test`` (aligned with the rows of each
    split), the ``trained`` model (threshold and test metrics attached) and ``best_epoch``.
    """
    cfg = (config or GNNConfig()).with_kind(kind)
    y = df["label"].to_numpy(dtype="float64")
    trained: TrainedGNN = fit_edge_gnn(
        features, y, masks, cfg, seed=seed, pipeline_stats=pipeline_stats, tracker=tracker
    )
    scores = score_splits(trained, features, masks, ("val", "test"))
    trained.threshold = threshold_from_validation(y[masks["val"]], scores["val"])
    trained.metrics = {
        k: float(v)
        for k, v in compute_metrics(y[masks["test"]], scores["test"], trained.threshold).items()
    }
    return {
        "scores_val": scores["val"],
        "scores_test": scores["test"],
        "trained": trained,
        "best_epoch": trained.best_epoch,
    }


__all__ = ["run_gnn_experiment"]
