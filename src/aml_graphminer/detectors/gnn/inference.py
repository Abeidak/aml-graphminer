"""Score new transactions with a trained GNN (the notebooks' inference section, packaged)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import ModelError
from ...features.pipeline import FeaturePipeline
from .._context import restore_order, stack_context
from .data import resolve_device
from .trainer import TrainedGNN, predict_split


def score_transactions(
    trained: TrainedGNN,
    df: pd.DataFrame,
    context: pd.DataFrame | None = None,
    *,
    device: str | None = None,
) -> np.ndarray:
    """Probability of laundering of every row of ``df`` (same order as ``df``).

    Parameters
    ----------
    trained:
        Model restored by :func:`load_checkpoint` or returned by training.
    df:
        Canonical transaction table (``validate_transactions``) to score.
    context:
        Optional extra history (already-seen transactions): it is part of the message-passing
        graph and of the rolling windows but is *not* scored.

    The feature statistics saved with the model are reused, so the data is transformed
    exactly like the training data; node aggregates use every provided transaction. Scores are
    reproducible: the same ``df`` / ``context`` always give the same scores, whatever the row
    order (see ``GNNConfig.inference`` for how neighbourhoods are aggregated).
    """
    if not trained.stats:
        raise ModelError(
            "this model carries no feature statistics: it cannot score raw transactions"
        )
    if len(df) == 0:
        return np.zeros(0, dtype="float32")
    pipeline = FeaturePipeline.from_dict(trained.stats)
    full, is_target, target_pos = stack_context(df, context)
    features = pipeline.transform(full)
    dev = resolve_device(device or trained.config.device)
    model = trained.model.to(dev).eval()
    graph = np.ones(len(full), dtype=bool)
    ids, probs = predict_split(model, features, graph, is_target, trained.config, dev)
    trained.model = model.cpu()

    by_row = np.zeros(len(full), dtype="float64")
    by_row[ids] = probs
    return restore_order(by_row, is_target, target_pos, len(df)).astype("float32")


__all__ = ["score_transactions"]
