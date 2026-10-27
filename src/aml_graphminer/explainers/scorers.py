"""Adapters turning any detector into the ``score_fn(df, context) -> scores`` used by explainers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ..detectors._context import restore_order, stack_context
from ..detectors.baselines import TabularBaseline
from ..detectors.patterns import PatternScanner
from ..features.pipeline import FeaturePipeline

ScoreFn = Callable[[pd.DataFrame, pd.DataFrame | None], np.ndarray]


def tabular_score_fn(
    pipeline: FeaturePipeline, baseline: TabularBaseline, with_nodes: bool = True
) -> ScoreFn:
    """Score with a tabular baseline; node aggregates use every provided transaction."""

    def score(df: pd.DataFrame, context: pd.DataFrame | None = None) -> np.ndarray:
        if len(df) == 0:
            return np.zeros(0)
        full, is_target, target_pos = stack_context(df, context)
        features = pipeline.transform(full)
        probs = np.zeros(len(full))
        probs[is_target] = baseline.predict_proba(features.tabular(is_target, with_nodes))
        return restore_order(probs, is_target, target_pos, len(df))

    return score


def pattern_score_fn(scanner: PatternScanner) -> ScoreFn:
    """Score with the structural-motif scanner (max calibrated alert score per transaction)."""

    def score(df: pd.DataFrame, context: pd.DataFrame | None = None) -> np.ndarray:
        if len(df) == 0:
            return np.zeros(0)
        full, is_target, target_pos = stack_context(df, context)
        # the scanner works on tx ids: make them unique even if df and context overlap
        full = full.reset_index(drop=True)
        scores, _ = scanner.scan(full).transaction_scores()
        return restore_order(scores, is_target, target_pos, len(df))

    return score


def detector_score_fn(detector: object) -> ScoreFn:
    """Score with a :class:`~aml_graphminer.detectors.gnn.GNNDetector` (or anything with ``score``)."""

    def score(df: pd.DataFrame, context: pd.DataFrame | None = None) -> np.ndarray:
        if len(df) == 0:
            return np.zeros(0)
        return np.asarray(detector.score(df, context), dtype="float64")  # type: ignore[attr-defined]

    return score


__all__ = ["ScoreFn", "detector_score_fn", "pattern_score_fn", "tabular_score_fn"]
