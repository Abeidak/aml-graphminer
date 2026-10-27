"""High-level GNN detector: ``fit`` → ``score`` → ``save`` / ``load`` (the toolkit's public GNN API)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...evaluation.splits import temporal_split
from ...exceptions import ConfigError
from ...features.pipeline import FeatureConfig, FeaturePipeline
from ...tracking import Tracker
from .checkpoint import load_checkpoint, save_checkpoint
from .config import GNNConfig
from .experiment import run_gnn_experiment
from .inference import score_transactions
from .models import count_parameters
from .trainer import TrainedGNN


class GNNDetector:
    """Edge-classification GNN bundled with its feature pipeline and decision threshold.

    Example
    -------
    >>> det = GNNDetector.fit(df, GNNConfig(kind="sage", epochs=5))   # doctest: +SKIP
    >>> det.save("runs/sage")                                         # doctest: +SKIP
    >>> alerts = GNNDetector.load("runs/sage").predict(new_df)        # doctest: +SKIP
    """

    model_type = "gnn"

    def __init__(self, trained: TrainedGNN) -> None:
        self.trained = trained

    # ------------------------------------------------------------------ training
    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        config: GNNConfig | None = None,
        *,
        fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
        feature_config: FeatureConfig | None = None,
        seed: int = 0,
        tracker: Tracker | None = None,
    ) -> GNNDetector:
        """Temporal split → features fitted on train → training → threshold on validation."""
        if df["label"].isna().any():
            raise ConfigError("training needs fully labelled transactions (column 'label')")
        masks = temporal_split(df, fractions)
        pipeline = FeaturePipeline(feature_config).fit(df, masks["train"])
        features = pipeline.transform_splits(df, masks)
        out = run_gnn_experiment(
            (config or GNNConfig()).kind,
            df,
            features,
            masks,
            config,
            seed,
            tracker=tracker,
            pipeline_stats=pipeline.to_dict(),
        )
        return cls(out["trained"])

    # ------------------------------------------------------------------ inference
    @property
    def threshold(self) -> float:
        return self.trained.threshold

    @property
    def config(self) -> GNNConfig:
        return self.trained.config

    @property
    def metrics(self) -> dict[str, float]:
        """Test metrics measured at training time (empty for a model trained elsewhere)."""
        return self.trained.metrics

    def describe(self) -> dict[str, Any]:
        """Summary used by the CLI (``amlgm train``) and the REST API (``/health``)."""
        return {
            "model_type": self.model_type,
            "kind": self.config.kind,
            "threshold": self.threshold,
            "parameters": count_parameters(self.trained.model),
            "best_epoch": self.trained.best_epoch,
            "inference": self.config.inference,
            "metrics": self.metrics,
        }

    def score(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.Series:
        """Laundering probability of each transaction (index = ``df.index``)."""
        values = score_transactions(self.trained, df, context)
        return pd.Series(values, index=df.index, name="score")

    def predict(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.DataFrame:
        """Scores plus the binary alert (``score >= threshold``)."""
        scores = self.score(df, context)
        return pd.DataFrame({"score": scores, "alert": (scores >= self.threshold).astype(int)})

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        return save_checkpoint(self.trained, directory)

    @classmethod
    def load(cls, directory: str | Path, map_location: str = "cpu") -> GNNDetector:
        return cls(load_checkpoint(directory, map_location))

    def top(
        self, df: pd.DataFrame, k: int = 20, context: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """The ``k`` most suspicious transactions of ``df``, with their score."""
        scores = self.score(df, context)
        order = np.argsort(-scores.to_numpy(), kind="stable")[:k]
        return df.iloc[order].assign(score=scores.iloc[order])


__all__ = ["GNNDetector"]
