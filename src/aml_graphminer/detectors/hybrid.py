"""Hybrid detector: a learned model and the structural-motif scanner, stacked on validation.

The two views are complementary: the model ranks every transaction from its features, the motif
scanner raises few alerts but with auditable evidence. :class:`HybridDetector` combines the
model's score with the scanner's calibrated alert score through a logistic regression learned on
the **validation** slice (three inputs: model logit, motif score, motif fired or not), so the
test slice stays untouched. The base model can be any saved detector (tabular, GNN, ...).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.metrics import best_threshold, compute_metrics
from ..evaluation.splits import temporal_split
from ..exceptions import ConfigError, ModelError
from .patterns import PatternConfig

EPS = 1e-6
INPUTS = ("model_logit", "motif_score", "motif_fired")


def stack_inputs(model_scores: np.ndarray, motif_scores: np.ndarray) -> np.ndarray:
    """Design matrix of the stacker: ``[logit(model score), motif score, motif fired]``."""
    p = np.clip(np.asarray(model_scores, dtype="float64"), EPS, 1.0 - EPS)
    m = np.asarray(motif_scores, dtype="float64")
    return np.column_stack([np.log(p / (1.0 - p)), m, (m > 0).astype("float64")])


def fit_stacker(
    y: np.ndarray, model_scores: np.ndarray, motif_scores: np.ndarray, c: float = 1.0
) -> tuple[np.ndarray, float]:
    """Logistic regression of the labels on :func:`stack_inputs`; returns ``(coef, intercept)``."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype="float64")
    if y.sum() == 0 or y.sum() == len(y):
        raise ConfigError("the stacker needs both classes in the validation slice")
    lr = LogisticRegression(C=c, max_iter=1000)
    lr.fit(stack_inputs(model_scores, motif_scores), y)
    return lr.coef_[0].astype("float64"), float(lr.intercept_[0])


def apply_stacker(
    coef: np.ndarray, intercept: float, model_scores: np.ndarray, motif_scores: np.ndarray
) -> np.ndarray:
    """Hybrid score (probability) of every transaction."""
    z = stack_inputs(model_scores, motif_scores) @ np.asarray(coef, dtype="float64") + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


class HybridDetector:
    """Stack a trained detector with the motif scanner (``amlgm train --model hybrid``)."""

    model_type = "hybrid"

    def __init__(
        self,
        base: Any,
        motifs: Any,
        coef: np.ndarray,
        intercept: float,
        threshold: float = 0.5,
        metrics: dict[str, float] | None = None,
        component_metrics: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.base = base
        self.motifs = motifs
        self.coef = np.asarray(coef, dtype="float64")
        self.intercept = float(intercept)
        self.threshold = float(threshold)
        self.metrics = metrics or {}
        # test metrics of the base model alone and of the motifs alone, on the same scores
        self.component_metrics = component_metrics or {}

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        base: Any,
        config: PatternConfig | None = None,
        *,
        fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
        motifs: Any = None,
    ) -> HybridDetector:
        """Calibrate the motifs and learn the stacker on the validation slice of ``df``.

        ``base`` must have been trained on the training slice of the same temporal split
        (``fractions``), otherwise its validation scores would be optimistic. Each slice is
        scored with the earlier transactions as context; the reported metrics are on the test
        slice.
        """
        from .bundle import MotifDetector

        if df["label"].isna().any():
            raise ConfigError("the hybrid detector needs fully labelled transactions")
        masks = temporal_split(df, fractions)
        motifs = motifs or MotifDetector.fit(df, config, fractions=fractions)
        y = df["label"].to_numpy(dtype="float64")
        scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for part, before in (("val", masks["train"]), ("test", masks["train"] | masks["val"])):
            rows, context = df[masks[part]], df[before]
            scores[part] = (
                base.score(rows, context).to_numpy(),
                motifs.score(rows, context).to_numpy(),
            )
        y_val, y_test = y[masks["val"]], y[masks["test"]]

        def test_metrics(val_scores: np.ndarray, test_scores: np.ndarray) -> dict[str, float]:
            threshold = best_threshold(y_val, val_scores)
            return {k: float(v) for k, v in compute_metrics(y_test, test_scores, threshold).items()}

        coef, intercept = fit_stacker(y_val, *scores["val"])
        val = apply_stacker(coef, intercept, *scores["val"])
        test = apply_stacker(coef, intercept, *scores["test"])
        components = {
            "base": test_metrics(scores["val"][0], scores["test"][0]),
            "motifs": test_metrics(scores["val"][1], scores["test"][1]),
        }
        threshold = best_threshold(y_val, val)
        return cls(base, motifs, coef, intercept, threshold, test_metrics(val, test), components)

    def score(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.Series:
        if len(df) == 0:
            return pd.Series(dtype="float64", name="score")
        model = self.base.score(df, context).to_numpy()
        motif = self.motifs.score(df, context).to_numpy()
        values = apply_stacker(self.coef, self.intercept, model, motif)
        return pd.Series(values, index=df.index, name="score")

    def predict(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.DataFrame:
        scores = self.score(df, context)
        return pd.DataFrame({"score": scores, "alert": (scores >= self.threshold).astype(int)})

    def save(self, directory: str | Path) -> Path:
        from .bundle import _write_config

        directory = Path(directory)
        self.base.save(directory / "base")
        self.motifs.save(directory / "motifs")
        return _write_config(
            directory,
            {
                "model_type": self.model_type,
                "base_model_type": self.base.describe().get("model_type"),
                "stacker": {
                    "inputs": list(INPUTS),
                    "coef": self.coef.tolist(),
                    "intercept": self.intercept,
                },
                "threshold": self.threshold,
                "metrics": self.metrics,
                "component_metrics": self.component_metrics,
            },
        )

    @classmethod
    def _from_payload(cls, directory: Path, payload: dict[str, Any]) -> HybridDetector:
        from .bundle import load_detector

        for part in ("base", "motifs"):
            if not (directory / part).is_dir():
                raise ModelError(f"{directory / part} is missing")
        stacker = payload.get("stacker") or {}
        return cls(
            load_detector(directory / "base"),
            load_detector(directory / "motifs"),
            np.asarray(stacker.get("coef", [1.0, 0.0, 0.0]), dtype="float64"),
            float(stacker.get("intercept", 0.0)),
            float(payload.get("threshold", 0.5)),
            payload.get("metrics", {}),
            payload.get("component_metrics", {}),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "base": self.base.describe(),
            "motif_detectors": self.motifs.describe().get("detectors", []),
            "stacker": dict(zip(INPUTS, np.round(self.coef, 4).tolist())),
            "threshold": self.threshold,
            "metrics": self.metrics,
            "component_metrics": self.component_metrics,
        }


__all__ = ["HybridDetector", "apply_stacker", "fit_stacker", "stack_inputs"]
