"""Tabular baselines (Random Forest, gradient boosting) on edge + node features.

The CDC asks to compare the GNNs with RF / XGBoost: these baselines use the very same
features as the GNNs (``[edge | node[src] | node[dst]]``), so the comparison isolates
what message passing adds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..exceptions import ConfigError, ModelError
from ..utils import require

BASELINE_KINDS = ("rf", "hgb", "xgb", "logreg")


class TabularBaseline:
    """Thin wrapper giving every baseline the same ``fit / predict_proba / save / load`` API.

    Parameters
    ----------
    kind:
        ``"rf"`` (RandomForest), ``"hgb"`` (scikit-learn HistGradientBoosting), ``"xgb"``
        (XGBoost, needs the ``xgboost`` extra) or ``"logreg"``.
    seed:
        Random seed.
    params:
        Extra keyword arguments forwarded to the underlying estimator.
    """

    def __init__(self, kind: str = "rf", seed: int = 0, **params: Any) -> None:
        if kind not in BASELINE_KINDS:
            raise ConfigError(f"unknown baseline {kind!r}; choose among {BASELINE_KINDS}")
        self.kind, self.seed, self.params = kind, seed, params
        self.model: Any = None
        self.columns: list[str] | None = None

    def _build(self, y: np.ndarray) -> Any:
        if self.kind == "rf":
            defaults: dict[str, Any] = dict(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=self.seed,
            )
            return RandomForestClassifier(**{**defaults, **self.params})
        if self.kind == "hgb":
            defaults = dict(
                max_iter=200,
                learning_rate=0.1,
                class_weight="balanced",
                random_state=self.seed,
            )
            return HistGradientBoostingClassifier(**{**defaults, **self.params})
        if self.kind == "logreg":
            defaults = dict(max_iter=500, class_weight="balanced", random_state=self.seed)
            return LogisticRegression(**{**defaults, **self.params})
        xgb = require("xgboost", "xgboost", "the XGBoost baseline")
        pos = max(float(y.sum()), 1.0)
        defaults = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=float(len(y) - pos) / pos,
            tree_method="hist",
            eval_metric="aucpr",
            random_state=self.seed,
            n_jobs=-1,
        )
        return xgb.XGBClassifier(**{**defaults, **self.params})

    def fit(
        self, X: np.ndarray, y: np.ndarray, columns: list[str] | None = None
    ) -> TabularBaseline:
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            raise ModelError("the training labels contain a single class; cannot fit a classifier")
        self.model = self._build(y)
        self.model.fit(X, y)
        self.columns = columns
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ModelError("baseline is not fitted")
        return np.asarray(self.model.predict_proba(X)[:, 1], dtype="float64")

    def feature_importance(self) -> dict[str, float]:
        """Impurity-based importances (RF / HGB / XGB) or ``abs(coef)`` (logreg), by column name."""
        if self.model is None:
            raise ModelError("baseline is not fitted")
        if hasattr(self.model, "feature_importances_"):
            values = np.asarray(self.model.feature_importances_, dtype="float64")
        elif hasattr(self.model, "coef_"):
            values = np.abs(np.asarray(self.model.coef_)).ravel()
        else:  # HistGradientBoosting has no impurity importances
            return {}
        names = self.columns or [f"f{i}" for i in range(len(values))]
        return dict(zip(names, values.tolist()))

    def save(self, path: str | Path) -> Path:
        """Pickle the wrapper; an XGBoost model goes to ``<path>.xgb.json`` (portable format).

        XGBoost's pickle is not portable across platforms (a model pickled on Linux may fail
        with "input stream corrupted" on Windows), its JSON model format is.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = self.model
        if self.kind == "xgb" and model is not None:
            model.save_model(str(path) + ".xgb.json")
            model = None
        joblib.dump(
            {
                "kind": self.kind,
                "seed": self.seed,
                "params": self.params,
                "model": model,
                "columns": self.columns,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> TabularBaseline:
        payload = joblib.load(path)
        obj = cls(payload["kind"], payload["seed"], **payload["params"])
        obj.model, obj.columns = payload["model"], payload["columns"]
        sidecar = Path(str(path) + ".xgb.json")
        if obj.kind == "xgb" and obj.model is None and sidecar.exists():
            xgb = require("xgboost", "xgboost", "the XGBoost baseline")
            obj.model = xgb.XGBClassifier()
            obj.model.load_model(str(sidecar))
        return obj
