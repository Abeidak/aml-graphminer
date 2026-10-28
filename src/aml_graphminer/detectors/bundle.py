"""Ready-to-use detectors with one common interface: ``fit / score / predict / save / load``.

* :class:`MotifDetector` — structural-motif scanner (no training needed, calibration optional);
* :class:`TabularDetector` — RF / HistGB / XGBoost / logistic regression on transaction + account features;
* ``GNNDetector`` — supervised GCN / GraphSAGE / GAT (``gnn`` extra);
* ``AutoEncoderDetector`` — unsupervised graph auto-encoder, trained without labels (``gnn`` extra);
* :class:`~aml_graphminer.detectors.hybrid.HybridDetector` — any of them stacked with the motifs.

:func:`load_detector` restores any of them from a directory written by ``save`` (this is what
the CLI ``score`` / ``explain`` / ``serve`` commands and the REST API use).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .. import __version__
from ..evaluation.metrics import best_threshold, compute_metrics
from ..evaluation.splits import temporal_split
from ..exceptions import ConfigError, ModelError
from ..features.pipeline import FeatureConfig, FeaturePipeline
from ._context import restore_order, stack_context
from .baselines import BASELINE_KINDS, TabularBaseline
from .patterns import PatternConfig, PatternScanner, lookback_history

BUNDLE_VERSION = 1
CONFIG_FILE = "config.json"


class Detector(Protocol):
    """What the CLI, the API and the explainers expect from a detector."""

    threshold: float

    def score(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.Series: ...

    def predict(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.DataFrame: ...

    def save(self, directory: str | Path) -> Path: ...

    def describe(self) -> dict[str, Any]: ...


def _predict(det: Any, df: pd.DataFrame, context: pd.DataFrame | None) -> pd.DataFrame:
    scores = det.score(df, context)
    return pd.DataFrame({"score": scores, "alert": (scores >= det.threshold).astype(int)})


def _write_config(directory: Path, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"bundle_version": BUNDLE_VERSION, "library_version": __version__, **payload}
    (directory / CONFIG_FILE).write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    return directory


# --------------------------------------------------------------------------- motifs
class MotifDetector:
    """Structural-motif scanner packaged as a detector (alerts → per-transaction scores)."""

    model_type = "patterns"

    def __init__(
        self,
        scanner: PatternScanner,
        threshold: float = 0.5,
        metrics: dict[str, float] | None = None,
    ):
        self.scanner = scanner
        self.threshold = float(threshold)
        self.metrics = metrics or {}

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame | None = None,
        config: PatternConfig | None = None,
        *,
        fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    ) -> MotifDetector:
        """Build the scanner; with labelled data calibrate it on the validation slice.

        Calibration learns the precision of each motif and the F1-optimal threshold on the
        validation slice; the test slice is only used for the reported metrics.
        """
        scanner = PatternScanner(config)
        if df is None or df["label"].isna().all():
            return cls(scanner)
        masks = temporal_split(df, fractions)
        val, test = df[masks["val"]], df[masks["test"]]
        lookback = scanner.config.lookback
        val_hist, test_hist = (lookback_history(df, part, lookback) for part in (val, test))
        scanner.fit_calibration(val, history=val_hist)
        val_scores, _ = scanner.scan(val, val_hist).transaction_scores()
        threshold = (
            best_threshold(val["label"].to_numpy(), val_scores) if val["label"].sum() else 0.5
        )
        test_scores, _ = scanner.scan(test, test_hist).transaction_scores()
        metrics = (
            {
                k: float(v)
                for k, v in compute_metrics(
                    test["label"].to_numpy(), test_scores, threshold
                ).items()
            }
            if test["label"].sum()
            else {}
        )
        return cls(scanner, threshold, metrics)

    def score(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.Series:
        if len(df) == 0:
            return pd.Series(dtype="float64", name="score")
        lookback = self.scanner.config.lookback
        if context is not None and lookback is not None:  # older context cannot change a motif
            context = lookback_history(pd.concat([context, df]), df, lookback)
        full, is_target, target_pos = stack_context(df, context)
        scores, _ = self.scanner.scan(full).transaction_scores()
        values = restore_order(scores, is_target, target_pos, len(df))
        return pd.Series(values, index=df.index, name="score")

    def predict(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.DataFrame:
        return _predict(self, df, context)

    def save(self, directory: str | Path) -> Path:
        return _write_config(
            Path(directory),
            {
                "model_type": self.model_type,
                "patterns": self.scanner.config.model_dump(),
                "calibration": self.scanner.calibration,
                "calibration_bins": {
                    k: [list(e), list(p)] for k, (e, p) in self.scanner.bin_calibration.items()
                },
                "threshold": self.threshold,
                "metrics": self.metrics,
            },
        )

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> MotifDetector:
        scanner = PatternScanner(PatternConfig.from_dict(payload.get("patterns")))
        scanner.calibration = {k: float(v) for k, v in payload.get("calibration", {}).items()}
        scanner.bin_calibration = {
            k: ([float(x) for x in e], [float(x) for x in p])
            for k, (e, p) in payload.get("calibration_bins", {}).items()
        }
        return cls(scanner, float(payload.get("threshold", 0.5)), payload.get("metrics", {}))

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "detectors": sorted(self.scanner.detectors),
            "threshold": self.threshold,
            "calibrated": bool(self.scanner.calibration),
            "metrics": self.metrics,
        }


# --------------------------------------------------------------------------- tabular
class TabularDetector:
    """Tabular baseline + the feature pipeline it was trained with."""

    model_type = "tabular"

    def __init__(
        self,
        pipeline: FeaturePipeline,
        baseline: TabularBaseline,
        threshold: float = 0.5,
        with_nodes: bool = True,
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.baseline = baseline
        self.threshold = float(threshold)
        self.with_nodes = with_nodes
        self.metrics = metrics or {}

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        kind: str = "rf",
        *,
        fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
        feature_config: FeatureConfig | None = None,
        seed: int = 0,
        with_nodes: bool = True,
        params: dict[str, Any] | None = None,
    ) -> TabularDetector:
        """Temporal split → features fitted on train → model → threshold on validation."""
        if kind not in BASELINE_KINDS:
            raise ConfigError(f"unknown baseline {kind!r}; choose among {BASELINE_KINDS}")
        if df["label"].isna().any():
            raise ConfigError("training needs fully labelled transactions (column 'label')")
        masks = temporal_split(df, fractions)
        y = df["label"].to_numpy(dtype="float64")
        pipeline = FeaturePipeline(feature_config).fit(df, masks["train"])
        feats = pipeline.transform_splits(df, masks)
        baseline = TabularBaseline(kind, seed=seed, **(params or {})).fit(
            feats["train"].tabular(masks["train"], with_nodes),
            y[masks["train"]],
            feats["train"].tabular_columns(with_nodes),
        )
        val = baseline.predict_proba(feats["val"].tabular(masks["val"], with_nodes))
        test = baseline.predict_proba(feats["test"].tabular(masks["test"], with_nodes))
        threshold = best_threshold(y[masks["val"]], val)
        metrics = {
            k: float(v) for k, v in compute_metrics(y[masks["test"]], test, threshold).items()
        }
        return cls(pipeline, baseline, threshold, with_nodes, metrics)

    def score(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.Series:
        if len(df) == 0:
            return pd.Series(dtype="float64", name="score")
        full, is_target, target_pos = stack_context(df, context)
        feats = self.pipeline.transform(full)
        probs = np.zeros(len(full))
        probs[is_target] = self.baseline.predict_proba(feats.tabular(is_target, self.with_nodes))
        return pd.Series(
            restore_order(probs, is_target, target_pos, len(df)), index=df.index, name="score"
        )

    def predict(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.DataFrame:
        return _predict(self, df, context)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.baseline.save(directory / "baseline.joblib")
        return _write_config(
            directory,
            {
                "model_type": self.model_type,
                "kind": self.baseline.kind,
                "with_nodes": self.with_nodes,
                "threshold": self.threshold,
                "metrics": self.metrics,
                "stats": self.pipeline.to_dict(),
            },
        )

    @classmethod
    def _from_payload(cls, directory: Path, payload: dict[str, Any]) -> TabularDetector:
        weights = directory / "baseline.joblib"
        if not weights.exists():
            raise ModelError(f"{weights} is missing")
        return cls(
            FeaturePipeline.from_dict(payload["stats"]),
            TabularBaseline.load(weights),
            float(payload.get("threshold", 0.5)),
            bool(payload.get("with_nodes", True)),
            payload.get("metrics", {}),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "kind": self.baseline.kind,
            "threshold": self.threshold,
            "n_features": len(self.baseline.columns or []),
            "metrics": self.metrics,
        }


# --------------------------------------------------------------------------- loading
def read_bundle_config(directory: str | Path) -> dict[str, Any]:
    """Parse ``config.json`` of a saved detector."""
    path = Path(directory) / CONFIG_FILE
    if not path.exists():
        raise ModelError(f"{directory} is not a saved detector (no {CONFIG_FILE})")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ModelError(f"{path} is not a valid detector configuration")
    return data


def load_detector(directory: str | Path) -> Any:
    """Restore any detector saved with ``save`` (motifs, tabular, GNN, auto-encoder, hybrid)."""
    directory = Path(directory)
    payload = read_bundle_config(directory)
    kind = payload.get("model_type") or ("gnn" if payload.get("model") == "EdgeGNN" else None)
    if kind == "patterns":
        return MotifDetector._from_payload(payload)
    if kind == "tabular":
        return TabularDetector._from_payload(directory, payload)
    if kind == "gnn":
        from .gnn import GNNDetector

        return GNNDetector.load(directory)
    if kind == "gnn_unsupervised":
        from .gnn import AutoEncoderDetector

        return AutoEncoderDetector.load(directory)
    if kind == "hybrid":
        from .hybrid import HybridDetector

        return HybridDetector._from_payload(directory, payload)
    raise ModelError(f"unknown detector type {kind!r} in {directory / CONFIG_FILE}")


__all__ = [
    "BUNDLE_VERSION",
    "CONFIG_FILE",
    "Detector",
    "MotifDetector",
    "TabularDetector",
    "load_detector",
    "read_bundle_config",
]
