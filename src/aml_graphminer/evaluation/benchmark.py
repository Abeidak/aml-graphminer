"""Reproducible benchmark runner: structural motifs vs tabular baselines vs GNNs.

The protocol is the one of the original notebooks and of the IBM AML paper:

* temporal split 60 / 20 / 20 (no shuffling);
* features and node aggregates fitted on the **training** period only;
* decision threshold chosen on the **validation** split (F1-optimal), metrics reported on **test**;
* several seeds → mean ± std.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..detectors.baselines import BASELINE_KINDS, TabularBaseline
from ..detectors.patterns import PatternConfig, PatternScanner, pattern_recall_table
from ..exceptions import ConfigError
from ..features.pipeline import FeatureConfig, FeaturePipeline, FeatureSet
from ..tracking import Tracker
from ..utils import logger, set_seed
from .metrics import aggregate_runs, best_threshold, compute_metrics, to_markdown
from .splits import temporal_split

GNN_KINDS = ("gcn", "sage", "gat")
UNSUPERVISED_KINDS = ("gae",)  # graph auto-encoder: trained without labels
ALL_MODELS = ("patterns", *BASELINE_KINDS, *GNN_KINDS, *UNSUPERVISED_KINDS)


@dataclass
class BenchmarkReport:
    """Everything a benchmark run produces."""

    results: pd.DataFrame
    per_pattern: pd.DataFrame | None = None
    config: dict[str, Any] = field(default_factory=dict)
    fitted: dict[str, Any] = field(default_factory=dict)

    def summary(
        self, metrics: Iterable[str] = ("pr_auc", "f1", "precision", "recall", "recall_at_k")
    ) -> pd.DataFrame:
        return aggregate_runs(self.results, metrics=metrics)

    def to_markdown(self) -> str:
        return to_markdown(self.summary())

    def save(self, directory: str | Path) -> Path:
        """Write ``results.csv``, ``summary.csv``, ``summary.md``, ``config.json`` (+ per-pattern recall)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(directory / "results.csv", index=False)
        summary = self.summary()
        summary.to_csv(directory / "summary.csv")
        (directory / "summary.md").write_text(self.to_markdown() + "\n", encoding="utf-8")
        (directory / "config.json").write_text(
            json.dumps(self.config, indent=2, default=str), encoding="utf-8"
        )
        if self.per_pattern is not None:
            self.per_pattern.to_csv(directory / "per_pattern_recall.csv", index=False)
        return directory


def _validate_inputs(df: pd.DataFrame, masks: dict[str, np.ndarray]) -> np.ndarray:
    if df["label"].isna().any():
        raise ConfigError("the benchmark needs fully labelled transactions (column 'label')")
    y = df["label"].to_numpy(dtype="float64")
    for name in ("train", "val", "test"):
        if y[masks[name]].sum() == 0:
            raise ConfigError(
                f"the {name} split contains no laundering transaction; use more data, "
                "different fractions or a higher illicit ratio"
            )
    return y


def run_benchmark(
    df: pd.DataFrame,
    models: Sequence[str] = ("patterns", "rf", "hgb"),
    seeds: Sequence[int] = (0,),
    *,
    dataset: str = "dataset",
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    feature_config: FeatureConfig | None = None,
    pattern_config: PatternConfig | None = None,
    gnn_config: Any = None,
    baseline_params: dict[str, dict[str, Any]] | None = None,
    tabular_node_features: bool = True,
    tracker: Tracker | None = None,
) -> BenchmarkReport:
    """Run the requested models on ``df`` and collect test metrics.

    Parameters
    ----------
    models:
        Any of ``"patterns"``, ``"rf"``, ``"hgb"``, ``"xgb"``, ``"logreg"``, ``"gcn"``,
        ``"sage"``, ``"gat"`` and ``"gae"`` (unsupervised graph auto-encoder, encoder
        ``gnn_config.kind``). GNNs need the ``gnn`` extra.
    seeds:
        Seeds for the stochastic models (the pattern scanner is deterministic and runs once).
    gnn_config:
        :class:`aml_graphminer.detectors.gnn.GNNConfig` (defaults used when ``None``).
    tabular_node_features:
        Append the account-level features of sender and receiver to the transaction
        features of the tabular baselines (``False`` → edge features only).
    """
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise ConfigError(f"unknown model(s) {unknown}; available: {ALL_MODELS}")
    tracker = tracker or Tracker()
    masks = temporal_split(df, fractions)
    y = _validate_inputs(df, masks)
    feature_config = feature_config or FeatureConfig()
    baseline_params = baseline_params or {}

    pipeline = FeaturePipeline(feature_config).fit(df, masks["train"])
    features = pipeline.transform_splits(df, masks)
    tracker.log_params(
        {
            "dataset": dataset,
            "n_transactions": len(df),
            "n_accounts": features["test"].n_nodes,
            "fractions": fractions,
            "features": feature_config.to_dict(),
        }
    )

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {"pipeline": pipeline}
    per_pattern: pd.DataFrame | None = None

    for model in models:
        run_seeds: Sequence[int] = (seeds[0],) if model == "patterns" else seeds
        for seed in run_seeds:
            set_seed(seed)
            t0 = time.perf_counter()
            if model == "patterns":
                scores_val, scores_test, extra = _run_patterns(df, masks, pattern_config)
                per_pattern = extra.pop("per_pattern", None)
                fitted["patterns"] = extra.pop("scanner")
            elif model in BASELINE_KINDS:
                scores_val, scores_test, extra = _run_tabular(
                    model,
                    features,
                    y,
                    masks,
                    seed,
                    baseline_params.get(model, {}),
                    with_nodes=tabular_node_features,
                )
                fitted[model] = extra.pop("baseline")
            elif model in UNSUPERVISED_KINDS:
                scores_val, scores_test, extra = _run_gae(
                    df, features, masks, gnn_config, seed, tracker, pipeline.to_dict()
                )
                fitted[model] = extra.pop("trained", None)
            else:
                scores_val, scores_test, extra = _run_gnn(
                    model, df, features, masks, gnn_config, seed, tracker, pipeline.to_dict()
                )
                fitted[model] = extra.pop("trained", None)
            fit_seconds = time.perf_counter() - t0
            thr = best_threshold(y[masks["val"]], scores_val)
            metrics = compute_metrics(y[masks["test"]], scores_test, thr)
            row = {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "run_seconds": round(fit_seconds, 3),
                "n_train": int(masks["train"].sum()),
                "n_test": int(masks["test"].sum()),
                **metrics,
                **extra,
            }
            rows.append(row)
            tracker.log_metrics(
                {f"{model}/{k}": v for k, v in metrics.items() if isinstance(v, float)}
            )
            logger.info(
                "%s seed=%s pr_auc=%.4f f1=%.4f", model, seed, metrics["pr_auc"], metrics["f1"]
            )

    report = BenchmarkReport(
        results=pd.DataFrame(rows),
        per_pattern=per_pattern,
        config={
            "dataset": dataset,
            "models": list(models),
            "seeds": list(seeds),
            "fractions": list(fractions),
            "features": feature_config.to_dict(),
            "tabular_node_features": tabular_node_features,
            "patterns": (pattern_config or PatternConfig()).model_dump(),
            "gnn": getattr(gnn_config, "to_dict", lambda: None)(),
            "split_sizes": {k: int(v.sum()) for k, v in masks.items()},
            "illicit_in_split": {k: int(y[v].sum()) for k, v in masks.items()},
        },
        fitted=fitted,
    )
    return report


# --------------------------------------------------------------------------- runners
def _run_patterns(
    df: pd.DataFrame, masks: dict[str, np.ndarray], config: PatternConfig | None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Calibrate on the validation slice, scan validation and test slices independently."""
    scanner = PatternScanner(config)
    val_df, test_df = df[masks["val"]], df[masks["test"]]
    scanner.fit_calibration(val_df)
    val_scores, _ = scanner.scan(val_df).transaction_scores()
    test_result = scanner.scan(test_df)
    test_scores, _ = test_result.transaction_scores()
    extra: dict[str, Any] = {"scanner": scanner, "n_alerts": len(test_result)}
    if (test_df["pattern"] != "").any():
        extra["per_pattern"] = pattern_recall_table(test_df, test_result)
    return val_scores, test_scores, extra


def _run_tabular(
    kind: str,
    features: dict[str, FeatureSet],
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    seed: int,
    params: dict[str, Any],
    *,
    with_nodes: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Train on the train split, score val / test with the node snapshot of each split."""
    cols = features["train"].tabular_columns(with_nodes)
    baseline = TabularBaseline(kind, seed=seed, **params).fit(
        features["train"].tabular(masks["train"], with_nodes), y[masks["train"]], cols
    )
    return (
        baseline.predict_proba(features["val"].tabular(masks["val"], with_nodes)),
        baseline.predict_proba(features["test"].tabular(masks["test"], with_nodes)),
        {"baseline": baseline},
    )


def _run_gnn(
    kind: str,
    df: pd.DataFrame,
    features: dict[str, FeatureSet],
    masks: dict[str, np.ndarray],
    gnn_config: Any,
    seed: int,
    tracker: Tracker,
    pipeline_stats: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:  # pragma: no cover - needs torch
    from ..detectors.gnn import run_gnn_experiment

    out = run_gnn_experiment(
        kind, df, features, masks, gnn_config, seed, tracker=tracker, pipeline_stats=pipeline_stats
    )
    return (
        out["scores_val"],
        out["scores_test"],
        {"trained": out["trained"], "best_epoch": out["best_epoch"]},
    )


def _run_gae(
    df: pd.DataFrame,
    features: dict[str, FeatureSet],
    masks: dict[str, np.ndarray],
    gnn_config: Any,
    seed: int,
    tracker: Tracker,
    pipeline_stats: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:  # pragma: no cover - needs torch
    """Unsupervised graph auto-encoder: no label is used for training (only for the threshold)."""
    from ..detectors.gnn import run_gae_experiment

    out = run_gae_experiment(
        df, features, masks, gnn_config, seed, tracker=tracker, pipeline_stats=pipeline_stats
    )
    return (
        out["scores_val"],
        out["scores_test"],
        {"trained": out["trained"], "best_epoch": out["best_epoch"]},
    )
