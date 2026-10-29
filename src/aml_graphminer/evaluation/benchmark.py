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
from ..detectors.hybrid import INPUTS as STACK_INPUTS
from ..detectors.hybrid import apply_stacker, fit_stacker
from ..detectors.patterns import PatternConfig, PatternScanner, pattern_recall_table
from ..detectors.patterns.scanner import lookback_history
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

    def __repr__(self) -> str:  # one line: notebooks store it next to the HTML view
        models = self.results["model"].nunique() if "model" in self.results else 0
        return f"BenchmarkReport({models} models, {len(self.results)} runs)"

    def _repr_html_(self) -> str:  # notebooks: the summary table
        from ..jupyter.display import benchmark_html

        return benchmark_html(self)


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
        ``gnn_config.kind``), and ``"hybrid-<model>"``: that model stacked with the motif scanner
        by a logistic regression fitted on the validation slice (scores of the model and of the
        scanner are reused when they are also requested). GNNs need the ``gnn`` extra.
    seeds:
        Seeds for the stochastic models (the pattern scanner is deterministic and runs once).
    gnn_config:
        :class:`aml_graphminer.detectors.gnn.GNNConfig` (defaults used when ``None``).
    tabular_node_features:
        Append the account-level features of sender and receiver to the transaction
        features of the tabular baselines (``False`` → edge features only).
    """
    unknown = [m for m in models if m not in ALL_MODELS and _hybrid_base(m) is None]
    if unknown:
        raise ConfigError(
            f"unknown model(s) {unknown}; available: {ALL_MODELS} and hybrid-<model> "
            "(a model stacked with the motif scanner)"
        )
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

    computed: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}

    def run(model: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Validation / test scores of one model and seed (each computed once)."""
        nonlocal per_pattern
        key = (model, seeds[0] if model == "patterns" else seed)
        if key in computed:
            return computed[key]
        set_seed(seed)
        base = _hybrid_base(model)
        if base is not None:
            base_val, base_test, _ = run(base, seed)
            motif_val, motif_test, _ = run("patterns", seed)
            coef, intercept = fit_stacker(y[masks["val"]], base_val, motif_val)
            scores_val = apply_stacker(coef, intercept, base_val, motif_val)
            scores_test = apply_stacker(coef, intercept, base_test, motif_test)
            extra: dict[str, Any] = {"stacker": dict(zip(STACK_INPUTS, np.round(coef, 4)))}
            fitted[model] = {"coef": coef, "intercept": intercept}
        elif model == "patterns":
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
        computed[key] = (scores_val, scores_test, extra)
        return computed[key]

    for model in models:
        run_seeds: Sequence[int] = (seeds[0],) if model == "patterns" else seeds
        for seed in run_seeds:
            t0 = time.perf_counter()
            scores_val, scores_test, extra = run(model, seed)
            extra = dict(extra)
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
def _hybrid_base(model: str) -> str | None:
    """``"hybrid-hgb"`` → ``"hgb"`` (``None`` when ``model`` is not a valid hybrid name)."""
    if not model.startswith("hybrid-"):
        return None
    base = model.removeprefix("hybrid-")
    return base if base in ALL_MODELS and base != "patterns" else None


def _run_patterns(
    df: pd.DataFrame, masks: dict[str, np.ndarray], config: PatternConfig | None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Calibrate on the validation slice, then score the validation and test slices.

    Each slice is scanned with the transactions of the ``lookback`` period before it as
    unlabelled context (``PatternConfig.lookback``), as a deployed scanner would see them.
    """
    scanner = PatternScanner(config)
    val_df, test_df = df[masks["val"]], df[masks["test"]]
    val_hist, test_hist = (
        lookback_history(df, part, scanner.config.lookback) for part in (val_df, test_df)
    )
    scanner.fit_calibration(val_df, history=val_hist)
    val_scores, _ = scanner.scan(val_df, val_hist).transaction_scores()
    test_result = scanner.scan(test_df, test_hist)
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
