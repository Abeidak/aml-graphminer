"""Benchmark on Elliptic Bitcoin (node classification): tabular baselines vs GNNs.

Protocol of Weber et al. (2019): temporal split on the *labelled* nodes, the first time steps
for training/validation and the last ones for testing; the decision threshold is selected on
the validation steps.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..detectors.baselines import BASELINE_KINDS, TabularBaseline
from ..exceptions import ConfigError
from ..ingestion.elliptic import ILLICIT, EllipticData
from ..tracking import Tracker
from ..utils import logger, set_seed
from .benchmark import GNN_KINDS, BenchmarkReport
from .metrics import best_threshold, compute_metrics

ELLIPTIC_MODELS = (*BASELINE_KINDS, *GNN_KINDS)


def run_elliptic_benchmark(
    data: EllipticData,
    models: Sequence[str] = ("rf", "hgb"),
    seeds: Sequence[int] = (0,),
    *,
    dataset: str = "elliptic",
    train_max: int = 29,
    val_max: int = 34,
    gnn_config: Any = None,
    baseline_params: dict[str, dict[str, Any]] | None = None,
    tracker: Tracker | None = None,
) -> BenchmarkReport:
    """Evaluate ``models`` on Elliptic and return a :class:`BenchmarkReport`."""
    unknown = [m for m in models if m not in ELLIPTIC_MODELS]
    if unknown:
        raise ConfigError(f"unknown model(s) {unknown}; available: {ELLIPTIC_MODELS}")
    masks = data.split_masks(train_max, val_max)
    y = (data.labels == ILLICIT).astype("float64")
    for name, mask in masks.items():
        if y[mask].sum() == 0:
            raise ConfigError(f"the {name} split contains no illicit node")
    tracker = tracker or Tracker()
    baseline_params = baseline_params or {}
    rows: list[dict[str, Any]] = []
    for model in models:
        for seed in seeds:
            set_seed(seed)
            t0 = time.perf_counter()
            extra: dict[str, Any] = {}
            if model in BASELINE_KINDS:
                clf = TabularBaseline(model, seed=seed, **baseline_params.get(model, {})).fit(
                    data.features[masks["train"]], y[masks["train"]]
                )
                val = clf.predict_proba(data.features[masks["val"]])
                test = clf.predict_proba(data.features[masks["test"]])
            else:  # pragma: no cover - needs torch
                from ..detectors.gnn import GNNConfig, run_node_experiment

                result = run_node_experiment(
                    model, data, gnn_config or GNNConfig(), seed, masks=masks, tracker=tracker
                )
                val, test = result.scores[masks["val"]], result.scores[masks["test"]]
                extra["best_epoch"] = result.best_epoch
            thr = best_threshold(y[masks["val"]], val)
            metrics = compute_metrics(y[masks["test"]], test, thr)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "run_seconds": round(time.perf_counter() - t0, 3),
                    "n_train": int(masks["train"].sum()),
                    "n_test": int(masks["test"].sum()),
                    **metrics,
                    **extra,
                }
            )
            logger.info(
                "%s seed=%s pr_auc=%.4f f1=%.4f", model, seed, metrics["pr_auc"], metrics["f1"]
            )
            tracker.log_metrics(
                {f"{model}/{k}": v for k, v in metrics.items() if isinstance(v, float)}
            )
    return BenchmarkReport(
        results=pd.DataFrame(rows),
        config={
            "dataset": dataset,
            "models": list(models),
            "seeds": list(seeds),
            "train_max": train_max,
            "val_max": val_max,
            "summary": data.summary(),
            "split_sizes": {k: int(np.sum(v)) for k, v in masks.items()},
            "gnn": getattr(gnn_config, "to_dict", lambda: None)(),
        },
    )


__all__ = ["ELLIPTIC_MODELS", "run_elliptic_benchmark"]
