"""Evaluation of the deterministic detectors against ground-truth labels and motif types."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...evaluation.metrics import compute_metrics
from .scanner import ScanResult


def evaluate_scan(df: pd.DataFrame, result: ScanResult) -> dict[str, Any]:
    """Transaction-level metrics of the scanner used as a binary / scoring classifier.

    * ranking metrics (PR-AUC, recall@k) use the alert score of every transaction;
    * ``precision`` / ``recall`` / ``f1`` use the decision rule *"flagged by at least one alert"*.
    """
    scores, _ = result.transaction_scores()
    y = df["label"].to_numpy(dtype="float64")
    return compute_metrics(y, scores, threshold=np.finfo(float).tiny)


def pattern_recall_table(df: pd.DataFrame, result: ScanResult) -> pd.DataFrame:
    """Per ground-truth motif: how many of its transactions are covered by an alert?

    ``recall_any`` counts coverage by *any* alert, ``recall_same`` only by an alert of the
    same detector (e.g. true ``fan_in`` covered by the ``fan_in`` detector).
    """
    if "pattern" not in df.columns:
        raise ValueError("the table has no 'pattern' column")
    index = pd.Index(result.tx_ids)
    covered_any = np.zeros(len(df), dtype=bool)
    covered_by: dict[str, np.ndarray] = {}
    for alert in result.alerts:
        pos = index.get_indexer(alert.tx_ids)
        pos = pos[pos >= 0]
        covered_any[pos] = True
        mask = covered_by.setdefault(alert.pattern, np.zeros(len(df), dtype=bool))
        mask[pos] = True
    rows = []
    for name, group in df[df["pattern"] != ""].groupby("pattern"):
        pos = index.get_indexer(group.index)
        same = covered_by.get(str(name), np.zeros(len(df), dtype=bool))
        rows.append(
            {
                "pattern": name,
                "n_tx": len(group),
                "recall_any": float(covered_any[pos].mean()),
                "recall_same": float(same[pos].mean()),
            }
        )
    return pd.DataFrame(rows, columns=["pattern", "n_tx", "recall_any", "recall_same"])


def alert_precision_table(df: pd.DataFrame, result: ScanResult) -> pd.DataFrame:
    """Per detector: number of alerts, covered transactions and share that is truly illicit."""
    index = pd.Index(result.tx_ids)
    labels = df["label"].fillna(0).to_numpy()
    rows = []
    names = sorted({a.pattern for a in result.alerts})
    for name in names:
        alerts = [a for a in result.alerts if a.pattern == name]
        covered = np.unique(np.concatenate([index.get_indexer(a.tx_ids) for a in alerts]))
        covered = covered[covered >= 0]
        illicit_share = [float(labels[index.get_indexer(a.tx_ids)].mean()) for a in alerts]
        rows.append(
            {
                "pattern": name,
                "n_alerts": len(alerts),
                "n_tx_covered": len(covered),
                "tx_precision": float(labels[covered].mean()) if len(covered) else float("nan"),
                "alert_precision": float(np.mean(np.asarray(illicit_share) >= 0.5)),
            }
        )
    return pd.DataFrame(
        rows, columns=["pattern", "n_alerts", "n_tx_covered", "tx_precision", "alert_precision"]
    )
