"""Evaluation metrics for highly imbalanced detection (PR-AUC, F1, recall@k, ...)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _clean(y: Any, p: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    if y.shape != p.shape:
        raise ValueError(f"y and p must have the same shape, got {y.shape} and {p.shape}")
    keep = ~np.isnan(y)
    return y[keep].astype(int), np.nan_to_num(p[keep], nan=0.0)


def best_threshold(y: Any, p: Any) -> float:
    """Score threshold maximising F1 (choose it on the *validation* split)."""
    y, p = _clean(y, p)
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def recall_at_k(y: Any, p: Any, k: int) -> float:
    """Fraction of all positives found among the ``k`` highest scores."""
    y, p = _clean(y, p)
    if y.sum() == 0 or k <= 0:
        return float("nan")
    top = np.argsort(-p, kind="stable")[: int(k)]
    return float(y[top].sum() / y.sum())


def precision_at_k(y: Any, p: Any, k: int) -> float:
    """Fraction of the ``k`` highest scores that are positives."""
    y, p = _clean(y, p)
    if len(y) == 0 or k <= 0:
        return float("nan")
    top = np.argsort(-p, kind="stable")[: int(k)]
    return float(y[top].mean())


def compute_metrics(
    y: Any,
    p: Any,
    threshold: float | None = None,
    ks: Sequence[int] = (100, 1000),
    alert_rates: Sequence[float] = (0.001, 0.01),
) -> dict[str, float]:
    """All metrics reported in the benchmarks.

    Notes
    -----
    * ``recall_at_k`` follows the original notebooks: ``k`` = number of positives, so it is the
      R-precision (precision@P, which equals recall@P).
    * ``recall_at_<k>`` uses an explicit alert budget of ``k`` transactions;
      ``recall_at_<rate>pct`` flags the top ``rate`` fraction of transactions.
    * When ``threshold`` is ``None`` it is chosen on the data given (do not report that as a
      test result: pass the validation threshold).
    """
    y_arr, p_arr = _clean(y, p)
    n_pos = int(y_arr.sum())
    thr = best_threshold(y_arr, p_arr) if threshold is None else float(threshold)
    pred = (p_arr >= thr).astype(int)
    tp = int(((pred == 1) & (y_arr == 1)).sum())
    fp = int(((pred == 1) & (y_arr == 0)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    nan = float("nan")
    out: dict[str, float] = {
        "threshold": thr,
        "n": float(len(y_arr)),
        "n_pos": float(n_pos),
        "pr_auc": float(average_precision_score(y_arr, p_arr)) if n_pos else nan,
        "roc_auc": float(roc_auc_score(y_arr, p_arr)) if 0 < n_pos < len(y_arr) else nan,
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "precision": float(precision),
        "recall": float(recall),
        "recall_at_k": precision_at_k(y_arr, p_arr, max(n_pos, 1)) if n_pos else nan,
    }
    for k in ks:
        out[f"recall_at_{k}"] = recall_at_k(y_arr, p_arr, k)
    for rate in alert_rates:
        out[f"recall_at_{rate * 100:g}pct"] = recall_at_k(
            y_arr, p_arr, max(1, round(rate * len(y_arr)))
        )
    return out


def bootstrap_ci(
    y: Any,
    p: Any,
    metric: Callable[[np.ndarray, np.ndarray], float] = average_precision_score,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of a metric (default PR-AUC)."""
    y_arr, p_arr = _clean(y, p)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if y_arr[idx].sum() == 0:
            continue
        values.append(metric(y_arr[idx], p_arr[idx]))
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def aggregate_runs(
    results: pd.DataFrame,
    by: Sequence[str] = ("dataset", "model"),
    metrics: Iterable[str] = ("pr_auc", "f1", "precision", "recall", "recall_at_k"),
) -> pd.DataFrame:
    """Mean and standard deviation over seeds (``std`` is NaN with a single seed)."""
    cols = [m for m in metrics if m in results.columns]
    return results.groupby(list(by))[cols].agg(["mean", "std"])


def to_markdown(table: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a (possibly multi-index) frame as a GitHub-flavoured Markdown table."""
    frame = table.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [f"{a} ({b})" if b else a for a, b in frame.columns]
    frame = frame.reset_index()
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    lines = [header, rule]
    for _, row in frame.iterrows():
        cells = [
            format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
