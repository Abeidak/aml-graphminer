"""Run every enabled structural-motif detector and aggregate the alerts."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ...exceptions import ConfigError
from ...utils import parse_duration
from .base import Alert, PatternDetector, TxArrays
from .community import CommunityDetector
from .config import PatternConfig
from .cycles import CycleDetector
from .fan import FanInDetector, FanOutDetector
from .layering import GatherScatterDetector, ScatterGatherDetector
from .relay import RelayDetector
from .smurfing import SmurfingDetector

DetectorFactory = Callable[[PatternConfig], PatternDetector]

DETECTOR_REGISTRY: dict[str, DetectorFactory] = {
    "fan_in": lambda c: FanInDetector(c.fan_in),
    "fan_out": lambda c: FanOutDetector(c.fan_out),
    "smurfing": lambda c: SmurfingDetector(c.smurfing),
    "cycle": lambda c: CycleDetector(c.cycle),
    "scatter_gather": lambda c: ScatterGatherDetector(c.scatter_gather),
    "gather_scatter": lambda c: GatherScatterDetector(c.gather_scatter),
    "community": lambda c: CommunityDetector(c.community),
    "relay": lambda c: RelayDetector(c.relay),
}


def lookback_history(
    df: pd.DataFrame, part: pd.DataFrame, lookback: str | None
) -> pd.DataFrame | None:
    """Transactions of ``df`` in the ``lookback`` period before ``part`` (``None``: no context).

    Used as unlabelled context when a time slice is scanned (:meth:`PatternScanner.scan`).
    """
    if lookback is None or part.empty:
        return None
    start = part["timestamp"].min()
    earlier = (df["timestamp"] >= start - pd.Timedelta(seconds=parse_duration(lookback))) & (
        df["timestamp"] < start
    )
    return df[earlier]


def register_detector(name: str, factory: DetectorFactory) -> None:
    """Plug a custom detector into the scanner (``factory(config) -> PatternDetector``).

    A researcher can thus test their own motif without touching the rest of the pipeline.
    """
    DETECTOR_REGISTRY[name] = factory


@dataclass
class ScanResult:
    """Alerts found by a scan, with helpers to turn them into scores and tables."""

    alerts: list[Alert]
    tx_ids: np.ndarray
    timings: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, float] = field(default_factory=dict)
    combine: str = "max"  # "max" or "noisy_or" (PatternConfig.combine)
    # per-pattern (inner bin edges of the alert score, precision of each bin)
    bins: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.alerts)

    def weight(self, alert: Alert) -> float:
        """Alert score after the optional precision calibration (per pattern or per score bin).

        With score bins the weight is the precision of the alert's bin (the score only breaks
        ties inside a bin).
        """
        if alert.pattern in self.bins:
            edges, precision = self.bins[alert.pattern]
            k = int(np.searchsorted(edges, alert.score, side="right"))
            return float(precision[k]) * (1.0 - 1e-3 + 1e-3 * alert.score)
        return alert.score * self.calibration.get(alert.pattern, 1.0)

    def to_frame(self) -> pd.DataFrame:
        """One row per alert, sorted by decreasing (calibrated) score."""
        cols = [
            "alert_id",
            "pattern",
            "score",
            "n_tx",
            "n_accounts",
            "start",
            "end",
            "total_amount",
        ]
        if not self.alerts:
            return pd.DataFrame(columns=[*cols, "accounts", "tx_ids"])
        rows = [
            {
                "alert_id": i,
                "pattern": a.pattern,
                "score": self.weight(a),
                "n_tx": len(a.tx_ids),
                "n_accounts": len(a.accounts),
                "start": a.start,
                "end": a.end,
                "total_amount": a.total_amount,
                "accounts": a.accounts,
                "tx_ids": a.tx_ids,
            }
            for i, a in enumerate(self.alerts)
        ]
        return pd.DataFrame(rows).sort_values("score", ascending=False, ignore_index=True)

    def transaction_scores(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-transaction score (max over alerts) and name of the strongest pattern.

        Returns ``(scores, top_pattern)`` aligned with the scanned frame; transactions covered
        by no alert get score ``0`` and pattern ``""``. With ``combine="noisy_or"`` the alerts
        of a transaction are independent pieces of evidence: ``1 - prod(1 - weight)``.
        """
        n = len(self.tx_ids)
        scores = np.zeros(n)
        miss = np.zeros(n)  # sum of log(1 - weight), for noisy-or
        top = np.full(n, "", dtype=object)
        index = pd.Index(self.tx_ids)
        for alert in self.alerts:
            pos = index.get_indexer(alert.tx_ids)
            pos = pos[pos >= 0]
            w = self.weight(alert)
            better = scores[pos] < w
            scores[pos[better]] = w
            top[pos[better]] = alert.pattern
            if self.combine == "noisy_or":
                np.add.at(miss, np.unique(pos), np.log1p(-min(w, 1.0 - 1e-12)))
        if self.combine == "noisy_or":
            scores = -np.expm1(miss)
        return scores, top

    def account_scores(self) -> dict[str, float]:
        """Highest alert score every account takes part in."""
        out: dict[str, float] = {}
        for alert in self.alerts:
            w = self.weight(alert)
            for acc in alert.accounts:
                if w > out.get(acc, 0.0):
                    out[acc] = w
        return out

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for a in self.alerts:
            counts[a.pattern] = counts.get(a.pattern, 0) + 1
        scores, _ = self.transaction_scores()
        return {
            "n_alerts": len(self.alerts),
            "alerts_by_pattern": counts,
            "n_flagged_transactions": int((scores > 0).sum()),
            "n_transactions": len(self.tx_ids),
            "timings_seconds": {k: round(v, 3) for k, v in self.timings.items()},
        }

    def to_json(self, path: str | None = None, indent: int | None = 2) -> str:
        """Serialise alerts (+ summary) to JSON; writes to ``path`` when given."""
        payload = {
            "summary": self.summary(),
            "alerts": [{**a.to_dict(), "score": round(self.weight(a), 6)} for a in self.alerts],
        }
        text = json.dumps(payload, indent=indent)
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        return text


class PatternScanner:
    """Orchestrates the deterministic detectors.

    Example
    -------
    >>> scanner = PatternScanner()                       # doctest: +SKIP
    >>> result = scanner.scan(df)                        # doctest: +SKIP
    >>> result.to_frame().head()                         # doctest: +SKIP
    """

    def __init__(
        self,
        config: PatternConfig | None = None,
        only: Iterable[str] | None = None,
    ) -> None:
        self.config = config or PatternConfig()
        names = list(only) if only is not None else list(DETECTOR_REGISTRY)
        unknown = [n for n in names if n not in DETECTOR_REGISTRY]
        if unknown:
            raise ConfigError(
                f"unknown detector(s) {unknown}; available: {sorted(DETECTOR_REGISTRY)}"
            )
        # sections switched off in the configuration are not even instantiated; custom
        # detectors (no section in the configuration) are always kept
        self.detectors: dict[str, PatternDetector] = {
            n: DETECTOR_REGISTRY[n](self.config)
            for n in names
            if getattr(getattr(self.config, n, None), "enabled", True)
        }
        self.calibration: dict[str, float] = {}
        self.bin_calibration: dict[str, tuple[list[float], list[float]]] = {}

    def scan(self, df: pd.DataFrame, history: pd.DataFrame | None = None) -> ScanResult:
        """Run all detectors on a canonical, time-sorted transaction table.

        ``history`` holds earlier transactions (their labels are never read): motifs that began
        before ``df`` are then seen whole. Only the transactions of ``df`` are scored, and only
        the alerts touching at least one of them are returned.
        """
        frame = df
        if history is not None and len(history):
            earlier = history[~history.index.isin(df.index)]
            frame = pd.concat([earlier, df]).sort_values("timestamp", kind="stable")
        currency = self.config.currency
        kept = self._kept(frame)
        if not kept.all():
            if currency.convert and currency.rates is None:
                # rates come from the cross-currency transactions, which may be excluded below
                currency = currency.model_copy(update={"rates": self._rates(frame)})
            frame = frame[kept]
        ctx = TxArrays.from_frame(frame, currency)
        alerts: list[Alert] = []
        timings: dict[str, float] = {}
        for name, detector in self.detectors.items():
            t0 = time.perf_counter()
            alerts.extend(detector.detect(ctx))
            timings[name] = time.perf_counter() - t0
        if history is not None and len(history):
            own = pd.Index(df.index)
            alerts = [a for a in alerts if (own.get_indexer(a.tx_ids) >= 0).any()]
        return ScanResult(
            alerts,
            df.index.to_numpy(),
            timings,
            dict(self.calibration),
            self.config.combine,
            dict(self.bin_calibration),
        )

    def _kept(self, frame: pd.DataFrame) -> np.ndarray:
        """Rows searched for motifs (``transaction_types`` and ``exclude`` settings)."""
        keep = np.ones(len(frame), dtype=bool)
        cfg = self.config
        if cfg.transaction_types is not None and "transaction_type" in frame.columns:
            keep &= frame["transaction_type"].isin(cfg.transaction_types).to_numpy()
        if "self_transfers" in cfg.exclude:
            keep &= (frame["sender"] != frame["receiver"]).to_numpy()
        if "same_bank" in cfg.exclude and "sender_bank" in frame.columns:
            bank = frame["sender_bank"].astype(str)
            keep &= ~((bank == frame["receiver_bank"].astype(str)) & (bank != "")).to_numpy()
        if "cross_currency" in cfg.exclude:
            keep &= (frame["currency"] == frame["currency_received"]).to_numpy()
        return keep

    def _rates(self, frame: pd.DataFrame) -> dict[str, float]:
        """Exchange rates inferred from all the transactions of ``frame`` (before filtering)."""
        ctx = TxArrays.from_frame(frame, self.config.currency)
        return {
            str(name): float(rate)
            for name, rate in zip(ctx.currency_names, ctx.fx)
            if np.isfinite(rate)
        }

    def fit_calibration(
        self, df: pd.DataFrame, prior: float = 1.0, history: pd.DataFrame | None = None
    ) -> dict[str, float]:
        """Learn the precision of every detector on a *labelled validation slice*.

        The precision of a pattern is the (Laplace-smoothed) share of the transactions its alerts
        cover that are labelled illicit. Calibrated scores ``precision x alert score`` let alerts
        of different detectors be ranked on a common scale. Only the labels of ``df`` are used
        (``history`` is unlabelled context, see :meth:`scan`).
        """
        if df["label"].isna().all():
            raise ConfigError("calibration needs labelled transactions")
        return self.calibrate(self.scan(df, history), df["label"].fillna(0).to_numpy(), prior)

    def calibrate(
        self, result: ScanResult, labels: np.ndarray, prior: float = 1.0
    ) -> dict[str, float]:
        """Calibration from an existing scan of a labelled slice (``labels`` aligned with it)."""
        labels = np.asarray(labels, dtype="float64")
        index = pd.Index(result.tx_ids)
        by_pattern: dict[str, set[int]] = {}
        for alert in result.alerts:
            pos = index.get_indexer(alert.tx_ids)
            by_pattern.setdefault(alert.pattern, set()).update(pos[pos >= 0].tolist())
        self.calibration = {}
        for name in self.detectors:
            covered = np.array(sorted(by_pattern.get(name, set())), dtype=int)
            positives = float(labels[covered].sum()) if len(covered) else 0.0
            self.calibration[name] = (positives + prior) / (len(covered) + 2.0 * prior)
        self.bin_calibration = self._fit_bins(result, labels, index)
        return dict(self.calibration)

    def _fit_bins(
        self, result: ScanResult, labels: np.ndarray, index: pd.Index, shrink: float = 20.0
    ) -> dict[str, tuple[list[float], list[float]]]:
        """Precision per quantile band of the alert score, shrunk towards the detector's own."""
        k = self.config.calibration_bins
        if k <= 1:
            return {}
        out: dict[str, tuple[list[float], list[float]]] = {}
        for name in self.detectors:
            alerts = [a for a in result.alerts if a.pattern == name]
            if len(alerts) < 2 * k:
                continue
            scores = np.array([a.score for a in alerts])
            edges = np.unique(np.quantile(scores, np.linspace(0, 1, k + 1)[1:-1]))
            band = np.searchsorted(edges, scores, side="right")
            base = self.calibration[name]
            precision = []
            for b in range(len(edges) + 1):
                rows: set[int] = set()
                for alert, a_band in zip(alerts, band):
                    if a_band == b:
                        pos = index.get_indexer(pd.Index(alert.tx_ids))
                        rows.update(pos[pos >= 0].tolist())
                covered = np.fromiter(rows, dtype=int, count=len(rows))
                hits = float(labels[covered].sum()) if len(covered) else 0.0
                precision.append((hits + shrink * base) / (len(covered) + shrink))
            out[name] = ([float(e) for e in edges], precision)
        return out
