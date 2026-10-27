"""Fan-in / fan-out detectors: abnormal convergence / divergence of money flows."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ...features.temporal import trailing_window_stats
from ...utils import parse_duration
from .base import Alert, PatternDetector, TxArrays, group_slices, saturating, sliding_windows
from .config import FanConfig


class _FanDetector(PatternDetector):
    """Shared logic: a *hub* exchanges money with many distinct counterparties in a short window."""

    role: ClassVar[str]  # "receiver" for fan-in, "sender" for fan-out

    def __init__(self, config: FanConfig | None = None) -> None:
        self.config = config or FanConfig()

    def _keys(self, ctx: TxArrays) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def detect(self, ctx: TxArrays) -> list[Alert]:
        cfg = self.config
        if not cfg.enabled or len(ctx) == 0:
            return []
        hub, other = self._keys(ctx)
        window = parse_duration(cfg.window)
        horizon = max(float(ctx.ts.max() - ctx.ts.min()), window * max(cfg.min_burst_ratio, 1.0))
        # cheap vectorised prefilter: a hub needs >= k transactions inside one window
        in_window = trailing_window_stats(hub, ctx.ts, np.ones(len(ctx)), window).count
        candidates = np.unique(hub[in_window >= cfg.min_counterparties])
        alerts: list[Alert] = []
        for key, rows in group_slices(hub, candidates):
            expected = len(rows) * window / horizon  # tx per window if the hub were stationary
            need = int(np.ceil(cfg.min_burst_ratio * expected))
            for i, j in sliding_windows(
                ctx.ts[rows],
                other[rows],
                window,
                min_distinct=cfg.min_counterparties,
                min_count=need,
            ):
                sel = rows[i:j]
                amount = ctx.paid[sel]
                if amount.sum() < cfg.min_total_amount:
                    continue
                counterparties = np.unique(other[sel])
                span = float(ctx.ts[sel].max() - ctx.ts[sel].min())
                cv = float(amount.std() / amount.mean()) if amount.mean() > 0 else 0.0
                size = saturating(len(counterparties), 2.0 * cfg.min_counterparties)
                tight = float(np.clip(1.0 - span / window, 0.2, 1.0))
                similar = 1.0 / (1.0 + cv)
                burst = (j - i) / max(expected, 1e-9)
                surge = saturating(burst, 10.0)
                score = 0.2 + 0.8 * (0.35 * size + 0.2 * tight + 0.2 * similar + 0.25 * surge)
                alerts.append(
                    self.make_alert(
                        ctx,
                        self.name,
                        sel,
                        [key, *counterparties.tolist()],
                        score,
                        {
                            "hub": str(ctx.names[key]),
                            "n_counterparties": len(counterparties),
                            "n_transactions": len(sel),
                            "span_seconds": span,
                            "window_seconds": window,
                            "amount_cv": cv,
                            "burst_ratio": burst,
                        },
                    )
                )
        return alerts


class FanInDetector(_FanDetector):
    """Many distinct senders pay the same receiver within the window."""

    name = "fan_in"
    role = "receiver"

    def _keys(self, ctx: TxArrays) -> tuple[np.ndarray, np.ndarray]:
        return ctx.dst, ctx.src


class FanOutDetector(_FanDetector):
    """One sender pays many distinct receivers within the window."""

    name = "fan_out"
    role = "sender"

    def _keys(self, ctx: TxArrays) -> tuple[np.ndarray, np.ndarray]:
        return ctx.src, ctx.dst
