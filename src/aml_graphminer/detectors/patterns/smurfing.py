"""Smurfing detector: structuring below a reporting threshold and splitting of a large inflow."""

from __future__ import annotations

import numpy as np

from ...features.temporal import trailing_window_stats
from ...utils import parse_duration
from .base import Alert, PatternDetector, TxArrays, group_slices, saturating, sliding_windows
from .config import SmurfingConfig


class SmurfingDetector(PatternDetector):
    """Two classic *placement* signatures.

    * **structuring** - an account sends (or receives) many payments whose amount sits just
      below the reporting ``threshold`` (default 10 000, last 25 % below the line) within a window;
    * **split** - an account receives an amount *A* and, within the window, forwards it in at
      least ``min_parts`` smaller parts whose sum matches *A* (within ``split_tolerance``).
    """

    name = "smurfing"

    def __init__(self, config: SmurfingConfig | None = None) -> None:
        self.config = config or SmurfingConfig()

    def detect(self, ctx: TxArrays) -> list[Alert]:
        cfg = self.config
        if not cfg.enabled or len(ctx) == 0:
            return []
        alerts: list[Alert] = []
        if cfg.structuring:
            alerts += self._structuring(ctx)
        if cfg.split:
            alerts += self._split(ctx)
        return alerts

    # ------------------------------------------------------------------ structuring
    def _structuring(self, ctx: TxArrays) -> list[Alert]:
        cfg = self.config
        low = cfg.threshold * (1.0 - cfg.band)
        in_band = (ctx.paid >= low) & (ctx.paid < cfg.threshold)
        if cfg.currency is not None:
            code = ctx.currency_code(cfg.currency)
            in_band &= ctx.cur_paid == code
        rows_band = np.flatnonzero(in_band)
        if len(rows_band) < cfg.min_transactions:
            return []
        window = parse_duration(cfg.window)
        alerts: list[Alert] = []
        for role, hub_all, other_all in (
            ("sender", ctx.src, ctx.dst),
            ("receiver", ctx.dst, ctx.src),
        ):
            hub, other = hub_all[rows_band], other_all[rows_band]
            all_rows = dict(group_slices(hub_all, np.unique(hub)))  # every payment of the hubs
            ones = np.ones(len(rows_band))
            cnt = trailing_window_stats(hub, ctx.ts[rows_band], ones, window).count
            candidates = np.unique(hub[cnt >= cfg.min_transactions])
            for key, local in group_slices(hub, candidates):
                rows = rows_band[local]
                every_ts = ctx.ts[all_rows[key]]

                def dominated(
                    i: int, j: int, rows: np.ndarray = rows, every_ts: np.ndarray = every_ts
                ) -> bool:
                    t0, t1 = ctx.ts[rows[i]], ctx.ts[rows[j - 1]]
                    total = np.searchsorted(every_ts, t1, side="right") - np.searchsorted(
                        every_ts, t0, side="left"
                    )
                    return (j - i) >= cfg.min_band_share * total

                for i, j in sliding_windows(
                    ctx.ts[rows],
                    other[local],
                    window,
                    min_distinct=1,
                    min_count=cfg.min_transactions,
                    accept=dominated,
                ):
                    sel = rows[i:j]
                    amounts = ctx.paid[sel]
                    closeness = float(np.mean((amounts - low) / (cfg.threshold - low)))
                    score = 0.3 + 0.7 * (
                        0.6 * saturating(len(sel), 2.0 * cfg.min_transactions) + 0.4 * closeness
                    )
                    alerts.append(
                        self.make_alert(
                            ctx,
                            self.name,
                            sel,
                            [key, *np.unique(other[local][i:j]).tolist()],
                            score,
                            {
                                "mode": "structuring",
                                "role": role,
                                "account": str(ctx.names[key]),
                                "n_transactions": len(sel),
                                "threshold": cfg.threshold,
                                "mean_amount": float(amounts.mean()),
                                "n_counterparties": len(np.unique(other[local][i:j])),
                            },
                        )
                    )
        return alerts

    # ------------------------------------------------------------------ split
    def _split(self, ctx: TxArrays) -> list[Alert]:
        cfg = self.config
        window = parse_duration(cfg.split_window)
        out_deg = np.bincount(ctx.src, minlength=ctx.n_nodes)
        in_deg = np.bincount(ctx.dst, minlength=ctx.n_nodes)
        candidates = np.flatnonzero((out_deg >= cfg.min_parts) & (in_deg >= 1))
        if len(candidates) == 0:
            return []
        out_groups = dict(group_slices(ctx.src, candidates))
        in_groups = dict(group_slices(ctx.dst, candidates))
        alerts: list[Alert] = []
        for node in candidates.tolist():
            out_rows = out_groups[node]
            out_ts = ctx.ts[out_rows]
            out_amt = ctx.paid_c[out_rows]
            csum = np.concatenate([[0.0], np.cumsum(out_amt)])
            used_until = -np.inf
            for r in in_groups[node]:  # chronological inflows
                t0 = ctx.ts[r]
                if t0 < used_until:
                    continue
                lo = int(np.searchsorted(out_ts, t0, side="right"))
                hi = int(np.searchsorted(out_ts, t0 + window, side="right"))
                if hi - lo < cfg.min_parts:
                    continue
                inflow = ctx.recv_c[r]
                total = csum[hi] - csum[lo]
                parts = out_amt[lo:hi]
                if (
                    abs(total - inflow) <= cfg.split_tolerance * inflow
                    and parts.max() <= cfg.max_part_fraction * inflow
                ):
                    sel = np.concatenate([[r], out_rows[lo:hi]])
                    used_until = float(out_ts[hi - 1])
                    fit = 1.0 - abs(total - inflow) / max(inflow, 1e-9) / max(
                        cfg.split_tolerance, 1e-9
                    )
                    score = 0.35 + 0.65 * (
                        0.5 * saturating(hi - lo, 2.0 * cfg.min_parts)
                        + 0.5 * float(np.clip(fit, 0, 1))
                    )
                    receivers = np.unique(ctx.dst[out_rows[lo:hi]])
                    alerts.append(
                        self.make_alert(
                            ctx,
                            self.name,
                            sel,
                            [node, int(ctx.src[r]), *receivers.tolist()],
                            score,
                            {
                                "mode": "split",
                                "account": str(ctx.names[node]),
                                "inflow_amount": float(inflow),
                                "n_parts": int(hi - lo),
                                "parts_total": float(total),
                                "largest_part_fraction": float(parts.max() / inflow),
                            },
                            amount=np.array([inflow]),
                        )
                    )
        return alerts
