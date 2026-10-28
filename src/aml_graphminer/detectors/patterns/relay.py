"""Relay detector: money that passes through an account almost intact (``A -> B -> C``)."""

from __future__ import annotations

import numpy as np

from ...utils import parse_duration
from .base import Alert, PatternDetector, TxArrays
from .config import RelayConfig


class RelayDetector(PatternDetector):
    """An account forwards, within ``hop_window``, (almost) the amount it has just received.

    Each incoming payment ``A -> B`` is paired with the outgoing payment ``B -> C`` (``C != A``)
    that follows it within ``hop_window`` and forwards between ``min_hop_ratio`` and
    ``max_hop_ratio`` of the received amount (compared in the reference currency); among several
    candidates the closest amount wins. With ``currency_change`` the forwarded payment must use
    another currency than the one received — the signature of the IBM "stack" motifs, where every
    hop converts the money. Relays through very active accounts (``max_degree``) are ignored.
    """

    name = "relay"

    def __init__(self, config: RelayConfig | None = None) -> None:
        self.config = config or RelayConfig()

    def detect(self, ctx: TxArrays) -> list[Alert]:
        cfg = self.config
        if not cfg.enabled or len(ctx) == 0:
            return []
        hop = parse_duration(cfg.hop_window)
        degree = np.bincount(ctx.src, minlength=ctx.n_nodes) + np.bincount(
            ctx.dst, minlength=ctx.n_nodes
        )
        ok = degree <= cfg.max_degree
        first = np.flatnonzero(ok[ctx.dst] & (ctx.src != ctx.dst))
        second = np.flatnonzero(ok[ctx.src] & (ctx.src != ctx.dst))
        if len(first) == 0 or len(second) == 0:
            return []
        # second legs sorted by (relay account, time) -> range search per first leg
        order = second[np.lexsort((ctx.ts[second], ctx.src[second]))]
        t0 = float(ctx.ts.min())
        big = float(ctx.ts.max() - t0) + hop + 1.0
        key = ctx.src[order].astype("float64") * big + (ctx.ts[order] - t0)
        start = ctx.dst[first].astype("float64") * big + (ctx.ts[first] - t0)
        lo = np.searchsorted(key, start, side="right")
        hi = np.searchsorted(key, start + hop, side="right")
        counts = hi - lo
        if counts.sum() == 0:
            return []
        leg1 = np.repeat(first, counts)
        offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        leg2 = order[np.repeat(lo, counts) + offsets]
        received, forwarded, comparable = ctx.hop_amounts(leg1, leg2)
        ratio = np.divide(
            forwarded, received, out=np.zeros(len(leg1)), where=comparable & (received > 0)
        )
        keep = (
            comparable
            & (ctx.dst[leg2] != ctx.src[leg1])
            & (ratio >= cfg.min_hop_ratio)
            & (ratio <= cfg.max_hop_ratio)
        )
        if cfg.currency_change:
            keep &= ctx.cur_paid[leg2] != ctx.cur_recv[leg1]
        leg1, leg2, ratio = leg1[keep], leg2[keep], ratio[keep]
        if len(leg1) == 0:
            return []
        # one relay per incoming payment: the forwarded amount closest to the received one
        best = np.lexsort((np.abs(1.0 - ratio), leg1))
        first_of = np.r_[True, leg1[best][1:] != leg1[best][:-1]]
        leg1, leg2, ratio = leg1[best][first_of], leg2[best][first_of], ratio[best][first_of]
        width = max(cfg.max_hop_ratio - cfg.min_hop_ratio, 1e-9)
        alerts: list[Alert] = []
        for a, b, r in zip(leg1.tolist(), leg2.tolist(), ratio.tolist()):
            delay = float(ctx.ts[b] - ctx.ts[a])
            closeness = float(np.clip(1.0 - abs(1.0 - r) / width, 0.0, 1.0))
            speed = float(np.clip(1.0 - delay / hop, 0.0, 1.0))
            rows = np.array([a, b])
            alerts.append(
                self.make_alert(
                    ctx,
                    self.name,
                    rows,
                    [int(ctx.src[a]), int(ctx.dst[a]), int(ctx.dst[b])],
                    0.5 + 0.5 * (0.6 * closeness + 0.4 * speed),
                    {
                        "relay": str(ctx.names[ctx.dst[a]]),
                        "forwarded_share": r,
                        "delay_seconds": delay,
                        "currency_in": str(ctx.currency_names[ctx.cur_recv[a]]),
                        "currency_out": str(ctx.currency_names[ctx.cur_paid[b]]),
                    },
                    amount=ctx.paid_c[rows[:1]],
                )
            )
        return alerts
