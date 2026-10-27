"""Layering detectors: scatter-gather and gather-scatter."""

from __future__ import annotations

import numpy as np

from ...utils import parse_duration
from .base import Alert, PatternDetector, TxArrays, group_slices, saturating, sliding_windows
from .config import GatherScatterConfig, ScatterGatherConfig


class ScatterGatherDetector(PatternDetector):
    """``source -> {m1..mk} -> sink``: money is dispersed then re-aggregated.

    Two-hop paths ``s -> m -> c`` are enumerated with a vectorised range join per intermediary
    (the second hop must happen within ``hop_window`` after the first and carry a comparable
    amount). Paths are then grouped by ``(s, c)``; a group with at least ``min_intermediaries``
    distinct intermediaries inside ``max_span`` is an alert.
    """

    name = "scatter_gather"

    def __init__(self, config: ScatterGatherConfig | None = None) -> None:
        self.config = config or ScatterGatherConfig()

    def detect(self, ctx: TxArrays) -> list[Alert]:
        cfg = self.config
        if not cfg.enabled or len(ctx) == 0:
            return []
        hop = parse_duration(cfg.hop_window)
        max_span = parse_duration(cfg.max_span)
        in_deg = np.bincount(ctx.dst, minlength=ctx.n_nodes)
        out_deg = np.bincount(ctx.src, minlength=ctx.n_nodes)
        ok_mid = (
            (in_deg >= 1)
            & (out_deg >= 1)
            & (in_deg <= cfg.max_hub_degree)
            & (out_deg <= cfg.max_hub_degree)
        )
        first_leg = np.flatnonzero(ok_mid[ctx.dst] & (ctx.src != ctx.dst))
        second_leg = np.flatnonzero(ok_mid[ctx.src] & (ctx.src != ctx.dst))
        if len(first_leg) == 0 or len(second_leg) == 0:
            return []

        # second legs sorted by (intermediary, time) → range search per first leg
        order2 = second_leg[np.lexsort((ctx.ts[second_leg], ctx.src[second_leg]))]
        big = float(ctx.ts.max() - ctx.ts.min()) + hop + 1.0
        t0 = float(ctx.ts.min())
        key2 = ctx.src[order2].astype("float64") * big + (ctx.ts[order2] - t0)
        mid = ctx.dst[first_leg]
        key_lo = mid.astype("float64") * big + (ctx.ts[first_leg] - t0)
        lo = np.searchsorted(key2, key_lo, side="right")
        hi = np.searchsorted(key2, key_lo + hop, side="right")
        counts = hi - lo
        if counts.sum() == 0:
            return []
        leg1 = np.repeat(first_leg, counts)
        offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        leg2 = order2[np.repeat(lo, counts) + offsets]

        src, sink = ctx.src[leg1], ctx.dst[leg2]
        received = ctx.recv[leg1]
        comparable = ctx.cur_recv[leg1] == ctx.cur_paid[leg2]
        ratio_ok = (ctx.paid[leg2] >= cfg.min_hop_ratio * received) & (
            ctx.paid[leg2] <= cfg.max_hop_ratio * received
        )
        keep = (src != sink) & (~comparable | ratio_ok)
        leg1, leg2, src, sink = leg1[keep], leg2[keep], src[keep], sink[keep]
        if len(leg1) == 0:
            return []

        alerts: list[Alert] = []
        pair_key = src.astype("int64") * ctx.n_nodes + sink
        for key, idx in group_slices(pair_key):
            mids = np.unique(ctx.dst[leg1[idx]])
            if len(mids) < cfg.min_intermediaries:
                continue
            rows = np.unique(np.concatenate([leg1[idx], leg2[idx]]))
            span = float(ctx.ts[rows].max() - ctx.ts[rows].min())
            if span > max_span:
                continue
            s_node, c_node = divmod(int(key), ctx.n_nodes)
            score = 0.4 + 0.6 * (
                0.7 * saturating(len(mids), 2.0 * cfg.min_intermediaries)
                + 0.3 * float(np.clip(1.0 - span / max_span, 0.0, 1.0))
            )
            alerts.append(
                self.make_alert(
                    ctx,
                    self.name,
                    rows,
                    [s_node, *mids.tolist(), c_node],
                    score,
                    {
                        "source": str(ctx.names[s_node]),
                        "sink": str(ctx.names[c_node]),
                        "n_intermediaries": len(mids),
                        "span_seconds": span,
                        "dispersed_amount": float(ctx.recv[leg1[idx]].sum()),
                    },
                    amount=ctx.paid[np.unique(leg1[idx])],
                )
            )
        return alerts


class GatherScatterDetector(PatternDetector):
    """``{s1..sk} -> hub -> {t1..tk'}``: money is gathered in one account then redistributed."""

    name = "gather_scatter"

    def __init__(self, config: GatherScatterConfig | None = None) -> None:
        self.config = config or GatherScatterConfig()

    def detect(self, ctx: TxArrays) -> list[Alert]:
        cfg = self.config
        if not cfg.enabled or len(ctx) == 0:
            return []
        w_in, w_out = parse_duration(cfg.gather_window), parse_duration(cfg.scatter_window)
        horizon = max(float(ctx.ts.max() - ctx.ts.min()), w_in * max(cfg.min_burst_ratio, 1.0))
        uniq_in = _distinct_per_node(ctx.dst, ctx.src, ctx.n_nodes)
        uniq_out = _distinct_per_node(ctx.src, ctx.dst, ctx.n_nodes)
        hubs = np.flatnonzero((uniq_in >= cfg.min_in) & (uniq_out >= cfg.min_out))
        if len(hubs) == 0:
            return []
        in_groups = dict(group_slices(ctx.dst, hubs))
        out_groups = dict(group_slices(ctx.src, hubs))
        alerts: list[Alert] = []
        for hub in hubs.tolist():
            in_rows, out_rows = in_groups[hub], out_groups[hub]
            out_ts = ctx.ts[out_rows]
            expected = len(in_rows) * w_in / horizon
            need = int(np.ceil(cfg.min_burst_ratio * expected))
            for i, j in sliding_windows(
                ctx.ts[in_rows], ctx.src[in_rows], w_in, cfg.min_in, min_count=need
            ):
                gather = in_rows[i:j]
                t_first, t_last = float(ctx.ts[gather].min()), float(ctx.ts[gather].max())
                lo = int(np.searchsorted(out_ts, t_first, side="right"))
                hi = int(np.searchsorted(out_ts, t_last + w_out, side="right"))
                scatter = out_rows[lo:hi]
                if len(scatter) == 0:
                    continue
                receivers = np.unique(ctx.dst[scatter])
                if len(receivers) < cfg.min_out:
                    continue
                total_in, total_out = float(ctx.recv[gather].sum()), float(ctx.paid[scatter].sum())
                ratio = total_out / total_in if total_in > 0 else 0.0
                if not (cfg.min_conservation <= ratio <= cfg.max_conservation):
                    continue
                everything = len(in_rows) + len(out_rows)
                episode = int(
                    np.searchsorted(ctx.ts[in_rows], t_last + w_out, side="right")
                    - np.searchsorted(ctx.ts[in_rows], t_first, side="left")
                ) + (hi - lo)
                if (
                    len(gather) + len(scatter)
                ) < cfg.min_activity_share * episode or everything == 0:
                    continue
                senders = np.unique(ctx.src[gather])
                rows = np.concatenate([gather, scatter])
                conservation = float(np.clip(1.0 - abs(1.0 - ratio), 0.0, 1.0))
                score = 0.4 + 0.6 * (
                    0.35 * saturating(len(senders), 2.0 * cfg.min_in)
                    + 0.35 * saturating(len(receivers), 2.0 * cfg.min_out)
                    + 0.3 * conservation
                )
                alerts.append(
                    self.make_alert(
                        ctx,
                        self.name,
                        rows,
                        [hub, *senders.tolist(), *receivers.tolist()],
                        score,
                        {
                            "hub": str(ctx.names[hub]),
                            "n_sources": len(senders),
                            "n_targets": len(receivers),
                            "gathered_amount": total_in,
                            "scattered_amount": total_out,
                            "conservation_ratio": ratio,
                        },
                        amount=ctx.paid[gather],
                    )
                )
        return alerts


def _distinct_per_node(node: np.ndarray, other: np.ndarray, n: int) -> np.ndarray:
    """Number of distinct ``other`` values per ``node`` (vectorised)."""
    pairs = np.unique(node.astype("int64") * n + other)
    return np.bincount(pairs // n, minlength=n)
