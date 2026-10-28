"""Risk-community detector: rings of pass-through accounts that circulate money among themselves."""

from __future__ import annotations

from collections.abc import Iterator

import networkx as nx
import numpy as np
import pandas as pd

from ...utils import parse_duration
from .base import Alert, PatternDetector, TxArrays, group_slices, saturating
from .config import CommunityConfig


class CommunityDetector(PatternDetector):
    """Detect *risk communities* (communautés à risque): groups that mostly trade among themselves.

    1. **Relays** — accounts that both receive and send money and forward at least
       ``min_flow_through`` of what they receive (``min(in, out) / max(in, out)``). Accounts with
       more than ``max_member_degree`` transactions (merchants, payroll or bank accounts) are
       ignored altogether: their flows are legitimately balanced and they would glue unrelated
       accounts together.
    2. **Groups** — connected components of the transactions that touch at least one relay
       (so the sources and sinks around the relays belong to the group); components larger than
       ``max_size`` are split into Louvain (modularity) communities.
    3. **Risk** — a group of ``min_size`` to ``max_size`` accounts with at least ``min_relays``
       relays is an alert when at least ``min_internal_tx`` transactions link its members within
       ``max_span`` and at least ``min_internal_share`` of the members' outgoing volume stays
       inside the group.

    The score grows with the internal share, the share of relays, the number of internal
    transactions per member, the speed of the flows and the presence of a directed cycle; the
    evidence reports all of them, so every alert can be audited.
    """

    name = "community"

    def __init__(self, config: CommunityConfig | None = None) -> None:
        self.config = config or CommunityConfig()

    def detect(self, ctx: TxArrays) -> list[Alert]:
        cfg = self.config
        if not cfg.enabled or len(ctx) == 0:
            return []
        n = ctx.n_nodes
        out_vol = np.bincount(ctx.src, weights=ctx.paid_c, minlength=n)
        in_vol = np.bincount(ctx.dst, weights=ctx.recv_c, minlength=n)
        degree = np.bincount(ctx.src, minlength=n) + np.bincount(ctx.dst, minlength=n)
        largest = np.maximum(in_vol, out_vol)
        flow = np.divide(np.minimum(in_vol, out_vol), largest, out=np.zeros(n), where=largest > 0)
        light = degree <= cfg.max_member_degree
        relay = (in_vol > 0) & (out_vol > 0) & (flow >= cfg.min_flow_through) & light
        edges = np.flatnonzero(
            (relay[ctx.src] | relay[ctx.dst])
            & light[ctx.src]
            & light[ctx.dst]
            & (ctx.src != ctx.dst)
        )
        if len(edges) < cfg.min_internal_tx:
            return []

        label = np.full(n, -1, dtype="int64")
        groups = list(self._groups(ctx, edges))
        for g, members in enumerate(groups):
            label[members] = g
        same = (label[ctx.src[edges]] == label[ctx.dst[edges]]) & (label[ctx.src[edges]] >= 0)
        internal = edges[same]
        max_span = parse_duration(cfg.max_span)
        alerts: list[Alert] = []
        for g, rows in group_slices(label[ctx.src[internal]]):
            alert = self._assess(ctx, groups[g], internal[rows], out_vol, flow, relay, max_span)
            if alert is not None:
                alerts.append(alert)
        return alerts

    # ------------------------------------------------------------------ helpers
    def _groups(self, ctx: TxArrays, edges: np.ndarray) -> Iterator[np.ndarray]:
        """Member arrays of the candidate groups (deterministic order)."""
        cfg = self.config
        u = np.minimum(ctx.src[edges], ctx.dst[edges])
        v = np.maximum(ctx.src[edges], ctx.dst[edges])
        weights = (
            pd.DataFrame({"u": u, "v": v, "w": ctx.paid_c[edges]}).groupby(["u", "v"])["w"].sum()
        )
        graph = nx.Graph()
        graph.add_weighted_edges_from(
            zip(
                weights.index.get_level_values(0).tolist(),
                weights.index.get_level_values(1).tolist(),
                weights.to_numpy().tolist(),
            )
        )
        components = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: c[0])
        for component in components:
            if len(component) < cfg.min_size:
                continue
            if len(component) <= cfg.max_size:
                yield np.asarray(component, dtype="int64")
                continue
            parts = nx.community.louvain_communities(
                graph.subgraph(component), weight="weight", resolution=cfg.resolution, seed=cfg.seed
            )
            for part in sorted((sorted(p) for p in parts), key=lambda p: p[0]):
                if cfg.min_size <= len(part) <= cfg.max_size:
                    yield np.asarray(part, dtype="int64")

    def _assess(
        self,
        ctx: TxArrays,
        members: np.ndarray,
        rows: np.ndarray,
        out_vol: np.ndarray,
        flow: np.ndarray,
        relay: np.ndarray,
        max_span: float,
    ) -> Alert | None:
        cfg = self.config
        rows = np.sort(rows)
        n_relays = int(relay[members].sum())
        if len(rows) < cfg.min_internal_tx or n_relays < cfg.min_relays:
            return None
        span = float(ctx.ts[rows].max() - ctx.ts[rows].min())
        if span > max_span:
            return None
        internal_amount = float(ctx.paid_c[rows].sum())
        outgoing = float(out_vol[members].sum())
        share = internal_amount / outgoing if outgoing > 0 else 0.0
        if share < cfg.min_internal_share:
            return None
        directed = nx.DiGraph()
        directed.add_edges_from(zip(ctx.src[rows].tolist(), ctx.dst[rows].tolist()))
        has_cycle = not nx.is_directed_acyclic_graph(directed)
        relay_share = n_relays / len(members)
        cohesion = saturating(len(rows) / len(members), 1.0)
        compact = float(np.clip(1.0 - span / max_span, 0.0, 1.0))
        score = 0.3 + 0.7 * (
            0.3 * min(share, 1.0)
            + 0.2 * relay_share
            + 0.1 * cohesion
            + 0.2 * compact
            + 0.2 * float(has_cycle)
        )
        return self.make_alert(
            ctx,
            self.name,
            rows,
            members.tolist(),
            score,
            {
                "n_members": len(members),
                "n_relays": n_relays,
                "n_internal_tx": len(rows),
                "internal_share": share,
                "mean_flow_through": float(flow[members].mean()),
                "span_seconds": span,
                "internal_amount": internal_amount,
                "has_cycle": has_cycle,
            },
        )


__all__ = ["CommunityDetector"]
