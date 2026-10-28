"""Cycle detector: time-respecting directed cycles ``A -> B -> C -> ... -> A`` with conserved amounts."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from ...utils import logger, parse_duration
from .base import Alert, PatternDetector, TxArrays
from .config import CycleConfig


class CycleDetector(PatternDetector):
    """Find money that flows around a loop and comes back to its origin.

    A cycle is a sequence of transactions ``e1, ..., ek`` with ``k`` between ``min_length`` and
    ``max_length`` such that

    * ``e_i`` pays from ``v_i`` to ``v_{i+1}`` and ``e_k`` returns to ``v_1`` (all nodes distinct);
    * times strictly increase along the path and the whole loop lasts at most ``max_span``;
    * what an account forwards is close to what it received (``min_hop_ratio`` ≤ paid / received
      ≤ ``max_hop_ratio``). Amounts in two currencies are compared in the reference currency
      (``currency`` section of the configuration); a hop whose rates are unknown is rejected,
      or accepted unchecked with ``cross_currency``.

    Because time must increase, a given cycle is discovered **once**, from its earliest
    transaction. The search is restricted to strongly-connected components of the aggregated
    graph and bounded by ``max_branching`` / ``max_cycles`` to stay tractable on dense data.
    """

    name = "cycle"

    def __init__(self, config: CycleConfig | None = None) -> None:
        self.config = config or CycleConfig()
        self.truncated = False  # True when the search budget was exhausted in the last run

    def detect(self, ctx: TxArrays) -> list[Alert]:
        cfg = self.config
        if not cfg.enabled or len(ctx) == 0:
            return []
        keep = self._rows_in_cycles(ctx)
        if len(keep) < cfg.min_length:
            return []
        max_span = parse_duration(cfg.max_span)

        # CSR-like adjacency over the kept rows, sorted by (source, time)
        order = keep[np.lexsort((ctx.ts[keep], ctx.src[keep]))]
        o_src, o_dst, o_ts = ctx.src[order], ctx.dst[order], ctx.ts[order]
        o_paid, o_recv = ctx.paid[order], ctx.recv[order]
        o_paid_ref, o_recv_ref = ctx.paid_ref[order], ctx.recv_ref[order]
        o_cp, o_cr = ctx.cur_paid[order], ctx.cur_recv[order]
        indptr = np.zeros(ctx.n_nodes + 1, dtype="int64")
        np.add.at(indptr, o_src + 1, 1)
        indptr = np.cumsum(indptr)

        alerts: list[Alert] = []
        found = 0
        expansions = 0
        self.truncated = False

        def extend(path: list[int], nodes: list[int]) -> None:
            nonlocal found, expansions
            if found >= cfg.max_cycles:
                return
            expansions += 1
            if expansions > cfg.max_expansions:
                self.truncated = True
                return
            last = path[-1]
            v = int(o_dst[last])
            lo, hi = int(indptr[v]), int(indptr[v + 1])
            if lo == hi:
                return
            t_first = o_ts[path[0]]
            seg = slice(lo, hi)
            left = lo + int(np.searchsorted(o_ts[seg], o_ts[last], side="right"))
            right = lo + int(np.searchsorted(o_ts[seg], t_first + max_span, side="right"))
            if left >= right:
                return
            cand = np.arange(left, min(right, left + cfg.max_branching))
            same_currency = o_cp[cand] == o_cr[last]
            # NaN (unknown rate) fails both comparisons: such hops are not comparable
            received = np.where(same_currency, o_recv[last], o_recv_ref[last])
            forwarded = np.where(same_currency, o_paid[cand], o_paid_ref[cand])
            accepted = (forwarded >= cfg.min_hop_ratio * received) & (
                forwarded <= cfg.max_hop_ratio * received
            )
            if cfg.cross_currency:
                accepted |= ~same_currency & ~(np.isfinite(received) & np.isfinite(forwarded))
            ok = cand[accepted]
            for e in ok.tolist():
                nxt = int(o_dst[e])
                depth = len(path) + 1
                if nxt == nodes[0]:
                    if depth >= cfg.min_length:
                        self._record(ctx, order, path + [e], nodes, alerts, max_span)
                        found += 1
                        if found >= cfg.max_cycles:
                            return
                elif nxt not in nodes and depth < cfg.max_length:
                    extend(path + [e], nodes + [nxt])

        for e0 in range(len(order)):
            if found >= cfg.max_cycles:
                break
            extend([e0], [int(o_src[e0]), int(o_dst[e0])])
        if self.truncated or found >= cfg.max_cycles:
            logger.warning(
                "cycle search stopped early (%d cycles, %d expansions): the graph is dense; "
                "tighten min_hop_ratio/max_span or lower max_length",
                found,
                expansions,
            )
            self.truncated = True
        return alerts

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _rows_in_cycles(ctx: TxArrays) -> np.ndarray:
        """Rows whose endpoints share a strongly-connected component of size >= 3."""
        mask = ctx.src != ctx.dst
        adj = sparse.coo_matrix(
            (np.ones(int(mask.sum())), (ctx.src[mask], ctx.dst[mask])),
            shape=(ctx.n_nodes, ctx.n_nodes),
        ).tocsr()
        _, comp = connected_components(adj, directed=True, connection="strong")
        sizes = np.bincount(comp)
        same = (comp[ctx.src] == comp[ctx.dst]) & (sizes[comp[ctx.src]] >= 3) & mask
        return np.flatnonzero(same)

    def _record(
        self,
        ctx: TxArrays,
        order: np.ndarray,
        path: list[int],
        nodes: list[int],
        alerts: list[Alert],
        max_span: float,
    ) -> None:
        rows = order[np.array(path)]
        span = float(ctx.ts[rows[-1]] - ctx.ts[rows[0]])
        first, last = float(ctx.paid_c[rows[0]]), float(ctx.paid_c[rows[-1]])
        conservation = float(np.clip(last / first, 0.0, 1.0)) if first > 0 else 0.0
        speed = float(np.clip(1.0 - span / max_span, 0.0, 1.0))
        short = 1.0 / (len(rows) - 1)
        score = 0.5 + 0.5 * (0.45 * conservation + 0.35 * speed + 0.2 * short)
        alerts.append(
            self.make_alert(
                ctx,
                self.name,
                rows,
                nodes,
                score,
                {
                    "length": len(rows),
                    "span_seconds": span,
                    "first_amount": first,
                    "last_amount": last,
                    "amount_retained": conservation,
                    "cycle": [str(ctx.names[n]) for n in nodes] + [str(ctx.names[nodes[0]])],
                },
                amount=ctx.paid_c[rows[:1]],
            )
        )
