"""Local transaction neighbourhoods around a transaction (the sub-graph an analyst inspects)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from ..exceptions import ConfigError
from ..utils import epoch_seconds


class TransactionIndex:
    """Account → transactions lookup built once, reused for many explanations.

    Parameters
    ----------
    df:
        Canonical transaction table (index = transaction id, which must be unique).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not df.index.is_unique:
            raise ConfigError("transaction ids (the index) must be unique to explain a transaction")
        self.df = df
        self._ts = epoch_seconds(df["timestamp"])
        n = len(df)
        codes, uniques = pd.factorize(pd.concat([df["sender"], df["receiver"]], ignore_index=True))
        positions = np.concatenate([np.arange(n), np.arange(n)])
        order = np.argsort(codes, kind="stable")
        self._sorted_positions = positions[order]
        sorted_codes = codes[order]
        ids = np.arange(len(uniques))
        self._starts = np.searchsorted(sorted_codes, ids, side="left")
        self._ends = np.searchsorted(sorted_codes, ids, side="right")
        self._account_code = {str(a): i for i, a in enumerate(uniques)}
        self._senders = df["sender"].astype(str).to_numpy()
        self._receivers = df["receiver"].astype(str).to_numpy()

    def position(self, tx_id: int) -> int:
        """Row position of a transaction id."""
        loc = self.df.index.get_indexer(pd.Index([tx_id]))[0]
        if loc < 0:
            raise KeyError(f"unknown transaction id {tx_id!r}")
        return int(loc)

    def rows_of(self, account: str) -> np.ndarray:
        """Row positions of every transaction touching ``account`` (sorted, unique)."""
        code = self._account_code.get(str(account))
        if code is None:
            return np.zeros(0, dtype="int64")
        return np.unique(self._sorted_positions[self._starts[code] : self._ends[code]])

    def neighbourhood(
        self,
        tx_id: int,
        hops: int = 2,
        max_edges: int = 200,
        time_window: float | None = None,
    ) -> pd.DataFrame:
        """Transactions within ``hops`` account-hops of ``tx_id`` (the transaction included).

        The result is capped at ``max_edges`` rows: when more are found, those closest in
        time to the explained transaction are kept. ``time_window`` (seconds) restricts the
        neighbourhood to transactions that happened within that distance of it.
        """
        pos = self.position(tx_id)
        t0 = self._ts[pos]
        frontier = {self._senders[pos], self._receivers[pos]}
        seen_accounts = set(frontier)
        keep = {pos}
        for _ in range(max(hops, 0)):
            rows = [self.rows_of(a) for a in frontier]
            if not rows:
                break
            cand = np.unique(np.concatenate(rows))
            if time_window is not None:
                cand = cand[np.abs(self._ts[cand] - t0) <= time_window]
            keep.update(int(r) for r in cand)
            nxt = set(self._senders[cand]) | set(self._receivers[cand])
            frontier = nxt - seen_accounts
            seen_accounts |= nxt
            if not frontier:
                break
        rows_arr = np.array(sorted(keep), dtype="int64")
        if len(rows_arr) > max_edges:
            gap = np.abs(self._ts[rows_arr] - t0)
            gap[rows_arr == pos] = -1.0  # the explained transaction is always kept
            rows_arr = np.sort(rows_arr[np.argsort(gap, kind="stable")[:max_edges]])
        return self.df.iloc[rows_arr]


def subgraph_to_networkx(frame: pd.DataFrame, target: int | None = None) -> nx.MultiDiGraph:
    """Directed multigraph of a set of transactions (``key`` = transaction id)."""
    graph = nx.MultiDiGraph()
    for tx_id, row in zip(frame.index.tolist(), frame.to_dict("records")):
        graph.add_edge(
            str(row["sender"]),
            str(row["receiver"]),
            key=int(tx_id),
            tx_id=int(tx_id),
            amount=float(row["amount"]),
            timestamp=str(row["timestamp"]),
            is_target=target is not None and int(tx_id) == int(target),
        )
    return graph


__all__ = ["TransactionIndex", "subgraph_to_networkx"]
