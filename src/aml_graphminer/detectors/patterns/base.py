"""Shared building blocks of the structural-motif detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ...features.network import index_accounts
from ...utils import epoch_seconds


@dataclass
class TxArrays:
    """Column-oriented view of a canonical transaction table (built once per scan)."""

    ts: np.ndarray  # float seconds, ascending
    src: np.ndarray  # int account ids
    dst: np.ndarray
    n_nodes: int
    names: np.ndarray  # names[i] = account id string
    paid: np.ndarray
    recv: np.ndarray
    cur_paid: np.ndarray  # int codes
    cur_recv: np.ndarray
    currency_names: np.ndarray
    tx_ids: np.ndarray  # values of df.index
    timestamps: pd.Series
    label: np.ndarray

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> TxArrays:
        ts = epoch_seconds(df["timestamp"])
        if len(ts) > 1 and (ts[1:] < ts[:-1]).any():
            raise ValueError("the transaction table must be sorted by timestamp")
        src, dst, n, names = index_accounts(df)
        codes, uniques = pd.factorize(
            pd.concat([df["currency"], df["currency_received"]], ignore_index=True)
        )
        m = len(df)
        return cls(
            ts=ts,
            src=src,
            dst=dst,
            n_nodes=n,
            names=names,
            paid=df["amount"].to_numpy(dtype="float64"),
            recv=df["amount_received"].to_numpy(dtype="float64"),
            cur_paid=np.asarray(codes[:m], dtype="int64"),
            cur_recv=np.asarray(codes[m:], dtype="int64"),
            currency_names=np.asarray(uniques, dtype=object),
            tx_ids=df.index.to_numpy(),
            timestamps=df["timestamp"].reset_index(drop=True),
            label=df["label"].to_numpy(dtype="float64"),
        )

    def __len__(self) -> int:
        return len(self.ts)

    def currency_code(self, name: str) -> int:
        hits = np.flatnonzero(self.currency_names == name)
        return int(hits[0]) if len(hits) else -1


@dataclass
class Alert:
    """One structural-motif alert with its evidence (auditable, JSON-serialisable)."""

    pattern: str
    score: float
    accounts: list[str]
    tx_ids: list[int]
    start: pd.Timestamp
    end: pd.Timestamp
    total_amount: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "score": round(float(self.score), 6),
            "accounts": list(self.accounts),
            "tx_ids": [int(t) for t in self.tx_ids],
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "total_amount": round(float(self.total_amount), 2),
            "evidence": _jsonable(self.evidence),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


class PatternDetector(ABC):
    """Base class: a detector turns a :class:`TxArrays` view into a list of :class:`Alert`."""

    name: ClassVar[str]

    @abstractmethod
    def detect(self, ctx: TxArrays) -> list[Alert]:
        """Return the alerts found in ``ctx``."""

    def detect_frame(self, df: pd.DataFrame) -> list[Alert]:
        """Convenience wrapper accepting a canonical DataFrame."""
        return self.detect(TxArrays.from_frame(df))

    # -- helpers shared by the detectors ------------------------------------------------
    @staticmethod
    def make_alert(
        ctx: TxArrays,
        pattern: str,
        rows: np.ndarray,
        accounts: list[int],
        score: float,
        evidence: dict[str, Any],
        amount: np.ndarray | None = None,
    ) -> Alert:
        rows = np.asarray(rows)
        amounts = ctx.paid[rows] if amount is None else amount
        return Alert(
            pattern=pattern,
            score=float(np.clip(score, 0.0, 1.0)),
            accounts=[str(ctx.names[a]) for a in accounts],
            tx_ids=[int(t) for t in ctx.tx_ids[rows]],
            start=ctx.timestamps.iloc[int(rows.min())],
            end=ctx.timestamps.iloc[int(rows.max())],
            total_amount=float(np.sum(amounts)),
            evidence=evidence,
        )


def group_slices(
    keys: np.ndarray, candidates: np.ndarray | None = None
) -> list[tuple[int, np.ndarray]]:
    """Group row positions by key (rows stay in their original, i.e. chronological, order).

    ``candidates`` restricts the groups returned to those keys.
    """
    if len(keys) == 0:
        return []
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    out = []
    wanted = None if candidates is None else set(np.asarray(candidates).tolist())
    for s, e in zip(starts, ends):
        k = int(sorted_keys[s])
        if wanted is None or k in wanted:
            out.append((k, order[s:e]))
    return out


def sliding_windows(
    ts: np.ndarray,
    others: np.ndarray,
    window: float,
    min_distinct: int,
    min_count: int = 0,
    accept: Callable[[int, int], bool] | None = None,
) -> list[tuple[int, int]]:
    """Greedy, non-overlapping windows ``[i, j)`` of duration ``<= window``.

    A window qualifies when it holds at least ``min_distinct`` distinct values of ``others``
    and at least ``min_count`` rows (and, when given, ``accept(i, j)`` returns True). ``ts``
    must be ascending. Runs in O(n) with an incremental counter, so very active accounts
    (hubs) are handled efficiently.
    """
    n = len(ts)
    need = max(min_distinct, min_count, 1)
    out: list[tuple[int, int]] = []
    if n < need:
        return out
    ts_l = ts.tolist()
    oth = others.tolist()
    counter: dict[Any, int] = {}
    distinct = 0
    i = j = 0
    while i < n:
        limit = ts_l[i] + window
        while j < n and ts_l[j] <= limit:
            c = counter.get(oth[j], 0)
            if c == 0:
                distinct += 1
            counter[oth[j]] = c + 1
            j += 1
        if (
            distinct >= min_distinct
            and (j - i) >= max(min_count, 1)
            and (accept is None or accept(i, j))
        ):
            out.append((i, j))
            i = j
            counter, distinct = {}, 0
            continue
        c = counter[oth[i]] - 1
        if c == 0:
            del counter[oth[i]]
            distinct -= 1
        else:
            counter[oth[i]] = c
        i += 1
    return out


def saturating(x: float, scale: float) -> float:
    """Monotone map of ``x >= 0`` to ``[0, 1)``: ``1 - exp(-x / scale)``."""
    return float(1.0 - np.exp(-max(x, 0.0) / max(scale, 1e-9)))
