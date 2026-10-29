"""Shared building blocks of the structural-motif detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from ...features.network import index_accounts
from ...utils import epoch_seconds

if TYPE_CHECKING:
    from .config import CurrencyConfig


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
    # value of one unit of each currency in the reference currency (NaN: unknown) and the paid /
    # received amounts converted with it (NaN where the rate is unknown)
    fx: np.ndarray = field(default_factory=lambda: np.zeros(0))
    paid_ref: np.ndarray = field(default_factory=lambda: np.zeros(0))
    recv_ref: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # amounts on one scale: the reference currency when the rate is known, raw otherwise
    paid_c: np.ndarray = field(default_factory=lambda: np.zeros(0))
    recv_c: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def from_frame(cls, df: pd.DataFrame, currency: CurrencyConfig | None = None) -> TxArrays:
        """Build the view; ``currency`` controls the conversion to a reference currency."""
        ts = epoch_seconds(df["timestamp"])
        if len(ts) > 1 and (ts[1:] < ts[:-1]).any():
            raise ValueError("the transaction table must be sorted by timestamp")
        src, dst, n, names = index_accounts(df)
        codes, uniques = pd.factorize(
            pd.concat([df["currency"], df["currency_received"]], ignore_index=True)
        )
        m = len(df)
        paid = df["amount"].to_numpy(dtype="float64")
        recv = df["amount_received"].to_numpy(dtype="float64")
        cur_paid = np.asarray(codes[:m], dtype="int64")
        cur_recv = np.asarray(codes[m:], dtype="int64")
        currency_names = np.asarray(uniques, dtype=object)
        if currency is None or currency.convert:
            fx = exchange_rates(
                paid,
                recv,
                cur_paid,
                cur_recv,
                currency_names,
                reference=None if currency is None else currency.reference,
                known=None if currency is None else currency.rates,
            )
        else:
            fx = np.full(len(currency_names), np.nan)
        rate_paid = fx[cur_paid] if len(fx) else np.full(m, np.nan)
        rate_recv = fx[cur_recv] if len(fx) else np.full(m, np.nan)
        return cls(
            ts=ts,
            src=src,
            dst=dst,
            n_nodes=n,
            names=names,
            paid=paid,
            recv=recv,
            cur_paid=cur_paid,
            cur_recv=cur_recv,
            currency_names=currency_names,
            tx_ids=df.index.to_numpy(),
            timestamps=df["timestamp"].reset_index(drop=True),
            label=df["label"].to_numpy(dtype="float64"),
            fx=fx,
            paid_ref=paid * rate_paid,
            recv_ref=recv * rate_recv,
            paid_c=np.where(np.isfinite(rate_paid), paid * rate_paid, paid),
            recv_c=np.where(np.isfinite(rate_recv), recv * rate_recv, recv),
        )

    def __len__(self) -> int:
        return len(self.ts)

    def currency_code(self, name: str) -> int:
        hits = np.flatnonzero(self.currency_names == name)
        return int(hits[0]) if len(hits) else -1

    def hop_amounts(
        self, received_rows: np.ndarray, paid_rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """What ``received_rows`` brought in and ``paid_rows`` forwarded, on a common scale.

        Pairs in the same currency keep their raw amounts (exact); pairs in two currencies are
        compared in the reference currency when both rates are known. Returns
        ``(received, forwarded, comparable)``.
        """
        same = self.cur_recv[received_rows] == self.cur_paid[paid_rows]
        r_ref, p_ref = self.recv_ref[received_rows], self.paid_ref[paid_rows]
        converted = ~same & np.isfinite(r_ref) & np.isfinite(p_ref)
        received = np.where(same, self.recv[received_rows], r_ref)
        forwarded = np.where(same, self.paid[paid_rows], p_ref)
        return received, forwarded, same | converted


def exchange_rates(
    paid: np.ndarray,
    recv: np.ndarray,
    cur_paid: np.ndarray,
    cur_recv: np.ndarray,
    currency_names: np.ndarray,
    reference: str | None = None,
    known: dict[str, float] | None = None,
) -> np.ndarray:
    """Value of one unit of every currency, in units of the ``reference`` currency.

    A payment of ``paid`` units of currency ``a`` that arrives as ``recv`` units of ``b`` reveals
    ``rate(a) / rate(b) = recv / paid``. The median log-ratio of every exchanged pair is taken
    (robust to the odd rounded amount) and the log-rates are solved by weighted least squares
    with ``rate(reference) = 1``. A currency never exchanged with the reference, directly or
    through other currencies, gets ``NaN`` (its amounts are not converted). ``reference``
    defaults to the most used payment currency; ``known`` rates (units of the reference per
    unit) replace the inferred ones.
    """
    k = len(currency_names)
    names = [str(c) for c in currency_names]
    if k == 0:
        return np.zeros(0)
    if reference not in names:
        counts = np.bincount(cur_paid, minlength=k) if len(cur_paid) else np.ones(k)
        reference = names[int(np.argmax(counts))]
    ref = names.index(str(reference))
    rates = np.full(k, np.nan)
    rates[ref] = 0.0  # log-rate
    cross = (cur_paid != cur_recv) & (paid > 0) & (recv > 0)
    if cross.any():
        pair = cur_paid[cross] * k + cur_recv[cross]
        grouped = pd.Series(np.log(recv[cross] / paid[cross])).groupby(pair)
        median, size = grouped.median(), grouped.size()
        a, b = np.divmod(median.index.to_numpy(dtype="int64"), k)
        # currencies connected to the reference through exchanged pairs
        reach, frontier = {ref}, [ref]
        while frontier:
            c = frontier.pop()
            linked = np.r_[b[a == c], a[b == c]].tolist()
            for other in linked:
                if other not in reach:
                    reach.add(other)
                    frontier.append(other)
        unknown = sorted(reach - {ref})
        if unknown:
            col = {c: i for i, c in enumerate(unknown)}
            keep = np.array([x in reach for x in a.tolist()])
            design = np.zeros((int(keep.sum()), len(unknown)))
            for i, (x, y) in enumerate(zip(a[keep].tolist(), b[keep].tolist())):
                if x in col:
                    design[i, col[x]] += 1.0
                if y in col:
                    design[i, col[y]] -= 1.0
            weight = np.sqrt(size.to_numpy(dtype="float64")[keep])
            solution = np.linalg.lstsq(
                design * weight[:, None], median.to_numpy()[keep] * weight, rcond=None
            )[0]
            for c, i in col.items():
                rates[c] = solution[i]
    rates = np.exp(rates)
    if known:
        scale = float(known.get(names[ref], 1.0))
        for name, value in known.items():
            if name in names and value > 0:
                rates[names.index(name)] = float(value) / scale
    return rates


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

    def _repr_html_(self) -> str:  # notebooks: the alert and its evidence
        from ...jupyter.display import alert_html

        return alert_html(self)


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
        amounts = ctx.paid_c[rows] if amount is None else amount
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
