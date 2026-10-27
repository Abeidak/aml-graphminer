"""Causal rolling-window features per account (Layer 2).

For every transaction we describe the *recent activity of its sender and of its receiver*
over trailing windows (1 h, 24 h, 7 d by default). Windows only look **backwards** (they
contain the current transaction and the ones that happened strictly less than ``window``
seconds before it), so the features can be computed once on the whole table without any
look-ahead leakage.

The implementation is fully vectorised (sort + ``searchsorted`` + cumulative sums, no
Python loop per account), so it scales to millions of transactions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils import epoch_seconds, parse_duration


@dataclass(frozen=True)
class WindowStats:
    """Trailing-window aggregates aligned with the input rows.

    ``variance`` is computed from values *centred* on the global mean before the cumulative
    sums are taken, which avoids the catastrophic cancellation of ``E[x²] - E[x]²``.
    """

    count: np.ndarray
    total: np.ndarray
    total_sq: np.ndarray
    variance: np.ndarray | None = None

    @property
    def mean(self) -> np.ndarray:
        return self.total / np.maximum(self.count, 1)

    @property
    def std(self) -> np.ndarray:
        if self.variance is not None:
            return np.sqrt(np.maximum(self.variance, 0.0))
        var = self.total_sq / np.maximum(self.count, 1) - self.mean**2
        return np.sqrt(np.maximum(var, 0.0))


def trailing_window_stats(
    group_ids: np.ndarray,
    ts: np.ndarray,
    values: np.ndarray,
    window: float,
) -> WindowStats:
    """Per-group trailing-window count / sum / sum of squares.

    For row *i* the window is ``(ts_i - window, ts_i]`` restricted to rows of the same group.
    The current row is included, so ``count >= 1``. Rows with *identical* timestamps are
    ordered by their position in the input (stable), so a row never sees a later row of the
    same instant: this keeps the features strictly causal.

    Parameters
    ----------
    group_ids:
        Integer group id of every row (e.g. the sender account index).
    ts:
        Time of every row, in seconds (float).
    values:
        Value to aggregate.
    window:
        Window length in seconds (> 0).
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    n = len(ts)
    if n == 0:
        empty = np.zeros(0)
        return WindowStats(empty, empty, empty, empty)
    group_ids = np.asarray(group_ids, dtype="int64")
    ts = np.asarray(ts, dtype="float64")
    values = np.asarray(values, dtype="float64")

    order = np.lexsort((ts, group_ids))
    g, t, v = group_ids[order], ts[order], values[order]

    t_min = float(t.min())
    span = float(t.max()) - t_min
    big = span + window + 1.0  # keeps the key ranges of different groups disjoint
    key = g.astype("float64") * big + (t - t_min)
    left = np.searchsorted(key, key - window, side="right")  # first row with ts > ts_i - window

    shift = float(v.mean())
    vc = v - shift  # centred values: keeps the cumulative sums of squares well conditioned
    csum = np.concatenate([[0.0], np.cumsum(v)])
    csq = np.concatenate([[0.0], np.cumsum(v * v)])
    csum_c = np.concatenate([[0.0], np.cumsum(vc)])
    csq_c = np.concatenate([[0.0], np.cumsum(vc * vc)])
    idx = np.arange(n)
    cnt = (idx - left + 1).astype("float64")
    tot = csum[idx + 1] - csum[left]
    tot_sq = csq[idx + 1] - csq[left]
    tot_c = csum_c[idx + 1] - csum_c[left]
    tot_sq_c = csq_c[idx + 1] - csq_c[left]
    var = tot_sq_c / cnt - (tot_c / cnt) ** 2

    out_c, out_t, out_s, out_v = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    out_c[order], out_t[order], out_s[order], out_v[order] = cnt, tot, tot_sq, var
    return WindowStats(out_c, out_t, out_s, out_v)


def previous_gap(group_ids: np.ndarray, ts: np.ndarray, default: float) -> np.ndarray:
    """Seconds since the previous event of the same group (``default`` for the first one)."""
    n = len(ts)
    if n == 0:
        return np.zeros(0)
    group_ids = np.asarray(group_ids, dtype="int64")
    ts = np.asarray(ts, dtype="float64")
    order = np.lexsort((ts, group_ids))
    g, t = group_ids[order], ts[order]
    gap = np.full(n, float(default))
    same = g[1:] == g[:-1]
    gap[1:][same] = t[1:][same] - t[:-1][same]
    out = np.empty(n)
    out[order] = gap
    return out


def temporal_feature_names(windows: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    for side in ("s", "r"):  # sender-side / receiver-side activity
        for w in windows:
            names += [f"{side}_cnt_{w}", f"{side}_sum_{w}", f"{side}_mean_{w}", f"{side}_std_{w}"]
        names.append(f"{side}_gap")
    return names


def temporal_features(
    df: pd.DataFrame,
    src: np.ndarray,
    dst: np.ndarray,
    windows: tuple[str, ...] = ("1h", "24h", "7d"),
) -> pd.DataFrame:
    """Trailing-window activity of the sender and receiver of every transaction.

    Sender side uses the account's *outgoing* history (amounts paid), receiver side its
    *incoming* history (amounts received). Counts and sums are ``log1p``-compressed;
    mean / std are computed on ``log1p(amount)``. A window named ``"all"`` is the whole
    history seen so far (expanding window).
    """
    ts = epoch_seconds(df["timestamp"])
    paid = df["amount"].to_numpy(dtype="float64")
    recv = df["amount_received"].to_numpy(dtype="float64")
    cols: dict[str, np.ndarray] = {}
    horizon = float(ts.max() - ts.min()) + 1.0 if len(ts) else 1.0
    seconds_of = {w: (horizon if w == "all" else parse_duration(w)) for w in windows}
    max_gap = max(seconds_of.values()) * 4.0
    for side, groups, amounts in (("s", src, paid), ("r", dst, recv)):
        for w in windows:
            seconds = seconds_of[w]
            raw = trailing_window_stats(groups, ts, amounts, seconds)
            logv = trailing_window_stats(groups, ts, np.log1p(amounts), seconds)
            cols[f"{side}_cnt_{w}"] = np.log1p(raw.count)
            cols[f"{side}_sum_{w}"] = np.log1p(raw.total)
            cols[f"{side}_mean_{w}"] = logv.mean
            cols[f"{side}_std_{w}"] = logv.std
        cols[f"{side}_gap"] = np.log1p(previous_gap(groups, ts, max_gap))
    ordered = temporal_feature_names(windows)
    return pd.DataFrame({k: cols[k] for k in ordered}, index=df.index)
