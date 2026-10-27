"""Helpers to score some transactions *in the context of* others (shared by all detectors)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Transactions of the same instant are put in a canonical order (context first, then by content,
# then by transaction id) so that scores never depend on the order in which rows were supplied.
_TIE_BREAK: tuple[str, ...] = (
    "sender",
    "receiver",
    "amount",
    "amount_received",
    "transaction_type",
    "currency",
    "currency_received",
)
_HELPERS = ["_target", "_pos", "_label"]


def stack_context(
    df: pd.DataFrame, context: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Merge the transactions to score with optional history, sorted by time.

    Returns ``(full, is_target, target_pos)``:

    * ``full`` — context + ``df`` sorted by timestamp, fresh ``RangeIndex``; rows of the same
      timestamp are ordered canonically (context first, then by content and transaction id),
      so shuffling the input never changes the result;
    * ``is_target`` — boolean mask of the rows of ``full`` that come from ``df``;
    * ``target_pos`` — for those rows, the position of the row in ``df``.
    """
    target = df.copy()
    target["_target"] = True
    target["_pos"] = np.arange(len(df))
    target["_label"] = df.index.astype(str)
    parts = [target]
    if context is not None and len(context):
        past = context.copy()
        past["_target"] = False
        past["_pos"] = -1
        past["_label"] = context.index.astype(str)
        parts.insert(0, past)
    full = pd.concat(parts, ignore_index=True)
    keys = ["timestamp", "_target", *[c for c in _TIE_BREAK if c in full.columns], "_label"]
    full = full.sort_values(keys, kind="stable").reset_index(drop=True)
    is_target = full["_target"].to_numpy(dtype=bool)
    target_pos = full.loc[is_target, "_pos"].to_numpy(dtype=int)
    return full.drop(columns=_HELPERS), is_target, target_pos


def restore_order(
    values_full: np.ndarray, is_target: np.ndarray, target_pos: np.ndarray, n: int
) -> np.ndarray:
    """Per-row values of ``full`` → values aligned with the original ``df`` (length ``n``)."""
    out = np.zeros(n, dtype="float64")
    out[target_pos] = np.asarray(values_full, dtype="float64")[is_target]
    return out


__all__ = ["restore_order", "stack_context"]
