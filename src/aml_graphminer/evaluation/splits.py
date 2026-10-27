"""Temporal train / validation / test split (no look-ahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..exceptions import ConfigError


def temporal_split(
    df: pd.DataFrame, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> dict[str, np.ndarray]:
    """Split by *time order* like the original notebooks (60 / 20 / 20 by default).

    ``df`` must be sorted by timestamp (``validate_transactions`` guarantees it). Returns
    boolean masks ``{"train", "val", "test"}`` over the rows of ``df``.
    """
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) <= 0:
        raise ConfigError("fractions must be three positive numbers summing to 1")
    ts = df["timestamp"].to_numpy()
    if len(ts) > 1 and (ts[1:] < ts[:-1]).any():
        raise ConfigError("the transaction table must be sorted by timestamp")
    n = len(df)
    i1 = int(fractions[0] * n)
    i2 = int((fractions[0] + fractions[1]) * n)
    idx = np.arange(n)
    return {"train": idx < i1, "val": (idx >= i1) & (idx < i2), "test": idx >= i2}
