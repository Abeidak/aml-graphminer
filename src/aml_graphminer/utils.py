"""Small helpers shared by several layers (time handling, seeding, optional imports)."""

from __future__ import annotations

import importlib
import logging
import random
from types import ModuleType

import numpy as np
import pandas as pd

from .exceptions import OptionalDependencyError

logger = logging.getLogger("aml_graphminer")

_EPOCH = pd.Timestamp("1970-01-01")


def epoch_seconds(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Return timestamps as float64 seconds since 1970-01-01.

    Unlike ``astype("int64")`` this does not depend on the datetime resolution
    (pandas >= 3 infers ``s``/``us``/``ns`` from the input), so it is safe to use
    everywhere a numeric time axis is needed.
    """
    values = pd.Series(ts) if not isinstance(ts, pd.Series) else ts
    return ((values - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype="float64")


def parse_duration(value: str | float | int | pd.Timedelta) -> float:
    """Parse ``"1h"``, ``"24h"``, ``"7d"``, ``"90min"``, ``"30s"`` or a number of seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    return pd.Timedelta(str(value)).total_seconds()


def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy`` and (if installed) ``torch``."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - torch is optional
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def require(package: str, extra: str, feature: str = "") -> ModuleType:
    """Import ``package`` or raise :class:`OptionalDependencyError` with install hint."""
    try:
        return importlib.import_module(package)
    except ImportError as exc:
        raise OptionalDependencyError(package, extra, feature) from exc


def has_module(package: str) -> bool:
    """True when ``package`` can be imported."""
    try:
        importlib.import_module(package)
    except ImportError:
        return False
    return True


def shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy (nats) of a vector of non-negative weights."""
    counts = np.asarray(counts, dtype="float64")
    total = counts.sum()
    if total <= 0 or len(counts) <= 1:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())
