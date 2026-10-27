"""One-call dataset loading: transactions (+ IBM ``*_Patterns.txt`` ground truth when present)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..exceptions import IngestionError
from ..utils import logger
from .ibm_patterns import attach_patterns, read_ibm_patterns
from .readers import read_transactions


def find_patterns_file(path: str | Path) -> Path | None:
    """``HI-Small_Trans.csv`` → ``HI-Small_Patterns.txt`` next to it (``None`` if absent)."""
    path = Path(path)
    name = path.name
    if "_Trans" not in name:
        return None
    candidate = path.with_name(name.split("_Trans")[0] + "_Patterns.txt")
    return candidate if candidate.exists() else None


def load_dataset(
    path: str | Path,
    *,
    schema: str = "auto",
    nrows: int | None = None,
    patterns_file: str | Path | bool | None = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read ``path`` into the canonical schema and attach laundering-pattern names.

    Parameters
    ----------
    patterns_file:
        ``True`` (default) → use the sibling ``*_Patterns.txt`` of an IBM file if it exists;
        ``False``/``None`` → do not attach patterns; a path → use that file.
    """
    df = read_transactions(path, schema=schema, nrows=nrows, **kwargs)
    if patterns_file is True:
        patterns_path = find_patterns_file(path)
    elif patterns_file:
        patterns_path = Path(patterns_file)
        if not patterns_path.exists():
            raise IngestionError(f"patterns file not found: {patterns_path}")
    else:
        patterns_path = None
    if patterns_path is not None:
        patterns = read_ibm_patterns(patterns_path)
        df, matched = attach_patterns(df, patterns)
        logger.info(
            "attached %d/%d laundering-pattern transactions from %s",
            matched,
            len(patterns),
            patterns_path.name,
        )
    return df


__all__ = ["find_patterns_file", "load_dataset", "summarize"]


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Descriptive statistics of a canonical transaction table (JSON-serialisable)."""
    labelled = df["label"].notna()
    illicit = int((df["label"] == 1.0).sum())
    accounts = pd.unique(pd.concat([df["sender"], df["receiver"]], ignore_index=True))
    summary: dict[str, Any] = {
        "transactions": len(df),
        "accounts": len(accounts),
        "first_timestamp": None if df.empty else str(df["timestamp"].min()),
        "last_timestamp": None if df.empty else str(df["timestamp"].max()),
        "currencies": int(df["currency"].nunique()),
        "payment_formats": sorted(df["transaction_type"].astype(str).unique().tolist()),
        "labelled": int(labelled.sum()),
        "illicit": illicit,
        "illicit_ratio": float(illicit / max(int(labelled.sum()), 1)),
        "total_amount": float(df["amount"].sum()),
    }
    if "pattern" in df.columns and (df["pattern"] != "").any():
        summary["patterns"] = {
            str(k): int(v) for k, v in df.loc[df["pattern"] != "", "pattern"].value_counts().items()
        }
    return summary
