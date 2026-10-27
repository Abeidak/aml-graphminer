"""Unified transactional schema (Layer 1).

Every adapter converts its source format to the same *canonical* table so that the
rest of the toolkit never has to know where the data came from.

Canonical columns
-----------------
======================  ===========================================================
``timestamp``           ``datetime64`` — when the transaction happened
``sender``              ``str`` — sending account identifier (bank + account for IBM)
``receiver``            ``str`` — receiving account identifier
``amount``              ``float`` — amount paid by the sender (>= 0)
``currency``            ``str`` — currency of ``amount``
``transaction_type``    ``str`` — payment format / channel (wire, cash, ACH, ...)
``amount_received``     ``float`` — amount credited to the receiver (default ``amount``)
``currency_received``   ``str`` — currency of ``amount_received``
``sender_bank``         ``str`` — bank of the sender ("" if unknown)
``receiver_bank``       ``str`` — bank of the receiver ("" if unknown)
``label``               ``float`` — 1 = laundering, 0 = legitimate, NaN = unknown
``pattern``             ``str`` — laundering motif name when known ("" otherwise)
``source_row``          ``int`` — row number in the original file (traceability)
======================  ===========================================================

The returned frame is sorted by ``timestamp`` (stable) and its index (``tx_id``) is the
transaction identifier used by detectors, models and explanations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..exceptions import SchemaError

REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "sender", "receiver", "amount")
CANONICAL_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "sender",
    "receiver",
    "amount",
    "currency",
    "transaction_type",
    "amount_received",
    "currency_received",
    "sender_bank",
    "receiver_bank",
    "label",
    "pattern",
    "source_row",
)
TEXT_COLUMNS: tuple[str, ...] = (
    "sender",
    "receiver",
    "currency",
    "transaction_type",
    "currency_received",
    "sender_bank",
    "receiver_bank",
    "pattern",
)


class Transaction(BaseModel):
    """Validated single transaction (used by the REST API and for sample validation)."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    timestamp: datetime
    sender: str = Field(min_length=1)
    receiver: str = Field(min_length=1)
    amount: float = Field(ge=0)
    currency: str = "UNK"
    transaction_type: str = "unknown"
    amount_received: float | None = Field(default=None, ge=0)
    currency_received: str | None = None
    sender_bank: str = ""
    receiver_bank: str = ""
    label: int | None = Field(default=None, ge=0, le=1)
    pattern: str = ""

    @field_validator("sender", "receiver", mode="before")
    @classmethod
    def _to_str(cls, value: Any) -> Any:
        return str(value) if isinstance(value, (int, np.integer)) else value


def _fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    if "currency" not in df:
        df["currency"] = "UNK"
    if "transaction_type" not in df:
        df["transaction_type"] = "unknown"
    if "amount_received" not in df:
        df["amount_received"] = df["amount"]
    if "currency_received" not in df:
        df["currency_received"] = df["currency"]
    for col in ("sender_bank", "receiver_bank", "pattern"):
        if col not in df:
            df[col] = ""
    if "label" not in df:
        df["label"] = np.nan
    if "source_row" not in df:
        df["source_row"] = np.arange(len(df), dtype="int64")
    return df


def validate_transactions(
    df: pd.DataFrame,
    *,
    strict: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """Validate and normalise a transaction table to the canonical schema.

    Parameters
    ----------
    df:
        Table with at least ``timestamp, sender, receiver, amount``.
    strict:
        When ``True`` (default) any invalid row raises :class:`SchemaError`. When
        ``False`` invalid rows (bad timestamp, missing account, negative or missing
        amount) are dropped and counted in ``result.attrs["dropped_rows"]``.
    sort:
        Sort by timestamp (stable) and reset the index (``tx_id``).

    Returns
    -------
    pandas.DataFrame
        A new frame with exactly :data:`CANONICAL_COLUMNS` (extra columns are kept after them).
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError(f"expected a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"missing required column(s): {missing}. Found columns: {list(df.columns)}"
        )

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    if "amount_received" in out:
        out["amount_received"] = pd.to_numeric(out["amount_received"], errors="coerce")
    for col in ("sender", "receiver"):
        out[col] = out[col].astype("string")

    bad = (
        out["timestamp"].isna()
        | out["amount"].isna()
        | (out["amount"] < 0)
        | out["sender"].isna()
        | out["receiver"].isna()
        | (out["sender"].str.len() == 0)
        | (out["receiver"].str.len() == 0)
    )
    n_bad = int(bad.sum())
    if n_bad:
        if strict:
            first = out.index[bad][0]
            raise SchemaError(
                f"{n_bad} invalid row(s) (bad timestamp / amount / account). "
                f"First offending index: {first!r}"
            )
        out = out.loc[~bad].copy()
    out.attrs["dropped_rows"] = n_bad

    out = _fill_defaults(out)
    if "amount_received" in out:
        out["amount_received"] = out["amount_received"].fillna(out["amount"])
    for col in TEXT_COLUMNS:
        out[col] = out[col].fillna("").astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="coerce").astype("float64")
    invalid_label = out["label"].notna() & ~out["label"].isin([0.0, 1.0])
    if invalid_label.any():
        raise SchemaError("label must be 0, 1 or empty")
    out["source_row"] = pd.to_numeric(out["source_row"], errors="coerce").fillna(-1).astype("int64")
    out["amount"] = out["amount"].astype("float64")
    out["amount_received"] = out["amount_received"].astype("float64")

    extra = [c for c in out.columns if c not in CANONICAL_COLUMNS]
    out = out[[*CANONICAL_COLUMNS, *extra]]
    if sort:
        out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    out.index.name = "tx_id"
    return out


def validate_records(
    records: Iterable[Mapping[str, Any]], *, max_records: int | None = None
) -> pd.DataFrame:
    """Validate JSON-like records with Pydantic and return a canonical frame.

    Used by the REST API where payloads are small and strict per-row validation is cheap.
    """
    rows = []
    for i, rec in enumerate(records):
        if max_records is not None and i >= max_records:
            raise SchemaError(f"too many records (max {max_records})")
        try:
            tx = Transaction.model_validate(dict(rec))
        except Exception as exc:
            raise SchemaError(f"record {i}: {exc}") from exc
        rows.append(tx.model_dump())
    if not rows:
        raise SchemaError("no records provided")
    return validate_transactions(pd.DataFrame(rows))


def sample_validate(df: pd.DataFrame, n: int = 1000, seed: int = 0) -> int:
    """Run the Pydantic model on a random sample of rows; return the number checked.

    Row-level Pydantic validation is too slow for millions of rows, so bulk validation
    is vectorised in :func:`validate_transactions` and this helper offers a cheap
    additional spot check.
    """
    if len(df) == 0:
        return 0
    sample = df.sample(min(n, len(df)), random_state=seed)
    cols = [c for c in Transaction.model_fields if c in sample.columns]
    for rec in sample[cols].to_dict("records"):
        rec = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
        if rec.get("label") is not None:
            rec["label"] = int(rec["label"])
        Transaction.model_validate(rec)
    return len(sample)
