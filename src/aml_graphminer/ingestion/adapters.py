"""Source-format adapters (Layer 1): everything is converted to the canonical schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

import numpy as np
import pandas as pd

from ..exceptions import IngestionError
from .schema import validate_transactions


class Adapter(ABC):
    """Convert a raw frame (as read from disk) to the canonical transactional schema."""

    name: ClassVar[str]

    @abstractmethod
    def matches(self, columns: list[str]) -> bool:
        """Return True when ``columns`` look like this adapter's source format."""

    @abstractmethod
    def to_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return a (not yet validated) frame with canonical column names."""

    def convert(self, raw: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
        try:
            frame = self.to_frame(raw)
        except IngestionError:
            raise
        except KeyError as exc:
            raise IngestionError(f"{self.name}: column {exc} not found in input") from exc
        return validate_transactions(frame, strict=strict)


class CanonicalAdapter(Adapter):
    """Input already uses the canonical column names."""

    name = "canonical"

    def matches(self, columns: list[str]) -> bool:
        return {"timestamp", "sender", "receiver", "amount"} <= set(columns)

    def to_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        frame = raw.copy()
        if "source_row" not in frame:
            frame["source_row"] = np.arange(len(frame), dtype="int64")
        return frame


class CsvAdapter(Adapter):
    """Generic adapter driven by a ``{canonical_name: source_column}`` mapping.

    Example
    -------
    >>> adapter = CsvAdapter({"timestamp": "date", "sender": "from", "receiver": "to",
    ...                       "amount": "value", "label": "is_fraud"})
    """

    name = "mapping"

    def __init__(self, mapping: Mapping[str, str]) -> None:
        required = {"timestamp", "sender", "receiver", "amount"}
        missing = required - set(mapping)
        if missing:
            raise IngestionError(f"mapping must define {sorted(missing)}")
        self.mapping = dict(mapping)

    def matches(self, columns: list[str]) -> bool:
        return all(src in columns for src in self.mapping.values())

    def to_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        absent = [src for src in self.mapping.values() if src not in raw.columns]
        if absent:
            raise IngestionError(f"mapped source column(s) not found: {absent}")
        frame = pd.DataFrame({dst: raw[src] for dst, src in self.mapping.items()})
        frame["source_row"] = np.arange(len(frame), dtype="int64")
        return frame


# --------------------------------------------------------------------------- IBM AML
IBM_COLUMNS: tuple[str, ...] = (
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
)
_IBM_SHORT = (
    "timestamp",
    "sender_bank",
    "sender_acc",
    "receiver_bank",
    "receiver_acc",
    "amount_received",
    "currency_received",
    "amount",
    "currency",
    "transaction_type",
    "label",
)
IBM_TIME_FORMAT = "%Y/%m/%d %H:%M"


def normalize_bank_id(value: object) -> str:
    """Bank ids appear as ``"011"`` in some IBM files and ``11`` in others → ``"11"``."""
    text = str(value).strip()
    return str(int(text)) if text.isdigit() else text


class IBMAdapter(Adapter):
    """IBM *Transactions for Anti-Money Laundering* (Altman et al., NeurIPS 2023).

    The files have 10 columns (+ ``Is Laundering`` in the labelled files) and two columns
    named ``Account``. The mapping is therefore done **by position**, like the original
    Colab notebooks, so it works with pandas (``Account.1``) and polars
    (``Account_duplicated_0``) alike. Account identifiers are ``"<bank>_<account>"``.
    """

    name = "ibm"

    def matches(self, columns: list[str]) -> bool:
        cols = [str(c) for c in columns]
        return len(cols) in (10, 11) and cols[0] == "Timestamp" and "Payment Format" in cols

    def to_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.shape[1] not in (10, 11):
            raise IngestionError(
                f"ibm: expected 10 or 11 columns (IBM AML format), got {raw.shape[1]}"
            )
        frame = raw.copy()
        names = list(_IBM_SHORT[: frame.shape[1]])
        frame.columns = names
        if "label" not in frame:
            frame["label"] = np.nan
        frame["timestamp"] = _parse_ibm_time(frame["timestamp"])
        frame["sender_bank"] = frame["sender_bank"].map(normalize_bank_id)
        frame["receiver_bank"] = frame["receiver_bank"].map(normalize_bank_id)
        frame["sender"] = frame["sender_bank"] + "_" + frame["sender_acc"].astype(str)
        frame["receiver"] = frame["receiver_bank"] + "_" + frame["receiver_acc"].astype(str)
        frame = frame.drop(columns=["sender_acc", "receiver_acc"])
        frame["source_row"] = np.arange(len(frame), dtype="int64")
        return frame


def _parse_ibm_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format=IBM_TIME_FORMAT, errors="coerce")
    if parsed.isna().any():  # fall back to inference (other date formats, seconds, ISO ...)
        fallback = pd.to_datetime(series, errors="coerce")
        parsed = parsed.fillna(fallback)
    return parsed


ADAPTERS: tuple[type[Adapter], ...] = (IBMAdapter, CanonicalAdapter)


def detect_adapter(columns: list[str]) -> Adapter:
    """Pick the first built-in adapter that recognises ``columns``."""
    for cls in ADAPTERS:
        adapter = cls()  # type: ignore[call-arg]
        if adapter.matches(list(columns)):
            return adapter
    raise IngestionError(
        "could not detect the input format from columns "
        f"{list(columns)[:12]}. Pass schema='ibm' or schema={{'timestamp': 'my_col', ...}}."
    )
