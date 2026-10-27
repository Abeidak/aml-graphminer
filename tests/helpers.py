"""Small utilities shared by the tests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from aml_graphminer.ingestion.schema import validate_transactions

BASE = pd.Timestamp("2022-09-01 08:00:00")


def make_tx(rows: Iterable[Sequence[object]], **defaults: object) -> pd.DataFrame:
    """Build a canonical table from ``(minutes, sender, receiver, amount[, label[, pattern]])`` rows."""
    records = []
    for row in rows:
        minutes, sender, receiver, amount, *rest = row
        record: dict[str, object] = {
            "timestamp": BASE + pd.Timedelta(minutes=float(minutes)),  # type: ignore[arg-type]
            "sender": sender,
            "receiver": receiver,
            "amount": float(amount),  # type: ignore[arg-type]
            "currency": "US Dollar",
            "transaction_type": "ACH",
            "label": rest[0] if rest else 0.0,
            "pattern": rest[1] if len(rest) > 1 else "",
            **defaults,
        }
        records.append(record)
    return validate_transactions(pd.DataFrame(records))
