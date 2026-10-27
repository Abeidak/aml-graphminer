"""Parser for the ``*_Patterns.txt`` files shipped with the IBM AML datasets.

These files list every laundering *attempt* (the injected motif) between
``BEGIN LAUNDERING ATTEMPT - <TYPE>`` / ``END LAUNDERING ATTEMPT - <TYPE>`` markers, one
transaction per line using the same comma-separated layout as ``*_Trans.csv``. They
allow **per-motif evaluation** (how many fan-in / cycle / ... transactions does a detector
recover?) on top of the binary ``Is Laundering`` label.

.. note::
   The parser is written from the published description of the file format and is
   tolerant (type names are normalised, 10- or 11-field lines are accepted). Validate it
   on your copy of the real files with ``amlgm inspect --patterns <file>``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..exceptions import IngestionError
from .adapters import IBMAdapter

_BEGIN = re.compile(r"^\s*BEGIN LAUNDERING ATTEMPT\s*-\s*(?P<kind>[^:]+?)\s*(?::.*)?$", re.I)
_END = re.compile(r"^\s*END LAUNDERING ATTEMPT", re.I)


def normalize_pattern_name(name: str) -> str:
    """``"FAN-OUT"`` → ``"fan_out"``, ``"GATHER-SCATTER"`` → ``"gather_scatter"``."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def read_ibm_patterns(path: str | Path) -> pd.DataFrame:
    """Read a ``*_Patterns.txt`` file.

    Returns
    -------
    pandas.DataFrame
        Canonical transaction columns plus ``pattern`` (normalised motif type) and
        ``attempt_id`` (one id per BEGIN/END block).
    """
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"patterns file not found: {path}")

    rows: list[list[str]] = []
    kinds: list[str] = []
    attempts: list[int] = []
    current: str | None = None
    attempt = -1
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            begin = _BEGIN.match(line)
            if begin:
                current = normalize_pattern_name(begin.group("kind"))
                attempt += 1
                continue
            if _END.match(line):
                current = None
                continue
            if current is None:
                continue
            fields = [f.strip() for f in line.split(",")]
            if len(fields) not in (10, 11):
                raise IngestionError(
                    f"unexpected line in patterns file (not 10/11 fields): {line!r}"
                )
            if len(fields) == 10:
                fields.append("1")
            rows.append(fields)
            kinds.append(current)
            attempts.append(attempt)

    if not rows:
        return pd.DataFrame(columns=["pattern", "attempt_id"])

    raw = pd.DataFrame(rows)
    frame = IBMAdapter().convert(raw, strict=False)
    frame = frame.drop(columns=["pattern"], errors="ignore")
    frame["pattern"] = np.asarray(kinds)[frame["source_row"].to_numpy()]
    frame["attempt_id"] = np.asarray(attempts)[frame["source_row"].to_numpy()]
    frame["label"] = 1.0
    return frame


_KEY = ["timestamp", "sender", "receiver", "amount", "transaction_type"]


def attach_patterns(transactions: pd.DataFrame, patterns: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill ``transactions['pattern']`` from the parsed patterns file.

    Matching is done on ``(timestamp, sender, receiver, amount, transaction_type)``.

    Returns
    -------
    (frame, n_matched)
        A copy of ``transactions`` with the ``pattern`` column filled where a match exists,
        and the number of pattern transactions that were found in ``transactions``.
    """
    if patterns.empty:
        return transactions.copy(), 0
    lookup = patterns[[*_KEY, "pattern"]].drop_duplicates(subset=_KEY, keep="first")
    out = transactions.copy()
    index_name = out.index.name or "tx_id"
    merged = (
        out.drop(columns=["pattern"])
        .reset_index()
        .merge(lookup, how="left", on=_KEY)
        .set_index(index_name)
    )
    merged["pattern"] = merged["pattern"].fillna("")
    out["pattern"] = merged["pattern"].to_numpy()
    n_matched = int((out["pattern"] != "").sum())
    return out, n_matched
