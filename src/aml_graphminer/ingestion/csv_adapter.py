"""Generic tabular adapters — CSV / TSV / JSON / JSON-lines / Parquet (CDC §5.3: ``csv_adapter``).

The implementation lives in :mod:`aml_graphminer.ingestion.adapters` (adapter classes) and
:mod:`aml_graphminer.ingestion.readers` (file readers); this module gathers the public names
under the file name used by the project specification.

Example
-------
>>> from aml_graphminer.ingestion.csv_adapter import CsvAdapter, read_transactions
>>> adapter = CsvAdapter({"timestamp": "date", "sender": "from", "receiver": "to",
...                       "amount": "value"})                       # doctest: +SKIP
>>> df = read_transactions("payments.csv", schema=adapter)            # doctest: +SKIP
"""

from __future__ import annotations

from .adapters import Adapter, CanonicalAdapter, CsvAdapter, detect_adapter
from .readers import SUPPORTED_FORMATS, infer_format, read_raw, read_transactions
from .schema import CANONICAL_COLUMNS, REQUIRED_COLUMNS, validate_transactions

__all__ = [
    "CANONICAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SUPPORTED_FORMATS",
    "Adapter",
    "CanonicalAdapter",
    "CsvAdapter",
    "detect_adapter",
    "infer_format",
    "read_raw",
    "read_transactions",
    "validate_transactions",
]
