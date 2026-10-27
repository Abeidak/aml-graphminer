"""IBM *Transactions for Anti-Money Laundering* adapter (CDC §5.3: ``ibm_adapter``).

Everything needed for the IBM AML files (Altman et al., NeurIPS 2023): the position-based
adapter of ``*_Trans.csv``, the parser of the ``*_Patterns.txt`` ground truth, the one-call
loader that attaches the motif names, and the writer of the IBM layout. The implementation lives
in :mod:`~aml_graphminer.ingestion.adapters`, :mod:`~aml_graphminer.ingestion.ibm_patterns`,
:mod:`~aml_graphminer.ingestion.dataset` and :mod:`~aml_graphminer.ingestion.readers`.

Example
-------
>>> from aml_graphminer.ingestion.ibm_adapter import load_dataset
>>> df = load_dataset("HI-Small_Trans.csv")   # + HI-Small_Patterns.txt when present  # doctest: +SKIP
"""

from __future__ import annotations

from .adapters import IBM_COLUMNS, IBM_TIME_FORMAT, IBMAdapter, normalize_bank_id
from .dataset import find_patterns_file, load_dataset
from .ibm_patterns import attach_patterns, normalize_pattern_name, read_ibm_patterns
from .readers import write_ibm_csv

__all__ = [
    "IBM_COLUMNS",
    "IBM_TIME_FORMAT",
    "IBMAdapter",
    "attach_patterns",
    "find_patterns_file",
    "load_dataset",
    "normalize_bank_id",
    "normalize_pattern_name",
    "read_ibm_patterns",
    "write_ibm_csv",
]
