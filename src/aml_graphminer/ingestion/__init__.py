"""Layer 1 — ingestion: adapters from CSV / JSON / Parquet / IBM AML / Elliptic to one schema."""

from .adapters import (
    ADAPTERS,
    Adapter,
    CanonicalAdapter,
    CsvAdapter,
    IBMAdapter,
    detect_adapter,
    normalize_bank_id,
)
from .dataset import find_patterns_file, load_dataset, summarize
from .elliptic import EllipticData, read_elliptic
from .ibm_patterns import attach_patterns, normalize_pattern_name, read_ibm_patterns
from .readers import SUPPORTED_FORMATS, infer_format, read_raw, read_transactions, write_ibm_csv
from .schema import (
    CANONICAL_COLUMNS,
    REQUIRED_COLUMNS,
    Transaction,
    sample_validate,
    validate_records,
    validate_transactions,
)

__all__ = [
    "ADAPTERS",
    "CANONICAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SUPPORTED_FORMATS",
    "Adapter",
    "CanonicalAdapter",
    "CsvAdapter",
    "EllipticData",
    "IBMAdapter",
    "Transaction",
    "attach_patterns",
    "detect_adapter",
    "find_patterns_file",
    "infer_format",
    "load_dataset",
    "normalize_bank_id",
    "normalize_pattern_name",
    "read_elliptic",
    "read_ibm_patterns",
    "read_raw",
    "read_transactions",
    "sample_validate",
    "summarize",
    "validate_records",
    "validate_transactions",
    "write_ibm_csv",
]
