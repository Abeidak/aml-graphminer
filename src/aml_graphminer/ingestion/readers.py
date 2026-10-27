"""File readers (CSV / JSON / JSON-lines / Parquet) and the :func:`read_transactions` entry point."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..exceptions import IngestionError
from ..utils import require
from .adapters import (
    IBM_COLUMNS,
    Adapter,
    CanonicalAdapter,
    CsvAdapter,
    IBMAdapter,
    detect_adapter,
)

_SUFFIX_FORMAT = {
    ".csv": "csv",
    ".txt": "csv",
    ".tsv": "tsv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
    ".pq": "parquet",
}
SUPPORTED_FORMATS = ("csv", "tsv", "json", "jsonl", "parquet")


def infer_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in _SUFFIX_FORMAT:
        raise IngestionError(
            f"cannot infer the file format from {Path(path).name!r}; "
            f"pass fmt= one of {SUPPORTED_FORMATS}"
        )
    return _SUFFIX_FORMAT[suffix]


def read_raw(
    path: str | Path,
    *,
    fmt: str | None = None,
    nrows: int | None = None,
    engine: str = "pandas",
    str_positions: Sequence[int] = (),
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a file into a raw frame without any schema interpretation.

    ``str_positions`` lists the (0-based) positions of CSV columns that must be read as text
    (identifiers such as ``"011"`` or ``"8000ECA90"``); positions are used instead of names
    because the IBM files contain two columns called ``Account``.

    ``engine="polars"`` uses the Polars CSV reader (much faster and lighter on memory for
    the 30-180 M-row IBM files) and converts the result to pandas; it needs the ``scale`` extra.
    """
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"file not found: {path}")
    fmt = fmt or infer_format(path)
    if fmt not in SUPPORTED_FORMATS:
        raise IngestionError(f"unsupported format {fmt!r}; choose among {SUPPORTED_FORMATS}")

    if fmt in ("csv", "tsv"):
        sep = "\t" if fmt == "tsv" else kwargs.pop("sep", ",")
        if engine == "polars":  # pragma: no cover - optional dependency
            pl = require("polars", "scale", "the polars CSV engine")
            names = pl.read_csv(path, separator=sep, n_rows=0).columns
            overrides = {names[i]: pl.Utf8 for i in str_positions if i < len(names)}
            return pl.read_csv(
                path, separator=sep, n_rows=nrows, schema_overrides=overrides or None
            ).to_pandas()
        dtype = None
        if str_positions:
            names = pd.read_csv(path, sep=sep, nrows=0, **kwargs).columns
            dtype = {names[i]: str for i in str_positions if i < len(names)}
        return pd.read_csv(path, sep=sep, nrows=nrows, dtype=dtype, **kwargs)
    if fmt == "json":
        frame = pd.read_json(path, **kwargs)
        return frame.head(nrows) if nrows else frame
    if fmt == "jsonl":
        return pd.read_json(path, lines=True, nrows=nrows, **kwargs)
    # parquet
    require("pyarrow", "parquet", "reading Parquet files")
    frame = pd.read_parquet(path, **kwargs)
    return frame.head(nrows) if nrows else frame


def _resolve_adapter(schema: str | Mapping[str, str] | Adapter, columns: list[str]) -> Adapter:
    if isinstance(schema, Adapter):
        return schema
    if isinstance(schema, Mapping):
        return CsvAdapter(schema)
    if schema == "auto":
        return detect_adapter(columns)
    if schema == "ibm":
        return IBMAdapter()
    if schema == "canonical":
        return CanonicalAdapter()
    raise IngestionError(
        f"unknown schema {schema!r}: use 'auto', 'ibm', 'canonical', a column mapping or an Adapter"
    )


def read_transactions(
    source: str | Path | pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    schema: str | Mapping[str, str] | Adapter = "auto",
    fmt: str | None = None,
    nrows: int | None = None,
    engine: str = "pandas",
    strict: bool = True,
    **read_kwargs: Any,
) -> pd.DataFrame:
    """Load transactions from any supported source into the canonical schema.

    Parameters
    ----------
    source:
        Path to a CSV / TSV / JSON / JSON-lines / Parquet file, a DataFrame or a list of records.
    schema:
        ``"auto"`` (detect IBM AML or canonical columns), ``"ibm"``, ``"canonical"``, a
        ``{canonical_name: source_column}`` mapping, or an :class:`Adapter` instance.
    fmt:
        File format when it cannot be inferred from the extension.
    nrows:
        Read only the first ``nrows`` rows (handy for quick experiments).
    engine:
        ``"pandas"`` (default) or ``"polars"`` (CSV only, needs the ``scale`` extra).
    strict:
        Raise on invalid rows (default) or drop them (``False``).
    """
    if isinstance(source, pd.DataFrame):
        raw = source.head(nrows) if nrows else source
    elif isinstance(source, (str, Path)):
        raw = _read_path(Path(source), schema, fmt, nrows, engine, read_kwargs)
    else:
        raw = pd.DataFrame(list(source))
        if nrows:
            raw = raw.head(nrows)
    adapter = _resolve_adapter(schema, list(raw.columns))
    return adapter.convert(raw, strict=strict)


def _read_path(
    path: Path,
    schema: Any,
    fmt: str | None,
    nrows: int | None,
    engine: str,
    kwargs: dict[str, Any],
) -> pd.DataFrame:
    """Read ``path``; for IBM-style CSV keep the four identifier columns as text."""
    is_ibm_candidate = isinstance(schema, str) and schema in ("auto", "ibm")
    if is_ibm_candidate and (fmt or infer_format(path)) in ("csv", "tsv"):
        header = read_raw(path, fmt=fmt, nrows=1, engine="pandas", **kwargs)
        if header.shape[1] in (10, 11) and str(header.columns[0]) == "Timestamp":
            return read_raw(
                path, fmt=fmt, nrows=nrows, engine=engine, str_positions=(1, 2, 3, 4), **kwargs
            )
    return read_raw(path, fmt=fmt, nrows=nrows, engine=engine, **kwargs)


def write_ibm_csv(df: pd.DataFrame, path: str | Path, *, with_label: bool = True) -> Path:
    """Write a canonical frame back to the IBM AML CSV layout (round-trip / interoperability)."""
    path = Path(path)
    out = pd.DataFrame(
        {
            "Timestamp": df["timestamp"].dt.strftime("%Y/%m/%d %H:%M"),
            "From Bank": df["sender_bank"],
            "Account": df["sender"].str.split("_", n=1).str[-1],
            "To Bank": df["receiver_bank"],
            "Account.1": df["receiver"].str.split("_", n=1).str[-1],
            "Amount Received": df["amount_received"],
            "Receiving Currency": df["currency_received"],
            "Amount Paid": df["amount"],
            "Payment Currency": df["currency"],
            "Payment Format": df["transaction_type"],
        }
    )
    if with_label:
        out["Is Laundering"] = df["label"].fillna(0).astype(int)
    columns = list(IBM_COLUMNS if with_label else IBM_COLUMNS[:10])
    out[columns].to_csv(path, index=False)
    return path
