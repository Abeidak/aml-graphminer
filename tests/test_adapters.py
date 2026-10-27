import pandas as pd
import pytest

from aml_graphminer.exceptions import IngestionError
from aml_graphminer.ingestion import (
    CanonicalAdapter,
    CsvAdapter,
    IBMAdapter,
    detect_adapter,
    normalize_bank_id,
)
from aml_graphminer.ingestion.adapters import IBM_COLUMNS

IBM_ROWS = [
    ["2022/09/01 10:05", "011", "ACC_A", 12, "ACC_B", 100.0, "Euro", 110.0, "US Dollar", "Wire", 1],
    [
        "2022/09/01 09:00",
        12,
        "ACC_B",
        "011",
        "ACC_A",
        50.0,
        "US Dollar",
        50.0,
        "US Dollar",
        "Cash",
        0,
    ],
]


def _ibm(n_cols=11):
    return pd.DataFrame([r[:n_cols] for r in IBM_ROWS], columns=list(IBM_COLUMNS[:n_cols]))


def test_normalize_bank_id():
    assert normalize_bank_id("011") == "11"
    assert normalize_bank_id(11) == "11"
    assert normalize_bank_id(" 70 ") == "70"
    assert normalize_bank_id("ABC") == "ABC"


def test_ibm_adapter_maps_by_position_and_builds_account_ids():
    out = IBMAdapter().convert(_ibm())
    assert out["timestamp"].is_monotonic_increasing
    first = out.iloc[1]
    assert first["sender"] == "11_ACC_A" and first["receiver"] == "12_ACC_B"
    assert first["amount"] == 110.0 and first["amount_received"] == 100.0
    assert first["currency"] == "US Dollar" and first["currency_received"] == "Euro"
    assert first["label"] == 1.0 and first["transaction_type"] == "Wire"
    assert first["sender_bank"] == "11"


def test_ibm_adapter_unlabelled_file():
    out = IBMAdapter().convert(_ibm(10))
    assert out["label"].isna().all()


def test_ibm_adapter_works_with_polars_style_duplicate_names():
    raw = _ibm()
    raw.columns = [c if c != "Account.1" else "Account_duplicated_0" for c in raw.columns]
    assert len(IBMAdapter().convert(raw)) == 2


def test_ibm_adapter_wrong_width():
    with pytest.raises(IngestionError, match="10 or 11"):
        IBMAdapter().to_frame(pd.DataFrame({"a": [1]}))


def test_detect_adapter():
    assert isinstance(detect_adapter(list(IBM_COLUMNS)), IBMAdapter)
    assert isinstance(
        detect_adapter(["timestamp", "sender", "receiver", "amount"]), CanonicalAdapter
    )
    with pytest.raises(IngestionError, match="could not detect"):
        detect_adapter(["x", "y"])


def test_csv_adapter_mapping():
    raw = pd.DataFrame({"d": ["2022-01-01"], "f": ["a"], "t": ["b"], "v": [3.0], "fraud": [1]})
    adapter = CsvAdapter(
        {"timestamp": "d", "sender": "f", "receiver": "t", "amount": "v", "label": "fraud"}
    )
    out = adapter.convert(raw)
    assert out.loc[0, "sender"] == "a" and out.loc[0, "label"] == 1.0
    assert adapter.matches(list(raw.columns))


def test_csv_adapter_validation():
    with pytest.raises(IngestionError, match="must define"):
        CsvAdapter({"timestamp": "d"})
    adapter = CsvAdapter({"timestamp": "d", "sender": "f", "receiver": "t", "amount": "v"})
    with pytest.raises(IngestionError, match="not found"):
        adapter.convert(pd.DataFrame({"d": [1]}))
