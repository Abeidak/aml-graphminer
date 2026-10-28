import numpy as np
import pandas as pd
import pytest

from aml_graphminer.exceptions import SchemaError
from aml_graphminer.ingestion.schema import (
    CANONICAL_COLUMNS,
    Transaction,
    sample_validate,
    validate_records,
    validate_transactions,
)


def _raw(**over):
    base = {
        "timestamp": ["2022-01-02 10:00", "2022-01-01 09:00"],
        "sender": ["a", "b"],
        "receiver": ["b", "c"],
        "amount": [10.0, 20.0],
    }
    base.update(over)
    return pd.DataFrame(base)


def test_validate_fills_defaults_and_sorts():
    out = validate_transactions(_raw())
    assert list(out.columns)[: len(CANONICAL_COLUMNS)] == list(CANONICAL_COLUMNS)
    assert out.index.name == "tx_id"
    assert out["timestamp"].is_monotonic_increasing
    assert out["sender"].tolist() == ["b", "a"]  # sorted by time
    assert out["source_row"].tolist() == [1, 0]  # original positions kept
    assert (out["amount_received"] == out["amount"]).all()
    assert out["label"].isna().all()
    assert (out["currency"] == "UNK").all()


def test_validate_keeps_extra_columns_after_canonical():
    out = validate_transactions(_raw(extra=[1, 2]))
    assert out.columns[-1] == "extra"


@pytest.mark.parametrize("missing", ["timestamp", "sender", "receiver", "amount"])
def test_missing_required_column(missing):
    with pytest.raises(SchemaError, match="missing required"):
        validate_transactions(_raw().drop(columns=missing))


@pytest.mark.parametrize(
    "bad",
    [
        {"amount": [-1.0, 5.0]},
        {"amount": [np.nan, 5.0]},
        {"timestamp": ["not a date", "2022-01-01"]},
        {"sender": ["", "b"]},
        {"receiver": [None, "c"]},
    ],
)
def test_invalid_rows_strict_raises_and_lenient_drops(bad):
    df = _raw(**bad)
    with pytest.raises(SchemaError, match="invalid row"):
        validate_transactions(df)
    out = validate_transactions(df, strict=False)
    assert len(out) == 1
    assert out.attrs["dropped_rows"] == 1


def test_invalid_label_rejected():
    with pytest.raises(SchemaError, match="label"):
        validate_transactions(_raw(label=[0, 2]))


def test_not_a_dataframe():
    with pytest.raises(SchemaError):
        validate_transactions([{"a": 1}])  # type: ignore[arg-type]


def test_sort_false_keeps_order():
    out = validate_transactions(_raw(), sort=False)
    assert out["sender"].tolist() == ["a", "b"]


def test_validate_records_and_errors():
    recs = [
        {"timestamp": "2022-01-01 10:00", "sender": "a", "receiver": "b", "amount": 5, "label": 1},
        {"timestamp": "2022-01-01 09:00", "sender": 7, "receiver": "b", "amount": 6},
    ]
    out = validate_records(recs)
    assert len(out) == 2 and out["sender"].tolist() == ["7", "a"]
    with pytest.raises(SchemaError, match="record 0"):
        validate_records([{"timestamp": "x", "sender": "a", "receiver": "b", "amount": 1}])
    with pytest.raises(SchemaError, match="no records"):
        validate_records([])
    with pytest.raises(SchemaError, match="too many"):
        validate_records(recs, max_records=1)


def test_transaction_model_strips_and_converts():
    tx = Transaction.model_validate(
        {"timestamp": "2022-01-01", "sender": " a ", "receiver": 5, "amount": 1}
    )
    assert tx.sender == "a" and tx.receiver == "5"
    with pytest.raises(ValueError):  # pydantic.ValidationError subclasses ValueError
        Transaction.model_validate(
            {"timestamp": "2022-01-01", "sender": "a", "receiver": "b", "amount": -1}
        )


def test_sample_validate_counts_rows(tiny_df):
    assert sample_validate(tiny_df, n=50) == 50
    assert sample_validate(tiny_df.iloc[:0]) == 0


def test_sample_validate_names_the_invalid_transaction(tiny_df):
    assert sample_validate(tiny_df, n=None) == len(tiny_df)
    bad = tiny_df.copy()
    bad.loc[bad.index[3], "amount_received"] = -1.0  # not caught by the vectorised checks
    with pytest.raises(SchemaError, match=f"transaction {bad.index[3]!r}"):
        sample_validate(bad, n=None)


def test_load_dataset_spot_checks_rows_with_pydantic(tmp_path, tiny_df, monkeypatch):
    from aml_graphminer.ingestion import dataset

    path = tmp_path / "t.csv"
    tiny_df.to_csv(path)
    checked = []
    monkeypatch.setattr(dataset, "sample_validate", lambda df, n: checked.append(n) or n)
    dataset.load_dataset(path)
    dataset.load_dataset(path, check_sample=0)
    assert checked == [1000]
