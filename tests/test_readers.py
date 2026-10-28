import json

import pandas as pd
import pytest

from aml_graphminer.exceptions import IngestionError
from aml_graphminer.ingestion import infer_format, read_raw, read_transactions, write_ibm_csv
from aml_graphminer.utils import has_module

RECORDS = [
    {"timestamp": "2022-01-01 10:00", "sender": "a", "receiver": "b", "amount": 5.0},
    {"timestamp": "2022-01-01 09:00", "sender": "b", "receiver": "c", "amount": 7.0},
]


def test_infer_format():
    assert infer_format("x.csv") == "csv"
    assert infer_format("x.tsv") == "tsv"
    assert infer_format("x.jsonl") == "jsonl"
    assert infer_format("x.parquet") == "parquet"
    with pytest.raises(IngestionError, match="cannot infer"):
        infer_format("x.unknown")


def test_read_records_and_dataframe():
    assert len(read_transactions(RECORDS)) == 2
    assert len(read_transactions(pd.DataFrame(RECORDS))) == 2
    assert len(read_transactions(RECORDS, nrows=1)) == 1
    assert len(read_transactions(pd.DataFrame(RECORDS), nrows=1)) == 1


def test_read_csv_json_jsonl_tsv(tmp_path):
    frame = pd.DataFrame(RECORDS)
    frame.to_csv(tmp_path / "t.csv", index=False)
    frame.to_csv(tmp_path / "t.tsv", index=False, sep="\t")
    (tmp_path / "t.json").write_text(json.dumps(RECORDS))
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(r) for r in RECORDS))
    for name in ("t.csv", "t.tsv", "t.json", "t.jsonl"):
        out = read_transactions(tmp_path / name)
        assert len(out) == 2 and out["timestamp"].is_monotonic_increasing, name
    assert len(read_transactions(tmp_path / "t.csv", nrows=1)) == 1


def test_read_parquet_roundtrip(tmp_path):
    if not has_module("pyarrow"):
        pytest.skip("pyarrow not installed")
    pd.DataFrame(RECORDS).to_parquet(tmp_path / "t.parquet")
    assert len(read_transactions(tmp_path / "t.parquet")) == 2


def test_parquet_without_pyarrow_gives_install_hint(tmp_path):
    if has_module("pyarrow"):
        pytest.skip("pyarrow is installed")
    (tmp_path / "t.parquet").write_bytes(b"x")
    with pytest.raises(Exception, match="parquet"):
        read_raw(tmp_path / "t.parquet")


def test_missing_file_and_bad_format(tmp_path):
    with pytest.raises(IngestionError, match="not found"):
        read_raw(tmp_path / "nope.csv")
    (tmp_path / "t.csv").write_text("a\n1\n")
    with pytest.raises(IngestionError, match="unsupported format"):
        read_raw(tmp_path / "t.csv", fmt="xml")


def test_unknown_schema_name(tmp_path):
    with pytest.raises(IngestionError, match="unknown schema"):
        read_transactions(RECORDS, schema="nope")  # type: ignore[arg-type]


def test_explicit_schemas():
    out = read_transactions(RECORDS, schema="canonical")
    assert len(out) == 2
    mapped = read_transactions(
        [{"d": "2022-01-01", "f": "a", "t": "b", "v": 1.0}],
        schema={"timestamp": "d", "sender": "f", "receiver": "t", "amount": "v"},
    )
    assert mapped.loc[0, "receiver"] == "b"


def test_ibm_csv_roundtrip_keeps_identifiers_as_text(tmp_path, tiny_df):
    path = write_ibm_csv(tiny_df, tmp_path / "X_Trans.csv")
    back = read_transactions(path)
    assert len(back) == len(tiny_df)
    assert set(back["sender"]) == set(tiny_df["sender"])  # no id mangled to a number
    assert back["label"].sum() == tiny_df["label"].sum()
    unlabeled = write_ibm_csv(tiny_df, tmp_path / "Y_Trans.csv", with_label=False)
    assert read_transactions(unlabeled)["label"].isna().all()


def test_ibm_ids_with_leading_zeros_survive(tmp_path):
    header = "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
    row = "2022/09/01 00:20,010,800104D70,001,80010E430,3697.34,US Dollar,3697.34,US Dollar,Reinvestment,0\n"
    path = tmp_path / "Z_Trans.csv"
    path.write_text(header + row)
    out = read_transactions(path)
    assert out.loc[0, "sender"] == "10_800104D70"
    assert out.loc[0, "receiver"] == "1_80010E430"


def test_polars_engine_reads_the_same_ibm_table_as_pandas(tmp_path, tiny_df):
    pytest.importorskip("polars")
    path = write_ibm_csv(tiny_df, tmp_path / "X-Small_Trans.csv")
    by_pandas = read_transactions(path)
    by_polars = read_transactions(path, engine="polars")
    pd.testing.assert_frame_equal(by_polars, by_pandas, check_dtype=False)
    assert by_polars["sender"].str.contains("_").all()  # bank ids kept as text ("0012" ...)
    assert len(read_transactions(path, engine="polars", nrows=5)) == 5
