import pandas as pd
import pytest

from aml_graphminer.exceptions import IngestionError
from aml_graphminer.ingestion import (
    attach_patterns,
    find_patterns_file,
    load_dataset,
    normalize_pattern_name,
    read_ibm_patterns,
    summarize,
)
from aml_graphminer.synthetic import write_ibm_files


@pytest.fixture()
def ibm_dir(tmp_path, tiny_df):
    write_ibm_files(tiny_df, tmp_path, "Mini")
    return tmp_path


def test_normalize_pattern_name():
    assert normalize_pattern_name("FAN-OUT") == "fan_out"
    assert normalize_pattern_name(" GATHER-SCATTER ") == "gather_scatter"


def test_read_patterns_and_attach(ibm_dir, tiny_df):
    patterns = read_ibm_patterns(ibm_dir / "Mini_Patterns.txt")
    assert len(patterns) == int(tiny_df["label"].sum())
    assert set(patterns["pattern"]) <= set(tiny_df["pattern"]) - {""}
    assert (patterns["attempt_id"] >= 0).all()
    plain = pd.read_csv(ibm_dir / "Mini_Trans.csv")
    assert len(plain) == len(tiny_df)
    out, n = attach_patterns(tiny_df.assign(pattern=""), patterns)
    assert n == len(patterns)
    assert (out["pattern"] != "").sum() == n


def test_attach_with_no_patterns(tiny_df):
    out, n = attach_patterns(tiny_df, pd.DataFrame(columns=["pattern", "attempt_id"]))
    assert n == 0 and len(out) == len(tiny_df)


def test_read_patterns_errors(tmp_path):
    with pytest.raises(IngestionError, match="not found"):
        read_ibm_patterns(tmp_path / "missing.txt")
    bad = tmp_path / "P.txt"
    bad.write_text("BEGIN LAUNDERING ATTEMPT - FAN-OUT\n1,2,3\nEND LAUNDERING ATTEMPT - FAN-OUT\n")
    with pytest.raises(IngestionError, match="not 10/11 fields"):
        read_ibm_patterns(bad)
    empty = tmp_path / "E.txt"
    empty.write_text("\n")
    assert read_ibm_patterns(empty).empty


def test_load_dataset_attaches_sibling_patterns(ibm_dir, tiny_df):
    assert find_patterns_file(ibm_dir / "Mini_Trans.csv") == ibm_dir / "Mini_Patterns.txt"
    df = load_dataset(ibm_dir / "Mini_Trans.csv")
    assert (df["pattern"] != "").sum() == int(tiny_df["label"].sum())
    plain = load_dataset(ibm_dir / "Mini_Trans.csv", patterns_file=False)
    assert (plain["pattern"] == "").all()
    explicit = load_dataset(ibm_dir / "Mini_Trans.csv", patterns_file=ibm_dir / "Mini_Patterns.txt")
    assert (explicit["pattern"] != "").any()
    with pytest.raises(IngestionError, match="patterns file not found"):
        load_dataset(ibm_dir / "Mini_Trans.csv", patterns_file=ibm_dir / "nope.txt")


def test_find_patterns_file_none(tmp_path):
    assert find_patterns_file(tmp_path / "data.csv") is None
    assert find_patterns_file(tmp_path / "A_Trans.csv") is None


def test_summarize(tiny_df):
    s = summarize(tiny_df)
    assert s["transactions"] == len(tiny_df)
    assert s["illicit"] == int(tiny_df["label"].sum())
    assert 0 < s["illicit_ratio"] < 1
    assert "patterns" in s
    assert summarize(tiny_df.iloc[:0])["first_timestamp"] is None
