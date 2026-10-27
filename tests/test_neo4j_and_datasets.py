"""Neo4j export, Kaggle download helpers and the CDC-named ingestion modules (no network)."""

import sys
import types
import zipfile

import numpy as np
import pandas as pd
import pytest

from aml_graphminer.datasets import download_elliptic, download_ibm, ibm_file_names
from aml_graphminer.detectors.patterns import PatternScanner
from aml_graphminer.exceptions import ConfigError, IngestionError
from aml_graphminer.graph import export_neo4j, neo4j_frames, push_to_neo4j


# --------------------------------------------------------------------------- Neo4j
def test_property_graph_tables(tiny_df):
    scan = PatternScanner().scan(tiny_df)
    scores = np.linspace(0, 1, len(tiny_df))
    accounts, transactions = neo4j_frames(tiny_df, scores=scores, scan=scan)
    assert len(transactions) == len(tiny_df)
    assert set(accounts["account_id"]) == set(tiny_df["sender"]) | set(tiny_df["receiver"])
    assert accounts["n_sent"].sum() == accounts["n_received"].sum() == len(tiny_df)
    assert accounts["max_score"].max() == pytest.approx(1.0)
    assert (accounts["n_alerts"] > 0).any() and (transactions["motif"] != "").any()
    assert set(transactions["label"]) <= {"0", "1"}
    with pytest.raises(ConfigError, match="scores"):
        neo4j_frames(tiny_df, scores=np.zeros(3))


def test_export_files(tiny_df, tmp_path):
    paths = export_neo4j(tiny_df, tmp_path / "neo4j")
    assert {p.name for p in paths.values()} == {
        "accounts.csv",
        "transactions.csv",
        "import.cypher",
        "queries.cypher",
    }
    transactions = pd.read_csv(paths["transactions"])
    assert len(transactions) == len(tiny_df) and transactions["score"].isna().all()
    script = paths["import"].read_text(encoding="utf-8")
    assert "LOAD CSV WITH HEADERS FROM 'file:///transactions.csv'" in script
    assert "IN TRANSACTIONS" in script and "CREATE CONSTRAINT" in script
    assert "TRANSFER*3..6" in paths["queries"].read_text(encoding="utf-8")


def test_push_uses_the_official_driver(tiny_df, monkeypatch):
    calls = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def run(self, query, **params):
            calls.append((query, params))

    class Driver:
        closed = False

        def session(self, database):
            assert database == "aml"
            return Session()

        def close(self):
            Driver.closed = True

    fake = types.SimpleNamespace(
        GraphDatabase=types.SimpleNamespace(driver=lambda uri, auth: Driver())
    )
    monkeypatch.setitem(sys.modules, "neo4j", fake)
    counts = push_to_neo4j(tiny_df, "bolt://x", "neo4j", "pw", database="aml", batch_size=500)
    assert counts["transactions"] == len(tiny_df) and Driver.closed
    transfer_batches = [p["rows"] for q, p in calls if "TRANSFER" in q]
    assert sum(len(rows) for rows in transfer_batches) == len(tiny_df)
    assert all(len(rows) <= 500 for rows in transfer_batches)
    first = transfer_batches[0][0]
    assert first["score"] is None and isinstance(first["timestamp"], str)
    with pytest.raises(ConfigError, match="batch_size"):
        push_to_neo4j(tiny_df, "bolt://x", "u", "p", batch_size=0)


# --------------------------------------------------------------------------- Kaggle
class FakeKaggle:
    """Mimics the two Kaggle API calls used by the toolkit (no network)."""

    def __init__(self, zipped=True):
        self.zipped = zipped
        self.files = []

    def dataset_download_file(self, dataset, name, path, force=False):
        self.files.append(name)
        target = f"{path}/{name}"
        if self.zipped:
            with zipfile.ZipFile(target + ".zip", "w") as archive:
                archive.writestr(name, "content")
        else:
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("content")

    def dataset_download_files(self, dataset, path, force=False, unzip=True):
        nested = f"{path}/elliptic_bitcoin_dataset"
        import os

        os.makedirs(nested, exist_ok=True)
        for name in (
            "elliptic_txs_features.csv",
            "elliptic_txs_classes.csv",
            "elliptic_txs_edgelist.csv",
        ):
            with open(f"{nested}/{name}", "w", encoding="utf-8") as handle:
                handle.write("x")


def test_ibm_file_names():
    assert ibm_file_names(("small",), ("HI",)) == ["HI-Small_Trans.csv", "HI-Small_Patterns.txt"]
    assert len(ibm_file_names(("small", "medium"))) == 8
    with pytest.raises(ConfigError):
        ibm_file_names(("huge",))
    with pytest.raises(ConfigError):
        ibm_file_names(("small",), ("XX",))


def test_download_ibm_unzips_and_skips_existing_files(tmp_path):
    api = FakeKaggle(zipped=True)
    paths = download_ibm(tmp_path, sizes=("small",), groups=("HI",), api=api)
    assert [p.name for p in paths] == ["HI-Small_Trans.csv", "HI-Small_Patterns.txt"]
    assert all(p.read_text() == "content" for p in paths) and not list(tmp_path.glob("*.zip"))
    again = FakeKaggle(zipped=False)
    download_ibm(tmp_path, sizes=("small",), groups=("HI",), api=again)
    assert again.files == []  # already present
    download_ibm(tmp_path, sizes=("small",), groups=("HI",), api=again, force=True)
    assert len(again.files) == 2

    class Broken(FakeKaggle):
        def dataset_download_file(self, *args, **kwargs):
            pass

    with pytest.raises(IngestionError, match="failed"):
        download_ibm(tmp_path / "other", groups=("LI",), api=Broken())


def test_download_elliptic(tmp_path):
    files = download_elliptic(tmp_path / "elliptic", api=FakeKaggle())
    assert sorted(p.name for p in files) == [
        "elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv",
        "elliptic_txs_features.csv",
    ]

    class Empty(FakeKaggle):
        def dataset_download_files(self, *args, **kwargs):
            pass

    with pytest.raises(IngestionError, match="not found"):
        download_elliptic(tmp_path / "empty", api=Empty())


# --------------------------------------------------------------------------- CDC structure
def test_cdc_named_ingestion_modules():
    from aml_graphminer.ingestion import csv_adapter, elliptic_adapter, ibm_adapter

    assert csv_adapter.CsvAdapter and csv_adapter.read_transactions
    assert ibm_adapter.IBMAdapter and ibm_adapter.load_dataset and ibm_adapter.read_ibm_patterns
    assert elliptic_adapter.read_elliptic and elliptic_adapter.EllipticData
    for module in (csv_adapter, ibm_adapter, elliptic_adapter):
        assert all(hasattr(module, name) for name in module.__all__)
