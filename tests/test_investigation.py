"""Back end of the web investigation interface (framework-free, no web server needed)."""

import io

import numpy as np
import pandas as pd
import pytest

from aml_graphminer.api.investigation import MAX_PAGE, InvestigationService
from aml_graphminer.detectors.bundle import MotifDetector, TabularDetector
from aml_graphminer.exceptions import ConfigError, NotFoundError
from aml_graphminer.ingestion import write_ibm_csv


@pytest.fixture(scope="module")
def tabular(synth_df):
    return InvestigationService(TabularDetector.fit(synth_df, "hgb"))


@pytest.fixture(scope="module")
def motifs(synth_df):
    return InvestigationService(MotifDetector.fit(synth_df))


@pytest.fixture(scope="module")
def dataset(tabular, synth_df):
    return tabular.add(synth_df, "synthetic")


def test_add_describe_and_list(tabular, dataset, synth_df):
    assert dataset["transactions"] == len(synth_df) and dataset["labelled"] is True
    assert dataset["n_alerts"] > 0 and dataset["n_flagged"] >= 0 and dataset["accounts"] > 0
    assert any(d["id"] == dataset["id"] for d in tabular.datasets())
    model = tabular.model()
    assert model["model"]["model_type"] == "tabular" and "occlusion" in model["explain_methods"]
    assert "community" in model["patterns"]


def test_overview_has_dashboard_data(tabular, dataset):
    ov = tabular.overview(dataset["id"])
    assert sum(ov["score_histogram"]["counts"]) == dataset["transactions"]
    assert ov["eda"]["transactions"] == dataset["transactions"]
    assert sum(ov["alerts_by_pattern"].values()) == dataset["n_alerts"]
    assert 0 <= ov["metrics"]["pr_auc"] <= 1 and 0 <= ov["motif_metrics"]["recall"] <= 1


def test_alerts_are_sorted_filtered_and_paginated(tabular, dataset):
    first = tabular.alerts(dataset["id"], limit=5)
    scores = [a["score"] for a in first["items"]]
    assert scores == sorted(scores, reverse=True) and len(first["items"]) <= 5
    second = tabular.alerts(dataset["id"], limit=5, offset=5)
    assert not {a["alert_id"] for a in first["items"]} & {a["alert_id"] for a in second["items"]}
    pattern = first["items"][0]["pattern"]
    only = tabular.alerts(dataset["id"], pattern=pattern, limit=MAX_PAGE)
    assert only["total"] >= 1 and {a["pattern"] for a in only["items"]} == {pattern}
    assert tabular.alerts(dataset["id"], min_score=1.01)["total"] == 0


def test_alert_detail_has_evidence_transactions_and_graph(tabular, dataset):
    alert_id = tabular.alerts(dataset["id"], limit=1)["items"][0]["alert_id"]
    detail = tabular.alert(dataset["id"], alert_id)
    assert detail["evidence"] and detail["description"] and detail["summary"]
    assert {t["tx_id"] for t in detail["transactions"]} == set(detail["tx_ids"])
    graph = detail["graph"]
    assert len(graph["edges"]) == len(detail["tx_ids"])
    assert {n["id"] for n in graph["nodes"]} >= {e["source"] for e in graph["edges"]}
    with pytest.raises(NotFoundError):
        tabular.alert(dataset["id"], 10**9)


def test_transactions_sort_filter_and_details(tabular, dataset, synth_df):
    page = tabular.transactions(dataset["id"], limit=20)
    scores = [t["score"] for t in page["items"]]
    assert page["total"] == len(synth_df) and scores == sorted(scores, reverse=True)
    ascending = tabular.transactions(dataset["id"], sort="amount", descending=False, limit=5)
    amounts = [t["amount"] for t in ascending["items"]]
    assert amounts == sorted(amounts) and amounts[0] == pytest.approx(synth_df["amount"].min())
    account = page["items"][0]["sender"]
    mine = tabular.transactions(dataset["id"], account=account, limit=MAX_PAGE)
    assert all(account in (t["sender"], t["receiver"]) for t in mine["items"])
    flagged = tabular.transactions(dataset["id"], only_alerts=True, limit=MAX_PAGE)
    assert all(t["alert"] for t in flagged["items"]) and flagged["total"] == dataset["n_flagged"]
    motif = tabular.transactions(dataset["id"], pattern="fan_out", limit=MAX_PAGE)
    assert all(t["motif"] == "fan_out" for t in motif["items"])
    tx = tabular.transaction(dataset["id"], page["items"][0]["tx_id"])
    assert tx["tx_id"] == page["items"][0]["tx_id"] and isinstance(tx["alerts"], list)
    with pytest.raises(ConfigError, match="sort"):
        tabular.transactions(dataset["id"], sort="sender")
    with pytest.raises(ConfigError, match="limit"):
        tabular.transactions(dataset["id"], limit=0)
    with pytest.raises(NotFoundError):
        tabular.transaction(dataset["id"], -5)


def test_explanations_come_with_a_graph(tabular, motifs, dataset, synth_df):
    tx = tabular.transactions(dataset["id"], limit=1)["items"][0]["tx_id"]
    exp = tabular.explain(dataset["id"], tx, max_edges=25)
    assert exp["method"] == "occlusion" and exp["edges"][0]["tx_id"] == tx
    assert any(e["role"] == "target" for e in exp["graph"]["edges"])
    assert all("importance" in n for n in exp["graph"]["nodes"])
    with pytest.raises(ConfigError, match="max_edges"):
        tabular.explain(dataset["id"], tx, max_edges=1)
    with pytest.raises(ConfigError, match="hops"):
        tabular.explain(dataset["id"], tx, hops=9)

    motif_set = motifs.add(synth_df, "motifs")
    alert = motifs.alerts(motif_set["id"], limit=1)["items"][0]
    in_alert = motifs.alert(motif_set["id"], alert["alert_id"])["tx_ids"][0]
    exp = motifs.explain(motif_set["id"], in_alert)
    assert exp["method"] == "patterns" and exp["patterns"]


def test_account_profile(tabular, dataset):
    tx = tabular.transactions(dataset["id"], limit=1)["items"][0]
    profile = tabular.account(dataset["id"], tx["sender"])
    assert profile["n_sent"] >= 1 and profile["max_score"] >= tx["score"] - 1e-12
    assert profile["transactions"][0]["score"] == pytest.approx(profile["max_score"])
    assert tx["sender"] in {n["id"] for n in profile["graph"]["nodes"]}
    with pytest.raises(NotFoundError):
        tabular.account(dataset["id"], "nobody")


def test_exports(tabular, dataset, synth_df):
    alerts = pd.read_csv(io.StringIO(tabular.export_alerts_csv(dataset["id"])))
    assert len(alerts) == dataset["n_alerts"] and {"pattern", "score", "accounts"} <= set(
        alerts.columns
    )
    scores = pd.read_csv(io.StringIO(tabular.export_scores_csv(dataset["id"])))
    assert len(scores) == len(synth_df) and {"score", "alert", "motif", "label"} <= set(
        scores.columns
    )


def test_upload_unlabelled_unsorted_and_limits(tabular, synth_df, tmp_path):
    sample = synth_df.iloc[:800].copy()
    write_ibm_csv(sample, tmp_path / "x_Trans.csv", with_label=False)
    created = tabular.add_upload((tmp_path / "x_Trans.csv").read_bytes(), "x_Trans.csv")
    assert created["name"] == "x_Trans.csv" and created["labelled"] is False
    ov = tabular.overview(created["id"])
    assert ov["metrics"] is None and ov["eda"]["labelled"] == 0
    with pytest.raises(ConfigError, match="unsupported"):
        tabular.add_upload(b"x", "data.xlsx")
    shuffled = synth_df.iloc[:300].sample(frac=1.0, random_state=0)
    added = tabular.add(shuffled.reset_index(drop=True), "shuffled")
    assert tabular.get(added["id"]).df["timestamp"].is_monotonic_increasing
    with pytest.raises(ConfigError, match="no transaction"):
        tabular.add(synth_df.iloc[:0])
    small = InvestigationService(tabular.detector, max_transactions=10)
    with pytest.raises(ConfigError, match="exceed"):
        small.add(synth_df.iloc[:20])


def test_oldest_investigations_are_dropped_and_removal(synth_df):
    service = InvestigationService(MotifDetector.fit(None), max_datasets=2)
    ids = [service.add(synth_df.iloc[:200], f"d{i}")["id"] for i in range(3)]
    kept = [d["id"] for d in service.datasets()]
    assert kept == [ids[2], ids[1]]
    service.remove(ids[2])
    assert [d["id"] for d in service.datasets()] == [ids[1]]
    with pytest.raises(NotFoundError):
        service.overview(ids[0])


def test_demo_dataset_and_its_validation():
    service = InvestigationService(MotifDetector.fit(None))
    demo = service.add_demo(accounts=200, background=1500, illicit_ratio=0.05, seed=1)
    assert demo["labelled"] and demo["transactions"] > 1500
    with pytest.raises(ConfigError):
        service.add_demo(accounts=10)
    with pytest.raises(ConfigError):
        service.add_demo(illicit_ratio=0.9)
    assert np.isfinite(service.overview(demo["id"])["metrics"]["pr_auc"])
