import json

import numpy as np
import pandas as pd
import pytest

from aml_graphminer.detectors.bundle import MotifDetector, TabularDetector
from aml_graphminer.detectors.patterns import PatternScanner
from aml_graphminer.exceptions import ConfigError
from aml_graphminer.explainers import (
    Explanation,
    OcclusionExplainer,
    TransactionIndex,
    describe_alert,
    detector_score_fn,
    explain_with_patterns,
    pattern_score_fn,
    subgraph_to_networkx,
    tabular_score_fn,
)
from aml_graphminer.explainers.base import EdgeImportance, NodeImportance
from helpers import make_tx


@pytest.fixture(scope="module")
def scanned(synth_df):
    scanner = PatternScanner()
    return scanner, scanner.scan(synth_df)


@pytest.fixture(scope="module")
def flagged_tx(synth_df, scanned):
    _, result = scanned
    scores, _ = result.transaction_scores()
    illicit = synth_df["label"].to_numpy() == 1
    return int(synth_df.index[np.argmax(scores * illicit)])


# --------------------------------------------------------------------------- index / subgraph
def test_index_rows_and_neighbourhood():
    df = make_tx(
        [
            (0, "a", "b", 1),
            (10, "b", "c", 2),
            (20, "c", "d", 3),
            (30, "x", "y", 4),
            (40, "a", "a", 5),
        ]
    )
    index = TransactionIndex(df)
    assert index.rows_of("a").tolist() == [0, 4]  # self-loop counted once
    assert index.rows_of("nobody").tolist() == []
    one_hop = index.neighbourhood(1, hops=1)
    assert sorted(one_hop.index) == [0, 1, 2]
    two_hops = index.neighbourhood(0, hops=2)
    assert sorted(two_hops.index) == [0, 1, 2, 4]
    assert sorted(index.neighbourhood(0, hops=0).index) == [0]  # the transaction alone
    assert 3 not in index.neighbourhood(0, hops=5).index  # disconnected component


def test_neighbourhood_caps_edges_keeping_the_target_and_nearest_in_time():
    df = make_tx([(i, "hub", f"x{i}", 1) for i in range(50)])
    index = TransactionIndex(df)
    hood = index.neighbourhood(25, hops=1, max_edges=7)
    assert len(hood) == 7 and 25 in hood.index
    assert set(hood.index) == set(range(22, 29))
    windowed = index.neighbourhood(25, hops=1, time_window=3 * 60)
    assert sorted(windowed.index) == [22, 23, 24, 25, 26, 27, 28]


def test_unknown_transaction_and_duplicate_ids():
    df = make_tx([(0, "a", "b", 1)])
    with pytest.raises(KeyError):
        TransactionIndex(df).neighbourhood(99)
    with pytest.raises(ConfigError, match="unique"):
        TransactionIndex(pd.concat([df, df]))


def test_subgraph_to_networkx_marks_target():
    df = make_tx([(0, "a", "b", 1), (1, "b", "c", 2)])
    g = subgraph_to_networkx(df, target=1)
    assert g.number_of_edges() == 2
    assert g.edges["b", "c", 1]["is_target"] and not g.edges["a", "b", 0]["is_target"]


# --------------------------------------------------------------------------- explanation object
def test_explanation_serialisation_roundtrip_and_views():
    exp = Explanation(
        tx_id=3,
        method="occlusion",
        score=0.91,
        threshold=0.5,
        edges=[
            EdgeImportance(3, "a", "b", 100.0, "2022-01-01", 0.91, "target"),
            EdgeImportance(4, "b", "c", 50.0, "2022-01-01", 0.3),
            EdgeImportance(5, "c", "a", 20.0, "2022-01-01", -0.5),
        ],
        nodes=[NodeImportance("a", 0.4, "sender")],
        patterns=[{"pattern": "cycle", "score": 0.8}],
        summary="demo",
    )
    data = json.loads(exp.to_json())
    assert Explanation.from_dict(data).to_dict() == exp.to_dict()
    assert [e.tx_id for e in exp.top_edges(2)] == [5, 4]  # by |importance|, target excluded
    graph = exp.to_networkx()
    assert graph.number_of_nodes() == 3 and graph.number_of_edges() == 3
    assert graph.nodes["a"]["role"] == "sender"
    md = exp.to_markdown()
    assert "Transaction 3" in md and "cycle" in md and "-0.500" in md


# --------------------------------------------------------------------------- pattern explanations
def test_pattern_explanation_lists_the_alert_members(synth_df, scanned, flagged_tx):
    _, result = scanned
    exp = explain_with_patterns(synth_df, flagged_tx, result, threshold=0.2)
    assert exp.method == "patterns" and exp.score > 0
    assert exp.patterns and exp.patterns[0]["description"]
    members = {tx for a in result.alerts if flagged_tx in a.tx_ids for tx in a.tx_ids}
    assert {e.tx_id for e in exp.edges} == members
    assert [e.role for e in exp.edges].count("target") == 1 and exp.edges[0].role == "target"
    assert "structural" in exp.summary
    json.loads(exp.to_json())


def test_pattern_explanation_for_an_unflagged_transaction(synth_df, scanned):
    _, result = scanned
    scores, _ = result.transaction_scores()
    quiet = int(synth_df.index[np.flatnonzero(scores == 0)[0]])
    exp = explain_with_patterns(synth_df, quiet, result)
    assert exp.score == 0.0 and exp.edges == [] and "No structural motif" in exp.summary


def test_describe_alert(scanned):
    _, result = scanned
    text = describe_alert(result.alerts[0])
    assert result.alerts[0].pattern in text


# --------------------------------------------------------------------------- occlusion
def _motif_scenario():
    rng = np.random.default_rng(0)
    rows = [
        (
            float(rng.uniform(0, 6 * 24 * 60)),
            f"payer{rng.integers(0, 300)}",
            f"payee{rng.integers(0, 300)}",
            float(rng.lognormal(5, 0.8)),
        )
        for _ in range(300)
    ]
    fan = [(3 * 24 * 60 + 10 * i, "hub", f"mule{i}", 800.0 + i, 1.0, "fan_out") for i in range(8)]
    ring = [
        (60, "A", "B", 5000.0, 1.0, "cycle"),
        (120, "B", "C", 4900.0, 1.0, "cycle"),
        (180, "C", "D", 4800.0, 1.0, "cycle"),
        (240, "D", "A", 4700.0, 1.0, "cycle"),
    ]
    return make_tx(rows + fan + ring)


def test_occlusion_with_motif_scanner_attributes_the_alert_to_its_members():
    df = _motif_scenario()
    scanner = PatternScanner()
    tx = int(df.index[(df["receiver"] == "mule3").to_numpy()][0])
    exp = OcclusionExplainer(pattern_score_fn(scanner), hops=1, max_edges=100).explain(df, tx)
    assert exp.method == "occlusion" and exp.score > 0.3
    assert exp.edges[0].role == "target" and exp.edges[0].tx_id == tx
    others = [e for e in exp.edges[1:] if e.sender == "hub"]
    assert len(others) == 7 and all(e.importance > 0 for e in others), (
        "each fan-out member supports the alert"
    )
    hub = next(n for n in exp.nodes if n.account == "hub")
    assert (
        hub.importance == pytest.approx(exp.score) and hub.role == "sender"
    )  # without the hub nothing is left
    assert "most influential" in exp.summary


def test_occlusion_needs_a_neighbourhood_large_enough_to_contain_the_motif():
    df = _motif_scenario()
    # cycle detector only: the 1-hop view (D -> A -> B -> C) is a legitimate *community* alert
    scanner = PatternScanner(only=["cycle"])
    tx = int(df.index[((df["sender"] == "A") & (df["receiver"] == "B")).to_numpy()][0])
    fn = pattern_score_fn(scanner)
    one = OcclusionExplainer(fn, hops=1).explain(df, tx)  # misses the C -> D edge: no 4-cycle
    two = OcclusionExplainer(fn, hops=2).explain(df, tx)
    assert one.score == 0.0 and two.score > 0.3
    assert {e.tx_id for e in two.edges if e.importance > 0} >= {tx}


def test_occlusion_is_exact_on_a_toy_score_function():
    df = make_tx([(0, "a", "b", 10), (5, "b", "c", 20), (6, "c", "d", 30)])

    def score_fn(target, context):
        # the target's score is 0.4 plus 0.5 if the (b -> c) transaction is in the context
        boost = (
            0.0
            if context is None
            else 0.5 * float(((context["sender"] == "b") & (context["receiver"] == "c")).any())
        )
        return np.full(len(target), 0.4 + boost)

    exp = OcclusionExplainer(score_fn, hops=2, threshold=0.5).explain(df, 0)
    by_id = {e.tx_id: e.importance for e in exp.edges}
    assert (
        by_id[0] == pytest.approx(0.9)
        and by_id[1] == pytest.approx(0.5)
        and by_id[2] == pytest.approx(0.0)
    )
    assert exp.threshold == 0.5
    nodes = {n.account: n.importance for n in exp.nodes}
    assert nodes["b"] == pytest.approx(0.5) and nodes["a"] == pytest.approx(0.0)


def test_occlusion_without_context_has_no_neighbours():
    df = make_tx([(0, "a", "b", 10)])
    exp = OcclusionExplainer(lambda t, c: np.array([0.7])).explain(df, 0)
    assert len(exp.edges) == 1 and "driven by the transaction itself" in exp.summary


def test_occlusion_with_tabular_model(synth_df):
    det = TabularDetector.fit(synth_df, "rf")
    fn = tabular_score_fn(det.pipeline, det.baseline)
    scores = det.score(synth_df)
    tx = int(scores.idxmax())
    index = TransactionIndex(synth_df)
    exp = OcclusionExplainer(fn, max_occlusions=8).explain(synth_df, tx, index)
    hood = index.neighbourhood(tx, hops=2, max_edges=60)
    expected = float(fn(hood.loc[[tx]], hood.drop(index=tx))[0])
    assert exp.score == pytest.approx(expected)
    assert len(exp.edges) <= 9 and exp.edges[0].tx_id == tx


def test_score_function_adapters_agree_with_detectors(synth_df):
    motif = MotifDetector.fit(synth_df)
    sample = synth_df.iloc[1000:1100]
    assert np.allclose(detector_score_fn(motif)(sample, None), motif.score(sample))
    assert len(detector_score_fn(motif)(sample.iloc[:0], None)) == 0
    assert len(pattern_score_fn(motif.scanner)(sample.iloc[:0], None)) == 0


def test_plot_explanation_writes_a_png(synth_df, scanned, flagged_tx, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from aml_graphminer.explainers import plot_explanation

    _, result = scanned
    exp = explain_with_patterns(synth_df, flagged_tx, result)
    out = tmp_path / "e.png"
    plot_explanation(exp, path=out)
    assert out.stat().st_size > 1000
    plot_explanation(Explanation(tx_id=0, method="x"))  # empty explanation does not crash


def test_gnn_explainers_validate_inputs_and_need_the_gnn_extra(synth_df):
    from types import SimpleNamespace

    from aml_graphminer.exceptions import ModelError, OptionalDependencyError
    from aml_graphminer.explainers import AttentionExplainer, GNNExplainerAdapter
    from aml_graphminer.utils import has_module

    with pytest.raises(ModelError, match="GAT"):
        AttentionExplainer(SimpleNamespace(config=SimpleNamespace(kind="sage")))
    if has_module("torch_geometric"):
        pytest.skip("the gnn extra is installed")
    with pytest.raises(OptionalDependencyError):
        GNNExplainerAdapter(SimpleNamespace(stats={})).explain(synth_df, 0)


def test_occlusion_reports_attribute_importance_on_a_toy_model():
    """The score only depends on the amount: replacing it by the typical value removes it."""
    df = make_tx([(0, "a", "b", 5000), (5, "b", "c", 20), (6, "c", "d", 30), (7, "d", "e", 40)])

    def score_fn(target, context):
        return np.where(target["amount"].to_numpy() > 1000, 0.9, 0.1)

    exp = OcclusionExplainer(score_fn, hops=3).explain(df, 0)
    by_name = {f.name: f for f in exp.features}
    assert by_name["amount"].importance == pytest.approx(0.8)
    assert by_name["amount"].value == 5000.0
    assert by_name["transaction_type"].importance == pytest.approx(0.0)
    assert exp.features[0].name == "amount"  # sorted by |importance|
    assert "| amount | 5,000.00 | +0.800 |" in exp.to_markdown()
    assert Explanation.from_dict(exp.to_dict()).features[0].name == "amount"


def test_attribute_importance_is_empty_without_neighbours():
    df = make_tx([(0, "a", "b", 10)])
    exp = OcclusionExplainer(lambda t, c: np.array([0.7])).explain(df, 0)
    assert exp.features == []


def test_attribute_importance_with_a_real_detector(synth_df):
    det = TabularDetector.fit(synth_df, "rf")
    tx = int(det.score(synth_df).idxmax())
    exp = OcclusionExplainer(detector_score_fn(det), max_occlusions=4).explain(synth_df, tx)
    assert exp.features and all(np.isfinite(f.importance) for f in exp.features)
    assert {f.name for f in exp.features} <= {
        "amount",
        "amount_received",
        "transaction_type",
        "currency",
        "currency_received",
        "sender_bank",
        "receiver_bank",
    }


def test_explain_alerts_shares_one_scan_and_ranks_by_score(synth_df, tmp_path):
    from aml_graphminer.detectors.bundle import MotifDetector
    from aml_graphminer.explainers import explain_alerts

    detector = MotifDetector.fit(synth_df)
    explanations = explain_alerts(detector, synth_df, top=4, out_dir=tmp_path / "case")
    assert 0 < len(explanations) <= 4
    scores = [e.score for e in explanations]
    assert scores == sorted(scores, reverse=True) and min(scores) >= detector.threshold
    index = (tmp_path / "case" / "index.csv").read_text(encoding="utf-8")
    assert index.startswith("rank,tx_id,score,method") and "patterns" in index
    everything = explain_alerts(detector, synth_df, top=2, method="occlusion", only_alerts=False)
    assert len(everything) == 2 and everything[0].method == "occlusion"
    with pytest.raises(ConfigError):
        explain_alerts(detector, synth_df, top=0)
    with pytest.raises(ConfigError):
        explain_alerts(detector, synth_df, method="nope")
