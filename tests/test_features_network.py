import networkx as nx
import numpy as np
import pytest

from aml_graphminer.features.network import (
    NODE_COLUMNS,
    approx_betweenness,
    index_accounts,
    node_features,
    pagerank,
)
from helpers import make_tx


def test_index_accounts_is_consistent(tiny_df):
    src, dst, n, names = index_accounts(tiny_df)
    assert len(src) == len(dst) == len(tiny_df)
    assert names[src].tolist() == tiny_df["sender"].tolist()
    assert names[dst].tolist() == tiny_df["receiver"].tolist()
    assert n == len(set(tiny_df["sender"]) | set(tiny_df["receiver"]))


def test_pagerank_matches_networkx(tiny_df):
    src, dst, n, _ = index_accounts(tiny_df)
    ours = pagerank(src, dst, n)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for u, v in zip(src.tolist(), dst.tolist()):
        if graph.has_edge(u, v):
            graph[u][v]["weight"] += 1
        else:
            graph.add_edge(u, v, weight=1)
    ref = nx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=500)
    assert np.allclose(ours, np.array([ref[i] for i in range(n)]) * n, atol=1e-5)
    assert ours.mean() == pytest.approx(1.0)


def test_pagerank_edge_cases():
    assert pagerank(np.zeros(0, dtype=int), np.zeros(0, dtype=int), 0).size == 0
    # a single dangling node and a star
    pr = pagerank(np.array([0, 0, 0]), np.array([1, 2, 3]), 4)
    assert pr[0] < pr[1] and pr[1] == pytest.approx(pr[2])


def test_node_features_columns_rates_and_zero_for_inactive_nodes():
    df = make_tx([(0, "a", "b", 100), (60, "a", "c", 300), (120, "b", "c", 50)])
    src, dst, n, names = index_accounts(df)
    rows = np.array([True, True, False])  # the last transaction is not used for aggregates
    feats = node_features(df, src, dst, n, rows)
    assert list(feats.columns[: len(NODE_COLUMNS)]) == list(NODE_COLUMNS)
    a = int(np.flatnonzero(names == "a")[0])
    b = int(np.flatnonzero(names == "b")[0])
    assert feats.loc[a, "out_rate"] > feats.loc[b, "out_rate"]
    assert feats.loc[b, "out_rate"] == 0.0  # b's only payment is excluded from `rows`
    assert np.isfinite(feats.to_numpy()).all()


def test_node_features_with_no_rows_are_zero():
    df = make_tx([(0, "a", "b", 100)])
    src, dst, n, _ = index_accounts(df)
    feats = node_features(df, src, dst, n, np.array([False]))
    assert (feats.to_numpy() == 0).all()


def test_entropy_is_higher_for_spread_counterparties():
    df = make_tx(
        [(i, "hub", f"x{i}", 10) for i in range(5)] + [(i + 10, "solo", "y", 10) for i in range(5)]
    )
    src, dst, n, names = index_accounts(df)
    feats = node_features(df, src, dst, n, use_pagerank=False)
    hub, solo = (int(np.flatnonzero(names == k)[0]) for k in ("hub", "solo"))
    assert feats.loc[hub, "ent_out"] > feats.loc[solo, "ent_out"] == 0.0


def test_betweenness_option(tiny_df):
    src, dst, n, _ = index_accounts(tiny_df.head(300))
    values = approx_betweenness(src, dst, n, k=20, seed=1)
    assert values.shape == (n,) and (values >= 0).all()
    feats = node_features(tiny_df.head(300), src, dst, n, use_betweenness=True, betweenness_k=10)
    assert "betweenness" in feats.columns and "pagerank" in feats.columns
