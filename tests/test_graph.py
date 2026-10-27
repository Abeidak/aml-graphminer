import networkx as nx
import numpy as np
import pytest

from aml_graphminer.exceptions import ConfigError
from aml_graphminer.graph import (
    arrays_to_nx,
    build_digraph,
    build_hetero_networkx,
    build_multidigraph,
    communities,
    graph_summary,
    node_type_counts,
    nx_to_arrays,
)
from helpers import make_tx


@pytest.fixture()
def small():
    return make_tx(
        [
            (0, "a", "b", 10, 1.0, "fan_out"),
            (5, "a", "b", 20),
            (10, "b", "c", 30),
            (15, "c", "a", 40),
        ],
        sender_bank="1",
        receiver_bank="2",
    )


def test_multidigraph_keeps_one_edge_per_transaction(small):
    g = build_multidigraph(small)
    assert g.number_of_edges() == 4 and g.number_of_nodes() == 3
    assert g.number_of_edges("a", "b") == 2
    assert {k for _, _, k in g.edges(keys=True)} == set(small.index)
    assert g.edges["a", "b", 0]["pattern"] == "fan_out"


def test_multidigraph_rows_selection_and_errors(small):
    mask = np.array([True, False, True, False])
    assert build_multidigraph(small, mask).number_of_edges() == 2
    assert build_multidigraph(small, np.array([0, 3])).number_of_edges() == 2
    with pytest.raises(ConfigError, match="edge attribute"):
        build_multidigraph(small, edge_attrs=("nope",))


def test_node_attributes_are_attached(small):
    import pandas as pd

    attrs = pd.DataFrame({"risk": [0.1, 0.2]}, index=["a", "zzz"])
    g = build_multidigraph(small, node_attrs=attrs)
    assert g.nodes["a"]["risk"] == 0.1 and "risk" not in g.nodes["b"]


def test_aggregated_digraph(small):
    g = build_digraph(small)
    assert g.number_of_edges() == 3
    assert g["a"]["b"]["count"] == 2 and g["a"]["b"]["total_amount"] == 30.0
    assert g["a"]["b"]["n_illicit"] == 1.0


def test_summary_and_communities(small):
    s = graph_summary(build_digraph(small))
    assert s["nodes"] == 3 and s["edges"] == 3 and s["components"] == 1 and s["max_degree"] == 2
    assert graph_summary(nx.DiGraph())["nodes"] == 0
    assert graph_summary(nx.Graph([(1, 2)]))["components"] == 1
    comms = communities(build_digraph(small))
    assert set().union(*comms) == {"a", "b", "c"}


def test_arrays_roundtrip(small):
    g = build_digraph(small)
    arrays = nx_to_arrays(g, edge_attrs=("count", "total_amount"))
    assert arrays["edge_index"].shape == (2, 3)
    assert arrays["edge_attr"].shape == (3, 2)
    back = arrays_to_nx(
        arrays["edge_index"],
        node_names=arrays["node_names"],
        edge_attr=arrays["edge_attr"],
        edge_attr_names=("count", "total_amount"),
    )
    assert set(back.edges) == set(g.edges)
    assert back["a"]["b"]["count"] == 2.0


def test_undirected_graph_is_expanded_in_both_directions():
    arrays = nx_to_arrays(nx.Graph([(0, 1)]))
    assert arrays["edge_index"].shape == (2, 2)
    assert arrays["x"] is None and arrays["edge_attr"] is None


def test_arrays_to_nx_node_features_and_multigraph():
    g = arrays_to_nx(np.array([[0, 0], [1, 1]]), x=np.array([[1.0], [2.0], [3.0]]), multigraph=True)
    assert g.number_of_nodes() == 3 and g.number_of_edges() == 2
    assert g.nodes[2]["x0"] == 3.0


def test_hetero_graph(small):
    g = build_hetero_networkx(small, account_types={"a": "company"})
    counts = node_type_counts(g)
    assert counts == {"company": 1, "account": 2, "bank": 2}
    etypes = [d["etype"] for _, _, d in g.edges(data=True)]
    assert etypes.count("transfers") == 4 and etypes.count("held_at") == 3


def test_pyg_conversion_when_torch_is_available(small):
    pytest.importorskip("torch_geometric")
    from aml_graphminer.graph import nx_to_pyg, pyg_to_nx

    g = build_digraph(small)
    data = nx_to_pyg(g, edge_attrs=("count",))
    assert data.edge_index.shape == (2, 3) and data.edge_attr.shape == (3, 1)
    assert pyg_to_nx(data).number_of_edges() == 3
