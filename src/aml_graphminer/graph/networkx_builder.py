"""NetworkX representations used by the classical algorithms (cycles, communities, paths)."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from ..exceptions import ConfigError

DEFAULT_EDGE_ATTRS: tuple[str, ...] = (
    "timestamp",
    "amount",
    "amount_received",
    "transaction_type",
    "label",
    "pattern",
)


def build_multidigraph(
    df: pd.DataFrame,
    rows: np.ndarray | None = None,
    edge_attrs: Sequence[str] = DEFAULT_EDGE_ATTRS,
    node_attrs: pd.DataFrame | None = None,
) -> nx.MultiDiGraph:
    """One edge per transaction. The edge *key* is the transaction id (``df.index``).

    Parameters
    ----------
    df:
        Canonical transaction table.
    rows:
        Optional boolean mask / integer positions restricting the transactions used.
    edge_attrs:
        Columns copied onto the edges.
    node_attrs:
        Optional frame indexed by account id whose columns become node attributes.

    Notes
    -----
    Building a Python graph is meant for sub-graphs and mid-size tables (up to a few million
    edges); use the PyG representation for training on the full IBM files.
    """
    sub = _select(df, rows)
    missing = [c for c in edge_attrs if c not in sub.columns]
    if missing:
        raise ConfigError(f"edge attribute column(s) not in the table: {missing}")
    graph = nx.MultiDiGraph()
    senders = sub["sender"].tolist()
    receivers = sub["receiver"].tolist()
    ids = sub.index.tolist()
    columns = {c: sub[c].tolist() for c in edge_attrs}
    for i, (s, r, tx) in enumerate(zip(senders, receivers, ids)):
        graph.add_edge(s, r, key=tx, **{c: columns[c][i] for c in edge_attrs})
    if node_attrs is not None:
        for account, attrs in node_attrs.to_dict("index").items():
            if account in graph:
                graph.nodes[account].update(attrs)
    return graph


def _select(df: pd.DataFrame, rows: np.ndarray | None) -> pd.DataFrame:
    """Rows given as a boolean mask or as integer positions (``None`` → everything)."""
    if rows is None:
        return df
    arr = np.asarray(rows)
    return df[arr] if arr.dtype == bool else df.iloc[arr]


def build_digraph(df: pd.DataFrame, rows: np.ndarray | None = None) -> nx.DiGraph:
    """Aggregated directed graph: one edge per ``(sender, receiver)`` pair.

    Edge attributes: ``count``, ``total_amount``, ``first_ts``, ``last_ts`` and, when labels
    are available, ``n_illicit``.
    """
    sub = _select(df, rows)
    agg = (
        sub.groupby(["sender", "receiver"])
        .agg(
            count=("amount", "size"),
            total_amount=("amount", "sum"),
            first_ts=("timestamp", "min"),
            last_ts=("timestamp", "max"),
            n_illicit=("label", lambda s: float(np.nansum(s.to_numpy(dtype="float64")))),
        )
        .reset_index()
    )
    graph = nx.DiGraph()
    for rec in agg.to_dict("records"):
        attrs = {str(k): v for k, v in rec.items()}
        s, r = attrs.pop("sender"), attrs.pop("receiver")
        graph.add_edge(s, r, **attrs)
    return graph


def graph_summary(graph: nx.Graph) -> dict[str, float | int]:
    """Quick structural description (size, density, weak components, degree extremes)."""
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    if graph.is_directed():
        components = nx.number_weakly_connected_components(graph) if n else 0
        degrees = [d for _, d in graph.degree()]
    else:
        components = nx.number_connected_components(graph) if n else 0
        degrees = [d for _, d in graph.degree()]
    return {
        "nodes": n,
        "edges": m,
        "density": float(nx.density(graph)) if n > 1 else 0.0,
        "components": int(components),
        "max_degree": int(max(degrees)) if degrees else 0,
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
    }


def communities(graph: nx.Graph, seed: int = 0) -> list[set[str]]:
    """Louvain communities on the undirected, weighted projection (``count`` as weight)."""
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for u, v, data in graph.edges(data=True):
        w = data.get("count", 1)
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += w
        else:
            undirected.add_edge(u, v, weight=w)
    return [
        set(c) for c in nx.community.louvain_communities(undirected, weight="weight", seed=seed)
    ]
