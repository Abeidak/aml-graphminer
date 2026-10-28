"""Bidirectional NetworkX <-> array / PyTorch-Geometric conversion (Layer 3).

The array-level functions are dependency-free; the PyG wrappers import torch lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

from ..utils import require

if TYPE_CHECKING:  # pragma: no cover
    from torch_geometric.data import Data


def nx_to_arrays(
    graph: nx.Graph,
    node_attrs: Sequence[str] = (),
    edge_attrs: Sequence[str] = (),
) -> dict[str, Any]:
    """Convert a NetworkX (multi)graph to plain numpy arrays.

    Returns a dict with ``node_names`` (list), ``x`` ``(n, len(node_attrs))`` or ``None``,
    ``edge_index`` ``(2, m)`` int64 (undirected graphs are expanded to both directions),
    ``edge_attr`` ``(m, len(edge_attrs))`` or ``None`` and ``edge_keys`` (list, ``None`` entries
    for simple graphs).
    """
    names = list(graph.nodes)
    position = {n: i for i, n in enumerate(names)}
    x = (
        np.array(
            [[float(graph.nodes[n].get(a, 0.0)) for a in node_attrs] for n in names],
            dtype="float32",
        )
        if node_attrs
        else None
    )
    src: list[int] = []
    dst: list[int] = []
    attrs: list[list[float]] = []
    keys: list[Any] = []
    multigraph = graph.is_multigraph()
    edges = graph.edges(keys=True, data=True) if multigraph else graph.edges(data=True)
    for item in edges:
        if multigraph:
            u, v, key, data = item
        else:
            u, v, data = item
            key = None
        pairs = [(u, v)] if graph.is_directed() else [(u, v), (v, u)]
        for a, b in pairs:
            src.append(position[a])
            dst.append(position[b])
            keys.append(key)
            if edge_attrs:
                attrs.append([float(data.get(e, 0.0)) for e in edge_attrs])
    return {
        "node_names": names,
        "x": x,
        "edge_index": np.array([src, dst], dtype="int64").reshape(2, -1),
        "edge_attr": np.array(attrs, dtype="float32") if edge_attrs else None,
        "edge_keys": keys,
    }


def arrays_to_nx(
    edge_index: np.ndarray,
    x: np.ndarray | None = None,
    edge_attr: np.ndarray | None = None,
    node_names: Sequence[Any] | None = None,
    node_attr_names: Sequence[str] | None = None,
    edge_attr_names: Sequence[str] | None = None,
    multigraph: bool = False,
) -> nx.DiGraph | nx.MultiDiGraph:
    """Inverse of :func:`nx_to_arrays` for directed graphs."""
    n = int(edge_index.max()) + 1 if edge_index.size else 0
    if x is not None:
        n = max(n, len(x))
    if node_names is not None:
        n = max(n, len(node_names))
    names = list(node_names) if node_names is not None else list(range(n))
    names += list(range(len(names), n))
    graph: nx.DiGraph | nx.MultiDiGraph = nx.MultiDiGraph() if multigraph else nx.DiGraph()
    for i, name in enumerate(names):
        attrs: dict[str, float] = {}
        if x is not None:
            labels = node_attr_names or [f"x{j}" for j in range(x.shape[1])]
            attrs = {lab: float(x[i, j]) for j, lab in enumerate(labels)}
        graph.add_node(name, **attrs)
    for k in range(edge_index.shape[1]):
        attrs = {}
        if edge_attr is not None:
            labels = edge_attr_names or [f"e{j}" for j in range(edge_attr.shape[1])]
            attrs = {lab: float(edge_attr[k, j]) for j, lab in enumerate(labels)}
        graph.add_edge(names[int(edge_index[0, k])], names[int(edge_index[1, k])], **attrs)
    return graph


def nx_to_pyg(
    graph: nx.Graph,
    node_attrs: Sequence[str] = (),
    edge_attrs: Sequence[str] = (),
) -> Data:
    """NetworkX → ``torch_geometric.data.Data`` (``data.node_names`` keeps the account ids)."""
    torch = require("torch", "gnn", "PyG conversion")
    data_mod = require("torch_geometric.data", "gnn", "PyG conversion")
    arrays = nx_to_arrays(graph, node_attrs, edge_attrs)
    n = len(arrays["node_names"])
    x = (
        torch.from_numpy(arrays["x"])
        if arrays["x"] is not None
        else torch.ones((n, 1), dtype=torch.float32)
    )
    data = data_mod.Data(x=x, edge_index=torch.from_numpy(arrays["edge_index"]), num_nodes=n)
    if arrays["edge_attr"] is not None:
        data.edge_attr = torch.from_numpy(arrays["edge_attr"])
    data.node_names = arrays["node_names"]
    return data


def pyg_to_nx(
    data: Data,
    node_names: Sequence[Any] | None = None,
    node_attr_names: Sequence[str] | None = None,
    edge_attr_names: Sequence[str] | None = None,
    multigraph: bool = False,
) -> nx.DiGraph | nx.MultiDiGraph:
    """``torch_geometric.data.Data`` → NetworkX directed graph."""
    names = node_names if node_names is not None else getattr(data, "node_names", None)
    x = data.x.detach().cpu().numpy() if getattr(data, "x", None) is not None else None
    edge_attr = (
        data.edge_attr.detach().cpu().numpy()
        if getattr(data, "edge_attr", None) is not None
        else None
    )
    return arrays_to_nx(
        data.edge_index.detach().cpu().numpy(),
        x=x,
        edge_attr=edge_attr,
        node_names=names,
        node_attr_names=node_attr_names,
        edge_attr_names=edge_attr_names,
        multigraph=multigraph,
    )
