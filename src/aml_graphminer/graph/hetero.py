"""Heterogeneous transaction graphs: accounts (typed) and banks (Layer 3).

Node types default to ``"account"`` and can be refined with a mapping such as
``{"3_00A1": "company", "3_00B7": "individual"}``. Banks become nodes of type ``"bank"``
linked to their accounts by ``held_at`` edges; money flows are ``transfers`` edges.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import networkx as nx
import pandas as pd

from ..utils import require

if TYPE_CHECKING:  # pragma: no cover
    from torch_geometric.data import HeteroData

TRANSFER, HELD_AT = "transfers", "held_at"


def build_hetero_networkx(
    df: pd.DataFrame,
    account_types: Mapping[str, str] | None = None,
    default_type: str = "account",
) -> nx.MultiDiGraph:
    """Typed multigraph: ``ntype`` on nodes, ``etype`` on edges.

    * node types: the values of ``account_types`` (default ``"account"``) and ``"bank"``;
    * ``transfers`` edges carry ``amount``, ``timestamp`` and ``label`` (key = transaction id);
    * ``held_at`` edges connect an account to its bank (when bank ids are known).
    """
    account_types = account_types or {}
    graph = nx.MultiDiGraph()
    banks: dict[str, str] = {}
    for tx_id, rec in zip(df.index.tolist(), df.to_dict("records")):
        for role in ("sender", "receiver"):
            account = rec[role]
            if account not in graph:
                graph.add_node(account, ntype=account_types.get(account, default_type))
            bank = rec[f"{role}_bank"]
            if bank:
                banks[account] = bank
        graph.add_edge(
            rec["sender"],
            rec["receiver"],
            key=tx_id,
            etype=TRANSFER,
            amount=rec["amount"],
            timestamp=rec["timestamp"],
            label=rec["label"],
        )
    for account, bank in banks.items():
        bank_node = f"bank:{bank}"
        if bank_node not in graph:
            graph.add_node(bank_node, ntype="bank")
        graph.add_edge(account, bank_node, key=f"held_at:{account}", etype=HELD_AT)
    return graph


def node_type_counts(graph: nx.MultiDiGraph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, ntype in graph.nodes(data="ntype"):
        counts[ntype] = counts.get(ntype, 0) + 1
    return counts


def to_pyg_hetero(graph: nx.MultiDiGraph, feature_dim: int = 1) -> HeteroData:
    """Typed NetworkX graph → ``torch_geometric.data.HeteroData`` (constant node features)."""
    torch = require("torch", "gnn", "heterogeneous graphs")
    hetero_mod = require("torch_geometric.data", "gnn", "heterogeneous graphs")
    data: Any = hetero_mod.HeteroData()
    index: dict[str, dict[Any, int]] = {}
    for node, ntype in graph.nodes(data="ntype"):
        index.setdefault(ntype, {})[node] = len(index.get(ntype, {}))
    for ntype, members in index.items():
        data[ntype].x = torch.ones((len(members), feature_dim), dtype=torch.float32)
    edges: dict[tuple[str, str, str], tuple[list[int], list[int]]] = {}
    for u, v, attrs in graph.edges(data=True):
        key = (graph.nodes[u]["ntype"], attrs["etype"], graph.nodes[v]["ntype"])
        src, dst = edges.setdefault(key, ([], []))
        src.append(index[key[0]][u])
        dst.append(index[key[2]][v])
    for key, (src, dst) in edges.items():
        data[key].edge_index = torch.tensor([src, dst], dtype=torch.long)
    return data
