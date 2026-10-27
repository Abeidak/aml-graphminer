"""PyTorch-Geometric representation for GNN training (Layer 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..features.pipeline import FeatureSet
from ..utils import require

if TYPE_CHECKING:  # pragma: no cover
    from torch_geometric.data import Data


def to_pyg_data(
    features: FeatureSet,
    rows: np.ndarray | None = None,
    bidirectional: bool = True,
    with_edge_attr: bool = False,
) -> Data:  # pragma: no cover - needs torch
    """Build the message-passing graph from the transactions flagged by ``rows``.

    * nodes = accounts, ``x`` = node features;
    * edges = transactions of ``rows`` (the *training* graph only contains training edges, so
      validation / test transactions never leak into message passing);
    * ``bidirectional=True`` adds the reverse of every edge (money flow is directed but
      suspicious structure is visible from both ends);
    * ``edge_id`` stores the transaction id of every edge so that edge features and labels
      can be looked up by id after neighbour sampling.
    """
    torch = require("torch", "gnn", "the PyG graph builder")
    data_mod = require("torch_geometric.data", "gnn", "the PyG graph builder")
    mask = np.ones(features.n_edges, dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    ids = np.flatnonzero(mask)
    src = torch.from_numpy(features.src[ids])
    dst = torch.from_numpy(features.dst[ids])
    edge_index = torch.stack([src, dst])
    edge_id = torch.from_numpy(ids)
    edge_attr = torch.from_numpy(features.edge[ids]) if with_edge_attr else None
    if bidirectional:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_id = torch.cat([edge_id, edge_id])
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr])
    data = data_mod.Data(
        x=torch.from_numpy(features.node), edge_index=edge_index, num_nodes=features.n_nodes
    )
    data.edge_id = edge_id
    if edge_attr is not None:
        data.edge_attr = edge_attr
    return data
