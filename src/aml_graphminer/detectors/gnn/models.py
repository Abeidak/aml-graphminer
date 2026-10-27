"""Graph neural networks (PyTorch Geometric): edge classifier and node classifier.

``EdgeGNN`` is the architecture of the original Colab notebooks: a stack of message-passing
layers encodes the *accounts*, then an MLP scores every *transaction* from
``[h_sender, h_receiver, transaction features]``. ``NodeGNN`` is the node-level counterpart
used for the Elliptic Bitcoin benchmark.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...exceptions import ConfigError
from ...utils import require
from .config import GNN_KINDS, GNNConfig


def make_conv(kind: str, in_dim: int, out_dim: int, heads: int = 4) -> nn.Module:
    """One message-passing layer producing ``out_dim`` features."""
    pyg_nn = require("torch_geometric.nn", "gnn", "graph neural networks")
    if kind == "gcn":
        return pyg_nn.GCNConv(in_dim, out_dim)  # type: ignore[no-any-return]
    if kind == "sage":
        return pyg_nn.SAGEConv(in_dim, out_dim)  # type: ignore[no-any-return]
    if kind == "gat":
        if out_dim % heads:
            raise ConfigError("for GAT, the output width must be divisible by the number of heads")
        return pyg_nn.GATConv(in_dim, out_dim // heads, heads=heads)  # type: ignore[no-any-return]
    raise ConfigError(f"unknown GNN kind {kind!r}; choose among {GNN_KINDS}")


class _Encoder(nn.Module):
    """Stack of ``layers`` message-passing layers with ReLU + dropout."""

    def __init__(
        self, kind: str, in_dim: int, hidden: int, layers: int, dropout: float, heads: int
    ):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList(
            make_conv(kind, dims[i], dims[i + 1], heads) for i in range(layers)
        )
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)), self.dropout, self.training)
        return x


class EdgeGNN(nn.Module):
    """GNN over accounts + MLP over ``[h_src, h_dst, edge features]`` (one logit per transaction)."""

    def __init__(
        self,
        kind: str,
        in_node: int,
        in_edge: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.encoder = _Encoder(kind, in_node, hidden, layers, dropout, heads)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden + in_edge, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    @classmethod
    def from_config(cls, config: GNNConfig, in_node: int, in_edge: int) -> EdgeGNN:
        return cls(
            config.kind,
            in_node,
            in_edge,
            config.hidden,
            config.layers,
            config.dropout,
            config.gat_heads,
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Account embeddings after message passing."""
        return self.encoder(x, edge_index)

    def decode(self, h: Tensor, label_index: Tensor, edge_feat: Tensor) -> Tensor:
        """Transaction logits from account embeddings and transaction features."""
        z = torch.cat([h[label_index[0]], h[label_index[1]], edge_feat], dim=1)
        return self.mlp(z).squeeze(-1)

    def forward(
        self, x: Tensor, edge_index: Tensor, label_index: Tensor, edge_feat: Tensor
    ) -> Tensor:
        return self.decode(self.encode(x, edge_index), label_index, edge_feat)


class NodeGNN(nn.Module):
    """GNN producing one logit per node (e.g. Elliptic Bitcoin transactions)."""

    def __init__(
        self,
        kind: str,
        in_node: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.encoder = _Encoder(kind, in_node, hidden, layers, dropout, heads)
        self.head = nn.Linear(hidden, 1)

    @classmethod
    def from_config(cls, config: GNNConfig, in_node: int) -> NodeGNN:
        return cls(
            config.kind, in_node, config.hidden, config.layers, config.dropout, config.gat_heads
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.head(self.encoder(x, edge_index)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> dict[str, Any]:
    return {"class": type(model).__name__, "parameters": count_parameters(model)}


__all__ = ["EdgeGNN", "NodeGNN", "count_parameters", "make_conv", "model_summary"]
