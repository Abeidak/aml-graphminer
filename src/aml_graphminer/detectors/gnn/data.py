"""Data plumbing for the GNNs: devices and neighbour-sampling loaders (PyTorch Geometric)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ...exceptions import ConfigError
from ...features.pipeline import FeatureSet
from ...utils import require
from ._compat import check_sampler_backend
from .config import GNNConfig

SPLITS = ("train", "val", "test")


def resolve_device(name: str = "auto") -> torch.device:
    """``"auto"`` → CUDA when available, else CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ConfigError("device='cuda' requested but CUDA is not available")
    return torch.device(name)


def make_link_loader(
    features: FeatureSet,
    graph_rows: np.ndarray,
    target_rows: np.ndarray,
    config: GNNConfig,
    *,
    shuffle: bool = False,
    batch_size: int | None = None,
) -> Any:
    """Neighbour-sampling loader that yields mini-batches of *transactions* to classify.

    Parameters
    ----------
    graph_rows:
        Transactions that form the message-passing graph (both directions are added).
        Training uses the training transactions only, validation adds the validation ones,
        test adds the test ones: a transaction never receives messages from the future.
    target_rows:
        Transactions to score. ``batch.edge_label`` holds their transaction ids (row position).
    """
    data_mod = require("torch_geometric.data", "gnn", "graph neural networks")
    loader_mod = require("torch_geometric.loader", "gnn", "graph neural networks")
    check_sampler_backend()
    # np.array (not asarray) copies: masks derived from pandas can be read-only views, which
    # torch.from_numpy only accepts with a warning
    graph_mask = torch.from_numpy(np.array(graph_rows, dtype=bool))
    target_mask = torch.from_numpy(np.array(target_rows, dtype=bool))
    if int(target_mask.sum()) == 0:
        raise ConfigError("no transaction to score: the target selection is empty")
    src = torch.from_numpy(np.ascontiguousarray(features.src))
    dst = torch.from_numpy(np.ascontiguousarray(features.dst))
    edge_index = torch.stack([src[graph_mask], dst[graph_mask]])
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # messages in both directions
    data = data_mod.Data(
        x=torch.from_numpy(np.ascontiguousarray(features.node)),
        edge_index=edge_index,
        num_nodes=features.n_nodes,
    )
    targets = target_mask.nonzero().view(-1)
    return loader_mod.LinkNeighborLoader(
        data,
        num_neighbors=list(config.num_neighbors),
        edge_label_index=torch.stack([src[targets], dst[targets]]),
        edge_label=targets,
        batch_size=batch_size or config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
    )


def split_graph_rows(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Message-passing rows of every split (train ⊂ val ⊂ test graphs)."""
    return {
        "train": masks["train"],
        "val": masks["train"] | masks["val"],
        "test": masks["train"] | masks["val"] | masks["test"],
    }


def build_loaders(
    features: dict[str, FeatureSet],
    masks: dict[str, np.ndarray],
    config: GNNConfig,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, Any]:
    """One loader per requested split (the training loader shuffles)."""
    graph_rows = split_graph_rows(masks)
    return {
        name: make_link_loader(
            features[name], graph_rows[name], masks[name], config, shuffle=(name == "train")
        )
        for name in splits
    }


__all__ = [
    "SPLITS",
    "build_loaders",
    "make_link_loader",
    "resolve_device",
    "split_graph_rows",
]
