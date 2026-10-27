"""Adapter for the Elliptic Bitcoin dataset (Weber et al., 2019).

In Elliptic the **transactions are the nodes** (and edges are the flow of bitcoins between
them), so this adapter returns a *node-classification* dataset instead of a table of
account-to-account transfers. It is consumed by
:class:`aml_graphminer.detectors.gnn.NodeGNN` and by the tabular baselines.

Expected files (Kaggle ``ellipticco/elliptic-data-set``)::

    elliptic_txs_features.csv   # no header: txId, time step, <feature columns...>
    elliptic_txs_classes.csv    # txId, class   (class: "1" illicit, "2" licit, "unknown")
    elliptic_txs_edgelist.csv   # txId1, txId2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..exceptions import IngestionError

ILLICIT, LICIT, UNKNOWN = 1, 0, -1


@dataclass
class EllipticData:
    """Node-level dataset: features, labels (1 illicit / 0 licit / -1 unknown) and edges."""

    node_ids: np.ndarray
    time_step: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    edge_index: np.ndarray  # shape (2, E), positions in node_ids

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    def split_masks(self, train_max: int = 29, val_max: int = 34) -> dict[str, np.ndarray]:
        """Temporal split on *labelled* nodes.

        Weber et al. train on the first 34 time steps and test on the last 15; we carve the
        validation set out of the end of the training period (steps ``train_max+1 .. val_max``).
        """
        known = self.labels != UNKNOWN
        return {
            "train": known & (self.time_step <= train_max),
            "val": known & (self.time_step > train_max) & (self.time_step <= val_max),
            "test": known & (self.time_step > val_max),
        }

    def summary(self) -> dict[str, float | int]:
        known = self.labels != UNKNOWN
        return {
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "features": self.num_features,
            "time_steps": len(np.unique(self.time_step)),
            "labelled": int(known.sum()),
            "illicit": int((self.labels == ILLICIT).sum()),
            "licit": int((self.labels == LICIT).sum()),
            "illicit_ratio_among_labelled": float(
                (self.labels == ILLICIT).sum() / max(int(known.sum()), 1)
            ),
        }


def _find(directory: Path, name: str) -> Path:
    direct = directory / name
    if direct.exists():
        return direct
    hits = sorted(directory.rglob(name))
    if not hits:
        raise IngestionError(f"{name} not found under {directory}")
    return hits[0]


def read_elliptic(directory: str | Path, *, nrows: int | None = None) -> EllipticData:
    """Read the three Elliptic CSV files located in (or below) ``directory``."""
    directory = Path(directory)
    if not directory.is_dir():
        raise IngestionError(f"not a directory: {directory}")
    feats = pd.read_csv(_find(directory, "elliptic_txs_features.csv"), header=None, nrows=nrows)
    classes = pd.read_csv(_find(directory, "elliptic_txs_classes.csv"))
    edges = pd.read_csv(_find(directory, "elliptic_txs_edgelist.csv"))

    if feats.shape[1] < 3:
        raise IngestionError("elliptic features file needs txId, time step and >= 1 feature")
    node_ids = feats.iloc[:, 0].to_numpy(dtype="int64")
    time_step = feats.iloc[:, 1].to_numpy(dtype="int64")
    features = feats.iloc[:, 2:].to_numpy(dtype="float32")

    classes = classes.rename(columns={classes.columns[0]: "txId", classes.columns[1]: "class"})
    mapping = {"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}
    cls = classes.set_index("txId")["class"].astype(str).str.strip().map(mapping)
    labels = pd.Series(node_ids).map(cls).fillna(UNKNOWN).to_numpy(dtype="int8")

    position = pd.Series(np.arange(len(node_ids)), index=node_ids)
    src = position.reindex(edges.iloc[:, 0].to_numpy()).to_numpy()
    dst = position.reindex(edges.iloc[:, 1].to_numpy()).to_numpy()
    keep = ~(np.isnan(src) | np.isnan(dst))
    edge_index = np.vstack([src[keep], dst[keep]]).astype("int64")
    return EllipticData(node_ids, time_step, features, labels, edge_index)
