"""Small Elliptic-like node-classification dataset (tests, demos, CI — not a scientific proxy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..ingestion.elliptic import ILLICIT, LICIT, UNKNOWN, EllipticData


def make_elliptic_like(
    n_nodes: int = 3000,
    n_features: int = 24,
    n_steps: int = 49,
    illicit_rate: float = 0.08,
    unknown_rate: float = 0.5,
    seed: int = 0,
) -> EllipticData:
    """Random graph whose illicit nodes have shifted features and prefer illicit neighbours.

    Time steps are assigned in order, edges only connect nodes of the same step (like Elliptic),
    ``unknown_rate`` of the nodes are unlabelled.
    """
    rng = np.random.default_rng(seed)
    time_step = np.sort(rng.integers(1, n_steps + 1, n_nodes))
    illicit = rng.random(n_nodes) < illicit_rate
    shift = rng.normal(0.0, 1.0, n_features) * 0.8
    features = rng.normal(0.0, 1.0, (n_nodes, n_features)) + illicit[:, None] * shift
    labels = np.where(illicit, ILLICIT, LICIT).astype("int8")
    labels[rng.random(n_nodes) < unknown_rate] = UNKNOWN

    src, dst = [], []
    for step in np.unique(time_step):
        nodes = np.flatnonzero(time_step == step)
        if len(nodes) < 2:
            continue
        n_edges = int(len(nodes) * 1.2)
        a = rng.choice(nodes, n_edges)
        same = illicit[a] & (rng.random(n_edges) < 0.6)
        pool = nodes[illicit[nodes]]
        b = np.where(
            same & (len(pool) > 0),
            rng.choice(pool if len(pool) else nodes, n_edges),
            rng.choice(nodes, n_edges),
        )
        keep = a != b
        src.append(a[keep])
        dst.append(b[keep])
    edge_index = np.vstack([np.concatenate(src), np.concatenate(dst)]).astype("int64")
    node_ids = (1_000_000 + np.arange(n_nodes)).astype("int64")
    return EllipticData(
        node_ids, time_step.astype("int64"), features.astype("float32"), labels, edge_index
    )


def write_elliptic_files(data: EllipticData, directory: str | Path) -> Path:
    """Write the three Kaggle-layout CSV files read by :func:`~aml_graphminer.ingestion.read_elliptic`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    feats = pd.DataFrame(data.features)
    feats.insert(0, "time", data.time_step)
    feats.insert(0, "txId", data.node_ids)
    feats.to_csv(directory / "elliptic_txs_features.csv", header=False, index=False)
    names = {ILLICIT: "1", LICIT: "2", UNKNOWN: "unknown"}
    pd.DataFrame({"txId": data.node_ids, "class": [names[int(v)] for v in data.labels]}).to_csv(
        directory / "elliptic_txs_classes.csv", index=False
    )
    pd.DataFrame(
        {"txId1": data.node_ids[data.edge_index[0]], "txId2": data.node_ids[data.edge_index[1]]}
    ).to_csv(directory / "elliptic_txs_edgelist.csv", index=False)
    return directory


__all__ = ["make_elliptic_like", "write_elliptic_files"]
