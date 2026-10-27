"""Node classification with GNNs — Elliptic Bitcoin benchmark (transactions are the nodes)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from ...evaluation.metrics import best_threshold, compute_metrics
from ...ingestion.elliptic import EllipticData
from ...tracking import Tracker
from ...utils import logger, require, set_seed
from .config import GNNConfig
from .data import resolve_device
from .losses import build_criterion
from .models import NodeGNN


@dataclass
class NodeResult:
    """Outcome of :func:`run_node_experiment` (scores are for *all* nodes)."""

    model: NodeGNN
    scores: np.ndarray
    metrics: dict[str, float]
    threshold: float
    best_epoch: int
    history: list[dict[str, float]]


def _standardise(features: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    mean = features[fit_rows].mean(axis=0)
    std = features[fit_rows].std(axis=0) + 1e-6
    return ((features - mean) / std).astype("float32")


def run_node_experiment(
    kind: str,
    data: EllipticData,
    config: GNNConfig | None = None,
    seed: int = 0,
    *,
    masks: dict[str, np.ndarray] | None = None,
    tracker: Tracker | None = None,
) -> NodeResult:
    """Full-batch node classification (Elliptic has ~200k nodes: it fits in memory).

    The model sees the *whole* graph (features of unlabeled neighbours are legitimate input,
    as in Weber et al.), the loss uses labelled **training** nodes only, the model is selected
    on validation PR-AUC and the threshold on validation F1; metrics are for the test steps.
    """
    data_mod = require("torch_geometric.data", "gnn", "graph neural networks")
    cfg = (config or GNNConfig()).with_kind(kind)
    tracker = tracker or Tracker()
    set_seed(seed)
    device = resolve_device(cfg.device)
    masks = masks or data.split_masks()

    x = torch.from_numpy(_standardise(data.features, masks["train"]))
    edge_index = torch.from_numpy(data.edge_index)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    graph = data_mod.Data(x=x, edge_index=edge_index, num_nodes=data.num_nodes).to(device)
    y = torch.from_numpy((data.labels == 1).astype("float32")).to(device)
    train_idx = torch.from_numpy(np.flatnonzero(masks["train"])).to(device)
    y_np = (data.labels == 1).astype("float64")

    model = NodeGNN.from_config(cfg, data.num_features).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = build_criterion(cfg)

    best_ap, best_state, best_epoch, stale = -1.0, None, 0, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimiser.zero_grad()
        logits = model(graph.x, graph.edge_index)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimiser.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(graph.x, graph.edge_index)).cpu().numpy()
        val = masks["val"]
        ap = float(average_precision_score(y_np[val], probs[val])) if y_np[val].sum() else 0.0
        history.append({"epoch": float(epoch), "train_loss": float(loss.item()), "val_pr_auc": ap})
        tracker.log_metrics(
            {f"{kind}/train_loss": float(loss.item()), f"{kind}/val_pr_auc": ap}, step=epoch
        )
        logger.info("[%s] epoch %3d | loss %.4f | val PR-AUC %.4f", kind, epoch, loss.item(), ap)
        if ap > best_ap or best_state is None:
            best_ap, best_epoch, stale = ap, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if cfg.patience and stale >= cfg.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(graph.x, graph.edge_index)).cpu().numpy()
    thr = best_threshold(y_np[masks["val"]], scores[masks["val"]])
    metrics = compute_metrics(y_np[masks["test"]], scores[masks["test"]], thr)
    return NodeResult(model.cpu(), scores, metrics, thr, best_epoch, history)


def node_experiment_row(result: NodeResult, kind: str, seed: int) -> dict[str, Any]:
    """Flatten a :class:`NodeResult` into a results-table row."""
    return {"model": kind, "seed": seed, "best_epoch": result.best_epoch, **result.metrics}


__all__ = ["NodeResult", "node_experiment_row", "run_node_experiment"]
