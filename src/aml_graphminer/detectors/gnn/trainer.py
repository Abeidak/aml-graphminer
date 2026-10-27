"""Plain-PyTorch training of the edge-classification GNNs (the notebooks' loop, packaged)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor, nn

from ...evaluation.metrics import best_threshold
from ...exceptions import ConfigError
from ...features.pipeline import FeatureSet
from ...tracking import Tracker
from ...utils import logger, set_seed
from .config import GNNConfig
from .data import build_loaders, make_link_loader, resolve_device, split_graph_rows
from .losses import build_criterion
from .models import EdgeGNN

# Seed of the neighbour sampling used to *score* transactions: fixed, so that the same
# transactions always get the same scores (an alert must be reproducible to be auditable).
INFERENCE_SEED = 0
# Transactions decoded per step by the full-neighbourhood inference.
FULL_INFERENCE_CHUNK = 65_536


@dataclass
class TrainedGNN:
    """A trained edge classifier together with everything needed to reuse it.

    Attributes
    ----------
    model:
        The :class:`EdgeGNN` (best validation epoch restored).
    config:
        Hyper-parameters used for training.
    stats:
        Feature-pipeline statistics (:meth:`FeaturePipeline.to_dict`) — reloaded at inference
        so user data is transformed exactly like the training data.
    in_node, in_edge:
        Input widths of the node / transaction features.
    threshold:
        Decision threshold chosen on the validation split.
    history:
        Per-epoch ``train_loss`` / ``val_pr_auc``.
    best_epoch:
        1-based epoch of the restored weights.
    metrics:
        Test metrics when the model was trained by :func:`run_gnn_experiment`.
    """

    model: nn.Module
    config: GNNConfig
    stats: dict[str, Any]
    in_node: int
    in_edge: int
    threshold: float = 0.5
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


@contextmanager
def seeded_sampling(device: torch.device, seed: int = INFERENCE_SEED) -> Iterator[None]:
    """Run the enclosed neighbour sampling with a fixed seed, leaving the global RNG untouched.

    PyG's samplers draw from PyTorch's random generator: seeding it inside ``fork_rng`` makes
    scoring reproducible without changing the random stream used by training.
    """
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


@torch.no_grad()
def predict_edges(
    model: nn.Module,
    loader: Any,
    edge_feat: Tensor,
    device: torch.device,
    *,
    seed: int = INFERENCE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities of the transactions served by ``loader`` (seeded, hence reproducible).

    Returns ``(transaction_ids, probabilities)`` sorted by transaction id.
    """
    model.eval()
    id_chunks: list[Tensor] = []
    prob_chunks: list[Tensor] = []
    with seeded_sampling(device, seed):
        for batch in loader:
            batch = batch.to(device)
            ids = batch.edge_label
            logits = model(batch.x, batch.edge_index, batch.edge_label_index, edge_feat[ids])
            prob_chunks.append(torch.sigmoid(logits).cpu())
            id_chunks.append(ids.cpu())
    ids_all = torch.cat(id_chunks).numpy()
    probs = torch.cat(prob_chunks).numpy()
    order = np.argsort(ids_all, kind="stable")
    return ids_all[order], probs[order]


@torch.no_grad()
def predict_edges_full(
    model: nn.Module,
    features: FeatureSet,
    graph_rows: np.ndarray,
    target_rows: np.ndarray,
    device: torch.device,
    *,
    chunk_size: int = FULL_INFERENCE_CHUNK,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact probabilities with *full* neighbourhoods (no sampling).

    Account embeddings are computed once over the whole message-passing graph formed by
    ``graph_rows`` (both directions), then the transactions of ``target_rows`` are decoded in
    chunks. The result does not depend on batching and is deterministic.
    """
    model.eval()
    graph = np.flatnonzero(np.asarray(graph_rows, dtype=bool))
    targets = np.flatnonzero(np.asarray(target_rows, dtype=bool))
    if len(targets) == 0:
        raise ConfigError("no transaction to score: the target selection is empty")
    src = torch.from_numpy(np.ascontiguousarray(features.src))
    dst = torch.from_numpy(np.ascontiguousarray(features.dst))
    picked = torch.from_numpy(graph)
    edge_index = torch.stack([src[picked], dst[picked]])
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(device)
    x = torch.from_numpy(np.ascontiguousarray(features.node)).to(device)
    h = model.encode(x, edge_index)  # type: ignore[operator]
    chunks: list[Tensor] = []
    for start in range(0, len(targets), chunk_size):
        rows = targets[start : start + chunk_size]
        index = torch.from_numpy(rows)
        label_index = torch.stack([src[index], dst[index]]).to(device)
        edge_feat = torch.from_numpy(np.ascontiguousarray(features.edge[rows])).to(device)
        logits = model.decode(h, label_index, edge_feat)  # type: ignore[operator]
        chunks.append(torch.sigmoid(logits).cpu())
    return targets, torch.cat(chunks).numpy()


def predict_split(
    model: nn.Module,
    features: FeatureSet,
    graph_rows: np.ndarray,
    target_rows: np.ndarray,
    config: GNNConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Score ``target_rows`` on the graph of ``graph_rows`` with the configured inference mode."""
    if config.inference == "full":
        return predict_edges_full(model, features, graph_rows, target_rows, device)
    loader = make_link_loader(features, graph_rows, target_rows, config)
    edge_feat = torch.from_numpy(np.ascontiguousarray(features.edge)).to(device)
    return predict_edges(model, loader, edge_feat, device)


def _val_pr_auc(y: np.ndarray, ids: np.ndarray, probs: np.ndarray) -> float:
    truth = y[ids]
    return float(average_precision_score(truth, probs)) if truth.sum() > 0 else float("nan")


def fit_edge_gnn(
    features: dict[str, FeatureSet],
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    config: GNNConfig,
    *,
    seed: int = 0,
    pipeline_stats: dict[str, Any] | None = None,
    tracker: Tracker | None = None,
    loaders: dict[str, Any] | None = None,
) -> TrainedGNN:
    """Train an :class:`EdgeGNN` on the training split; keep the best validation PR-AUC epoch.

    Parameters
    ----------
    features:
        ``{"train": ..., "val": ...}`` feature sets (see ``FeaturePipeline.transform_splits``).
    y:
        Labels of every transaction (1 = laundering).
    masks:
        Boolean split masks (``train`` / ``val`` / ``test``).
    """
    if config.trainer == "lightning":
        from .lightning_trainer import fit_edge_gnn_lightning

        return fit_edge_gnn_lightning(
            features, y, masks, config, seed=seed, pipeline_stats=pipeline_stats, tracker=tracker
        )
    tracker = tracker or Tracker()
    set_seed(seed)
    device = resolve_device(config.device)
    full_eval = config.inference == "full"
    loaders = loaders or build_loaders(
        features, masks, config, ("train",) if full_eval else ("train", "val")
    )
    val_graph = split_graph_rows(masks)["val"]
    in_node = int(features["train"].node.shape[1])
    in_edge = int(features["train"].edge.shape[1])
    edge_feat = {
        name: torch.from_numpy(np.ascontiguousarray(features[name].edge)).to(device)
        for name in ("train", "val")
    }
    labels = torch.from_numpy(np.nan_to_num(y.astype("float32"))).to(device)

    model = EdgeGNN.from_config(config, in_node, in_edge).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = build_criterion(config)

    best_ap, best_state, best_epoch, stale = -1.0, None, 0, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            ids = batch.edge_label
            optimiser.zero_grad()
            logits = model(
                batch.x, batch.edge_index, batch.edge_label_index, edge_feat["train"][ids]
            )
            loss = criterion(logits, labels[ids])
            loss.backward()
            optimiser.step()
            total += float(loss.item())
            batches += 1
        if full_eval:
            ids_v, p_v = predict_edges_full(model, features["val"], val_graph, masks["val"], device)
        else:
            ids_v, p_v = predict_edges(model, loaders["val"], edge_feat["val"], device)
        val_ap = _val_pr_auc(y, ids_v, p_v)
        entry = {"epoch": float(epoch), "train_loss": total / max(batches, 1), "val_pr_auc": val_ap}
        history.append(entry)
        tracker.log_metrics(
            {f"{config.kind}/train_loss": entry["train_loss"], f"{config.kind}/val_pr_auc": val_ap},
            step=epoch,
        )
        logger.info(
            "[%s] epoch %2d | loss %.4f | val PR-AUC %.4f",
            config.kind,
            epoch,
            entry["train_loss"],
            val_ap,
        )
        score = -1.0 if np.isnan(val_ap) else val_ap
        if score > best_ap or best_state is None:
            best_ap, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if config.patience and stale >= config.patience:
                logger.info("[%s] early stopping at epoch %d", config.kind, epoch)
                break
    if best_state is None:  # pragma: no cover - epochs >= 1 guarantees a state
        raise ConfigError("training produced no model")
    model.load_state_dict(best_state)
    model.eval()
    return TrainedGNN(
        model=model.cpu(),
        config=config,
        stats=pipeline_stats or {},
        in_node=in_node,
        in_edge=in_edge,
        history=history,
        best_epoch=best_epoch,
    )


def score_splits(
    trained: TrainedGNN,
    features: dict[str, FeatureSet],
    masks: dict[str, np.ndarray],
    splits: tuple[str, ...] = ("val", "test"),
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Scores of the transactions of each split (aligned with ``np.flatnonzero(mask)``)."""
    dev = resolve_device(device or trained.config.device)
    graph_rows = split_graph_rows(masks)
    model = trained.model.to(dev).eval()
    out: dict[str, np.ndarray] = {}
    for name in splits:
        ids, probs = predict_split(
            model, features[name], graph_rows[name], masks[name], trained.config, dev
        )
        expected = np.flatnonzero(masks[name])
        if not np.array_equal(ids, expected):  # pragma: no cover - defensive
            raise ConfigError(f"loader returned unexpected transactions for the {name} split")
        out[name] = probs
    trained.model = model.cpu()
    return out


def threshold_from_validation(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    """F1-optimal threshold on the validation split."""
    return best_threshold(y_val, scores_val)


__all__ = [
    "INFERENCE_SEED",
    "TrainedGNN",
    "fit_edge_gnn",
    "predict_edges",
    "predict_edges_full",
    "predict_split",
    "score_splits",
    "seeded_sampling",
    "threshold_from_validation",
]
