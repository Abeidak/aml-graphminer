"""Unsupervised GNN detector: a graph auto-encoder that flags *unusual* transactions.

Labels are rare and late in anti-money-laundering (an investigation can take months), so the
toolkit also offers a detector that learns **without labels**. It is a graph auto-encoder in the
spirit of GAE (Kipf & Welling, 2016) and DOMINANT (Ding et al., 2019), adapted to transactions:

* a GCN / GraphSAGE / GAT **encoder** embeds the accounts from their features and neighbourhood;
* a **structure decoder** scores how plausible the link ``sender -> receiver`` is
  (inner product of the two embeddings, trained against randomly re-wired pairs);
* an **attribute decoder** reconstructs the features of the transaction (amount, format, time,
  recent activity of both accounts ...) from the two embeddings.

A transaction is anomalous when it is badly reconstructed or links accounts the model finds
unlikely to trade. Both errors are standardised on the training transactions, mixed with weight
``alpha`` and turned into a **percentile** of the training distribution: a score of 0.99 means
"more unusual than 99 % of the training transactions". Labels, when present, are only used to
choose the alert threshold on the validation split and to report metrics — never to train.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ... import __version__
from ...evaluation.metrics import best_threshold, compute_metrics
from ...evaluation.splits import temporal_split
from ...exceptions import ConfigError, ModelError
from ...features.pipeline import FeatureConfig, FeaturePipeline, FeatureSet
from ...tracking import Tracker
from ...utils import logger, set_seed
from .._context import restore_order, stack_context
from .config import GNNConfig
from .data import build_loaders, make_link_loader, resolve_device, split_graph_rows
from .models import _Encoder, count_parameters
from .trainer import FULL_INFERENCE_CHUNK, seeded_sampling

FORMAT_VERSION = 1
MODEL_TYPE = "gnn_unsupervised"
WEIGHTS_FILE = "model.pt"
CONFIG_FILE = "config.json"
N_QUANTILES = 1001
# Weight of the attribute error in the anomaly score (1 - DEFAULT_ALPHA for the structural error).
DEFAULT_ALPHA = 0.7


class EdgeAutoEncoder(nn.Module):
    """Account encoder + structure (link) decoder + transaction-attribute decoder."""

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
        self.attr_decoder = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, in_edge)
        )

    @classmethod
    def from_config(cls, config: GNNConfig, in_node: int, in_edge: int) -> EdgeAutoEncoder:
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
        return self.encoder(x, edge_index)

    @staticmethod
    def link_logits(h: Tensor, pairs: Tensor) -> Tensor:
        """Inner-product decoder: plausibility of ``pairs[0] -> pairs[1]`` (one logit per pair)."""
        return (h[pairs[0]] * h[pairs[1]]).sum(dim=-1)

    def reconstruct(self, h: Tensor, pairs: Tensor) -> Tensor:
        """Reconstructed transaction features of ``pairs``."""
        return self.attr_decoder(torch.cat([h[pairs[0]], h[pairs[1]]], dim=-1))

    def errors(self, h: Tensor, pairs: Tensor, edge_feat: Tensor) -> tuple[Tensor, Tensor]:
        """Per-transaction ``(attribute error, structural surprise)``."""
        attr = (self.reconstruct(h, pairs) - edge_feat).pow(2).mean(dim=-1)
        surprise = F.softplus(-self.link_logits(h, pairs))  # = -log sigmoid(logit)
        return attr, surprise


@dataclass
class TrainedAutoEncoder:
    """A trained :class:`EdgeAutoEncoder` with everything needed to score new transactions."""

    model: EdgeAutoEncoder
    config: GNNConfig
    stats: dict[str, Any]
    in_node: int
    in_edge: int
    alpha: float = DEFAULT_ALPHA
    score_stats: dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.99
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ scoring helpers
    def raw_scores(self, attr: np.ndarray, surprise: np.ndarray) -> np.ndarray:
        """Mix the two standardised errors (``alpha`` weights the attribute error)."""
        s = self.score_stats
        z_attr = (attr - s["attr_mean"]) / s["attr_std"]
        z_struct = (surprise - s["struct_mean"]) / s["struct_std"]
        return np.asarray(self.alpha * z_attr + (1.0 - self.alpha) * z_struct, dtype="float64")

    def to_percentile(self, raw: np.ndarray) -> np.ndarray:
        """Empirical CDF of the training raw scores: 0.99 = above 99 % of training transactions."""
        quantiles = np.asarray(self.score_stats["quantiles"], dtype="float64")
        levels = np.linspace(0.0, 1.0, len(quantiles))
        return np.interp(raw, quantiles, levels, left=0.0, right=1.0)


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def _errors_sampled(
    model: EdgeAutoEncoder,
    features: FeatureSet,
    graph_rows: np.ndarray,
    target_rows: np.ndarray,
    config: GNNConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = make_link_loader(features, graph_rows, target_rows, config)
    edge_feat = torch.from_numpy(np.ascontiguousarray(features.edge)).to(device)
    ids, attrs, structs = [], [], []
    with seeded_sampling(device):
        for batch in loader:
            batch = batch.to(device)
            h = model.encode(batch.x, batch.edge_index)
            a, s = model.errors(h, batch.edge_label_index, edge_feat[batch.edge_label])
            ids.append(batch.edge_label.cpu())
            attrs.append(a.cpu())
            structs.append(s.cpu())
    ids_all = torch.cat(ids).numpy()
    order = np.argsort(ids_all, kind="stable")
    return ids_all[order], torch.cat(attrs).numpy()[order], torch.cat(structs).numpy()[order]


@torch.no_grad()
def _errors_full(
    model: EdgeAutoEncoder,
    features: FeatureSet,
    graph_rows: np.ndarray,
    target_rows: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graph = torch.from_numpy(np.flatnonzero(np.asarray(graph_rows, dtype=bool)))
    targets = np.flatnonzero(np.asarray(target_rows, dtype=bool))
    src = torch.from_numpy(np.ascontiguousarray(features.src))
    dst = torch.from_numpy(np.ascontiguousarray(features.dst))
    edge_index = torch.stack([src[graph], dst[graph]])
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(device)
    h = model.encode(torch.from_numpy(np.ascontiguousarray(features.node)).to(device), edge_index)
    attrs, structs = [], []
    for start in range(0, len(targets), FULL_INFERENCE_CHUNK):
        rows = targets[start : start + FULL_INFERENCE_CHUNK]
        index = torch.from_numpy(rows)
        pairs = torch.stack([src[index], dst[index]]).to(device)
        feat = torch.from_numpy(np.ascontiguousarray(features.edge[rows])).to(device)
        attr, surprise = model.errors(h, pairs, feat)
        attrs.append(attr.cpu())
        structs.append(surprise.cpu())
    return targets, torch.cat(attrs).numpy(), torch.cat(structs).numpy()


def transaction_errors(
    model: EdgeAutoEncoder,
    features: FeatureSet,
    graph_rows: np.ndarray,
    target_rows: np.ndarray,
    config: GNNConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(row ids, attribute errors, structural surprise)`` of ``target_rows`` (sorted by id)."""
    if not np.asarray(target_rows, dtype=bool).any():
        raise ConfigError("no transaction to score: the target selection is empty")
    model.eval()
    if config.inference == "full":
        return _errors_full(model, features, graph_rows, target_rows, device)
    return _errors_sampled(model, features, graph_rows, target_rows, config, device)


# --------------------------------------------------------------------------- training
def fit_autoencoder(
    features: dict[str, FeatureSet],
    masks: dict[str, np.ndarray],
    config: GNNConfig,
    *,
    alpha: float = DEFAULT_ALPHA,
    trim: float = 0.05,
    seed: int = 0,
    pipeline_stats: dict[str, Any] | None = None,
    tracker: Tracker | None = None,
) -> TrainedAutoEncoder:
    """Train on the training transactions, select the epoch with the lowest validation loss.

    No label is used: the validation loss is the reconstruction loss of the validation
    transactions on the train + validation graph.

    ``trim`` makes the training *robust to contamination*: in every mini-batch the ``trim``
    fraction of transactions with the largest reconstruction loss is left out of the gradient
    (the "small-loss" trick). Without it, the auto-encoder slowly learns to reconstruct the
    laundering transactions hidden in the training data too and they stop standing out.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ConfigError("alpha must be in [0, 1]")
    if not 0.0 <= trim < 0.5:
        raise ConfigError("trim must be in [0, 0.5)")
    tracker = tracker or Tracker()
    set_seed(seed)
    device = resolve_device(config.device)
    graph_rows = split_graph_rows(masks)
    loader = build_loaders(features, masks, config, ("train",))["train"]
    in_node = int(features["train"].node.shape[1])
    in_edge = int(features["train"].edge.shape[1])
    edge_feat = torch.from_numpy(np.ascontiguousarray(features["train"].edge)).to(device)
    model = EdgeAutoEncoder.from_config(config, in_node, in_edge).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def loss_of(attr: Tensor, pos: Tensor, neg: Tensor) -> Tensor:
        # per-transaction loss: attribute error + structural error of the real and the fake link
        per_tx = alpha * attr + (1.0 - alpha) * (F.softplus(-pos) + F.softplus(neg))
        if trim > 0 and per_tx.numel() > 1:
            keep = max(1, round(per_tx.numel() * (1.0 - trim)))
            per_tx = torch.topk(per_tx, keep, largest=False).values
        return per_tx.mean()

    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            h = model.encode(batch.x, batch.edge_index)
            pairs = batch.edge_label_index
            attr = (model.reconstruct(h, pairs) - edge_feat[batch.edge_label]).pow(2).mean(dim=-1)
            # negatives: the same senders paired with random accounts of the sampled sub-graph
            fake = torch.randint(0, h.size(0), (pairs.size(1),), device=device)
            negatives = torch.stack([pairs[0], fake])
            loss = loss_of(attr, model.link_logits(h, pairs), model.link_logits(h, negatives))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.item())
            batches += 1
        val_loss = _validation_loss(
            model, features["val"], graph_rows["val"], masks["val"], config, device, alpha
        )
        entry = {"epoch": float(epoch), "train_loss": total / max(batches, 1), "val_loss": val_loss}
        history.append(entry)
        tracker.log_metrics(
            {
                f"gae-{config.kind}/train_loss": entry["train_loss"],
                f"gae-{config.kind}/val_loss": val_loss,
            },
            step=epoch,
        )
        logger.info(
            "[gae-%s] epoch %2d | loss %.4f | val loss %.4f",
            config.kind,
            epoch,
            entry["train_loss"],
            val_loss,
        )
        if val_loss < best or best_state is None:
            best, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if config.patience and stale >= config.patience:
                logger.info("[gae-%s] early stopping at epoch %d", config.kind, epoch)
                break
    if best_state is None:  # pragma: no cover - epochs >= 1 guarantees a state
        raise ConfigError("training produced no model")
    model.load_state_dict(best_state)
    model.eval()

    # standardisation and percentile table from the *training* transactions
    _, attr_tr, struct_tr = transaction_errors(
        model, features["train"], graph_rows["train"], masks["train"], config, device
    )
    trained = TrainedAutoEncoder(
        model=model.cpu(),
        config=config,
        stats=pipeline_stats or {},
        in_node=in_node,
        in_edge=in_edge,
        alpha=alpha,
        history=history,
        best_epoch=best_epoch,
    )
    trained.score_stats = {
        "attr_mean": float(attr_tr.mean()),
        "attr_std": float(attr_tr.std()) + 1e-9,
        "struct_mean": float(struct_tr.mean()),
        "struct_std": float(struct_tr.std()) + 1e-9,
    }
    raw_tr = trained.raw_scores(attr_tr, struct_tr)
    trained.score_stats["quantiles"] = np.quantile(raw_tr, np.linspace(0, 1, N_QUANTILES)).tolist()
    return trained


@torch.no_grad()
def _validation_loss(
    model: EdgeAutoEncoder,
    features: FeatureSet,
    graph_rows: np.ndarray,
    target_rows: np.ndarray,
    config: GNNConfig,
    device: torch.device,
    alpha: float,
) -> float:
    _, attr, surprise = transaction_errors(model, features, graph_rows, target_rows, config, device)
    return float(alpha * attr.mean() + (1.0 - alpha) * surprise.mean())


def score_autoencoder_splits(
    trained: TrainedAutoEncoder,
    features: dict[str, FeatureSet],
    masks: dict[str, np.ndarray],
    splits: tuple[str, ...] = ("val", "test"),
) -> dict[str, np.ndarray]:
    """Percentile scores of the transactions of each split (aligned with ``np.flatnonzero``)."""
    device = resolve_device(trained.config.device)
    graph_rows = split_graph_rows(masks)
    model = trained.model.to(device)
    out: dict[str, np.ndarray] = {}
    for name in splits:
        _, attr, surprise = transaction_errors(
            model, features[name], graph_rows[name], masks[name], trained.config, device
        )
        out[name] = trained.to_percentile(trained.raw_scores(attr, surprise))
    trained.model = model.cpu()
    return out


def run_gae_experiment(
    df: pd.DataFrame,
    features: dict[str, FeatureSet],
    masks: dict[str, np.ndarray],
    config: GNNConfig | None = None,
    seed: int = 0,
    *,
    alpha: float = DEFAULT_ALPHA,
    contamination: float = 0.01,
    tracker: Tracker | None = None,
    pipeline_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train without labels, score validation / test, choose the threshold, report metrics.

    The threshold is F1-optimal on the validation labels when they exist, otherwise the
    ``1 - contamination`` percentile (e.g. the 1 % most unusual transactions are alerts).
    """
    cfg = config or GNNConfig()
    trained = fit_autoencoder(
        features, masks, cfg, alpha=alpha, seed=seed, pipeline_stats=pipeline_stats, tracker=tracker
    )
    scores = score_autoencoder_splits(trained, features, masks)
    y = df["label"].to_numpy(dtype="float64")
    y_val, y_test = y[masks["val"]], y[masks["test"]]
    labelled_val = not np.isnan(y_val).any() and np.nansum(y_val) > 0
    trained.threshold = (
        best_threshold(y_val, scores["val"]) if labelled_val else float(1.0 - contamination)
    )
    if not np.isnan(y_test).any() and np.nansum(y_test) > 0:
        trained.metrics = {
            k: float(v)
            for k, v in compute_metrics(y_test, scores["test"], trained.threshold).items()
        }
    return {
        "scores_val": scores["val"],
        "scores_test": scores["test"],
        "trained": trained,
        "best_epoch": trained.best_epoch,
    }


# --------------------------------------------------------------------------- detector
class AutoEncoderDetector:
    """Unsupervised GNN detector with the common ``fit / score / predict / save / load`` API.

    Example
    -------
    >>> det = AutoEncoderDetector.fit(df, GNNConfig(kind="sage", epochs=5))   # doctest: +SKIP
    >>> det.predict(new_df).head()                                          # doctest: +SKIP
    """

    model_type = MODEL_TYPE

    def __init__(self, trained: TrainedAutoEncoder) -> None:
        self.trained = trained

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        config: GNNConfig | None = None,
        *,
        alpha: float = DEFAULT_ALPHA,
        contamination: float = 0.01,
        fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
        feature_config: FeatureConfig | None = None,
        seed: int = 0,
        tracker: Tracker | None = None,
    ) -> AutoEncoderDetector:
        """Temporal split → features fitted on train → unsupervised training → threshold.

        Works on **unlabelled** data; labels (if any) only set the threshold and the metrics.
        """
        if not 0.0 < contamination < 1.0:
            raise ConfigError("contamination must be in (0, 1)")
        masks = temporal_split(df, fractions)
        pipeline = FeaturePipeline(feature_config).fit(df, masks["train"])
        features = pipeline.transform_splits(df, masks)
        out = run_gae_experiment(
            df,
            features,
            masks,
            config,
            seed,
            alpha=alpha,
            contamination=contamination,
            tracker=tracker,
            pipeline_stats=pipeline.to_dict(),
        )
        return cls(out["trained"])

    # ------------------------------------------------------------------ properties
    @property
    def threshold(self) -> float:
        return self.trained.threshold

    @property
    def config(self) -> GNNConfig:
        return self.trained.config

    @property
    def metrics(self) -> dict[str, float]:
        return self.trained.metrics

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "kind": self.config.kind,
            "threshold": self.threshold,
            "alpha": self.trained.alpha,
            "parameters": count_parameters(self.trained.model),
            "best_epoch": self.trained.best_epoch,
            "inference": self.config.inference,
            "metrics": self.metrics,
        }

    # ------------------------------------------------------------------ inference
    def score(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.Series:
        """Anomaly percentile of each transaction (index = ``df.index``)."""
        if len(df) == 0:
            return pd.Series(dtype="float64", name="score")
        if not self.trained.stats:
            raise ModelError(
                "this model carries no feature statistics: it cannot score raw transactions"
            )
        full, is_target, target_pos = stack_context(df, context)
        features = FeaturePipeline.from_dict(self.trained.stats).transform(full)
        device = resolve_device(self.config.device)
        model = self.trained.model.to(device)
        ids, attr, surprise = transaction_errors(
            model, features, np.ones(len(full), dtype=bool), is_target, self.config, device
        )
        self.trained.model = model.cpu()
        by_row = np.zeros(len(full))
        by_row[ids] = self.trained.to_percentile(self.trained.raw_scores(attr, surprise))
        values = restore_order(by_row, is_target, target_pos, len(df))
        return pd.Series(values, index=df.index, name="score")

    def predict(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.DataFrame:
        scores = self.score(df, context)
        return pd.DataFrame({"score": scores, "alert": (scores >= self.threshold).astype(int)})

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        t = self.trained
        torch.save({k: v.cpu() for k, v in t.model.state_dict().items()}, directory / WEIGHTS_FILE)
        payload = {
            "format_version": FORMAT_VERSION,
            "library_version": __version__,
            "model_type": MODEL_TYPE,
            "model": "EdgeAutoEncoder",
            "gnn": t.config.to_dict(),
            "in_node": t.in_node,
            "in_edge": t.in_edge,
            "alpha": t.alpha,
            "threshold": float(t.threshold),
            "score_stats": t.score_stats,
            "best_epoch": int(t.best_epoch),
            "history": t.history,
            "metrics": t.metrics,
            "stats": t.stats,
        }
        (directory / CONFIG_FILE).write_text(
            json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path, map_location: str = "cpu") -> AutoEncoderDetector:
        directory = Path(directory)
        cfg_path, weights_path = directory / CONFIG_FILE, directory / WEIGHTS_FILE
        if not cfg_path.exists() or not weights_path.exists():
            raise ModelError(f"{directory} does not contain {CONFIG_FILE} and {WEIGHTS_FILE}")
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        if (
            payload.get("format_version") != FORMAT_VERSION
            or payload.get("model_type") != MODEL_TYPE
        ):
            raise ModelError(
                f"{directory} is not an unsupervised GNN detector of format {FORMAT_VERSION}"
            )
        config = GNNConfig.from_dict(payload["gnn"])
        model = EdgeAutoEncoder.from_config(
            config, int(payload["in_node"]), int(payload["in_edge"])
        )
        model.load_state_dict(
            torch.load(weights_path, map_location=map_location, weights_only=True)
        )
        model.eval()
        return cls(
            TrainedAutoEncoder(
                model=model,
                config=config,
                stats=payload.get("stats", {}),
                in_node=int(payload["in_node"]),
                in_edge=int(payload["in_edge"]),
                alpha=float(payload.get("alpha", DEFAULT_ALPHA)),
                score_stats=payload["score_stats"],
                threshold=float(payload.get("threshold", 0.99)),
                history=payload.get("history", []),
                best_epoch=int(payload.get("best_epoch", 0)),
                metrics=payload.get("metrics", {}),
            )
        )


__all__ = [
    "MODEL_TYPE",
    "AutoEncoderDetector",
    "EdgeAutoEncoder",
    "TrainedAutoEncoder",
    "fit_autoencoder",
    "run_gae_experiment",
    "score_autoencoder_splits",
    "transaction_errors",
]
