"""PyTorch-Lightning flavour of the edge-classifier training (CDC §Layer 4: Lightning).

Same model, loss and model-selection rule as :mod:`.trainer`; Lightning provides the
training loop, checkpointing (best validation PR-AUC), early stopping and logging.
"""

from __future__ import annotations

import importlib
import tempfile
import warnings
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor

from ...exceptions import OptionalDependencyError
from ...features.pipeline import FeatureSet
from ...tracking import Tracker
from ...utils import has_module, logger, set_seed
from .config import GNNConfig
from .data import build_loaders, resolve_device
from .losses import build_criterion
from .models import EdgeGNN
from .trainer import TrainedGNN


def _lightning() -> Any:
    """Import ``lightning.pytorch`` (new name) or ``pytorch_lightning`` (legacy name)."""
    for name in ("lightning.pytorch", "pytorch_lightning"):
        if has_module(name):
            return importlib.import_module(name)
    raise OptionalDependencyError("lightning", "gnn", "the Lightning trainer")


def make_lightning_module(
    pl: Any,
    model: EdgeGNN,
    config: GNNConfig,
    edge_feat: dict[str, Tensor],
    labels: Tensor,
    y_numpy: np.ndarray,
) -> Any:
    """Build a ``LightningModule`` around ``model`` (defined here to import ``pl`` lazily)."""

    class EdgeGNNLightning(pl.LightningModule):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.criterion = build_criterion(config)
            self._val_ids: list[Tensor] = []
            self._val_probs: list[Tensor] = []

        def forward(self, batch: Any, split: str = "train") -> Tensor:
            ids = batch.edge_label
            feats = edge_feat[split].to(self.device)[ids]
            return self.model(batch.x, batch.edge_index, batch.edge_label_index, feats)

        def training_step(self, batch: Any, batch_idx: int) -> Tensor:
            logits = self(batch, "train")
            targets = labels.to(self.device)[batch.edge_label]
            loss = self.criterion(logits, targets)
            self.log("train_loss", loss, prog_bar=True, batch_size=len(targets))
            return loss

        def validation_step(self, batch: Any, batch_idx: int) -> None:
            self._val_ids.append(batch.edge_label.detach().cpu())
            self._val_probs.append(torch.sigmoid(self(batch, "val")).detach().cpu())

        def on_validation_epoch_end(self) -> None:
            if not self._val_ids:
                return
            ids = torch.cat(self._val_ids).numpy()
            probs = torch.cat(self._val_probs).numpy()
            self._val_ids.clear()
            self._val_probs.clear()
            truth = y_numpy[ids]
            ap = float(average_precision_score(truth, probs)) if truth.sum() > 0 else 0.0
            self.log("val_pr_auc", ap, prog_bar=True)

        def configure_optimizers(self) -> Any:
            return torch.optim.Adam(
                self.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )

    return EdgeGNNLightning()


def fit_edge_gnn_lightning(
    features: dict[str, FeatureSet],
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    config: GNNConfig,
    *,
    seed: int = 0,
    pipeline_stats: dict[str, Any] | None = None,
    tracker: Tracker | None = None,
) -> TrainedGNN:
    """Train with PyTorch Lightning and return the best-PR-AUC checkpoint as a :class:`TrainedGNN`."""
    pl = _lightning()
    tracker = tracker or Tracker()
    set_seed(seed)
    pl.seed_everything(seed, workers=True)
    device = resolve_device(config.device)
    loaders = build_loaders(features, masks, config, ("train", "val"))
    in_node = int(features["train"].node.shape[1])
    in_edge = int(features["train"].edge.shape[1])
    model = EdgeGNN.from_config(config, in_node, in_edge)
    edge_feat = {
        name: torch.from_numpy(np.ascontiguousarray(features[name].edge))
        for name in ("train", "val")
    }
    labels = torch.from_numpy(np.nan_to_num(y.astype("float32")))
    module = make_lightning_module(pl, model, config, edge_feat, labels, np.nan_to_num(y))

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=tmp, monitor="val_pr_auc", mode="max", save_top_k=1
        )
        callbacks: list[Any] = [checkpoint]
        if config.patience:
            callbacks.append(
                pl.callbacks.EarlyStopping(
                    monitor="val_pr_auc", mode="max", patience=config.patience
                )
            )
        trainer = pl.Trainer(
            max_epochs=config.epochs,
            accelerator="gpu" if device.type == "cuda" else "cpu",
            devices=1,
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        with warnings.catch_warnings():
            # the neighbour loaders deliberately use config.num_workers (0 by default)
            warnings.filterwarnings("ignore", message=".*does not have many workers.*")
            warnings.filterwarnings("ignore", message=".*treespec.*")
            trainer.fit(module, loaders["train"], loaders["val"])
        best_path = checkpoint.best_model_path
        if best_path:
            state = torch.load(best_path, map_location="cpu", weights_only=False)["state_dict"]
            prefix = "model."
            module.model.load_state_dict(
                {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
            )
        best_score = checkpoint.best_model_score
        history = [{"epoch": float(trainer.current_epoch), "val_pr_auc": float(best_score or 0.0)}]
    logger.info("[%s] lightning best val PR-AUC %.4f", config.kind, history[-1]["val_pr_auc"])
    tracker.log_metrics({f"{config.kind}/val_pr_auc": history[-1]["val_pr_auc"]})
    module.model.eval()
    return TrainedGNN(
        model=module.model.cpu(),
        config=config,
        stats=pipeline_stats or {},
        in_node=in_node,
        in_edge=in_edge,
        history=history,
        best_epoch=int(trainer.current_epoch),
    )


__all__ = ["fit_edge_gnn_lightning", "make_lightning_module"]
