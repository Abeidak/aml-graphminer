"""Save / load trained GNNs: weights + JSON configuration (feature statistics, threshold)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ... import __version__
from ...exceptions import ModelError
from .config import GNNConfig
from .models import EdgeGNN
from .trainer import TrainedGNN

FORMAT_VERSION = 1
WEIGHTS_FILE = "model.pt"
CONFIG_FILE = "config.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def save_checkpoint(trained: TrainedGNN, directory: str | Path) -> Path:
    """Write ``model.pt`` (state dict) and ``config.json`` into ``directory``.

    ``config.json`` is the single source of truth to rebuild the model and to transform new
    data exactly like the training data (feature statistics) and to apply the decision
    ``threshold``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {k: v.cpu() for k, v in trained.model.state_dict().items()}, directory / WEIGHTS_FILE
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "library_version": __version__,
        "model": "EdgeGNN",
        "model_type": "gnn",
        "gnn": trained.config.to_dict(),
        "in_node": trained.in_node,
        "in_edge": trained.in_edge,
        "threshold": float(trained.threshold),
        "best_epoch": int(trained.best_epoch),
        "history": trained.history,
        "metrics": trained.metrics,
        "stats": trained.stats,
    }
    (directory / CONFIG_FILE).write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=True), encoding="utf-8"
    )
    return directory


def load_checkpoint(directory: str | Path, map_location: str = "cpu") -> TrainedGNN:
    """Inverse of :func:`save_checkpoint`."""
    directory = Path(directory)
    cfg_path, weights_path = directory / CONFIG_FILE, directory / WEIGHTS_FILE
    if not cfg_path.exists() or not weights_path.exists():
        raise ModelError(f"{directory} does not contain {CONFIG_FILE} and {WEIGHTS_FILE}")
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != FORMAT_VERSION:
        raise ModelError(
            f"unsupported checkpoint format {payload.get('format_version')!r} "
            f"(this release reads version {FORMAT_VERSION})"
        )
    config = GNNConfig.from_dict(payload["gnn"])
    model = EdgeGNN.from_config(config, int(payload["in_node"]), int(payload["in_edge"]))
    state = torch.load(weights_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return TrainedGNN(
        model=model,
        config=config,
        stats=payload.get("stats", {}),
        in_node=int(payload["in_node"]),
        in_edge=int(payload["in_edge"]),
        threshold=float(payload.get("threshold", 0.5)),
        history=payload.get("history", []),
        best_epoch=int(payload.get("best_epoch", 0)),
        metrics=payload.get("metrics", {}),
    )


__all__ = ["CONFIG_FILE", "FORMAT_VERSION", "WEIGHTS_FILE", "load_checkpoint", "save_checkpoint"]
