"""Hyper-parameters of the GNN detectors (no ``torch`` import: usable everywhere)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from ...exceptions import ConfigError

GNN_KINDS: tuple[str, ...] = ("gcn", "sage", "gat")
LOSSES: tuple[str, ...] = ("bce", "focal")
TRAINERS: tuple[str, ...] = ("torch", "lightning")
INFERENCE_MODES: tuple[str, ...] = ("sampled", "full")


@dataclass
class GNNConfig:
    """Configuration of an edge-classification GNN (defaults = the original Colab notebooks).

    Attributes
    ----------
    kind:
        Message-passing layer: ``"gcn"``, ``"sage"`` or ``"gat"``.
    hidden, layers, dropout:
        Width / depth / dropout of the node encoder (``layers`` message-passing rounds).
    gat_heads:
        Attention heads of the GAT layers (``hidden`` is split across the heads).
    num_neighbors:
        Neighbours sampled per hop by ``LinkNeighborLoader`` (one entry per layer).
    batch_size, epochs, lr, weight_decay:
        Optimisation settings.
    loss:
        ``"bce"`` (weighted binary cross-entropy, ``pos_weight``) or ``"focal"``.
    pos_weight:
        Weight of the laundering class in the BCE (≈ class imbalance; 30 in the notebooks).
    focal_gamma, focal_alpha:
        Focal-loss parameters.
    patience:
        Early stopping on the validation PR-AUC (``0`` → disabled, keep the best epoch only).
    trainer:
        ``"torch"`` (plain PyTorch loop) or ``"lightning"`` (PyTorch Lightning).
    num_workers:
        Workers of the neighbour loaders.
    device:
        ``"auto"`` (cuda if available), ``"cpu"`` or ``"cuda"``.
    inference:
        How validation / test / new transactions are scored. ``"sampled"`` (default) uses the
        same neighbour sampling as training with a **fixed seed**, so scoring the same data twice
        gives the same scores; it scales to the largest IBM files. ``"full"`` aggregates over
        *all* neighbours in one message-passing pass (exact, independent of how transactions
        are batched) and needs memory proportional to the number of edges of the graph.
    """

    kind: str = "sage"
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    gat_heads: int = 4
    num_neighbors: list[int] = field(default_factory=lambda: [10, 5])
    batch_size: int = 4096
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "bce"
    pos_weight: float = 30.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    patience: int = 0
    trainer: str = "torch"
    num_workers: int = 0
    device: str = "auto"
    inference: str = "sampled"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise :class:`~aml_graphminer.exceptions.ConfigError` on inconsistent settings."""
        if self.kind not in GNN_KINDS:
            raise ConfigError(f"unknown GNN kind {self.kind!r}; choose among {GNN_KINDS}")
        if self.loss not in LOSSES:
            raise ConfigError(f"unknown loss {self.loss!r}; choose among {LOSSES}")
        if self.trainer not in TRAINERS:
            raise ConfigError(f"unknown trainer {self.trainer!r}; choose among {TRAINERS}")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ConfigError("device must be 'auto', 'cpu' or 'cuda'")
        if self.inference not in INFERENCE_MODES:
            raise ConfigError(
                f"unknown inference mode {self.inference!r}; choose among {INFERENCE_MODES}"
            )
        if self.layers < 1 or self.hidden < 1:
            raise ConfigError("layers and hidden must be >= 1")
        if len(self.num_neighbors) != self.layers:
            raise ConfigError(
                f"num_neighbors has {len(self.num_neighbors)} entries but layers={self.layers}"
            )
        if any(k < 1 for k in self.num_neighbors):
            raise ConfigError("num_neighbors entries must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("dropout must be in [0, 1)")
        if self.kind == "gat" and (self.gat_heads < 1 or self.hidden % self.gat_heads != 0):
            raise ConfigError("for GAT, hidden must be divisible by gat_heads")
        if self.pos_weight <= 0 or self.lr <= 0 or self.batch_size < 1 or self.epochs < 1:
            raise ConfigError("pos_weight, lr, batch_size and epochs must be positive")
        if self.patience < 0:
            raise ConfigError("patience must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GNNConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown GNNConfig field(s): {sorted(unknown)}")
        return cls(**data)

    def with_kind(self, kind: str) -> GNNConfig:
        """Copy of this configuration with another message-passing layer."""
        data = self.to_dict()
        data["kind"] = kind
        return GNNConfig.from_dict(data)


__all__ = ["GNN_KINDS", "INFERENCE_MODES", "LOSSES", "TRAINERS", "GNNConfig"]
