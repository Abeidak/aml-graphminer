"""Project-level YAML configuration (features, patterns, GNN, benchmark, tracking)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .detectors.patterns.config import PatternConfig
from .exceptions import ConfigError
from .features.pipeline import FeatureConfig
from .tracking import MLflowTracker, Tracker


class BenchmarkSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[str] = Field(default_factory=lambda: ["patterns", "rf", "hgb"])
    seeds: list[int] = Field(default_factory=lambda: [0])
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    tabular_node_features: bool = True


class TrackingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "none"  # "none" | "mlflow"
    experiment: str = "aml-graphminer"
    uri: str | None = None


class AppConfig(BaseModel):
    """Everything a run needs, loadable from one YAML file (see ``configs/default.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(default_factory=dict)
    patterns: dict[str, Any] = Field(default_factory=dict)
    gnn: dict[str, Any] = Field(default_factory=dict)
    benchmark: BenchmarkSection = Field(default_factory=BenchmarkSection)
    tracking: TrackingSection = Field(default_factory=TrackingSection)

    # ------------------------------------------------------------------ typed accessors
    def feature_config(self) -> FeatureConfig:
        return FeatureConfig.from_dict(self.features)

    def pattern_config(self) -> PatternConfig:
        return PatternConfig.from_dict(self.patterns)

    def gnn_config(self) -> Any:
        from .detectors.gnn.config import GNNConfig

        return GNNConfig.from_dict(self.gnn)

    def tracker(self, run_name: str | None = None) -> Tracker:
        """No-op tracker, or an MLflow run when ``tracking.backend == "mlflow"``."""
        if self.tracking.backend == "none":
            return Tracker()
        if self.tracking.backend == "mlflow":
            return MLflowTracker(self.tracking.experiment, run_name, self.tracking.uri)
        raise ConfigError(f"unknown tracking backend {self.tracking.backend!r}")

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AppConfig:
        data = dict(data or {})
        # a file that only holds detector sections (configs/patterns.yaml) is a `patterns` section
        if data and set(data) <= set(PatternConfig.model_fields):
            data = {"patterns": data}
        try:
            config = cls.model_validate(data)
        except Exception as exc:
            raise ConfigError(f"invalid configuration: {exc}") from exc
        # validate nested sections eagerly so mistakes surface at load time
        config.feature_config()
        config.pattern_config()
        if config.gnn:
            config.gnn_config()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"configuration file not found: {path}")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{path} must contain a mapping at the top level")
        return cls.from_dict(data)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load ``path`` (YAML) or return the defaults."""
    return AppConfig.from_yaml(path) if path else AppConfig()


def default_config_yaml() -> str:
    """Fully-populated default configuration as YAML text (``amlgm init-config``)."""
    from .detectors.gnn.config import GNNConfig

    payload = {
        "features": FeatureConfig().to_dict(),
        "patterns": PatternConfig().model_dump(),
        "gnn": GNNConfig().to_dict(),
        "benchmark": BenchmarkSection().model_dump(),
        "tracking": TrackingSection().model_dump(),
    }
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


__all__ = ["AppConfig", "BenchmarkSection", "TrackingSection", "default_config_yaml", "load_config"]
