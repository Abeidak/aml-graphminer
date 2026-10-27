"""Exception hierarchy used across the toolkit."""

from __future__ import annotations


class AMLGraphMinerError(Exception):
    """Base class for every error raised on purpose by the toolkit."""


class SchemaError(AMLGraphMinerError, ValueError):
    """The transaction table does not respect the unified transactional schema."""


class IngestionError(AMLGraphMinerError):
    """A file could not be read or converted to the unified schema."""


class OptionalDependencyError(AMLGraphMinerError, ImportError):
    """An optional dependency (torch, fastapi, xgboost, ...) is required but missing."""

    def __init__(self, package: str, extra: str, feature: str = "") -> None:
        what = f" for {feature}" if feature else ""
        super().__init__(
            f"'{package}' is required{what}. Install it with: pip install 'aml-graphminer[{extra}]'"
        )
        self.package = package
        self.extra = extra


class ConfigError(AMLGraphMinerError, ValueError):
    """Invalid configuration (YAML / CLI / API)."""


class ModelError(AMLGraphMinerError):
    """A model checkpoint is missing, corrupted or incompatible with the data."""


class NotFoundError(AMLGraphMinerError, LookupError):
    """A dataset, transaction, alert or account requested through an interface does not exist."""
