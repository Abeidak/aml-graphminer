"""Experiment tracking: a no-op default and an optional MLflow backend."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

from .exceptions import ConfigError
from .utils import require

# MLflow >= 3 refuses the plain-directory store (./mlruns) unless explicitly allowed; a SQLite
# database is the backend it recommends and needs no server (`mlflow ui --backend-store-uri ...`).
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def is_file_store(uri: str) -> bool:
    """True for ``file:`` URIs and bare paths (``./mlruns``, ``C:\\runs``), i.e. directory stores."""
    scheme = urlparse(uri).scheme
    return scheme in ("", "file") or len(scheme) == 1  # one letter = Windows drive ("C:\\...")


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    """Explicit URI, else ``$MLFLOW_TRACKING_URI``, else :data:`DEFAULT_TRACKING_URI`."""
    return tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


class Tracker:
    """No-op tracker (the default) — also the interface the real backends implement."""

    def __enter__(self) -> Tracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.end(failed=exc is not None)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record hyper-parameters / configuration."""

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Record numeric metrics (optionally at a training step)."""

    def log_artifact(self, path: str | Path) -> None:
        """Attach a file (model checkpoint, results table, figure)."""

    def end(self, failed: bool = False) -> None:
        """Close the run."""


class MLflowTracker(Tracker):  # pragma: no cover - requires mlflow
    """MLflow backend (``pip install 'aml-graphminer[tracking]'``).

    Without ``tracking_uri`` (and without ``$MLFLOW_TRACKING_URI``) runs go to the SQLite
    database ``mlflow.db`` of the working directory; browse them with
    ``mlflow ui --backend-store-uri sqlite:///mlflow.db``. A directory store (``file:...`` or a
    path) is still accepted: it is explicitly allowed for MLflow versions that deprecate it.
    """

    def __init__(
        self,
        experiment: str = "aml-graphminer",
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        self._mlflow = require("mlflow", "tracking", "MLflow experiment tracking")
        uri = resolve_tracking_uri(tracking_uri)
        if is_file_store(uri):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        self.tracking_uri = uri
        self._mlflow.set_tracking_uri(uri)
        self._mlflow.set_experiment(experiment)
        self._run = self._mlflow.start_run(run_name=run_name)

    def log_params(self, params: Mapping[str, Any]) -> None:
        flat = {str(k): str(v)[:250] for k, v in params.items()}
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v == v}  # drop NaN
        self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path) -> None:
        self._mlflow.log_artifact(str(path))

    def end(self, failed: bool = False) -> None:
        self._mlflow.end_run(status="FAILED" if failed else "FINISHED")


def get_tracker(backend: str | None = None, **kwargs: Any) -> Tracker:
    """``None`` / ``"none"`` → no-op tracker, ``"mlflow"`` → :class:`MLflowTracker`."""
    if backend in (None, "", "none"):
        return Tracker()
    if backend == "mlflow":
        return MLflowTracker(**kwargs)
    raise ConfigError(f"unknown tracking backend {backend!r} (use 'none' or 'mlflow')")
