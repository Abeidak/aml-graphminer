"""REST API (FastAPI) and web investigation interface around the detectors — ``amlgm serve``."""

from __future__ import annotations

from typing import Any

from .investigation import InvestigationService
from .service import ScoringService

__all__ = ["InvestigationService", "ScoringService", "create_app"]


def create_app(model: object, **options: Any) -> object:
    """Create the FastAPI app (imports FastAPI lazily; needs the ``api`` extra).

    ``options`` are forwarded to :func:`aml_graphminer.api.app.create_app` (``datasets``,
    ``demo``, ``max_upload_mb``).
    """
    from .app import create_app as _create

    return _create(model, **options)  # type: ignore[arg-type]
