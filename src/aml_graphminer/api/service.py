"""Framework-free scoring service (the REST layer is a thin wrapper around it).

Keeping the logic here means it is fully unit-tested without a web framework.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .. import __version__
from ..detectors.bundle import MotifDetector, load_detector
from ..exceptions import AMLGraphMinerError, ConfigError
from ..explainers import available_methods, explain_transaction
from ..ingestion import read_transactions

Record = Mapping[str, Any]

MAX_TRANSACTIONS = 50_000


class ScoringService:
    """Load one detector and answer scoring / explanation requests on lists of records.

    Records may use the canonical column names (``timestamp, sender, receiver, amount ...``)
    or the IBM AML column names.
    """

    def __init__(self, detector: Any, model_dir: str | Path | None = None) -> None:
        self.detector = detector
        self.model_dir = None if model_dir is None else str(model_dir)

    @classmethod
    def from_dir(cls, directory: str | Path) -> ScoringService:
        return cls(load_detector(directory), directory)

    # ------------------------------------------------------------------ helpers
    def _frame(self, records: Sequence[Record], what: str = "transactions") -> pd.DataFrame:
        if not records:
            raise ConfigError(f"no {what} provided")
        if len(records) > MAX_TRANSACTIONS:
            raise ConfigError(f"too many {what}: {len(records)} > {MAX_TRANSACTIONS}")
        return read_transactions([dict(r) for r in records], strict=True)

    # ------------------------------------------------------------------ endpoints
    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "library_version": __version__,
            "model": self.detector.describe(),
            "explain_methods": available_methods(self.detector),
        }

    def score(
        self,
        records: Sequence[Record],
        context: Sequence[Record] | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Score ``records`` (optionally in the context of earlier ``context`` records)."""
        df = self._frame(records)
        ctx = self._frame(context, "context transactions") if context else None
        thr = self.detector.threshold if threshold is None else float(threshold)
        scores = self.detector.score(df, ctx).reindex(df.index)
        items = [
            {"row": int(row), "tx_id": int(tx_id), "score": float(s), "alert": bool(s >= thr)}
            for row, tx_id, s in zip(
                df["source_row"].tolist(), df.index.tolist(), scores.to_numpy(dtype="float64")
            )
        ]
        items.sort(key=lambda item: item["row"])
        return {
            "threshold": thr,
            "n_transactions": len(items),
            "n_alerts": sum(item["alert"] for item in items),
            "results": items,
        }

    def explain(
        self,
        records: Sequence[Record],
        row: int,
        method: str = "auto",
        hops: int | None = None,
        max_edges: int = 80,
    ) -> dict[str, Any]:
        """Explain the score of the ``row``-th posted transaction."""
        df = self._frame(records)
        hits = df.index[df["source_row"] == row]
        if len(hits) == 0:
            raise ConfigError(f"row {row} is outside the {len(df)} posted transactions")
        explanation = explain_transaction(
            self.detector, df, int(hits[0]), method, hops=hops, max_edges=max_edges
        )
        return explanation.to_dict()

    def scan(self, records: Sequence[Record], top: int = 20) -> dict[str, Any]:
        """Structural-motif alerts of a batch (available for every detector type)."""
        from ..detectors.patterns import PatternScanner

        df = self._frame(records)
        scanner = (
            self.detector.scanner if isinstance(self.detector, MotifDetector) else PatternScanner()
        )
        result = scanner.scan(df)
        frame = result.to_frame().head(top)
        alerts = [result.alerts[int(i)].to_dict() for i in frame["alert_id"]] if len(frame) else []
        return {"n_alerts": len(result), "alerts": alerts}


def error_payload(exc: Exception) -> dict[str, str]:
    """JSON body used by the API for library errors."""
    kind = "invalid_request" if isinstance(exc, AMLGraphMinerError) else "internal_error"
    return {"error": kind, "detail": str(exc)}


__all__ = ["MAX_TRANSACTIONS", "ScoringService", "error_payload"]
