"""Framework-free back end of the web investigation interface (``amlgm serve`` → ``/ui``).

An *investigation* is a transaction table loaded in memory (uploaded file, file given on the
command line or synthetic demo data) that has been **analysed** once: every structural-motif
detector has been run on it and the loaded detector has scored every transaction. The methods
below answer the questions an analyst asks while triaging alerts: what is in the data, which
alerts are the strongest, why was this transaction flagged, who is this account, what does its
neighbourhood look like. Everything returned is JSON-ready; the FastAPI layer (:mod:`.app`) is a
thin wrapper, so this logic is unit-tested without a web server.
"""

from __future__ import annotations

import io
import math
import tempfile
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import __version__
from ..detectors.bundle import MotifDetector
from ..detectors.patterns import PatternScanner, ScanResult
from ..eda import eda_summary
from ..evaluation.metrics import compute_metrics
from ..exceptions import ConfigError, NotFoundError
from ..explainers import (
    PATTERN_DESCRIPTIONS,
    Explanation,
    TransactionIndex,
    available_methods,
    describe_alert,
    explain_transaction,
)
from ..ingestion import load_dataset

MAX_PAGE = 500
SORTABLE = ("score", "amount", "timestamp")


@dataclass
class Investigation:
    """One analysed dataset."""

    id: str
    name: str
    df: pd.DataFrame
    created: float
    scan: ScanResult
    scores: np.ndarray  # detector score per row of ``df``
    tx_pattern: np.ndarray  # strongest motif per row ("" when none)
    tx_motif_score: np.ndarray  # strongest (calibrated) motif score per row
    seconds: float
    index: TransactionIndex | None = None
    alerts_by_tx: dict[int, list[int]] = field(default_factory=dict)
    # where the table comes from: {"type": "demo", "seed": 3}, {"type": "file"} or
    # {"type": "table"}; lets a client name a demo in its own language
    origin: dict[str, Any] = field(default_factory=lambda: {"type": "table"})

    def transaction_index(self) -> TransactionIndex:
        if self.index is None:
            self.index = TransactionIndex(self.df)
        return self.index


def _clean(value: Any) -> Any:
    """JSON-safe scalars (NaN → None, numpy → Python, timestamps → ISO strings)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


class InvestigationService:
    """In-memory investigations around one loaded detector.

    Parameters
    ----------
    detector:
        Any detector restored by :func:`~aml_graphminer.detectors.bundle.load_detector`.
    max_datasets:
        Investigations kept in memory (the oldest is dropped beyond).
    max_transactions:
        Largest table accepted (larger files should be scored with ``amlgm score``).
    """

    def __init__(
        self,
        detector: Any,
        *,
        max_datasets: int = 5,
        max_transactions: int = 2_000_000,
    ) -> None:
        self.detector = detector
        self.max_datasets = max_datasets
        self.max_transactions = max_transactions
        self.scanner = detector.scanner if isinstance(detector, MotifDetector) else PatternScanner()
        self._items: OrderedDict[str, Investigation] = OrderedDict()

    # ------------------------------------------------------------------ datasets
    def add(
        self,
        df: pd.DataFrame,
        name: str = "dataset",
        *,
        origin: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Analyse a canonical transaction table and keep it as a new investigation."""
        if len(df) == 0:
            raise ConfigError("the table contains no transaction")
        if len(df) > self.max_transactions:
            raise ConfigError(
                f"{len(df):,} transactions exceed the limit of the interface "
                f"({self.max_transactions:,}); score large files with `amlgm score`"
            )
        if not df["timestamp"].is_monotonic_increasing:
            df = df.sort_values("timestamp", kind="stable")
        if not df.index.is_unique:
            df = df.reset_index(drop=True)
            df.index.name = "tx_id"
        start = time.perf_counter()
        scan = self.scanner.scan(df)
        motif_scores, tx_pattern = scan.transaction_scores()
        scores = self.detector.score(df).reindex(df.index).to_numpy(dtype="float64")
        item = Investigation(
            id=uuid.uuid4().hex[:12],
            name=name,
            df=df,
            created=time.time(),
            scan=scan,
            scores=scores,
            tx_pattern=np.asarray(tx_pattern, dtype=object),
            tx_motif_score=np.asarray(motif_scores, dtype="float64"),
            seconds=time.perf_counter() - start,
            origin=origin or {"type": "table"},
        )
        position = pd.Index(df.index)
        for alert_id, alert in enumerate(scan.alerts):
            for pos in position.get_indexer(alert.tx_ids):
                if pos >= 0:
                    item.alerts_by_tx.setdefault(int(pos), []).append(alert_id)
        self._items[item.id] = item
        while len(self._items) > self.max_datasets:
            self._items.popitem(last=False)
        return self.describe(item.id)

    def add_file(
        self, path: str | Path, name: str | None = None, *, schema: str = "auto"
    ) -> dict[str, Any]:
        """Load a CSV / JSON / Parquet file (IBM AML or canonical columns) and analyse it."""
        path = Path(path)
        df = load_dataset(path, schema=schema)
        return self.add(df, name or path.name, origin={"type": "file"})

    def add_upload(self, content: bytes, filename: str, *, schema: str = "auto") -> dict[str, Any]:
        """Same as :meth:`add_file` for the bytes of an uploaded file."""
        suffix = Path(filename).suffix.lower() or ".csv"
        if suffix not in (".csv", ".txt", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet", ".pq"):
            raise ConfigError(
                f"unsupported file type {suffix!r} (CSV, JSON, JSON-lines or Parquet)"
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(content)
            return self.add_file(path, Path(filename).name, schema=schema)

    def add_demo(
        self,
        *,
        accounts: int = 1500,
        background: int = 15000,
        illicit_ratio: float = 0.02,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Analyse a synthetic IBM-AML-like dataset (labels and ground-truth motifs included)."""
        from ..synthetic import generate_transactions

        if not (50 <= accounts <= 200_000 and 100 <= background <= 2_000_000):
            raise ConfigError("demo size out of range (50-200k accounts, 100-2M transactions)")
        if not 0.0 < illicit_ratio < 0.5:
            raise ConfigError("illicit_ratio must be in (0, 0.5)")
        df = generate_transactions(accounts, background, illicit_ratio=illicit_ratio, seed=seed)
        return self.add(df, f"synthetic demo (seed {seed})", origin={"type": "demo", "seed": seed})

    def datasets(self) -> list[dict[str, Any]]:
        """Summaries of the investigations in memory, most recent first."""
        return [self.describe(key) for key in reversed(self._items)]

    def get(self, dataset_id: str) -> Investigation:
        try:
            return self._items[dataset_id]
        except KeyError:
            raise NotFoundError(f"unknown dataset {dataset_id!r}") from None

    def remove(self, dataset_id: str) -> None:
        self.get(dataset_id)
        del self._items[dataset_id]

    def describe(self, dataset_id: str) -> dict[str, Any]:
        item = self.get(dataset_id)
        df = item.df
        return _clean(
            {
                "id": item.id,
                "name": item.name,
                "transactions": len(df),
                "accounts": int(pd.unique(pd.concat([df["sender"], df["receiver"]])).size),
                "first_timestamp": df["timestamp"].min(),
                "last_timestamp": df["timestamp"].max(),
                "labelled": bool(df["label"].notna().all()),
                "n_alerts": len(item.scan.alerts),
                "n_flagged": int((item.scores >= self.detector.threshold).sum()),
                "analysis_seconds": round(item.seconds, 2),
                "origin": item.origin,
            }
        )

    # ------------------------------------------------------------------ overview
    def model(self) -> dict[str, Any]:
        return _clean(
            {
                "library_version": __version__,
                "model": self.detector.describe(),
                "threshold": self.detector.threshold,
                "explain_methods": available_methods(self.detector),
                # the motifs this service looks for (relay is opt-in, for example)
                "patterns": {
                    name: PATTERN_DESCRIPTIONS.get(name, name) for name in self.scanner.detectors
                },
            }
        )

    def overview(self, dataset_id: str) -> dict[str, Any]:
        """Dashboard: dataset statistics, alerts per motif, score distribution, metrics."""
        item = self.get(dataset_id)
        df = item.df
        eda = eda_summary(df)
        threshold = self.detector.threshold
        edges = np.linspace(0.0, 1.0, 21)
        counts, _ = np.histogram(np.clip(item.scores, 0.0, 1.0), bins=edges)
        out: dict[str, Any] = {
            "dataset": self.describe(dataset_id),
            "eda": eda,
            "alerts_by_pattern": item.scan.summary()["alerts_by_pattern"],
            "score_histogram": {"edges": edges.round(3).tolist(), "counts": counts.tolist()},
            "threshold": threshold,
            "metrics": None,
            "motif_metrics": None,
        }
        y = df["label"].to_numpy(dtype="float64")
        if not np.isnan(y).any() and 0 < y.sum() < len(y):
            out["metrics"] = compute_metrics(y, item.scores, threshold)
            out["motif_metrics"] = compute_metrics(y, item.tx_motif_score, np.finfo(float).tiny)
        return _clean(out)

    # ------------------------------------------------------------------ alerts
    def alerts(
        self,
        dataset_id: str,
        *,
        pattern: str | None = None,
        min_score: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Motif alerts sorted by decreasing (calibrated) score."""
        item = self.get(dataset_id)
        scan = item.scan
        rows = [
            (i, a)
            for i, a in enumerate(scan.alerts)
            if (not pattern or a.pattern == pattern) and scan.weight(a) >= min_score
        ]
        rows.sort(key=lambda ia: -scan.weight(ia[1]))
        page = rows[offset : offset + _page(limit)]
        return _clean(
            {
                "total": len(rows),
                "items": [
                    {
                        "alert_id": i,
                        "pattern": a.pattern,
                        "score": scan.weight(a),
                        "n_tx": len(a.tx_ids),
                        "n_accounts": len(a.accounts),
                        "start": a.start,
                        "end": a.end,
                        "total_amount": a.total_amount,
                        "evidence": a.evidence,
                        "summary": describe_alert(a),
                    }
                    for i, a in page
                ],
            }
        )

    def alert(self, dataset_id: str, alert_id: int) -> dict[str, Any]:
        """One alert: evidence, transactions and the graph they form."""
        item = self.get(dataset_id)
        if not 0 <= alert_id < len(item.scan.alerts):
            raise NotFoundError(f"unknown alert {alert_id}")
        alert = item.scan.alerts[alert_id]
        frame = item.df.loc[item.df.index.intersection(alert.tx_ids)]
        payload = alert.to_dict()
        payload.update(
            {
                "alert_id": alert_id,
                "score": item.scan.weight(alert),
                "description": PATTERN_DESCRIPTIONS.get(alert.pattern, alert.pattern),
                "summary": describe_alert(alert),
                "transactions": self._rows(item, frame.index),
                "graph": self._graph(item, frame, highlight=set(alert.accounts)),
            }
        )
        return _clean(payload)

    # ------------------------------------------------------------------ transactions
    def transactions(
        self,
        dataset_id: str,
        *,
        sort: str = "score",
        descending: bool = True,
        account: str | None = None,
        only_alerts: bool = False,
        pattern: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Transactions with their model score and strongest motif (filter, sort, paginate)."""
        if sort not in SORTABLE:
            raise ConfigError(f"sort must be one of {SORTABLE}")
        item = self.get(dataset_id)
        df = item.df
        mask = np.ones(len(df), dtype=bool)
        if account:
            mask &= ((df["sender"] == account) | (df["receiver"] == account)).to_numpy()
        if only_alerts:
            mask &= item.scores >= self.detector.threshold
        if pattern:
            mask &= item.tx_pattern == pattern
        positions = np.flatnonzero(mask)
        key = {
            "score": item.scores,
            "amount": df["amount"].to_numpy(dtype="float64"),
            "timestamp": np.arange(len(df), dtype="float64"),  # the table is sorted by time
        }[sort][positions]
        order = np.argsort(-key if descending else key, kind="stable")
        page = positions[order][offset : offset + _page(limit)]
        return _clean({"total": len(positions), "items": self._rows(item, df.index[page])})

    def transaction(self, dataset_id: str, tx_id: int) -> dict[str, Any]:
        """Details of one transaction and the alerts it belongs to."""
        item = self.get(dataset_id)
        pos = self._position(item, tx_id)
        row = self._rows(item, item.df.index[[pos]])[0]
        row["alerts"] = [
            {
                "alert_id": a,
                "pattern": item.scan.alerts[a].pattern,
                "score": item.scan.weight(item.scan.alerts[a]),
                "evidence": item.scan.alerts[a].evidence,
                "summary": describe_alert(item.scan.alerts[a]),
            }
            for a in item.alerts_by_tx.get(pos, [])
        ]
        return _clean(row)

    def explain(
        self,
        dataset_id: str,
        tx_id: int,
        *,
        method: str = "auto",
        hops: int | None = None,
        max_edges: int = 60,
    ) -> dict[str, Any]:
        """Explanation of one transaction plus a graph ready to draw."""
        item = self.get(dataset_id)
        self._position(item, tx_id)
        if not 5 <= max_edges <= 300:
            raise ConfigError("max_edges must be between 5 and 300")
        if hops is not None and not 1 <= hops <= 4:
            raise ConfigError("hops must be between 1 and 4")
        explanation = explain_transaction(
            self.detector,
            item.df,
            tx_id,
            method,
            hops=hops,
            max_edges=max_edges,
            index=item.transaction_index(),
            scan=item.scan if isinstance(self.detector, MotifDetector) else None,
        )
        payload = explanation.to_dict()
        payload["graph"] = self._explanation_graph(item, explanation)
        return _clean(payload)

    # ------------------------------------------------------------------ accounts
    def account(self, dataset_id: str, account: str, *, limit: int = 100) -> dict[str, Any]:
        """Profile of an account: activity, counterparties, risk, alerts and neighbourhood."""
        item = self.get(dataset_id)
        df = item.df
        sent = (df["sender"] == account).to_numpy()
        received = (df["receiver"] == account).to_numpy()
        touched = sent | received
        if not touched.any():
            raise NotFoundError(f"unknown account {account!r}")
        positions = np.flatnonzero(touched)
        alert_ids = sorted({a for p in positions.tolist() for a in item.alerts_by_tx.get(p, [])})
        order = positions[np.argsort(-item.scores[positions], kind="stable")][: _page(limit)]
        frame = df.iloc[positions]
        times = frame["timestamp"]
        return _clean(
            {
                "account": account,
                "bank": (
                    frame.loc[frame["sender"] == account, "sender_bank"].tolist()
                    + frame.loc[frame["receiver"] == account, "receiver_bank"].tolist()
                    + [""]
                )[0],
                "n_sent": int(sent.sum()),
                "n_received": int(received.sum()),
                "amount_sent": float(df.loc[sent, "amount"].sum()),
                "amount_received": float(df.loc[received, "amount_received"].sum()),
                "counterparties_out": int(df.loc[sent, "receiver"].nunique()),
                "counterparties_in": int(df.loc[received, "sender"].nunique()),
                "first_seen": times.min(),
                "last_seen": times.max(),
                "max_score": float(item.scores[positions].max()),
                "n_flagged": int((item.scores[positions] >= self.detector.threshold).sum()),
                "alerts": [
                    {
                        "alert_id": a,
                        "pattern": item.scan.alerts[a].pattern,
                        "score": item.scan.weight(item.scan.alerts[a]),
                    }
                    for a in alert_ids
                ],
                "transactions": self._rows(item, df.index[order]),
                "graph": self._graph(item, frame.iloc[: 4 * _page(limit)], highlight={account}),
            }
        )

    # ------------------------------------------------------------------ exports
    def export_alerts_csv(self, dataset_id: str) -> str:
        item = self.get(dataset_id)
        frame = item.scan.to_frame()
        if len(frame):
            frame = frame.assign(
                accounts=frame["accounts"].map(lambda a: ";".join(a)),
                tx_ids=frame["tx_ids"].map(lambda t: ";".join(map(str, t))),
            )
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        return buffer.getvalue()

    def export_scores_csv(self, dataset_id: str) -> str:
        item = self.get(dataset_id)
        df = item.df
        table = df[
            ["timestamp", "sender", "receiver", "amount", "currency", "transaction_type"]
        ].assign(
            score=item.scores,
            alert=(item.scores >= self.detector.threshold).astype(int),
            motif=item.tx_pattern,
            motif_score=item.tx_motif_score,
        )
        if df["label"].notna().all():
            table["label"] = df["label"].astype(int)
        buffer = io.StringIO()
        table.to_csv(buffer, index_label="tx_id")
        return buffer.getvalue()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _position(item: Investigation, tx_id: int) -> int:
        pos = int(item.df.index.get_indexer(pd.Index([tx_id]))[0])
        if pos < 0:
            raise NotFoundError(f"unknown transaction {tx_id}")
        return pos

    def _rows(self, item: Investigation, ids: pd.Index) -> list[dict[str, Any]]:
        positions = item.df.index.get_indexer(ids)
        frame = item.df.iloc[positions]
        labelled = bool(item.df["label"].notna().all())
        rows = []
        for pos, (tx_id, rec) in zip(
            positions.tolist(), zip(frame.index.tolist(), frame.to_dict("records"))
        ):
            rows.append(
                {
                    "tx_id": int(tx_id),
                    "timestamp": rec["timestamp"],
                    "sender": str(rec["sender"]),
                    "receiver": str(rec["receiver"]),
                    "amount": float(rec["amount"]),
                    "currency": str(rec["currency"]),
                    "amount_received": float(rec["amount_received"]),
                    "currency_received": str(rec["currency_received"]),
                    "transaction_type": str(rec["transaction_type"]),
                    "score": float(item.scores[pos]),
                    "alert": bool(item.scores[pos] >= self.detector.threshold),
                    "motif": str(item.tx_pattern[pos]),
                    "motif_score": float(item.tx_motif_score[pos]),
                    "label": (int(rec["label"]) if labelled else None),
                    "true_pattern": str(rec.get("pattern", "")),
                }
            )
        return rows

    def _graph(
        self, item: Investigation, frame: pd.DataFrame, highlight: set[str] | None = None
    ) -> dict[str, Any]:
        """Nodes (accounts) and edges (transactions) of ``frame`` for the graph view."""
        highlight = highlight or set()
        positions = item.df.index.get_indexer(frame.index)
        scores = item.scores[positions]
        nodes: dict[str, dict[str, Any]] = {}
        edges = []
        for tx_id, rec, score in zip(
            frame.index.tolist(), frame.to_dict("records"), scores.tolist()
        ):
            for acc in (str(rec["sender"]), str(rec["receiver"])):
                node = nodes.setdefault(
                    acc, {"id": acc, "score": 0.0, "highlight": acc in highlight}
                )
                node["score"] = max(node["score"], score)
            edges.append(
                {
                    "id": int(tx_id),
                    "source": str(rec["sender"]),
                    "target": str(rec["receiver"]),
                    "amount": float(rec["amount"]),
                    "currency": str(rec["currency"]),
                    "timestamp": rec["timestamp"],
                    "score": float(score),
                    "alert": bool(score >= self.detector.threshold),
                }
            )
        return {"nodes": list(nodes.values()), "edges": edges}

    def _explanation_graph(self, item: Investigation, explanation: Explanation) -> dict[str, Any]:
        positions = item.df.index.get_indexer(pd.Index([e.tx_id for e in explanation.edges]))
        node_info = {n.account: n for n in explanation.nodes}
        nodes: dict[str, dict[str, Any]] = {}
        edges = []
        for e, pos in zip(explanation.edges, positions.tolist()):
            score = float(item.scores[pos]) if pos >= 0 else None
            for acc in (e.sender, e.receiver):
                if acc not in nodes:
                    info = node_info.get(acc)
                    nodes[acc] = {
                        "id": acc,
                        "importance": 0.0 if info is None else info.importance,
                        "role": "neighbour" if info is None else info.role,
                        "highlight": info is not None and info.role in ("sender", "receiver"),
                    }
            edges.append(
                {
                    "id": e.tx_id,
                    "source": e.sender,
                    "target": e.receiver,
                    "amount": e.amount,
                    "timestamp": e.timestamp,
                    "importance": e.importance,
                    "role": e.role,
                    "score": score,
                    "alert": score is not None and score >= self.detector.threshold,
                }
            )
        return {"nodes": list(nodes.values()), "edges": edges}


def _page(limit: int) -> int:
    if limit < 1:
        raise ConfigError("limit must be >= 1")
    return min(int(limit), MAX_PAGE)


__all__ = ["MAX_PAGE", "Investigation", "InvestigationService"]
