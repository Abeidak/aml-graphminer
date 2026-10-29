"""Explanation data model (Layer 5): auditable, JSON-serialisable, networkx-convertible."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx


@dataclass
class EdgeImportance:
    """One transaction of the explanation with its contribution to the alert."""

    tx_id: int
    sender: str
    receiver: str
    amount: float
    timestamp: str
    importance: float
    role: str = "neighbour"  # "target" | "neighbour" | "pattern"


@dataclass
class NodeImportance:
    """One account of the explanation with its contribution to the alert."""

    account: str
    importance: float
    role: str = "neighbour"  # "sender" | "receiver" | "neighbour"


@dataclass
class FeatureImportance:
    """One raw attribute of the explained transaction (``value``) and its contribution."""

    name: str
    value: float | str
    importance: float


@dataclass
class Explanation:
    """Why a transaction was (or was not) flagged.

    ``importance`` values are method specific but always *signed in the direction of the
    alert*: positive → pushes the score up, negative → pushes it down. ``details`` holds the
    method-specific figures quoted by ``summary`` that no other field carries (number of
    neighbours, measured neighbourhood effect ...), so a client can word the summary itself —
    the French web interface does.
    """

    tx_id: int
    method: str
    score: float | None = None
    threshold: float | None = None
    edges: list[EdgeImportance] = field(default_factory=list)
    nodes: list[NodeImportance] = field(default_factory=list)
    features: list[FeatureImportance] = field(default_factory=list)
    patterns: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "tx_id": int(self.tx_id),
            "method": self.method,
            "score": None if self.score is None else round(float(self.score), 6),
            "threshold": None if self.threshold is None else round(float(self.threshold), 6),
            "summary": self.summary,
            "edges": [_round_fields(asdict(e)) for e in self.edges],
            "nodes": [_round_fields(asdict(n)) for n in self.nodes],
            "features": [_round_fields(asdict(f)) for f in self.features],
            "patterns": self.patterns,
            "details": _round_fields(self.details),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Explanation:
        return cls(
            tx_id=int(data["tx_id"]),
            method=str(data["method"]),
            score=data.get("score"),
            threshold=data.get("threshold"),
            edges=[EdgeImportance(**e) for e in data.get("edges", [])],
            nodes=[NodeImportance(**n) for n in data.get("nodes", [])],
            features=[FeatureImportance(**f) for f in data.get("features", [])],
            patterns=list(data.get("patterns", [])),
            summary=str(data.get("summary", "")),
            details=dict(data.get("details", {})),
        )

    # ------------------------------------------------------------------ views
    def top_edges(self, k: int = 5) -> list[EdgeImportance]:
        """The ``k`` neighbouring transactions with the largest absolute importance."""
        rest = [e for e in self.edges if e.role != "target"]
        return sorted(rest, key=lambda e: abs(e.importance), reverse=True)[:k]

    def to_networkx(self) -> nx.MultiDiGraph:
        """Explanation sub-graph: nodes = accounts, edges = transactions (with importance)."""
        graph = nx.MultiDiGraph(tx_id=int(self.tx_id), method=self.method, score=self.score)
        weights = {n.account: n for n in self.nodes}
        for e in self.edges:
            for account in (e.sender, e.receiver):
                if account not in graph:
                    node = weights.get(account)
                    graph.add_node(
                        account,
                        importance=0.0 if node is None else node.importance,
                        role="neighbour" if node is None else node.role,
                    )
            graph.add_edge(
                e.sender,
                e.receiver,
                key=int(e.tx_id),
                tx_id=int(e.tx_id),
                amount=e.amount,
                timestamp=e.timestamp,
                importance=e.importance,
                role=e.role,
            )
        return graph

    def to_markdown(self, top: int = 5) -> str:
        """Short human-readable report (used by the CLI and the API)."""
        lines = [f"### Transaction {self.tx_id} — {self.method}"]
        if self.score is not None:
            thr = "" if self.threshold is None else f" (threshold {self.threshold:.3g})"
            lines.append(f"- score: **{self.score:.3f}**{thr}")
        if self.summary:
            lines.append(f"- {self.summary}")
        for p in self.patterns:
            lines.append(f"- pattern `{p.get('pattern')}` (alert score {p.get('score', 0):.2f})")
        edges = self.top_edges(top)
        if edges:
            lines += ["", "| tx | sender → receiver | amount | importance |", "|---|---|---:|---:|"]
            lines += [
                f"| {e.tx_id} | {e.sender} → {e.receiver} | {e.amount:,.2f} | {e.importance:+.3f} |"
                for e in edges
            ]
        features = sorted(self.features, key=lambda f: abs(f.importance), reverse=True)[:top]
        if features:
            lines += ["", "| attribute | value | importance |", "|---|---|---:|"]
            lines += [
                f"| {f.name} | {f.value:,.2f} | {f.importance:+.3f} |"
                if isinstance(f.value, float)
                else f"| {f.name} | {f.value} | {f.importance:+.3f} |"
                for f in features
            ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # one line: notebooks store it next to the HTML view
        score = "" if self.score is None else f", score {self.score:.3f}"
        return (
            f"Explanation(tx {self.tx_id}, {self.method}{score}, {len(self.edges)} transactions, "
            f"{len(self.features)} attributes, {len(self.patterns)} motifs)"
        )

    def _repr_html_(self) -> str:  # notebooks: verdict, graph and what matters most
        from ..jupyter.display import explanation_html

        return explanation_html(self)


def _round_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in data.items()}


__all__ = ["EdgeImportance", "Explanation", "FeatureImportance", "NodeImportance"]
