"""Pattern-based explanations: *which laundering motif does this transaction belong to?*"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..detectors.patterns import Alert, ScanResult
from .base import EdgeImportance, Explanation, NodeImportance

PATTERN_DESCRIPTIONS: dict[str, str] = {
    "fan_out": "one account sends money to many accounts in a short time (dispersion)",
    "fan_in": "many accounts send money to one account in a short time (collection)",
    "smurfing": (
        "many transfers just below a reporting threshold (structuring), or one inflow forwarded "
        "in small parts (splitting)"
    ),
    "cycle": "money returns to its origin through a chain of accounts (round-tripping)",
    "scatter_gather": "money is split across intermediaries then re-grouped at a final account",
    "gather_scatter": "money is collected by a hub then redistributed to other accounts",
    "community": "a small group of pass-through accounts moves money mostly among themselves",
    "relay": "an account forwards almost at once nearly all it has just received (pass-through)",
}

_SCALARS = (int, float, str, bool)


def describe_alert(alert: Alert, max_items: int = 6) -> str:
    """One-sentence description of an alert from its evidence."""
    meaning = PATTERN_DESCRIPTIONS.get(alert.pattern, alert.pattern)
    facts = [f"{k}={_fmt(v)}" for k, v in alert.evidence.items() if isinstance(v, _SCALARS)][
        :max_items
    ]
    detail = f" ({', '.join(facts)})" if facts else ""
    return f"{alert.pattern}: {meaning}{detail}."


def explain_with_patterns(
    df: pd.DataFrame,
    tx_id: int,
    result: ScanResult,
    threshold: float | None = None,
) -> Explanation:
    """Explain ``tx_id`` with the structural-motif alerts that contain it.

    The importance of every transaction of an alert is the (calibrated) alert score, so the
    explanation reads "this transaction is flagged because it belongs to a fan-out of score
    0.62 involving these accounts".
    """
    alerts = [a for a in result.alerts if tx_id in set(a.tx_ids)]
    alerts.sort(key=result.weight, reverse=True)
    score = max((result.weight(a) for a in alerts), default=0.0)
    edges: dict[int, EdgeImportance] = {}
    nodes: dict[str, NodeImportance] = {}
    patterns: list[dict[str, Any]] = []
    for alert in alerts:
        weight = result.weight(alert)
        patterns.append({**alert.to_dict(), "score": weight, "description": describe_alert(alert)})
        sub = df.loc[df.index.intersection(alert.tx_ids)]
        for t, row in zip(sub.index.tolist(), sub.to_dict("records")):
            key = int(t)
            role = "target" if key == int(tx_id) else "pattern"
            previous = edges.get(key)
            if previous is None or weight > previous.importance:
                edges[key] = EdgeImportance(
                    tx_id=key,
                    sender=str(row["sender"]),
                    receiver=str(row["receiver"]),
                    amount=float(row["amount"]),
                    timestamp=str(row["timestamp"]),
                    importance=weight,
                    role=role,
                )
        for account in alert.accounts:
            nodes[account] = NodeImportance(
                account,
                max(weight, nodes.get(account, NodeImportance(account, 0.0)).importance),
                "neighbour",
            )
    if tx_id in df.index:
        explained = df.loc[[tx_id]].iloc[0]
        for account, role in (
            (str(explained["sender"]), "sender"),
            (str(explained["receiver"]), "receiver"),
        ):
            if account in nodes:
                nodes[account].role = role
    summary = (
        "No structural motif contains this transaction."
        if not alerts
        else f"Part of {len(alerts)} structural alert(s). "
        + " ".join(describe_alert(a) for a in alerts[:2])
    )
    return Explanation(
        tx_id=int(tx_id),
        method="patterns",
        score=score,
        threshold=threshold,
        edges=sorted(edges.values(), key=lambda e: (e.role != "target", -e.importance)),
        nodes=sorted(nodes.values(), key=lambda n: -n.importance),
        patterns=patterns,
        summary=summary,
    )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) >= 100 else f"{value:.3g}"
    return str(value)


__all__ = ["PATTERN_DESCRIPTIONS", "describe_alert", "explain_with_patterns"]
