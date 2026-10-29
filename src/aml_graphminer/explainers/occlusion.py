"""Model-agnostic occlusion explainer: *how much does the score drop without this context?*

For a transaction ``t`` the explainer removes, one at a time, each neighbouring transaction
(and each neighbouring account with all its transactions) from the context and measures the
change of the score of ``t``. Works with any detector exposed as a ``score_fn`` (GNN, tabular
baseline, structural-motif scanner), needs no gradients and gives faithful, easy-to-audit
explanations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import EdgeImportance, Explanation, FeatureImportance, NodeImportance
from .scorers import ScoreFn
from .subgraph import TransactionIndex

_NUMERIC_ATTRIBUTES = ("amount", "amount_received")
ATTRIBUTE_COLUMNS = (
    *_NUMERIC_ATTRIBUTES,
    "transaction_type",
    "currency",
    "currency_received",
    "sender_bank",
    "receiver_bank",
)


class OcclusionExplainer:
    """Explain a score by removing parts of the transaction's neighbourhood.

    Parameters
    ----------
    score_fn:
        ``score_fn(df, context) -> scores`` (see :mod:`aml_graphminer.explainers.scorers`).
    hops, max_edges:
        Size of the explained neighbourhood (account hops / maximum number of transactions).
    max_occlusions:
        Maximum number of transactions (and of accounts) that are occluded one by one; the
        ones closest in time to the explained transaction are tried first.
    threshold:
        Decision threshold stored in the explanation (informational).

    Notes
    -----
    The explanation is *local*: the detector only sees the transaction's neighbourhood.
    ``hops`` must therefore be large enough to contain the structure the detector reacts to
    (a GNN with ``L`` layers has a receptive field of ``L`` hops; a laundering cycle of length
    ``k`` needs about ``k / 2`` hops). With the structural-motif scanner prefer
    :func:`~aml_graphminer.explainers.explain_with_patterns`, which is exact and cheap.
    """

    method = "occlusion"

    def __init__(
        self,
        score_fn: ScoreFn,
        *,
        hops: int = 2,
        max_edges: int = 60,
        max_occlusions: int = 40,
        threshold: float | None = None,
    ) -> None:
        self.score_fn = score_fn
        self.hops = hops
        self.max_edges = max_edges
        self.max_occlusions = max_occlusions
        self.threshold = threshold

    def explain(
        self, df: pd.DataFrame, tx_id: int, index: TransactionIndex | None = None
    ) -> Explanation:
        """Explain the score of transaction ``tx_id`` of ``df``."""
        index = index or TransactionIndex(df)
        hood = index.neighbourhood(tx_id, self.hops, self.max_edges)
        target = hood.loc[[tx_id]]
        context = hood.drop(index=tx_id)
        base = float(self.score_fn(target, context if len(context) else None)[0])

        sender, receiver = str(target["sender"].iloc[0]), str(target["receiver"].iloc[0])
        t0 = target["timestamp"].iloc[0]
        order = np.argsort(
            np.abs((context["timestamp"] - t0).dt.total_seconds().to_numpy()), kind="stable"
        )
        ranked = context.iloc[order]

        edges = [_edge(tx_id, target.iloc[0], base, "target")]
        for j in ranked.index[: self.max_occlusions]:
            rest = context.drop(index=j)
            without = float(self.score_fn(target, rest if len(rest) else None)[0])
            edges.append(_edge(int(j), ranked.loc[j], base - without, "neighbour"))

        nodes = self._node_importance(target, context, base, sender, receiver)
        features = self._attribute_importance(target, context, base)
        explanation = Explanation(
            tx_id=int(tx_id),
            method=self.method,
            score=base,
            threshold=self.threshold,
            edges=edges,
            nodes=nodes,
            features=features,
            details={"neighbours": len(context)},
        )
        explanation.summary = _summary(explanation, len(context))
        return explanation

    def _attribute_importance(
        self, target: pd.DataFrame, context: pd.DataFrame, base: float
    ) -> list[FeatureImportance]:
        """Sensitivity of the score to each *raw attribute* of the explained transaction.

        Every attribute is replaced in turn by the typical value of the neighbourhood (median
        amount, most frequent type / currency / bank) and the score is recomputed: the
        importance is ``score - score_with_typical_value``, positive when the attribute pushes
        the score up. Needs neighbours to define "typical", so it is empty for isolated
        transactions. The perturbation keeps the other columns unchanged, hence derived features
        such as the paid / received ratio change too — read it as an attribute-level sensitivity.
        """
        if len(context) == 0:
            return []
        row = target.iloc[0]
        out: list[FeatureImportance] = []
        for column in ATTRIBUTE_COLUMNS:
            if column not in target.columns or column not in context.columns:
                continue
            values = context[column].dropna()
            if values.empty or pd.isna(row[column]):
                continue
            if column in _NUMERIC_ATTRIBUTES:
                typical: Any = float(values.astype("float64").median())
                shown: float | str = float(row[column])
            else:
                typical = values.mode().iloc[0]
                shown = str(row[column])
            modified = target.copy()
            modified[column] = typical
            without = float(self.score_fn(modified, context)[0])
            out.append(FeatureImportance(column, shown, base - without))
        return sorted(out, key=lambda f: abs(f.importance), reverse=True)

    def _node_importance(
        self,
        target: pd.DataFrame,
        context: pd.DataFrame,
        base: float,
        sender: str,
        receiver: str,
    ) -> list[NodeImportance]:
        accounts: dict[str, str] = {sender: "sender", receiver: "receiver"}
        if len(context):
            touched = pd.concat([context["sender"], context["receiver"]]).astype(str).value_counts()
            for acc in touched.index:
                accounts.setdefault(str(acc), "neighbour")
        out: list[NodeImportance] = []
        for acc, role in list(accounts.items())[: self.max_occlusions + 2]:
            if len(context):
                mask = (context["sender"].astype(str) == acc) | (
                    context["receiver"].astype(str) == acc
                )
                rest = context[~mask]
            else:
                rest = context
            without = float(self.score_fn(target, rest if len(rest) else None)[0])
            out.append(NodeImportance(account=acc, importance=base - without, role=role))
        return sorted(out, key=lambda n: abs(n.importance), reverse=True)


def _edge(tx_id: int, row: pd.Series, importance: float, role: str) -> EdgeImportance:
    return EdgeImportance(
        tx_id=int(tx_id),
        sender=str(row["sender"]),
        receiver=str(row["receiver"]),
        amount=float(row["amount"]),
        timestamp=str(row["timestamp"]),
        importance=float(importance),
        role=role,
    )


def _summary(explanation: Explanation, n_context: int) -> str:
    top = explanation.top_edges(1)
    base = f"Score {explanation.score:.3f} computed with {n_context} neighbouring transactions."
    if not top or abs(top[0].importance) < 1e-9:
        return (
            base
            + " Removing any single neighbour does not change the score: it is driven by the transaction itself."
        )
    e = top[0]
    direction = "raises" if e.importance > 0 else "lowers"
    return (
        f"{base} The most influential neighbour is transaction {e.tx_id} "
        f"({e.sender} → {e.receiver}, {e.amount:,.2f}): it {direction} the score by {abs(e.importance):.3f}."
    )


__all__ = ["OcclusionExplainer"]
