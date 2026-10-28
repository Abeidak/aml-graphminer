"""Explainers specific to the GNN detectors.

* :class:`GNNExplainerAdapter` — PyG ``GNNExplainer`` (Ying et al., 2019): learned soft masks
  over the transactions and account features of the neighbourhood;
* :class:`PGExplainerAdapter` — PyG ``PGExplainer`` (Luo et al., 2020): a small network,
  trained once on a set of alerts, that predicts the important transactions of any alert;
* :class:`CaptumExplainerAdapter` — Captum attributions (Integrated Gradients by default) of the
  neighbouring transactions, of the account features and of the transaction's own features;
* :class:`AttentionExplainer` — GAT attention weights (descriptive, not causal).

All explain the score of one transaction on its *local sub-graph* (``hops`` account hops
around it). Features of that sub-graph are recomputed with the model's saved feature
statistics, so rolling-window and node aggregates are those of the explained neighbourhood —
an approximation of the full-graph features that keeps explanations fast.

These classes need the ``gnn`` extra (PyTorch + PyTorch Geometric); Captum needs the
``explain`` extra.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ConfigError, ModelError
from ..features.pipeline import FeaturePipeline
from ..utils import require
from .base import EdgeImportance, Explanation, FeatureImportance, NodeImportance
from .subgraph import TransactionIndex


@dataclass
class _LocalView:  # pragma: no cover - needs torch
    """Tensors of the explained sub-graph."""

    frame: pd.DataFrame
    target: int
    x: Any
    edge_index: Any
    label_index: Any
    edge_feat: Any
    accounts: np.ndarray
    n_tx: int
    edge_cols: list[str] = field(default_factory=list)
    node_cols: list[str] = field(default_factory=list)


def reset_explain_state(model: Any) -> None:  # pragma: no cover - needs torch
    """Undo the explanation state PyG may leave on the message-passing layers of ``model``.

    PyG's GNNExplainer assigns its edge mask (a ``Parameter``) to every layer, which registers
    ``_edge_mask`` as a parameter of the layer; afterwards PyG wraps every new mask into a *fresh*
    ``Parameter``, cutting it from the autograd graph. Captum then fails ("tensor not used in the
    graph") and PGExplainer silently stops learning. Removing the registration restores the plain
    attribute, so explainers can run in any order on the same model.
    """
    for module in model.modules():
        if "_edge_mask" in getattr(module, "_parameters", {}):
            del module._parameters["_edge_mask"]
            module._edge_mask = None


def _local_view(
    trained: Any,
    df: pd.DataFrame,
    tx_id: int,
    index: TransactionIndex,
    hops: int,
    max_edges: int,
) -> _LocalView:  # pragma: no cover - needs torch
    torch = require("torch", "gnn", "GNN explainers")
    if not trained.stats:
        raise ModelError("this model carries no feature statistics: it cannot be explained")
    reset_explain_state(trained.model)
    hood = index.neighbourhood(tx_id, hops, max_edges)
    target = int(hood.index.get_indexer(pd.Index([tx_id]))[0])
    features = FeaturePipeline.from_dict(trained.stats).transform(hood.reset_index(drop=True))
    src = torch.from_numpy(np.ascontiguousarray(features.src))
    dst = torch.from_numpy(np.ascontiguousarray(features.dst))
    forward = torch.stack([src, dst])
    return _LocalView(
        frame=hood,
        target=target,
        x=torch.from_numpy(np.ascontiguousarray(features.node)),
        edge_index=torch.cat([forward, forward.flip(0)], dim=1),
        label_index=torch.stack([src[target : target + 1], dst[target : target + 1]]),
        edge_feat=torch.from_numpy(np.ascontiguousarray(features.edge[target : target + 1])),
        accounts=features.accounts,
        n_tx=len(hood),
        edge_cols=list(features.edge_cols),
        node_cols=list(features.node_cols),
    )


def neighbourhood_effect(model: Any, view: _LocalView) -> tuple[float, float]:  # pragma: no cover
    """Score with the whole local sub-graph and with the explained transaction alone.

    The difference is the measured influence of the neighbouring transactions. On the IBM AML
    models it is often small (the decision rests on the transaction and its two accounts), and
    then an edge mask, however it looks, cannot be a faithful explanation.
    """
    torch = require("torch", "gnn", "GNN explainers")
    keep = torch.tensor([view.target, view.n_tx + view.target])
    with torch.no_grad():
        full = torch.sigmoid(model(view.x, view.edge_index, view.label_index, view.edge_feat))
        alone = torch.sigmoid(
            model(view.x, view.edge_index[:, keep], view.label_index, view.edge_feat)
        )
    return float(full[0]), float(alone[0])


def _neighbourhood_note(model: Any, view: _LocalView) -> str:  # pragma: no cover - needs torch
    full, alone = neighbourhood_effect(model, view)
    return (
        f" Without its {view.n_tx - 1} neighbouring transactions the score would be {alone:.3f}"
        f" (measured neighbourhood effect {full - alone:+.3f})."
    )


def _build_explanation(
    method: str,
    view: _LocalView,
    edge_scores: np.ndarray,
    node_scores: np.ndarray,
    score: float,
    threshold: float | None,
) -> Explanation:  # pragma: no cover - needs torch
    frame = view.frame
    edges = []
    for pos, (tx_id, row) in enumerate(zip(frame.index.tolist(), frame.to_dict("records"))):
        edges.append(
            EdgeImportance(
                tx_id=int(tx_id),
                sender=str(row["sender"]),
                receiver=str(row["receiver"]),
                amount=float(row["amount"]),
                timestamp=str(row["timestamp"]),
                importance=float(edge_scores[pos]),
                role="target" if pos == view.target else "neighbour",
            )
        )
    target_row = frame.iloc[view.target]
    roles = {str(target_row["sender"]): "sender", str(target_row["receiver"]): "receiver"}
    nodes = [
        NodeImportance(str(acc), float(node_scores[i]), roles.get(str(acc), "neighbour"))
        for i, acc in enumerate(view.accounts)
    ]
    explanation = Explanation(
        tx_id=int(frame.index[view.target]),  # type: ignore[arg-type]
        method=method,
        score=score,
        threshold=threshold,
        edges=edges,
        nodes=sorted(nodes, key=lambda n: -abs(n.importance)),
    )
    top = explanation.top_edges(1)
    explanation.summary = f"{method}: score {score:.3f}; " + (
        f"most influential neighbouring transaction {top[0].tx_id} "
        f"({top[0].sender} → {top[0].receiver}, importance {top[0].importance:.3f})."
        if top
        else "no neighbouring transaction."
    )
    return explanation


class GNNExplainerAdapter:  # pragma: no cover - needs torch
    """PyG ``GNNExplainer``: learns soft masks over edges and node features of the sub-graph.

    Parameters
    ----------
    trained:
        A :class:`~aml_graphminer.detectors.gnn.TrainedGNN` (or ``GNNDetector.trained``).
    hops, max_edges:
        Size of the explained sub-graph.
    epochs:
        Optimisation steps of the explainer.
    """

    method = "gnnexplainer"

    def __init__(
        self, trained: Any, *, hops: int = 2, max_edges: int = 80, epochs: int = 100
    ) -> None:
        self.trained = trained
        self.hops, self.max_edges, self.epochs = hops, max_edges, epochs

    def explain(
        self, df: pd.DataFrame, tx_id: int, index: TransactionIndex | None = None
    ) -> Explanation:
        torch = require("torch", "gnn", "GNNExplainer")
        explain_mod = require("torch_geometric.explain", "gnn", "GNNExplainer")
        index = index or TransactionIndex(df)
        view = _local_view(self.trained, df, tx_id, index, self.hops, self.max_edges)
        model = self.trained.model.eval()

        explainer = explain_mod.Explainer(
            model=model,
            algorithm=explain_mod.GNNExplainer(epochs=self.epochs),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config={
                "mode": "binary_classification",
                "task_level": "edge",
                "return_type": "raw",
            },
        )
        result = explainer(
            view.x,
            view.edge_index,
            index=torch.tensor([0]),
            label_index=view.label_index,
            edge_feat=view.edge_feat,
        )
        reset_explain_state(model)  # leave the model usable by the other explainers
        n = view.n_tx
        mask = result.edge_mask.detach().cpu().numpy()
        edge_scores = np.maximum(mask[:n], mask[n : 2 * n])  # forward / reverse copy of every tx
        node_scores = result.node_mask.detach().abs().sum(dim=1).cpu().numpy()
        node_scores = node_scores / (node_scores.max() or 1.0)
        with torch.no_grad():
            score = float(
                torch.sigmoid(model(view.x, view.edge_index, view.label_index, view.edge_feat))[0]
            )
        explanation = _build_explanation(
            self.method, view, edge_scores, node_scores, score, self.trained.threshold
        )
        explanation.summary += _neighbourhood_note(model, view)
        return explanation


class AttentionExplainer:  # pragma: no cover - needs torch
    """Attention weights of a GAT detector (``kind="gat"``), averaged over heads and layers.

    Attention is a *descriptive* signal (which neighbours the model listened to), not a causal
    attribution: use :class:`~aml_graphminer.explainers.OcclusionExplainer` to test causality.
    """

    method = "attention"

    def __init__(self, trained: Any, *, hops: int = 2, max_edges: int = 80) -> None:
        if trained.config.kind != "gat":
            raise ModelError("AttentionExplainer needs a GAT model (kind='gat')")
        self.trained = trained
        self.hops, self.max_edges = hops, max_edges

    def explain(
        self, df: pd.DataFrame, tx_id: int, index: TransactionIndex | None = None
    ) -> Explanation:
        torch = require("torch", "gnn", "attention explainer")
        functional = require("torch.nn.functional", "gnn", "attention explainer")
        index = index or TransactionIndex(df)
        view = _local_view(self.trained, df, tx_id, index, self.hops, self.max_edges)
        model = self.trained.model.eval()
        n_edges = view.edge_index.size(1)
        not_loop = view.edge_index[0] != view.edge_index[1]
        accum = torch.zeros(n_edges)
        with torch.no_grad():
            h = view.x
            for conv in model.encoder.convs:
                out, (_, alpha) = conv(h, view.edge_index, return_attention_weights=True)
                per_edge = alpha.mean(dim=1)[: int(not_loop.sum())]  # drop the added self loops
                layer = torch.zeros(n_edges)
                layer[not_loop] = per_edge
                accum += layer / len(model.encoder.convs)
                h = functional.relu(out)
            score = float(
                torch.sigmoid(model(view.x, view.edge_index, view.label_index, view.edge_feat))[0]
            )
        att = accum.numpy()
        n = view.n_tx
        edge_scores = np.maximum(att[:n], att[n : 2 * n])
        node_scores = np.zeros(view.x.size(0))
        src_dst = view.edge_index.numpy()
        np.add.at(node_scores, src_dst[0][:n], edge_scores)
        np.add.at(node_scores, src_dst[1][:n], edge_scores)
        node_scores = node_scores / (node_scores.max() or 1.0)
        explanation = _build_explanation(
            self.method, view, edge_scores, node_scores, score, self.trained.threshold
        )
        explanation.summary += _neighbourhood_note(model, view)
        return explanation


def _probability(model: Any) -> Any:  # pragma: no cover - needs torch
    """Wrap an :class:`EdgeGNN` so that it returns probabilities (Captum's binary convention)."""
    torch = require("torch", "gnn", "GNN explainers")

    class ProbabilityModel(torch.nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: Any, edge_index: Any, label_index: Any, edge_feat: Any) -> Any:
            return torch.sigmoid(self.inner(x, edge_index, label_index, edge_feat))

    return ProbabilityModel(model).eval()


def _per_transaction(values: np.ndarray, n_tx: int, combine: str = "sum") -> np.ndarray:
    """Merge the forward and reverse copy of every transaction of the message-passing graph."""
    forward, reverse = values[:n_tx], values[n_tx : 2 * n_tx]
    return forward + reverse if combine == "sum" else np.maximum(forward, reverse)


class CaptumExplainerAdapter:  # pragma: no cover - needs torch + captum
    """Captum attributions of an alert (Integrated Gradients by default).

    Three kinds of attributions are returned, all *signed* (positive = raises the laundering
    probability) and expressed in probability units:

    * every **neighbouring transaction** (Captum on the edge mask of the sub-graph, via PyG's
      ``CaptumExplainer``);
    * every **account** (sum of the attributions of its features);
    * the **features** of the explained transaction itself (``tx.<feature>``) and of its sender
      and receiver (``sender.<feature>`` / ``receiver.<feature>``). Values shown are the
      standardised model inputs.

    Parameters
    ----------
    attribution:
        ``"IntegratedGradients"`` (default, baseline = an average transaction), ``"Saliency"``
        or ``"InputXGradient"``.
    top_features:
        Number of feature attributions kept in the explanation.
    """

    method = "captum"
    ATTRIBUTIONS = ("IntegratedGradients", "Saliency", "InputXGradient")

    def __init__(
        self,
        trained: Any,
        *,
        hops: int = 2,
        max_edges: int = 80,
        attribution: str = "IntegratedGradients",
        top_features: int = 10,
    ) -> None:
        if attribution not in self.ATTRIBUTIONS:
            raise ConfigError(
                f"unknown attribution {attribution!r}; choose among {self.ATTRIBUTIONS}"
            )
        self.trained = trained
        self.hops, self.max_edges = hops, max_edges
        self.attribution, self.top_features = attribution, top_features

    def explain(
        self, df: pd.DataFrame, tx_id: int, index: TransactionIndex | None = None
    ) -> Explanation:
        torch = require("torch", "gnn", "the Captum explainer")
        explain_mod = require("torch_geometric.explain", "gnn", "the Captum explainer")
        captum_attr = require("captum.attr", "explain", "Captum attributions")
        index = index or TransactionIndex(df)
        view = _local_view(self.trained, df, tx_id, index, self.hops, self.max_edges)
        model = self.trained.model.eval()
        wrapped = _probability(model)

        explainer = explain_mod.Explainer(
            model=wrapped,
            algorithm=explain_mod.algorithm.CaptumExplainer(self.attribution),
            explanation_type="phenomenon",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config={
                "mode": "binary_classification",
                "task_level": "edge",
                "return_type": "probs",
            },
        )
        result = explainer(
            view.x,
            view.edge_index,
            target=torch.tensor([1]),
            index=torch.tensor([0]),
            label_index=view.label_index,
            edge_feat=view.edge_feat,
        )
        edge_attr = result.edge_mask.detach().cpu().numpy()
        node_attr = result.node_mask.detach().cpu().numpy()  # (n_accounts, n_node_features)
        edge_scores = _per_transaction(edge_attr, view.n_tx, "sum")
        node_scores = node_attr.sum(axis=1)

        # attributions of the transaction's own features (the decoder input)
        def from_edge_features(edge_feat: Any) -> Any:
            copies = view.label_index.repeat(1, edge_feat.size(0))
            return wrapped(view.x, view.edge_index, copies, edge_feat)

        method = getattr(captum_attr, self.attribution)(from_edge_features)
        kwargs = (
            {"baselines": torch.zeros_like(view.edge_feat)}
            if self.attribution == "IntegratedGradients"
            else {}
        )
        tx_attr = method.attribute(view.edge_feat, **kwargs).detach().cpu().numpy()[0]

        with torch.no_grad():
            score = float(wrapped(view.x, view.edge_index, view.label_index, view.edge_feat)[0])
        explanation = _build_explanation(
            self.method, view, edge_scores, node_scores, score, self.trained.threshold
        )
        explanation.features = self._feature_importances(view, tx_attr, node_attr)
        top = explanation.features[0] if explanation.features else None
        if top is not None:
            direction = "raises" if top.importance > 0 else "lowers"
            explanation.summary += (
                f" Most influential feature: {top.name} (standardised value {float(top.value):.2f})"
                f" {direction} the probability by {abs(top.importance):.3f}."
            )
        explanation.summary += _neighbourhood_note(self.trained.model.eval(), view)
        return explanation

    def _feature_importances(
        self, view: _LocalView, tx_attr: np.ndarray, node_attr: np.ndarray
    ) -> list[FeatureImportance]:
        items = [
            FeatureImportance(f"tx.{name}", float(view.edge_feat[0, j]), float(tx_attr[j]))
            for j, name in enumerate(view.edge_cols)
        ]
        src, dst = int(view.label_index[0, 0]), int(view.label_index[1, 0])
        for role, node in (("sender", src), ("receiver", dst)):
            items += [
                FeatureImportance(
                    f"{role}.{name}", float(view.x[node, j]), float(node_attr[node, j])
                )
                for j, name in enumerate(view.node_cols)
            ]
        items.sort(key=lambda f: abs(f.importance), reverse=True)
        return items[: self.top_features]


class PGExplainerAdapter:  # pragma: no cover - needs torch
    """PyG ``PGExplainer`` (Luo et al., 2020): a parametric explainer of the *phenomenon* "alert".

    Unlike GNNExplainer, which optimises one mask per explained transaction, PGExplainer trains
    once — on the local sub-graphs of a set of alerts (:meth:`fit`) — a small network that
    predicts how important every transaction of a sub-graph is from the embeddings of its two
    accounts. Explaining a new alert is then a single forward pass, and explanations of different
    alerts are consistent with each other.

    Its size penalty drives the masks to zero when removing the neighbourhood does not change the
    prediction. All-zero masks are then the faithful answer, not a failure: on the IBM AML models
    the measured neighbourhood effect (reported in the summary) is usually a few hundredths, the
    decision resting on the transaction and its two accounts (see ``docs/explainability.md``).

    The sub-graph of the explained transaction plays the role of the "graph" of PyG's
    graph-level PGExplainer, whose single output is the logit of that transaction.
    """

    method = "pgexplainer"

    def __init__(
        self,
        trained: Any,
        *,
        hops: int = 2,
        max_edges: int = 80,
        epochs: int = 30,
        lr: float = 0.003,
    ) -> None:
        if epochs < 1:
            raise ConfigError("epochs must be >= 1")
        self.trained = trained
        self.hops, self.max_edges = hops, max_edges
        self.epochs, self.lr = epochs, lr
        self._explainer: Any = None
        self.fitted_on = 0

    def _explainer_for(self, explain_mod: Any) -> Any:
        algorithm = explain_mod.algorithm.PGExplainer(epochs=self.epochs, lr=self.lr)
        return explain_mod.Explainer(
            model=self.trained.model.eval(),
            algorithm=algorithm,
            explanation_type="phenomenon",
            edge_mask_type="object",
            model_config={
                "mode": "binary_classification",
                "task_level": "graph",
                "return_type": "raw",
            },
        )

    def fit(
        self, df: pd.DataFrame, tx_ids: Iterable[int], index: TransactionIndex | None = None
    ) -> PGExplainerAdapter:
        """Train the explainer network on the neighbourhoods of ``tx_ids`` (typically alerts)."""
        torch = require("torch", "gnn", "PGExplainer")
        explain_mod = require("torch_geometric.explain", "gnn", "PGExplainer")
        index = index or TransactionIndex(df)
        views = [
            _local_view(self.trained, df, int(t), index, self.hops, self.max_edges) for t in tx_ids
        ]
        if not views:
            raise ConfigError("PGExplainer needs at least one transaction to learn from")
        explainer = self._explainer_for(explain_mod)
        target = torch.tensor([1])
        with warnings.catch_warnings():
            # PyG's PGExplainer.train() returns float(loss) of a tensor that requires grad
            warnings.filterwarnings("ignore", message=".*requires_grad=True to a scalar.*")
            for epoch in range(self.epochs):
                for view in views:
                    explainer.algorithm.train(
                        epoch,
                        explainer.model,
                        view.x,
                        view.edge_index,
                        target=target,
                        label_index=view.label_index,
                        edge_feat=view.edge_feat,
                    )
        self._explainer = explainer
        self.fitted_on = len(views)
        return self

    def explain(
        self, df: pd.DataFrame, tx_id: int, index: TransactionIndex | None = None
    ) -> Explanation:
        torch = require("torch", "gnn", "PGExplainer")
        if self._explainer is None:
            raise ModelError("PGExplainerAdapter is not fitted: call fit(df, alert_ids) first")
        index = index or TransactionIndex(df)
        view = _local_view(self.trained, df, tx_id, index, self.hops, self.max_edges)
        result = self._explainer(
            view.x,
            view.edge_index,
            target=torch.tensor([1]),
            label_index=view.label_index,
            edge_feat=view.edge_feat,
        )
        edge_scores = _per_transaction(result.edge_mask.detach().cpu().numpy(), view.n_tx, "max")
        node_scores = np.zeros(view.x.size(0))
        endpoints = view.edge_index.numpy()
        np.add.at(node_scores, endpoints[0][: view.n_tx], edge_scores)
        np.add.at(node_scores, endpoints[1][: view.n_tx], edge_scores)
        node_scores = node_scores / (node_scores.max() or 1.0)
        model = self.trained.model.eval()
        with torch.no_grad():
            score = float(
                torch.sigmoid(model(view.x, view.edge_index, view.label_index, view.edge_feat))[0]
            )
        explanation = _build_explanation(
            self.method, view, edge_scores, node_scores, score, self.trained.threshold
        )
        if len(edge_scores) and float(edge_scores.max() - edge_scores.min()) < 1e-3:
            # every transaction got the same mask: the prediction barely depends on the
            # neighbourhood structure (it is driven by the transaction's own features)
            explanation.summary = (
                f"pgexplainer: score {score:.3f}; no neighbouring transaction stands out (all masks "
                f"≈ {float(edge_scores.mean()):.2f}): the prediction is driven by the transaction "
                "and account features rather than by the neighbourhood — see the captum method."
            )
        explanation.summary += f" (explainer trained on {self.fitted_on} alert neighbourhoods)."
        explanation.summary += _neighbourhood_note(model, view)
        return explanation


__all__ = [
    "AttentionExplainer",
    "CaptumExplainerAdapter",
    "GNNExplainerAdapter",
    "PGExplainerAdapter",
    "neighbourhood_effect",
    "reset_explain_state",
]
