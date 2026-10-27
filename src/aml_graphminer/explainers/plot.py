"""Matplotlib rendering of an explanation sub-graph (``pip install 'aml-graphminer[viz]'``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from ..utils import require
from .base import Explanation


def plot_explanation(
    explanation: Explanation,
    ax: Any = None,
    path: str | Path | None = None,
    max_labels: int = 12,
) -> Any:
    """Draw the explanation: edge width/colour ∝ importance, the explained transaction in red."""
    plt = require("matplotlib.pyplot", "viz", "plotting explanations")
    graph = explanation.to_networkx()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "empty explanation", ha="center", va="center")
        ax.set_axis_off()
        return ax
    layout = nx.spring_layout(nx.DiGraph(graph), seed=7)
    edges = list(graph.edges(keys=True, data=True))
    top = max((abs(d["importance"]) for _, _, _, d in edges), default=1.0) or 1.0
    widths = [0.8 + 4.0 * abs(d["importance"]) / top for _, _, _, d in edges]
    colours = [
        "#d62728" if d["role"] == "target" else ("#1f77b4" if d["importance"] >= 0 else "#7f7f7f")
        for _, _, _, d in edges
    ]
    node_sizes = [
        260 + 900 * abs(graph.nodes[n].get("importance", 0.0)) / (top or 1.0) for n in graph.nodes
    ]
    nx.draw_networkx_nodes(
        graph, layout, node_size=node_sizes, node_color="#f2f2f2", edgecolors="#333", ax=ax
    )
    nx.draw_networkx_edges(
        graph,
        layout,
        edgelist=[(u, v, k) for u, v, k, _ in edges],
        width=widths,
        edge_color=colours,
        arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.12",
        ax=ax,
    )
    labels = {n: str(n)[-8:] for n in list(graph.nodes)[:max_labels]}
    nx.draw_networkx_labels(graph, layout, labels=labels, font_size=7, ax=ax)
    score = "" if explanation.score is None else f" — score {explanation.score:.2f}"
    ax.set_title(f"Transaction {explanation.tx_id} ({explanation.method}){score}", fontsize=10)
    ax.set_axis_off()
    if path is not None and fig is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return ax


__all__ = ["plot_explanation"]
