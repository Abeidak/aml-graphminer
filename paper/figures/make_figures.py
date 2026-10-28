#!/usr/bin/env python
"""Regenerate the figures of the manuscript (architecture diagram, example explanation).

    python paper/figures/make_figures.py

The explanation figure is produced on SYNTHETIC data (the generator shipped with the package) and
is captioned as such in the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import aml_graphminer as aml
from aml_graphminer.detectors.bundle import MotifDetector
from aml_graphminer.explainers import explain_with_patterns, plot_explanation

OUT = Path(__file__).resolve().parent


def architecture() -> None:
    layers = [
        (
            "6  Interfaces",
            "web investigation UI  ·  REST API  ·  CLI amlgm  ·  notebooks  ·  MLflow  ·  HF Hub",
        ),
        (
            "5  Explainers",
            "pattern evidence  ·  occlusion  ·  GNNExplainer  ·  PGExplainer  ·  Captum  ·  attention",
        ),
        (
            "4  Detectors",
            "6 motifs + risk communities  ·  tabular  ·  GCN / SAGE / GAT  ·  graph auto-encoder",
        ),
        (
            "3  Graphs",
            "NetworkX (Louvain)  ·  PyTorch Geometric  ·  heterogeneous  ·  Neo4j export",
        ),
        (
            "2  Features",
            "causal temporal windows  ·  account rates  ·  PageRank  ·  leak-free pipeline",
        ),
        (
            "1  Ingestion",
            "IBM AML  ·  Elliptic  ·  CSV / JSON / Parquet  ·  adapters  ·  synthetic data",
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers) * 1.1)
    ax.axis("off")
    palette = ["#DCE6F2", "#E4EEDD", "#F6E7CC", "#EADCF0", "#F3DADA", "#DDE9E9"]
    for i, (name, text) in enumerate(layers):
        y = (len(layers) - 1 - i) * 1.1
        ax.add_patch(
            FancyBboxPatch(
                (0.1, y + 0.05),
                9.8,
                0.9,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                fc=palette[i],
                ec="#555555",
                lw=0.8,
            )
        )
        ax.text(0.35, y + 0.5, name, va="center", ha="left", fontsize=9, fontweight="bold")
        ax.text(2.35, y + 0.5, text, va="center", ha="left", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "architecture.pdf")
    fig.savefig(OUT / "architecture.png", dpi=200)  # slides and the thesis
    plt.close(fig)


def example_explanation() -> None:
    df = aml.generate_transactions(
        n_accounts=900, n_background=7000, days=10, illicit_ratio=0.02, seed=21
    )
    detector = MotifDetector.fit(df)
    result = detector.scanner.scan(df)
    # the strongest cycle alert: a round trip is the easiest motif to read on a figure
    cycle = max((a for a in result.alerts if a.pattern == "cycle"), key=lambda a: a.score)
    tx_id = int(cycle.tx_ids[0])
    explanation = explain_with_patterns(df, tx_id, result, detector.threshold)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    plot_explanation(explanation, ax=ax)
    ax.set_title(f"Transaction {tx_id}: " + ", ".join(p["pattern"] for p in explanation.patterns))
    fig.tight_layout()
    fig.savefig(OUT / "example_explanation.pdf")
    fig.savefig(OUT / "example_explanation.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    example_explanation()
    print("figures written to", OUT)
