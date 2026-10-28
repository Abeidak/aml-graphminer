"""One entry point to explain a transaction with any detector (CLI, REST API, web interface)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ConfigError
from ..utils import has_module
from .base import Explanation
from .occlusion import OcclusionExplainer
from .patterns import explain_with_patterns
from .scorers import detector_score_fn
from .subgraph import TransactionIndex

EXPLAIN_METHODS: tuple[str, ...] = (
    "auto",
    "patterns",
    "occlusion",
    "gnnexplainer",
    "pgexplainer",
    "captum",
    "attention",
)
GNN_METHODS: tuple[str, ...] = ("gnnexplainer", "pgexplainer", "captum", "attention")
# PGExplainer learns from the neighbourhoods of this many top-scored transactions.
PG_TRAINING_ALERTS = 16


def _is_motif(detector: Any) -> bool:
    return getattr(detector, "model_type", None) == "patterns"


def _is_supervised_gnn(detector: Any) -> bool:
    return getattr(detector, "model_type", None) == "gnn"


def available_methods(detector: Any) -> list[str]:
    """Explanation methods that make sense for ``detector`` (``"auto"`` excluded)."""
    if _is_motif(detector):
        return ["patterns", "occlusion"]
    methods = ["occlusion"]
    if _is_supervised_gnn(detector):
        methods += ["gnnexplainer", "pgexplainer"]
        if has_module("captum"):
            methods.append("captum")
        if detector.config.kind == "gat":
            methods.append("attention")
    return methods


def default_method(detector: Any) -> str:
    """``patterns`` for the motif scanner (exact evidence), ``occlusion`` otherwise."""
    return "patterns" if _is_motif(detector) else "occlusion"


def explain_transaction(
    detector: Any,
    df: pd.DataFrame,
    tx_id: int,
    method: str = "auto",
    *,
    hops: int | None = None,
    max_edges: int = 80,
    index: TransactionIndex | None = None,
    scan: Any = None,
) -> Explanation:
    """Explain why ``detector`` scored transaction ``tx_id`` of ``df`` the way it did.

    Parameters
    ----------
    method:
        ``"auto"`` (pattern evidence for the motif scanner, occlusion otherwise) or one of
        :data:`EXPLAIN_METHODS`. The GNN methods need a supervised GNN detector.
    hops, max_edges:
        Size of the explained neighbourhood (default 1 hop for motifs, 2 for models).
    index:
        A :class:`TransactionIndex` of ``df`` to reuse across explanations.
    scan:
        A precomputed :class:`~aml_graphminer.detectors.patterns.ScanResult` of ``df`` (motif
        detector only) to avoid rescanning.
    """
    if method not in EXPLAIN_METHODS:
        raise ConfigError(f"unknown explanation method {method!r}; choose among {EXPLAIN_METHODS}")
    if tx_id not in df.index:
        raise ConfigError(f"transaction {tx_id} is not in the table")
    method = default_method(detector) if method == "auto" else method
    index = index or TransactionIndex(df)
    if method == "patterns":
        if not _is_motif(detector):
            raise ConfigError(
                "method 'patterns' needs a motif detector (amlgm train --model patterns)"
            )
        result = scan if scan is not None else detector.scanner.scan(df)
        return explain_with_patterns(df, tx_id, result, detector.threshold)
    if method == "occlusion":
        return OcclusionExplainer(
            detector_score_fn(detector),
            hops=hops or (1 if _is_motif(detector) else 2),
            max_edges=max_edges,
            threshold=detector.threshold,
        ).explain(df, tx_id, index)
    if not _is_supervised_gnn(detector):
        raise ConfigError(f"method {method!r} needs a supervised GNN detector (gcn, sage or gat)")

    from . import gnn_explainers as gx

    trained, radius = detector.trained, hops or 2
    if method == "gnnexplainer":
        return gx.GNNExplainerAdapter(trained, hops=radius, max_edges=max_edges).explain(
            df, tx_id, index
        )
    if method == "attention":
        return gx.AttentionExplainer(trained, hops=radius, max_edges=max_edges).explain(
            df, tx_id, index
        )
    if method == "captum":
        return gx.CaptumExplainerAdapter(trained, hops=radius, max_edges=max_edges).explain(
            df, tx_id, index
        )
    # pgexplainer: learn from the neighbourhoods of the most suspicious transactions
    scores = detector.score(df).to_numpy()
    top = df.index[np.argsort(-scores, kind="stable")[:PG_TRAINING_ALERTS]].tolist()
    learn_from = list(dict.fromkeys([int(tx_id), *(int(t) for t in top)]))
    explainer = gx.PGExplainerAdapter(trained, hops=radius, max_edges=max_edges)
    return explainer.fit(df, learn_from, index).explain(df, tx_id, index)


def explain_alerts(
    detector: Any,
    df: pd.DataFrame,
    top: int = 20,
    method: str = "auto",
    *,
    hops: int | None = None,
    max_edges: int = 80,
    only_alerts: bool = True,
    out_dir: str | Path | None = None,
) -> list[Explanation]:
    """Explain the ``top`` highest-scored transactions of ``df`` in one go (a case file).

    The table is scored once, the neighbourhood index (and, for the motif scanner, the scan or,
    for PGExplainer, the trained explainer) is shared by every explanation. With
    ``only_alerts`` only transactions at or above the detector threshold are explained. With
    ``out_dir`` every explanation is written as ``<rank>_tx<id>.json`` / ``.md`` next to an
    ``index.csv`` and a ``README.md`` that list them.
    """
    if top < 1:
        raise ConfigError("top must be at least 1")
    if method not in EXPLAIN_METHODS:
        raise ConfigError(f"unknown explanation method {method!r}; choose among {EXPLAIN_METHODS}")
    method = default_method(detector) if method == "auto" else method
    scores = detector.score(df).to_numpy()
    order = np.argsort(-scores, kind="stable")
    if only_alerts:
        order = order[scores[order] >= detector.threshold]
    chosen = [int(t) for t in df.index[order[:top]]]
    index = TransactionIndex(df)
    scan = detector.scanner.scan(df) if method == "patterns" and _is_motif(detector) else None
    pg = None
    if method == "pgexplainer" and chosen:
        if not _is_supervised_gnn(detector):
            raise ConfigError("method 'pgexplainer' needs a supervised GNN detector")
        from . import gnn_explainers as gx

        learn_from = [int(t) for t in df.index[np.argsort(-scores, kind="stable")]]
        learn_from = list(dict.fromkeys([*chosen, *learn_from[:PG_TRAINING_ALERTS]]))
        pg = gx.PGExplainerAdapter(detector.trained, hops=hops or 2, max_edges=max_edges)
        pg.fit(df, learn_from, index)
    explanations = []
    for tx_id in chosen:
        if pg is not None:
            explanations.append(pg.explain(df, tx_id, index))
        else:
            explanations.append(
                explain_transaction(
                    detector,
                    df,
                    tx_id,
                    method,
                    hops=hops,
                    max_edges=max_edges,
                    index=index,
                    scan=scan,
                )
            )
    if out_dir is not None:
        _write_case_file(Path(out_dir), explanations, method)
    return explanations


def _write_case_file(directory: Path, explanations: list[Explanation], method: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, exp in enumerate(explanations, start=1):
        stem = f"{rank:03d}_tx{exp.tx_id}"
        (directory / f"{stem}.json").write_text(exp.to_json(), encoding="utf-8")
        (directory / f"{stem}.md").write_text(exp.to_markdown() + "\n", encoding="utf-8")
        rows.append(
            {
                "rank": rank,
                "tx_id": exp.tx_id,
                "score": exp.score,
                "method": exp.method,
                "patterns": ";".join(sorted({str(p.get("pattern", "")) for p in exp.patterns})),
                "summary": exp.summary,
                "file": f"{stem}.md",
            }
        )
    table = pd.DataFrame(
        rows, columns=["rank", "tx_id", "score", "method", "patterns", "summary", "file"]
    )
    table.to_csv(directory / "index.csv", index=False)
    lines = [f"# Explained alerts ({len(rows)}, method `{method}`)", ""]
    lines += ["| rank | transaction | score | patterns | summary |", "|---|---|---|---|---|"]
    for r in rows:
        score = "" if r["score"] is None else f"{r['score']:.3f}"
        summary = str(r["summary"]).replace("|", "\\|")
        lines.append(
            f"| {r['rank']} | [{r['tx_id']}]({r['file']}) | {score} | {r['patterns']} | {summary} |"
        )
    (directory / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "EXPLAIN_METHODS",
    "GNN_METHODS",
    "available_methods",
    "default_method",
    "explain_alerts",
    "explain_transaction",
]
