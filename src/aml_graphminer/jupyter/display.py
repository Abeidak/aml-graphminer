"""HTML views of the results for notebooks (Jupyter, JupyterLab, VS Code, Colab).

Pure functions that return HTML strings: ``ScanResult``, ``Alert``, ``Explanation`` and
``BenchmarkReport`` call them from ``_repr_html_``, so a result left at the end of a cell is
shown as tables and an interactive graph (tooltips on every account and transaction) without
loading anything. Text inherits the colour of the notebook theme, the accents are those of the
web interface and stay readable on light and dark themes.

Every value that comes from the data (account names, evidence) is escaped: an account called
``<script>`` is displayed, never executed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection
from html import escape
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from ..detectors.patterns.base import Alert
    from ..detectors.patterns.scanner import ScanResult
    from ..evaluation.benchmark import BenchmarkReport
    from ..explainers.base import Explanation

# accents of the web interface (api/static/style.css)
ALERT = "#dc2626"
POSITIVE = "#3b82f6"
NEUTRAL = "#94a3b8"
PATTERN_COLOURS = {
    "fan_in": "#e8590c",
    "fan_out": "#d6336c",
    "smurfing": "#7048e8",
    "cycle": "#1c7ed6",
    "scatter_gather": "#0ca678",
    "gather_scatter": "#2f9e44",
    "community": "#f08c00",
    "relay": "#ae3ec9",
}
_MUTED = "opacity:.7"
_ZERO = 5e-4  # importances that round to 0.000 are shown as 0, in grey, without a bar


def _box(body: str) -> str:
    return f'<div class="aml-graphminer" style="line-height:1.45">{body}</div>'


def _badge(pattern: str) -> str:
    colour = PATTERN_COLOURS.get(pattern, NEUTRAL)
    return (
        f'<span style="padding:0 7px;border:1px solid {colour};border-radius:9px;'
        f'color:{colour};white-space:nowrap">{escape(pattern)}</span>'
    )


def _bar(value: float, top: float, colour: str, width: int = 80) -> str:
    length = 0 if top <= 0 else round(width * min(abs(value) / top, 1.0))
    return (
        f'<span style="display:inline-block;width:{length}px;height:8px;background:{colour};'
        'border-radius:2px;vertical-align:middle;margin-right:6px"></span>'
    )


def _amount(value: float) -> str:
    return f"{value:,.2f}"


def _signed(value: float) -> str:
    return "0.000" if abs(value) < _ZERO else f"{value:+.3f}"


def _importance(value: float, top: float) -> str:
    """Signed importance with a bar relative to the largest one shown (blue: raises the score)."""
    if abs(value) < _ZERO:
        return _signed(value)
    colour = POSITIVE if value > 0 else NEUTRAL
    return _bar(value, max(top, 1e-3), colour) + _signed(value)


def _short(account: Any, size: int = 12) -> str:
    text = str(account)
    return text if len(text) <= size else "…" + text[-(size - 1) :]


def _table(header: list[str], rows: list[list[str]], numeric: Collection[int] = ()) -> str:
    def cell(tag: str, i: int, text: str) -> str:
        align = ' style="text-align:right"' if i in numeric else ' style="text-align:left"'
        return f"<{tag}{align}>{text}</{tag}>"

    head = "".join(cell("th", i, escape(h)) for i, h in enumerate(header))
    body = "".join(
        "<tr>" + "".join(cell("td", i, c) for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _value(value: Any, limit: int = 120) -> str:
    """Short escaped text of an evidence value (lists are truncated)."""
    if isinstance(value, float):
        text = f"{value:,.2f}" if abs(value) >= 1000 else f"{value:.4g}"
    elif isinstance(value, (list, tuple)):
        items = [str(v) for v in value[:8]]
        text = ", ".join(items) + (f" … (+{len(value) - 8})" if len(value) > 8 else "")
    else:
        text = str(value)
    return escape(text if len(text) <= limit else text[: limit - 1] + "…")


# ----------------------------------------------------------------------------- scans and alerts
def scan_html(result: ScanResult, top: int = 10) -> str:
    """Summary of a scan: alerts per motif and the ``top`` strongest alerts."""
    summary = result.summary()
    head = (
        f"<b>{summary['n_alerts']:,} alert(s)</b> · {summary['n_flagged_transactions']:,} of "
        f"{summary['n_transactions']:,} transactions flagged"
    )
    if not result.alerts:
        return _box(f"{head}<p style='{_MUTED}'>No motif found in these transactions.</p>")
    counts = sorted(summary["alerts_by_pattern"].items(), key=lambda kv: -kv[1])
    most = counts[0][1]
    per_pattern = "".join(
        f"<div>{_bar(n, most, PATTERN_COLOURS.get(p, NEUTRAL), 120)}{_badge(p)} {n:,}</div>"
        for p, n in counts
    )
    strongest = sorted(result.alerts, key=result.weight, reverse=True)[:top]
    rows = []
    for rank, alert in enumerate(strongest, start=1):
        score = result.weight(alert)
        accounts = [escape(_short(a)) for a in alert.accounts[:3]]
        if len(alert.accounts) > 3:
            accounts.append(f"+{len(alert.accounts) - 3}")
        rows.append(
            [
                str(rank),
                _badge(alert.pattern),
                f"{_bar(score, 1.0, ALERT, 50)}{score:.3f}",
                f"{len(alert.tx_ids):,}",
                f"{len(alert.accounts):,}",
                _amount(alert.total_amount),
                escape(f"{alert.start:%Y-%m-%d %H:%M} → {alert.end:%Y-%m-%d %H:%M}"),
                ", ".join(accounts),
            ]
        )
    table = _table(
        [
            "#",
            "pattern",
            "score",
            "transactions",
            "accounts",
            "amount",
            "period",
            "accounts involved",
        ],
        rows,
        numeric={0, 3, 4, 5},
    )
    more = (
        f"<p style='{_MUTED}'>Top {len(strongest)} of {len(result):,} alerts — "
        "<code>.to_frame()</code> lists them all, <code>.alerts[i]</code> shows one.</p>"
    )
    return _box(f"{head}<div style='margin:6px 0'>{per_pattern}</div>{table}{more}")


def alert_html(alert: Alert) -> str:
    """One motif alert with its evidence (what a compliance officer reads first)."""
    rows = [
        ["score", f"{_bar(alert.score, 1.0, ALERT, 50)}{alert.score:.3f}"],
        ["period", escape(f"{alert.start:%Y-%m-%d %H:%M} → {alert.end:%Y-%m-%d %H:%M}")],
        ["total amount", _amount(alert.total_amount)],
        ["transactions", f"{len(alert.tx_ids):,}: {_value(list(alert.tx_ids))}"],
        ["accounts", f"{len(alert.accounts):,}: {_value(list(alert.accounts))}"],
    ]
    rows += [[escape(str(k)), _value(v)] for k, v in alert.evidence.items()]
    return _box(f"<div>{_badge(alert.pattern)} <b>alert</b></div>{_table(['', ''], rows)}")


# ----------------------------------------------------------------------------- explanations
def _layout(graph: nx.MultiDiGraph) -> dict[Any, Any]:
    """Kamada-Kawai (even spacing, a hub does not crush its neighbours), spring otherwise."""
    simple = nx.Graph(graph)
    if simple.number_of_nodes() > 2 and nx.is_connected(simple):
        try:
            return nx.kamada_kawai_layout(simple)
        except ImportError:  # pragma: no cover - SciPy comes with scikit-learn
            pass
    return nx.spring_layout(simple, seed=7)


def _towards(x: float, y: float, tx: float, ty: float, distance: float) -> tuple[float, float]:
    """Point at ``distance`` from (x, y) in the direction of (tx, ty)."""
    length = math.hypot(tx - x, ty - y) or 1.0
    return x + (tx - x) * distance / length, y + (ty - y) * distance / length


def graph_svg(
    explanation: Explanation, width: int = 640, height: int = 360, labels: int = 14
) -> str:
    """Explanation sub-graph as inline SVG.

    Accounts are circles (the two accounts of the explained transaction in red, size ∝
    importance), transactions are arrows: red for the explained one, blue when they push the
    score up, grey otherwise, width ∝ absolute importance. Hovering shows the details.
    """
    graph = explanation.to_networkx()
    if graph.number_of_nodes() == 0:
        return f"<p style='{_MUTED}'>Empty explanation: no transaction to show.</p>"
    pos = _layout(graph)
    pad = 30.0
    xs = [float(p[0]) for p in pos.values()]
    ys = [float(p[1]) for p in pos.values()]

    def scale(value: float, low: float, high: float, size: int) -> float:
        if high - low < 1e-9:
            return size / 2
        return pad + (value - low) / (high - low) * (size - 2 * pad)

    at = {
        n: (
            scale(float(p[0]), min(xs), max(xs), width),
            scale(float(p[1]), min(ys), max(ys), height),
        )
        for n, p in pos.items()
    }
    target = {a for e in explanation.edges if e.role == "target" for a in (e.sender, e.receiver)}
    node_top = max([abs(d.get("importance", 0.0)) for _, d in graph.nodes(data=True)] + [1e-3])
    radius = {
        n: 5.0 + 7.0 * abs(d.get("importance", 0.0)) / node_top for n, d in graph.nodes(data=True)
    }
    edges = list(graph.edges(keys=True, data=True))
    edge_top = max([abs(d["importance"]) for *_, d in edges] + [1e-3])
    uid = "aml-" + re.sub(r"[^a-z0-9]", "", f"{int(explanation.tx_id)}{explanation.method}".lower())
    kinds = {"target": ALERT, "pos": POSITIVE, "other": NEUTRAL}
    defs = "".join(
        f'<marker id="{uid}-{kind}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
        f'markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{colour}"/></marker>'
        for kind, colour in kinds.items()
    )
    parts = [f"<defs>{defs}</defs>"]
    parallel: dict[tuple[Any, Any], int] = {}
    for u, v, _key, d in edges:
        kind = "pos" if d["importance"] >= _ZERO else "other"
        kind = "target" if d["role"] == "target" else kind
        stroke = 1.0 + 3.5 * abs(d["importance"]) / edge_top
        if kind == "target":
            stroke = max(stroke, 3.0)
        (x1, y1), (x2, y2) = at[u], at[v]
        if u == v:  # self-transfer: a loop above the account
            r = radius[u]
            path = (
                f"M {x1 - r * 0.6:.1f} {y1 - r * 0.8:.1f} C {x1 - r * 2.4:.1f} {y1 - r * 3.6:.1f} "
                f"{x1 + r * 2.4:.1f} {y1 - r * 3.6:.1f} {x1 + r * 0.6:.1f} {y1 - r * 0.8:.1f}"
            )
        else:  # parallel transfers are bent apart, the arrow stops at the circle
            k = parallel.get((u, v), 0)
            parallel[(u, v)] = k + 1
            bend = 0.12 + 0.1 * k
            cx = (x1 + x2) / 2 - bend * (y2 - y1)
            cy = (y1 + y2) / 2 + bend * (x2 - x1)
            sx, sy = _towards(x1, y1, cx, cy, radius[u])
            ex, ey = _towards(x2, y2, cx, cy, radius[v] + 1.5)
            path = f"M {sx:.1f} {sy:.1f} Q {cx:.1f} {cy:.1f} {ex:.1f} {ey:.1f}"
        tip = (
            f"tx {d['tx_id']}: {u} → {v}\n{_amount(float(d['amount']))} at {d['timestamp']}\n"
            f"importance {_signed(d['importance'])} ({d['role']})"
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{kinds[kind]}" stroke-width="{stroke:.2f}" '
            f'stroke-opacity="0.85" marker-end="url(#{uid}-{kind})"><title>{escape(tip)}</title></path>'
        )
    ranked = sorted(
        graph.nodes, key=lambda n: abs(graph.nodes[n].get("importance", 0.0)), reverse=True
    )
    named = target | set(ranked[: max(labels - len(target), 0)])
    for n, d in graph.nodes(data=True):
        x, y = at[n]
        fill = ALERT if n in target else NEUTRAL
        tip = f"{n}\n{d.get('role', 'neighbour')}, importance {_signed(d.get('importance', 0.0))}"
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius[n]:.1f}" fill="{fill}" fill-opacity="0.9" '
            f'stroke="currentColor" stroke-opacity="0.35"><title>{escape(tip)}</title></circle>'
        )
        if n in named:
            parts.append(
                f'<text x="{x:.1f}" y="{y - radius[n] - 3:.1f}" text-anchor="middle" font-size="10" '
                f'fill="currentColor">{escape(_short(n))}</text>'
            )
    label = escape(f"Explanation graph of transaction {explanation.tx_id}")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{label}" style="width:100%;max-width:{width}px;height:auto">'
        + "".join(parts)
        + "</svg>"
    )


def _legend() -> str:
    def line(colour: str, text: str) -> str:
        return (
            f'<span style="margin-right:14px"><span style="display:inline-block;width:18px;height:3px;'
            f'background:{colour};vertical-align:3px;margin-right:5px"></span>{text}</span>'
        )

    return (
        f"<div style='font-size:.9em;{_MUTED}'>"
        + line(ALERT, "explained transaction")
        + line(POSITIVE, "raises the score")
        + line(NEUTRAL, "lowers it / neutral")
        + "width ∝ |importance|, hover for details</div>"
    )


def explanation_html(explanation: Explanation, top: int = 8) -> str:
    """Verdict, summary, graph and the most important transactions, attributes and motifs."""
    title = f"<div style='font-size:1.1em;font-weight:600'>Transaction {explanation.tx_id} — {escape(explanation.method)}</div>"
    verdict = ""
    if explanation.score is not None:
        verdict = f"score <b>{explanation.score:.3f}</b>"
        if explanation.threshold is not None:
            flagged = explanation.score >= explanation.threshold
            word = (
                f"<b style='color:{ALERT}'>alert</b>"
                if flagged
                else f"<span style='{_MUTED}'>no alert</span>"
            )
            verdict += f" · threshold {explanation.threshold:.3f} · {word}"
    target = next((e for e in explanation.edges if e.role == "target"), None)
    if target is not None:
        verdict += escape(
            f"{' · ' if verdict else ''}{target.sender} → {target.receiver}, "
            f"{_amount(target.amount)} at {target.timestamp}"
        )
    body = [title, f"<div>{verdict}</div>"] if verdict else [title]
    if explanation.summary:
        body.append(f"<p>{escape(explanation.summary)}</p>")
    body += [graph_svg(explanation), _legend()]
    edges = explanation.top_edges(top)
    if edges:
        most = max(abs(e.importance) for e in edges)
        rows = [
            [
                str(e.tx_id),
                escape(f"{e.sender} → {e.receiver}"),
                _amount(e.amount),
                escape(str(e.timestamp)),
                _importance(e.importance, most),
            ]
            for e in edges
        ]
        body.append("<h4>Transactions that matter most</h4>")
        body.append(
            _table(["tx", "sender → receiver", "amount", "time", "importance"], rows, {0, 2})
        )
    features = sorted(explanation.features, key=lambda f: abs(f.importance), reverse=True)[:top]
    if features:
        most = max(abs(f.importance) for f in features)
        rows = [
            [escape(f.name), _value(f.value), _importance(f.importance, most)] for f in features
        ]
        body.append("<h4>Attributes</h4>")
        body.append(_table(["attribute", "value", "importance"], rows))
    if explanation.patterns:
        items = []
        for p in explanation.patterns:
            evidence = p.get("evidence", {})
            facts = ", ".join(
                f"{escape(str(k))} {_value(v, 40)}"
                for k, v in list(evidence.items())[:6]
                if not isinstance(v, (dict, list, tuple))
            )
            items.append(
                f"<li>{_badge(str(p.get('pattern', '?')))} alert score "
                f"{float(p.get('score', 0.0)):.3f}{' — ' + facts if facts else ''}</li>"
            )
        body.append(f"<h4>Motifs</h4><ul>{''.join(items)}</ul>")
    return _box("".join(body))


# ----------------------------------------------------------------------------- benchmarks
def benchmark_html(report: BenchmarkReport) -> str:
    """Mean ± std of the test metrics per model (the table of ``amlgm benchmark``)."""
    summary = report.summary()
    n_models = report.results["model"].nunique() if "model" in report.results else len(summary)
    head = f"<b>Benchmark</b> · {n_models} model(s), {len(report.results)} run(s) — test metrics"
    table = summary.to_html(float_format=lambda v: f"{v:.4f}", na_rep="—", border=0)
    return _box(f"{head}{table}")


__all__ = ["alert_html", "benchmark_html", "explanation_html", "graph_svg", "scan_html"]
