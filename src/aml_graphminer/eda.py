"""Exploratory data analysis of a transaction table (CDC milestone J1: "EDA produite").

* :func:`eda_summary` — JSON-serialisable statistics (volumes per day and hour, amount
  distribution, payment formats, currencies, degree distribution, reciprocity, motifs);
* :func:`eda_figures` — the matching matplotlib figures (``viz`` extra);
* :func:`eda_report` — both, plus a Markdown report, written to a directory
  (``amlgm eda data.csv --out eda/``).

The same statistics feed the dashboard of the web investigation interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features.network import index_accounts
from .ingestion.dataset import summarize
from .utils import require


def _labelled(df: pd.DataFrame) -> bool:
    return bool(df["label"].notna().any())


def _histogram(values: np.ndarray, bins: np.ndarray) -> list[int]:
    counts, _ = np.histogram(values, bins=bins)
    return [int(c) for c in counts]


def _rate(part: float, whole: float) -> float | None:
    return None if whole == 0 else float(part / whole)


def eda_summary(df: pd.DataFrame, *, top: int = 10, bins: int = 30) -> dict[str, Any]:
    """Descriptive statistics of a canonical transaction table (no figure, JSON-ready).

    Parameters
    ----------
    top:
        Number of most active accounts / most frequent categories reported.
    bins:
        Number of bins of the (log10) amount histogram.
    """
    out: dict[str, Any] = dict(summarize(df))
    labelled = _labelled(df)
    illicit = (df["label"] == 1.0).to_numpy()
    amounts = df["amount"].to_numpy(dtype="float64")

    # ---------------------------------------------------------------- time
    day = df["timestamp"].dt.floor("D")
    daily = (
        pd.DataFrame({"day": day, "illicit": illicit, "amount": amounts})
        .groupby("day")
        .agg(transactions=("amount", "size"), illicit=("illicit", "sum"), amount=("amount", "sum"))
    )
    out["daily"] = [
        {
            "day": str(pd.Timestamp(day_start).date()),
            "transactions": int(n),
            "illicit": int(bad) if labelled else None,
            "amount": float(volume),
        }
        for day_start, n, bad, volume in zip(
            daily.index.tolist(),
            daily["transactions"].tolist(),
            daily["illicit"].tolist(),
            daily["amount"].tolist(),
        )
    ]
    hours = df["timestamp"].dt.hour.to_numpy()
    out["hourly"] = [int(c) for c in np.bincount(hours, minlength=24)]

    # ---------------------------------------------------------------- amounts
    logs = np.log10(amounts + 1.0)
    upper = float(np.ceil(logs.max())) if len(logs) else 1.0
    edges = np.linspace(0.0, max(upper, 1.0), bins + 1)
    out["amounts"] = {
        "log10_bins": [round(float(e), 4) for e in edges],
        "all": _histogram(logs, edges),
        "illicit": _histogram(logs[illicit], edges) if labelled else None,
        "quantiles": {
            q: float(np.quantile(amounts, p)) if len(amounts) else 0.0
            for q, p in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0))
        },
    }

    # ---------------------------------------------------------------- categories
    def categories(column: str) -> list[dict[str, Any]]:
        grouped = pd.DataFrame({"c": df[column].astype(str), "i": illicit}).groupby("c")["i"]
        table = grouped.agg(["size", "sum"]).sort_values("size", ascending=False).head(top)
        return [
            {
                "name": str(name),
                "transactions": int(size),
                "illicit_rate": _rate(float(bad), float(size)) if labelled else None,
            }
            for name, size, bad in zip(
                table.index.tolist(), table["size"].tolist(), table["sum"].tolist()
            )
        ]

    out["payment_formats"] = categories("transaction_type")
    out["currency_counts"] = categories("currency")

    # ---------------------------------------------------------------- graph structure
    out_deg = df.groupby("sender").size()
    in_deg = df.groupby("receiver").size()
    degree = out_deg.add(in_deg, fill_value=0)
    out["degree"] = {
        "max_out": int(out_deg.max()) if len(out_deg) else 0,
        "max_in": int(in_deg.max()) if len(in_deg) else 0,
        "mean": float(degree.mean()) if len(degree) else 0.0,
        "median": float(degree.median()) if len(degree) else 0.0,
        "p99": float(degree.quantile(0.99)) if len(degree) else 0.0,
        "ccdf_out": _ccdf(out_deg.to_numpy()),
        "ccdf_in": _ccdf(in_deg.to_numpy()),
    }
    out["top_accounts"] = [
        {"account": str(acc), "transactions": int(n)}
        for acc, n in degree.sort_values(ascending=False).head(top).items()
    ]
    src, dst, n_accounts, _ = index_accounts(df)
    pairs = np.unique(src * max(n_accounts, 1) + dst)  # vectorised: fine for the IBM files
    reverse = (pairs % max(n_accounts, 1)) * max(n_accounts, 1) + pairs // max(n_accounts, 1)
    reciprocal = int((np.isin(reverse, pairs) & (reverse != pairs)).sum())
    out["structure"] = {
        "self_loops": int((df["sender"] == df["receiver"]).sum()),
        "distinct_pairs": len(pairs),
        "reciprocal_pair_share": _rate(reciprocal, len(pairs)) or 0.0,
        "cross_bank_share": float(
            ((df["sender_bank"] != df["receiver_bank"]) & (df["sender_bank"] != "")).mean()
        )
        if len(df)
        else 0.0,
        "cross_currency_share": float((df["currency"] != df["currency_received"]).mean())
        if len(df)
        else 0.0,
    }
    return out


def _ccdf(degrees: np.ndarray, points: int = 25) -> list[dict[str, float]]:
    """Complementary CDF ``P(degree >= d)`` on a log grid (for a log-log plot)."""
    if len(degrees) == 0:
        return []
    grid = np.unique(np.round(np.logspace(0, np.log10(max(degrees.max(), 1)), points)))
    total = float(len(degrees))
    return [{"degree": float(d), "share": float((degrees >= d).sum() / total)} for d in grid]


def eda_figures(
    df: pd.DataFrame, directory: str | Path, summary: dict[str, Any] | None = None
) -> list[Path]:
    """Write the EDA figures (PNG) into ``directory``; returns their paths (needs ``viz``)."""
    plt = require("matplotlib.pyplot", "viz", "EDA figures")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = summary or eda_summary(df)
    labelled = _labelled(df)
    paths: list[Path] = []

    def save(fig: Any, name: str) -> None:
        path = directory / name
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)

    # amounts
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    amounts = summary["amounts"]
    edges = np.asarray(amounts["log10_bins"])
    centres, width = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
    total = np.asarray(amounts["all"], dtype=float)
    ax.bar(centres, total / max(total.sum(), 1), width=width, alpha=0.6, label="all")
    if labelled and amounts["illicit"] is not None and sum(amounts["illicit"]):
        bad = np.asarray(amounts["illicit"], dtype=float)
        ax.bar(centres, bad / bad.sum(), width=width, alpha=0.6, label="laundering")
        ax.legend()
    ax.set_xlabel("log10(amount + 1)")
    ax.set_ylabel("share of transactions")
    ax.set_title("Amount distribution")
    save(fig, "amounts.png")

    # daily volume
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    days = [d["day"] for d in summary["daily"]]
    ax.bar(range(len(days)), [d["transactions"] for d in summary["daily"]], color="#4c78a8")
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("transactions")
    ax.set_title("Transactions per day")
    if labelled:
        twin = ax.twinx()
        shares = [
            (d["illicit"] or 0) / d["transactions"] if d["transactions"] else 0.0
            for d in summary["daily"]
        ]
        twin.plot(range(len(days)), shares, color="#e45756", marker="o", ms=3)
        twin.set_ylabel("laundering share", color="#e45756")
    save(fig, "daily.png")

    # hour of day
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(range(24), summary["hourly"], color="#72b7b2")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("transactions")
    ax.set_title("Activity by hour")
    save(fig, "hourly.png")

    # payment formats
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    formats = summary["payment_formats"]
    ax.barh([f["name"] for f in formats][::-1], [f["transactions"] for f in formats][::-1])
    ax.set_xlabel("transactions")
    title = "Payment formats"
    if labelled:
        for i, f in enumerate(formats[::-1]):
            if f["illicit_rate"] is not None:
                ax.text(f["transactions"], i, f"  {f['illicit_rate']:.2%}", va="center", fontsize=7)
        title += " (laundering rate)"
    ax.set_title(title)
    save(fig, "payment_formats.png")

    # degree distribution
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for key, label in (("ccdf_out", "out-degree"), ("ccdf_in", "in-degree")):
        points = summary["degree"][key]
        if points:
            ax.loglog(
                [p["degree"] for p in points], [p["share"] for p in points], marker=".", label=label
            )
    ax.set_xlabel("degree d")
    ax.set_ylabel("P(degree >= d)")
    ax.set_title("Degree distribution (CCDF)")
    ax.legend()
    save(fig, "degrees.png")

    # ground-truth motifs
    if summary.get("patterns"):
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        names = list(summary["patterns"])
        ax.barh(names[::-1], [summary["patterns"][n] for n in names][::-1], color="#f58518")
        ax.set_xlabel("transactions")
        ax.set_title("Laundering transactions per ground-truth motif")
        save(fig, "patterns.png")
    return paths


def eda_report(
    df: pd.DataFrame, directory: str | Path, *, name: str = "dataset", figures: bool = True
) -> Path:
    """Write ``summary.json``, the figures and ``EDA.md`` into ``directory``; returns ``EDA.md``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = eda_summary(df)
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    images = eda_figures(df, directory, summary) if figures else []
    lines = [
        f"# Exploratory data analysis — {name}",
        "",
        "| statistic | value |",
        "|---|---:|",
        f"| transactions | {summary['transactions']:,} |",
        f"| accounts | {summary['accounts']:,} |",
        f"| period | {summary['first_timestamp']} → {summary['last_timestamp']} |",
        f"| currencies | {summary['currencies']} |",
        f"| labelled transactions | {summary['labelled']:,} |",
        f"| laundering transactions | {summary['illicit']:,} ({summary['illicit_ratio']:.4%}) |",
        f"| median / p99 amount | {summary['amounts']['quantiles']['p50']:,.2f} / "
        f"{summary['amounts']['quantiles']['p99']:,.2f} |",
        f"| max in / out degree | {summary['degree']['max_in']:,} / {summary['degree']['max_out']:,} |",
        f"| reciprocal account pairs | {summary['structure']['reciprocal_pair_share']:.2%} |",
        f"| cross-currency payments | {summary['structure']['cross_currency_share']:.2%} |",
        "",
        "## Payment formats",
        "",
        "| format | transactions | laundering rate |",
        "|---|---:|---:|",
    ]
    for f in summary["payment_formats"]:
        rate = "n/a" if f["illicit_rate"] is None else f"{f['illicit_rate']:.3%}"
        lines.append(f"| {f['name']} | {f['transactions']:,} | {rate} |")
    if summary.get("patterns"):
        lines += ["", "## Ground-truth motifs", "", "| motif | transactions |", "|---|---:|"]
        lines += [f"| {k} | {v:,} |" for k, v in summary["patterns"].items()]
    if images:
        lines += ["", "## Figures", ""]
        lines += [f"![{p.stem}]({p.name})" for p in images]
    report = directory / "EDA.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


__all__ = ["eda_figures", "eda_report", "eda_summary"]
