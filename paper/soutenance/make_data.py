#!/usr/bin/env python
# ruff: noqa: RUF001  (French typography on the slides: en dashes)
"""Collect the MEASURED numbers shown on the defence slides into deck_data.json.

    python paper/soutenance/make_data.py      # then: cd paper/soutenance && npm install && node build_deck.js

Sources: benchmarks/results_parts/*/*/results.csv (3-seed benchmark), per_pattern_recall.csv,
benchmarks/results_parts/hybrid/hybrid.json (hybrid detector), results/eda-*/summary.json (data).
The few figures that are not in a result file are quoted, with their source, in MEASURED below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARTS = ROOT / "benchmarks" / "results_parts"

# measured elsewhere in the repository (source in the comment)
MEASURED = {
    # docs/limitations.md: test PR-AUC of the motif scanner along the J2 work
    "motif_history": [
        {"step": "Détecteurs d'origine", "value": 0.005},
        {"step": "Devises + contexte + ACH", "value": 0.340},
        {"step": "Fans plus longs + exclusions", "value": 0.399},
        {"step": "Relais + calibration par tranche", "value": 0.503},
    ],
    "motif_ci": [0.479, 0.526],  # bootstrap 95 % interval, docs/limitations.md
    "neighbourhood": {"with": "0,988–1,000", "without": "0,960–1,000"},  # docs/explainability.md
    "unodc": "1 600 Md $",  # UNODC 2011: laundered in 2009, 2.7 % of world GDP
    "tests": 475,  # pytest, CHANGELOG / last full run
    "coverage": 98,  # %, last full run
    "no_motif_share": 36,  # % of HI-Small validation laundering outside any motif
}


def benchmark() -> list[dict]:
    frames = [pd.read_csv(p) for p in PARTS.glob("*/*/results.csv")]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["dataset", "model", "seed"], keep="last"
    )
    g = df.groupby(["dataset", "model"])["pr_auc"].agg(["mean", "std", "count"]).reset_index()
    return [
        {
            "dataset": r.dataset,
            "model": r.model,
            "mean": float(r["mean"]),
            "std": None if r["count"] < 2 else float(r["std"]),
        }
        for _, r in g.iterrows()
    ]


def per_pattern() -> dict:
    out = {}
    for ds in ("HI-Small", "LI-Small"):
        t = pd.read_csv(PARTS / "patterns" / ds / "per_pattern_recall.csv")
        out[ds] = {r.pattern: round(float(r.recall_same), 3) for r in t.itertuples()}
    return out


def hybrid() -> list[dict]:
    data = json.loads((PARTS / "hybrid" / "hybrid.json").read_text(encoding="utf-8"))
    return [
        {
            "dataset": ds,
            "base": base,
            "alone": row["components"]["base"]["pr_auc"],
            "motifs": row["components"]["motifs"]["pr_auc"],
            "hybrid": row["hybrid"]["pr_auc"],
        }
        for ds, block in data.items()
        for base, row in block["hybrids"].items()
    ]


def eda(name: str) -> dict:
    s = json.loads((ROOT / "results" / f"eda-{name}" / "summary.json").read_text(encoding="utf-8"))
    return {k: s[k] for k in ("transactions", "accounts", "illicit", "illicit_ratio")}


def main() -> Path:
    out = Path(__file__).resolve().parent / "deck_data.json"
    payload = {
        "benchmark": benchmark(),
        "per_pattern": per_pattern(),
        "hybrid": hybrid(),
        "eda": {"HI-Small": eda("hi-small"), "LI-Small": eda("li-small")},
        **MEASURED,
    }
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    print("wrote", main())
