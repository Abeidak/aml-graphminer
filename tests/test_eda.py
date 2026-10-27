"""Exploratory data analysis: statistics, figures and Markdown report."""

import json

import numpy as np
import pytest

from aml_graphminer.eda import eda_report, eda_summary
from helpers import make_tx


def test_summary_is_consistent_with_the_table(synth_df):
    summary = eda_summary(synth_df)
    n = len(synth_df)
    assert summary["transactions"] == n
    assert sum(d["transactions"] for d in summary["daily"]) == n
    assert sum(d["illicit"] for d in summary["daily"]) == int(synth_df["label"].sum())
    assert sum(summary["hourly"]) == n and len(summary["hourly"]) == 24
    amounts = summary["amounts"]
    assert sum(amounts["all"]) == n and len(amounts["log10_bins"]) == len(amounts["all"]) + 1
    assert sum(amounts["illicit"]) == int(synth_df["label"].sum())
    assert amounts["quantiles"]["max"] == pytest.approx(synth_df["amount"].max())
    formats = summary["payment_formats"]
    assert formats[0]["transactions"] >= formats[-1]["transactions"]
    assert all(0 <= f["illicit_rate"] <= 1 for f in formats)
    degree = summary["degree"]
    assert degree["max_out"] == synth_df.groupby("sender").size().max()
    assert degree["ccdf_out"][0]["share"] == pytest.approx(1.0)
    assert 0 <= summary["structure"]["reciprocal_pair_share"] <= 1
    assert summary["top_accounts"][0]["transactions"] >= summary["top_accounts"][-1]["transactions"]
    json.dumps(summary, default=str)  # JSON-ready


def test_reciprocity_and_unlabelled_tables():
    df = make_tx([(0, "a", "b", 10), (1, "b", "a", 10), (2, "a", "c", 5), (3, "c", "c", 1)])
    df["label"] = np.nan
    summary = eda_summary(df)
    structure = summary["structure"]
    assert structure["self_loops"] == 1 and structure["distinct_pairs"] == 4
    assert structure["reciprocal_pair_share"] == pytest.approx(0.5)  # a->b and b->a
    assert summary["daily"][0]["illicit"] is None and summary["amounts"]["illicit"] is None
    assert summary["payment_formats"][0]["illicit_rate"] is None


def test_report_without_figures(tmp_path, tiny_df):
    report = eda_report(tiny_df, tmp_path / "eda", name="tiny", figures=False)
    text = report.read_text(encoding="utf-8")
    assert report.name == "EDA.md" and "Exploratory data analysis — tiny" in text
    assert "## Payment formats" in text and "## Ground-truth motifs" in text
    assert json.loads((tmp_path / "eda" / "summary.json").read_text())["transactions"] == len(
        tiny_df
    )


def test_report_with_figures(tmp_path, tiny_df):
    pytest.importorskip("matplotlib")
    report = eda_report(tiny_df, tmp_path / "eda", name="tiny")
    names = {p.name for p in (tmp_path / "eda").glob("*.png")}
    assert {
        "amounts.png",
        "daily.png",
        "hourly.png",
        "payment_formats.png",
        "degrees.png",
        "patterns.png",
    } <= names
    assert "![amounts](amounts.png)" in report.read_text(encoding="utf-8")
