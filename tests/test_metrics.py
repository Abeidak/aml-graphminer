import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aml_graphminer.evaluation.metrics import (
    aggregate_runs,
    best_threshold,
    bootstrap_ci,
    compute_metrics,
    precision_at_k,
    recall_at_k,
    to_markdown,
)

Y = np.array([0, 0, 1, 0, 1, 0, 0, 1])
P = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.15, 0.05, 0.4])


def test_perfect_ranking():
    m = compute_metrics(Y, P, threshold=0.35)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0) and m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["recall_at_k"] == pytest.approx(1.0)  # R-precision, notebook semantics


def test_recall_at_k_and_precision_at_k():
    assert recall_at_k(Y, P, 2) == pytest.approx(2 / 3)
    assert recall_at_k(Y, P, 3) == pytest.approx(1.0)
    assert precision_at_k(Y, P, 2) == pytest.approx(1.0)
    assert precision_at_k(Y, P, 4) == pytest.approx(0.75)
    assert np.isnan(recall_at_k(np.zeros(5), np.arange(5), 2))
    assert np.isnan(recall_at_k(Y, P, 0))
    assert np.isnan(precision_at_k([], [], 3))


def test_best_threshold_degenerate_labels():
    assert best_threshold([0, 0, 0], [0.1, 0.2, 0.3]) == 0.5
    assert best_threshold([1, 1], [0.1, 0.2]) == 0.5


def test_best_threshold_maximises_f1():
    thr = best_threshold(Y, P)
    f1 = compute_metrics(Y, P, thr)["f1"]
    for alt in (0.05, 0.2, 0.5, 0.85):
        assert f1 >= compute_metrics(Y, P, alt)["f1"] - 1e-9


def test_nan_labels_are_ignored_and_shape_checked():
    y = np.array([np.nan, 1, 0, 1])
    p = np.array([0.5, 0.9, 0.1, 0.8])
    assert compute_metrics(y, p, 0.5)["n"] == 3
    with pytest.raises(ValueError, match="same shape"):
        compute_metrics([0, 1], [0.1])


def test_no_positive_gives_nan_ranking_metrics():
    m = compute_metrics(np.zeros(6), np.linspace(0, 1, 6), 0.5)
    assert np.isnan(m["pr_auc"]) and np.isnan(m["roc_auc"]) and np.isnan(m["recall_at_k"])


def test_threshold_none_is_chosen_on_data():
    assert compute_metrics(Y, P)["threshold"] == pytest.approx(best_threshold(Y, P))


def test_alert_budget_metrics_present():
    m = compute_metrics(Y, P, 0.5, ks=(2,), alert_rates=(0.25,))
    assert m["recall_at_2"] == pytest.approx(2 / 3)
    assert m["recall_at_25pct"] == pytest.approx(2 / 3)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.1).astype(int)
    p = y * 0.5 + rng.random(400) * 0.6
    lo, hi = bootstrap_ci(y, p, n_boot=100)
    point = compute_metrics(y, p, 0.5)["pr_auc"]
    assert lo <= point <= hi
    lo_nan, hi_nan = bootstrap_ci(np.zeros(10), np.zeros(10), n_boot=5)
    assert np.isnan(lo_nan) and np.isnan(hi_nan)


def test_aggregate_and_markdown():
    results = pd.DataFrame(
        {
            "dataset": ["d"] * 4,
            "model": ["a", "a", "b", "b"],
            "seed": [0, 1, 0, 1],
            "pr_auc": [0.5, 0.7, 0.2, 0.2],
            "f1": [0.1, 0.3, 0.2, 0.2],
        }
    )
    table = aggregate_runs(results)
    assert table.loc[("d", "a"), ("pr_auc", "mean")] == pytest.approx(0.6)
    assert table.loc[("d", "b"), ("pr_auc", "std")] == pytest.approx(0.0)
    md = to_markdown(table)
    assert md.splitlines()[0].startswith("| dataset | model | pr_auc (mean)")
    assert "0.6000" in md


# Scores live on a 1/1000 grid: squaring two *distinct* grid values never makes them equal,
# whereas squaring arbitrary floats can underflow (1e-209 ** 2 == 0.0) and create artificial ties.
_GRID = st.integers(min_value=0, max_value=1000).map(lambda k: k / 1000.0)


@settings(max_examples=40)
@given(scores=st.lists(_GRID, min_size=20, max_size=60), shift=_GRID)
def test_metrics_are_bounded_and_invariant_to_monotone_transform(scores, shift):
    p = np.array(scores)
    y = (np.arange(len(p)) % 4 == 0).astype(int)
    m = compute_metrics(y, p, shift)
    for key in ("pr_auc", "roc_auc", "f1", "precision", "recall", "recall_at_k"):
        assert 0.0 <= m[key] <= 1.0
    m2 = compute_metrics(y, p**2, shift**2)  # strictly monotone on [0, 1]: same ranking
    for key in ("pr_auc", "roc_auc", "f1", "precision", "recall"):
        assert m[key] == pytest.approx(m2[key], abs=1e-9, nan_ok=True)
