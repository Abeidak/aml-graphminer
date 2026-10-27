import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aml_graphminer.features.network import index_accounts
from aml_graphminer.features.temporal import (
    previous_gap,
    temporal_feature_names,
    temporal_features,
    trailing_window_stats,
)
from aml_graphminer.utils import epoch_seconds


def brute_force(groups, ts, values, window):
    """Reference implementation; identical timestamps are ordered by row position."""
    n = len(ts)
    cnt, tot = np.zeros(n), np.zeros(n)
    for i in range(n):
        for j in range(n):
            if groups[j] != groups[i]:
                continue
            earlier = ts[j] < ts[i] or (ts[j] == ts[i] and j <= i)
            if earlier and ts[j] > ts[i] - window:
                cnt[i] += 1
                tot[i] += values[j]
    return cnt, tot


@settings(max_examples=40)
@given(
    n=st.integers(min_value=1, max_value=40),
    n_groups=st.integers(min_value=1, max_value=4),
    window=st.floats(min_value=1.0, max_value=500.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_trailing_window_matches_brute_force(n, n_groups, window, seed):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, n_groups, n)
    ts = rng.integers(0, 600, n).astype(float)  # integers → plenty of ties
    values = rng.uniform(0, 100, n)
    got = trailing_window_stats(groups, ts, values, window)
    cnt, tot = brute_force(groups, ts, values, window)
    assert np.allclose(got.count, cnt)
    assert np.allclose(got.total, tot)
    assert (got.count >= 1).all()


def test_window_mean_and_std():
    ws = trailing_window_stats(
        np.zeros(4, dtype=int), np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0]), 10.0
    )
    assert ws.mean.tolist() == pytest.approx([1.0, 1.5, 2.0, 2.5])
    assert ws.std[3] == pytest.approx(np.std([1, 2, 3, 4]))
    assert ws.std[0] == 0.0


def test_window_must_be_positive_and_empty_input():
    with pytest.raises(ValueError, match="window"):
        trailing_window_stats(np.zeros(1, dtype=int), np.zeros(1), np.zeros(1), 0)
    empty = trailing_window_stats(np.zeros(0, dtype=int), np.zeros(0), np.zeros(0), 5)
    assert len(empty.count) == 0


def test_window_is_causal_future_rows_never_change_the_past():
    rng = np.random.default_rng(1)
    g = rng.integers(0, 3, 60)
    t = np.sort(rng.uniform(0, 1000, 60))
    v = rng.uniform(0, 10, 60)
    full = trailing_window_stats(g, t, v, 100.0)
    head = trailing_window_stats(g[:30], t[:30], v[:30], 100.0)
    assert np.allclose(full.count[:30], head.count)
    assert np.allclose(full.total[:30], head.total)


def test_variance_is_numerically_stable_for_large_offsets():
    values = 1e9 + np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ws = trailing_window_stats(np.zeros(5, dtype=int), np.arange(5.0), values, 100.0)
    assert ws.std[4] == pytest.approx(np.std(values), rel=1e-6)


def test_previous_gap():
    g = np.array([0, 1, 0, 0])
    t = np.array([0.0, 5.0, 10.0, 40.0])
    assert previous_gap(g, t, default=999.0).tolist() == [999.0, 999.0, 10.0, 30.0]
    assert len(previous_gap(np.zeros(0, dtype=int), np.zeros(0), 1.0)) == 0


def test_temporal_features_shape_names_and_causality(tiny_df):
    src, dst, _, _ = index_accounts(tiny_df)
    feats = temporal_features(tiny_df, src, dst, ("1h", "24h", "all"))
    assert list(feats.columns) == temporal_feature_names(("1h", "24h", "all"))
    assert feats.shape == (len(tiny_df), 2 * (3 * 4 + 1))
    assert np.isfinite(feats.to_numpy()).all()
    half = len(tiny_df) // 2
    head = tiny_df.iloc[:half]
    s2, d2, _, _ = index_accounts(head)
    # account ids are re-factorised on the head, so compare the account-independent counts
    feats_head = temporal_features(head, s2, d2, ("1h", "24h", "all"))
    cols = [c for c in feats.columns if "_cnt_" in c or "_sum_" in c]
    assert np.allclose(feats.iloc[:half][cols].to_numpy(), feats_head[cols].to_numpy(), atol=1e-9)


def test_all_window_is_expanding_history():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-03-01"]),
            "amount": [10.0, 10.0, 10.0],
            "amount_received": [10.0, 10.0, 10.0],
        }
    )
    src = np.array([0, 0, 0])
    dst = np.array([1, 1, 1])
    feats = temporal_features(df, src, dst, ("1h", "all"))
    assert np.expm1(feats["s_cnt_all"]).round().tolist() == [1.0, 2.0, 3.0]
    assert np.expm1(feats["s_cnt_1h"]).round().tolist() == [1.0, 1.0, 1.0]
    assert epoch_seconds(df["timestamp"]).shape == (3,)
