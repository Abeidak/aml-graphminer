import numpy as np
import pytest

from aml_graphminer.ingestion.schema import CANONICAL_COLUMNS
from aml_graphminer.synthetic import PATTERN_TYPES, generate_transactions


def test_generator_is_deterministic_and_seed_sensitive():
    a = generate_transactions(200, 800, seed=1)
    b = generate_transactions(200, 800, seed=1)
    c = generate_transactions(200, 800, seed=2)
    assert a.equals(b)
    assert not a.equals(c)


def test_canonical_schema_and_ground_truth(tiny_df):
    assert list(tiny_df.columns[: len(CANONICAL_COLUMNS)]) == list(CANONICAL_COLUMNS)
    assert tiny_df["timestamp"].is_monotonic_increasing
    assert tiny_df["label"].isin([0.0, 1.0]).all()
    illicit = tiny_df[tiny_df["label"] == 1.0]
    assert set(illicit["pattern"]) <= set(PATTERN_TYPES)
    assert (tiny_df.loc[tiny_df["label"] == 0.0, "pattern"] == "").all()
    assert (tiny_df.loc[tiny_df["label"] == 0.0, "attempt_id"] == -1).all()
    assert (illicit["attempt_id"] >= 0).all()


def test_all_motifs_appear_in_a_larger_sample(synth_df):
    assert set(synth_df.loc[synth_df["label"] == 1, "pattern"]) == set(PATTERN_TYPES)


def test_illicit_ratio_is_respected_roughly():
    df = generate_transactions(600, 6000, illicit_ratio=0.05, seed=4)
    assert 0.025 < df["label"].mean() < 0.09


def test_custom_pattern_counts():
    df = generate_transactions(300, 1500, patterns={"cycle": 3}, seed=1)
    assert set(df.loc[df["label"] == 1, "pattern"]) == {"cycle"}
    assert df.loc[df["label"] == 1, "attempt_id"].nunique() == 3


def test_unknown_pattern_rejected():
    with pytest.raises(Exception, match="pattern"):
        generate_transactions(100, 300, patterns={"nope": 1})


def test_hard_negatives_can_be_disabled():
    with_hn = generate_transactions(300, 1500, seed=1, hard_negatives=True)
    without = generate_transactions(300, 1500, seed=1, hard_negatives=False)
    assert len(with_hn) > len(without)


def test_amounts_and_accounts_are_sane(tiny_df):
    assert (tiny_df["amount"] >= 0).all() and np.isfinite(tiny_df["amount"]).all()
    assert (tiny_df["sender"] != "").all()
    assert tiny_df["sender_bank"].ne("").all()
