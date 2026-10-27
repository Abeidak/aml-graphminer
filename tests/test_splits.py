import numpy as np
import pytest

from aml_graphminer.evaluation.splits import temporal_split
from aml_graphminer.exceptions import ConfigError


def test_split_is_disjoint_complete_and_temporal(synth_df):
    masks = temporal_split(synth_df)
    total = sum(m.sum() for m in masks.values())
    assert total == len(synth_df)
    assert not (masks["train"] & masks["val"]).any() and not (masks["val"] & masks["test"]).any()
    ts = synth_df["timestamp"]
    assert ts[masks["train"]].max() <= ts[masks["val"]].min()
    assert ts[masks["val"]].max() <= ts[masks["test"]].min()
    assert masks["train"].sum() == int(0.6 * len(synth_df))


@pytest.mark.parametrize(
    "fractions", [(0.5, 0.5), (0.6, 0.3, 0.2), (0.0, 0.5, 0.5), (1.0, 0.0, 0.0)]
)
def test_bad_fractions(synth_df, fractions):
    with pytest.raises(ConfigError, match="fractions"):
        temporal_split(synth_df, fractions)  # type: ignore[arg-type]


def test_unsorted_table_rejected(synth_df):
    shuffled = synth_df.sample(frac=1.0, random_state=0)
    with pytest.raises(ConfigError, match="sorted"):
        temporal_split(shuffled)


def test_custom_fractions(synth_df):
    masks = temporal_split(synth_df, (0.8, 0.1, 0.1))
    assert masks["train"].sum() == int(0.8 * len(synth_df))
    assert np.isclose(masks["test"].mean(), 0.1, atol=0.01)
