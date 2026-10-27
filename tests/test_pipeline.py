import json

import numpy as np
import pytest

from aml_graphminer.evaluation.splits import temporal_split
from aml_graphminer.exceptions import ConfigError, ModelError
from aml_graphminer.features import FeatureConfig, FeaturePipeline


@pytest.fixture(scope="module")
def fitted(synth_df, synth_masks):
    return FeaturePipeline().fit(synth_df, synth_masks["train"])


def test_transform_shapes_and_finiteness(synth_df, synth_masks, fitted):
    fs = fitted.transform(synth_df, node_rows=synth_masks["train"])
    assert fs.edge.shape == (len(synth_df), len(fs.edge_cols))
    assert fs.node.shape == (fs.n_nodes, len(fs.node_cols))
    assert fs.edge.dtype == np.float32 and fs.node.dtype == np.float32
    assert np.isfinite(fs.edge).all() and np.isfinite(fs.node).all()
    assert fs.tabular().shape[1] == fs.edge.shape[1] + 2 * fs.node.shape[1]
    assert fs.tabular(with_nodes=False).shape[1] == fs.edge.shape[1]
    assert len(fs.tabular_columns()) == fs.tabular().shape[1]
    assert fs.edge_frame().shape == fs.edge.shape and fs.node_frame().shape == fs.node.shape


def test_standardisation_uses_training_statistics(synth_df, synth_masks, fitted):
    fs = fitted.transform(synth_df)
    j = fs.edge_cols.index("log_paid")
    train = fs.edge[synth_masks["train"], j]
    assert abs(train.mean()) < 1e-3 and train.std() == pytest.approx(1.0, abs=1e-2)


def test_json_roundtrip_gives_identical_features(synth_df, synth_masks, fitted):
    stats = json.loads(json.dumps(fitted.to_dict()))
    clone = FeaturePipeline.from_dict(stats)
    a = fitted.transform_splits(synth_df, synth_masks)
    b = clone.transform_splits(synth_df, synth_masks)
    for split in a:
        assert np.allclose(a[split].edge, b[split].edge)
        assert np.allclose(a[split].node, b[split].node)


def test_no_leakage_from_validation_and_test_into_training_features(synth_df, synth_masks, fitted):
    base = fitted.transform_splits(synth_df, synth_masks)
    future = synth_masks["val"] | synth_masks["test"]
    altered = synth_df.copy()
    altered.loc[future, ["amount", "amount_received"]] *= 100.0
    changed = fitted.transform_splits(altered, synth_masks)
    assert np.allclose(base["train"].node, changed["train"].node)
    train = synth_masks["train"]
    assert np.allclose(base["train"].edge[train], changed["train"].edge[train], atol=1e-4)
    # the test snapshot does see the (modified) future, as it should
    assert not np.allclose(base["test"].node, changed["test"].node)


def test_transform_splits_graph_rows_are_nested(synth_df, synth_masks, fitted):
    parts = fitted.transform_splits(synth_df, synth_masks)
    assert set(parts) == {"train", "val", "test"}
    assert parts["train"].node.shape == parts["test"].node.shape
    only = fitted.transform_splits(
        synth_df,
        {"train": synth_masks["train"], "val": synth_masks["val"], "test": synth_masks["test"]},
    )
    assert only["val"].n_edges == len(synth_df)


def test_unseen_payment_format_is_ignored_at_inference(synth_df, synth_masks, fitted):
    other = synth_df.copy()
    other["transaction_type"] = "Teleport"
    fs = fitted.transform(other)
    cols = [c for c in fs.edge_cols if c.startswith("fmt_")]
    assert cols and (fs.edge[:, [fs.edge_cols.index(c) for c in cols]] == 0).all()


def test_fit_transform_and_unfitted_errors(tiny_df):
    pipe = FeaturePipeline(FeatureConfig(windows=("1h", "all"), pagerank=False))
    with pytest.raises(ModelError, match="not fitted"):
        pipe.transform(tiny_df)
    with pytest.raises(ModelError, match="not fitted"):
        pipe.to_dict()
    fs = pipe.fit_transform(tiny_df, np.ones(len(tiny_df), dtype=bool))
    assert "pagerank" not in fs.node_cols
    assert any(c.endswith("_all") for c in fs.edge_cols)


def test_fit_on_zero_rows_rejected(tiny_df):
    with pytest.raises(ConfigError, match="zero rows"):
        FeaturePipeline().fit(tiny_df, np.zeros(len(tiny_df), dtype=bool))


def test_from_dict_validates_version_and_config(fitted):
    stats = fitted.to_dict()
    with pytest.raises(ModelError, match="version"):
        FeaturePipeline.from_dict({**stats, "version": 99})
    with pytest.raises(ConfigError, match="unknown FeatureConfig"):
        FeatureConfig.from_dict({"nope": 1})


def test_temporal_features_can_be_disabled(tiny_df):
    pipe = FeaturePipeline(FeatureConfig(temporal=False, pagerank=False)).fit(tiny_df)
    fs = pipe.transform(tiny_df)
    assert not any("_cnt_" in c for c in fs.edge_cols)


def test_feature_config_roundtrip():
    cfg = FeatureConfig(windows=("30min", "all"), betweenness=True, betweenness_k=7)
    assert FeatureConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


def test_split_masks_are_used_by_default_masks(tiny_df):
    masks = temporal_split(tiny_df)
    fs = FeaturePipeline().fit(tiny_df, masks["train"]).transform(tiny_df, node_rows=masks["train"])
    assert fs.n_edges == len(tiny_df)
