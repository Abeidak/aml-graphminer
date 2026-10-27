import numpy as np
import pytest

from aml_graphminer.exceptions import IngestionError
from aml_graphminer.ingestion import read_elliptic
from aml_graphminer.ingestion.elliptic import ILLICIT, LICIT, UNKNOWN
from aml_graphminer.synthetic import make_elliptic_like, write_elliptic_files


@pytest.fixture(scope="module")
def data():
    return make_elliptic_like(1500, n_features=10, seed=2)


def test_roundtrip_through_kaggle_layout(tmp_path, data):
    write_elliptic_files(data, tmp_path / "ell")
    back = read_elliptic(tmp_path / "ell")
    assert np.array_equal(back.node_ids, data.node_ids)
    assert np.array_equal(back.labels, data.labels)
    assert np.array_equal(back.edge_index, data.edge_index)
    assert np.allclose(back.features, data.features, atol=1e-5)


def test_read_finds_files_in_subdirectories(tmp_path, data):
    write_elliptic_files(data, tmp_path / "a" / "b")
    assert read_elliptic(tmp_path).num_nodes == data.num_nodes
    assert read_elliptic(tmp_path, nrows=100).num_nodes == 100


def test_read_errors(tmp_path):
    with pytest.raises(IngestionError, match="not a directory"):
        read_elliptic(tmp_path / "missing")
    with pytest.raises(IngestionError, match="not found"):
        read_elliptic(tmp_path)


def test_split_masks_are_temporal_disjoint_and_labelled(data):
    masks = data.split_masks()
    known = data.labels != UNKNOWN
    assert not (masks["train"] & masks["val"]).any()
    assert not (masks["val"] & masks["test"]).any()
    assert (masks["train"] | masks["val"] | masks["test"]).sum() == known.sum()
    assert data.time_step[masks["train"]].max() <= 29 < data.time_step[masks["val"]].min()
    assert data.time_step[masks["test"]].min() > 34


def test_summary_and_label_values(data):
    s = data.summary()
    assert s["nodes"] == 1500 and s["features"] == 10
    assert s["illicit"] + s["licit"] == s["labelled"]
    assert set(np.unique(data.labels)) <= {ILLICIT, LICIT, UNKNOWN}
    assert data.num_edges == data.edge_index.shape[1]
