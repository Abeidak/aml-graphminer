"""GNN configuration and lazy imports — runs without torch."""

import pytest

from aml_graphminer.detectors.gnn import GNN_KINDS, GNNConfig
from aml_graphminer.exceptions import ConfigError


def test_defaults_match_the_original_notebooks():
    cfg = GNNConfig()
    assert (cfg.hidden, cfg.layers, cfg.dropout, cfg.lr, cfg.epochs) == (64, 2, 0.2, 1e-3, 10)
    assert cfg.num_neighbors == [10, 5] and cfg.batch_size == 4096 and cfg.pos_weight == 30.0


def test_dict_roundtrip_and_with_kind():
    cfg = GNNConfig(kind="gat", hidden=32, num_neighbors=[5, 5])
    assert GNNConfig.from_dict(cfg.to_dict()) == cfg
    other = cfg.with_kind("gcn")
    assert other.kind == "gcn" and other.hidden == 32 and cfg.kind == "gat"
    assert set(GNN_KINDS) == {"gcn", "sage", "gat"}


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "transformer"},
        {"loss": "mse"},
        {"trainer": "keras"},
        {"device": "tpu"},
        {"layers": 0},
        {"layers": 3},  # num_neighbors has 2 entries
        {"num_neighbors": [10, 0]},
        {"dropout": 1.0},
        {"kind": "gat", "hidden": 30},
        {"pos_weight": 0},
        {"epochs": 0},
        {"patience": -1},
        {"inference": "approximate"},
        {"nope": 1},
    ],
)
def test_invalid_configurations(bad):
    with pytest.raises(ConfigError):
        GNNConfig.from_dict(bad)


def test_inference_modes():
    from aml_graphminer.detectors.gnn import INFERENCE_MODES

    assert GNNConfig().inference == "sampled" and set(INFERENCE_MODES) == {"sampled", "full"}
    assert GNNConfig(inference="full").to_dict()["inference"] == "full"
    # checkpoints written before the option existed load with the default
    legacy = GNNConfig().to_dict()
    legacy.pop("inference")
    assert GNNConfig.from_dict(legacy).inference == "sampled"


def test_heavy_symbols_are_lazy_and_give_install_hint():
    import aml_graphminer.detectors.gnn as gnn
    from aml_graphminer.exceptions import OptionalDependencyError
    from aml_graphminer.utils import has_module

    assert "GNNDetector" in dir(gnn)
    with pytest.raises(AttributeError):
        getattr(gnn, "does_not_exist")  # noqa: B009
    if not has_module("torch"):
        with pytest.raises(OptionalDependencyError, match="aml-graphminer\\[gnn\\]"):
            getattr(gnn, "EdgeGNN")  # noqa: B009
