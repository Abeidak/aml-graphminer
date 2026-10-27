import pytest
import yaml

from aml_graphminer.detectors.patterns import PatternConfig
from aml_graphminer.exceptions import ConfigError


def test_defaults_and_roundtrip(tmp_path):
    cfg = PatternConfig()
    assert cfg.smurfing.threshold == 10_000.0
    assert set(cfg.enabled_names()) == {
        "fan_in",
        "fan_out",
        "smurfing",
        "cycle",
        "scatter_gather",
        "gather_scatter",
        "community",
    }
    path = tmp_path / "p.yaml"
    path.write_text(cfg.to_yaml())
    assert PatternConfig.from_yaml(path) == cfg


def test_overrides_from_dict_and_nested_yaml(tmp_path):
    cfg = PatternConfig.from_dict(
        {"fan_out": {"min_counterparties": 8, "window": "2h"}, "cycle": {"enabled": False}}
    )
    assert cfg.fan_out.min_counterparties == 8 and cfg.fan_out.window == "2h"
    assert "cycle" not in cfg.enabled_names()
    path = tmp_path / "shared.yaml"
    path.write_text(yaml.safe_dump({"patterns": {"smurfing": {"threshold": 5000}}, "other": 1}))
    assert PatternConfig.from_yaml(path).smurfing.threshold == 5000


@pytest.mark.parametrize(
    "bad",
    [
        {"fan_in": {"min_counterparties": 1}},
        {"fan_in": {"window": "soon"}},
        {"fan_in": {"window": "-3h"}},
        {"smurfing": {"band": 1.5}},
        {"cycle": {"max_length": 2}},
        {"unknown_section": {}},
        {"fan_out": {"typo_key": 1}},
    ],
)
def test_invalid_configuration_is_rejected(bad):
    with pytest.raises(ConfigError, match="invalid pattern configuration"):
        PatternConfig.from_dict(bad)


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        PatternConfig.from_yaml("/nonexistent/patterns.yaml")
