import json

import pytest
import yaml

from aml_graphminer.config import AppConfig, default_config_yaml, load_config
from aml_graphminer.detectors.bundle import TabularDetector
from aml_graphminer.exceptions import ConfigError, ModelError
from aml_graphminer.hub import export_model, render_model_card
from aml_graphminer.tracking import MLflowTracker, Tracker


def test_default_config_is_valid_and_round_trips(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(default_config_yaml())
    cfg = AppConfig.from_yaml(path)
    assert cfg.gnn_config().kind == "sage" and cfg.feature_config().windows == (
        "1h",
        "24h",
        "7d",
        "all",
    )
    assert cfg.pattern_config().fan_in.min_counterparties == 5
    assert cfg.benchmark.fractions == (0.6, 0.2, 0.2)


def test_partial_config_keeps_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump(
            {"gnn": {"kind": "gat", "hidden": 32}, "benchmark": {"models": ["rf"], "seeds": [1, 2]}}
        )
    )
    cfg = load_config(path)
    assert cfg.gnn_config().kind == "gat" and cfg.gnn_config().hidden == 32
    assert cfg.benchmark.models == ["rf"] and cfg.benchmark.seeds == [1, 2]
    assert load_config(None).benchmark.models == ["patterns", "rf", "hgb"]


def test_file_holding_only_detector_sections_is_a_patterns_section(tmp_path):
    path = tmp_path / "patterns.yaml"
    path.write_text(
        yaml.safe_dump({"fan_in": {"min_counterparties": 9}, "cycle": {"enabled": False}})
    )
    cfg = load_config(path)
    assert cfg.pattern_config().fan_in.min_counterparties == 9
    assert cfg.pattern_config().cycle.enabled is False
    assert cfg.benchmark.models == ["patterns", "rf", "hgb"]  # other sections keep defaults
    nested = AppConfig.from_dict({"patterns": {"fan_in": {"min_counterparties": 9}}})
    assert nested.pattern_config().fan_in.min_counterparties == 9
    # mixing detector sections with other sections is still an error (ambiguous file)
    with pytest.raises(ConfigError):
        AppConfig.from_dict({"fan_in": {"min_counterparties": 9}, "gnn": {"kind": "gat"}})


@pytest.mark.parametrize(
    "content",
    [
        "gnn: {kind: transformer}",
        "gnn: {layers: 3}",
        "features: {windows: [1h], nope: 1}",
        "patterns: {fan_in: {min_counterparties: 0}}",
        "benchmark: {fractions: [0.5, 0.5]}",
        "unknown_section: 1",
        "- just\n- a list",
        "a: [unclosed",
    ],
)
def test_invalid_configs_are_rejected(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_config_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent.yaml")


def test_tracker_selection(tmp_path, monkeypatch):
    assert type(AppConfig().tracker()) is Tracker
    cfg = AppConfig.from_dict({"tracking": {"backend": "carrier-pigeon"}})
    with pytest.raises(ConfigError, match="tracking backend"):
        cfg.tracker()
    cfg = AppConfig.from_dict({"tracking": {"backend": "mlflow"}})
    from aml_graphminer.utils import has_module

    # never write an MLflow database into the working directory during the tests
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    if not has_module("mlflow"):
        with pytest.raises(Exception, match="mlflow"):
            cfg.tracker()
    else:  # pragma: no cover
        with cfg.tracker() as tracker:
            assert isinstance(tracker, MLflowTracker)
            assert tracker.tracking_uri.startswith("sqlite:///")


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("./mlruns", True),
        ("file:///tmp/mlruns", True),
        ("C:\\runs\\mlruns", True),
        ("sqlite:///mlflow.db", False),
        ("http://127.0.0.1:5000", False),
    ],
)
def test_file_store_detection(uri, expected):
    from aml_graphminer.tracking import is_file_store

    assert is_file_store(uri) is expected


def test_tracking_uri_resolution(monkeypatch):
    from aml_graphminer.tracking import DEFAULT_TRACKING_URI, resolve_tracking_uri

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri() == DEFAULT_TRACKING_URI
    assert resolve_tracking_uri("http://h:5000") == "http://h:5000"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///x.db")
    assert resolve_tracking_uri() == "sqlite:///x.db"


def test_model_card_content():
    card = render_model_card(
        {
            "model_type": "gnn",
            "gnn": {"kind": "sage"},
            "threshold": 0.31,
            "metrics": {"pr_auc": 0.5, "f1": 0.4, "ignored": 9},
            "library_version": "9.9",
        },
        "me/aml-sage",
    )
    assert card.startswith("---\nlibrary_name: aml-graphminer") and "me/aml-sage" in card
    assert "| pr_auc | 0.5000 |" in card and "ignored" not in card
    assert "v9.9" in card and "(sage)" in card
    empty = render_model_card({"model_type": "patterns"})
    assert "No evaluation metrics" in empty


def test_export_model_copies_files_and_writes_card(tmp_path, synth_df):
    det = TabularDetector.fit(synth_df, "hgb")
    det.save(tmp_path / "model")
    target = export_model(tmp_path / "model", tmp_path / "hub", repo_id="me/x")
    names = {p.name for p in target.iterdir()}
    assert {"config.json", "baseline.joblib", "README.md"} <= names
    assert json.loads((target / "config.json").read_text())["kind"] == "hgb"
    with pytest.raises(ModelError, match="differ"):
        export_model(tmp_path / "model", tmp_path / "model")
    with pytest.raises(ModelError, match="not a saved detector"):
        export_model(tmp_path / "hub" / "nothing", tmp_path / "o")
