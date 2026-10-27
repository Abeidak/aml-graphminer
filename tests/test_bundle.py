import json

import numpy as np
import pandas as pd
import pytest

from aml_graphminer.detectors.bundle import (
    MotifDetector,
    TabularDetector,
    load_detector,
    read_bundle_config,
)
from aml_graphminer.exceptions import ConfigError, ModelError


@pytest.fixture(scope="module")
def trained(synth_df):
    return {
        "patterns": MotifDetector.fit(synth_df),
        "rf": TabularDetector.fit(synth_df, "rf"),
        "logreg": TabularDetector.fit(synth_df, "logreg", with_nodes=False),
    }


@pytest.mark.parametrize("name", ["patterns", "rf", "logreg"])
def test_save_load_roundtrip_gives_identical_scores(name, trained, synth_df, tmp_path):
    det = trained[name]
    det.save(tmp_path / name)
    back = load_detector(tmp_path / name)
    test = synth_df.iloc[int(0.8 * len(synth_df)) :]
    assert type(back) is type(det)
    assert np.allclose(det.score(test), back.score(test))
    assert back.threshold == pytest.approx(det.threshold)
    assert back.describe()["model_type"] == det.model_type


@pytest.mark.parametrize("name", ["patterns", "rf"])
def test_predict_columns_and_alert_rule(name, trained, synth_df):
    out = trained[name].predict(synth_df.head(300))
    assert list(out.columns) == ["score", "alert"]
    assert out["alert"].tolist() == (out["score"] >= trained[name].threshold).astype(int).tolist()
    assert out.index.equals(synth_df.head(300).index)


def test_scores_use_context_and_keep_order(trained, synth_df):
    det = trained["rf"]
    target = synth_df.iloc[3000:3100]
    history = synth_df.iloc[:3000]
    with_ctx = det.score(target, history)
    without = det.score(target)
    assert len(with_ctx) == 100 and with_ctx.index.equals(target.index)
    assert not np.allclose(with_ctx, without)
    shuffled = target.sample(frac=1.0, random_state=0)
    assert np.allclose(det.score(shuffled, history).reindex(target.index), with_ctx)


def test_empty_input(trained, synth_df):
    for det in trained.values():
        assert len(det.score(synth_df.iloc[:0])) == 0


def test_motif_detector_reports_metrics_when_labelled(trained):
    assert trained["patterns"].metrics["pr_auc"] > 0.2
    assert trained["rf"].metrics["pr_auc"] > 0.5
    assert 0 < trained["rf"].threshold < 1


def test_motif_detector_without_labels_is_uncalibrated(synth_df):
    det = MotifDetector.fit(synth_df.assign(label=np.nan))
    assert det.scanner.calibration == {} and det.threshold == 0.5 and det.metrics == {}
    assert MotifDetector.fit().describe()["calibrated"] is False


def test_tabular_detector_validation(synth_df):
    with pytest.raises(ConfigError, match="unknown baseline"):
        TabularDetector.fit(synth_df, "svm")
    with pytest.raises(ConfigError, match="labelled"):
        TabularDetector.fit(synth_df.assign(label=np.nan))


def test_load_errors(tmp_path):
    with pytest.raises(ModelError, match="not a saved detector"):
        load_detector(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mystery"}))
    with pytest.raises(ModelError, match="unknown detector type"):
        load_detector(tmp_path)
    (tmp_path / "config.json").write_text("[1]")
    with pytest.raises(ModelError, match="not a valid"):
        read_bundle_config(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "tabular", "stats": {}}))
    with pytest.raises(ModelError):
        load_detector(tmp_path)


def test_config_json_is_human_readable(trained, tmp_path):
    trained["rf"].save(tmp_path / "m")
    payload = json.loads((tmp_path / "m" / "config.json").read_text())
    assert payload["model_type"] == "tabular" and payload["kind"] == "rf"
    assert {"threshold", "metrics", "stats", "library_version"} <= set(payload)
    assert isinstance(pd.Series(payload["metrics"]).get("pr_auc"), float)
