"""Hybrid detector: a trained model stacked with the motif scanner on the validation slice."""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from aml_graphminer.cli import main
from aml_graphminer.detectors.bundle import TabularDetector, load_detector
from aml_graphminer.detectors.hybrid import (
    HybridDetector,
    apply_stacker,
    fit_stacker,
    stack_inputs,
)
from aml_graphminer.evaluation import run_benchmark
from aml_graphminer.exceptions import ConfigError, ModelError


@pytest.fixture(scope="module")
def base(synth_df):
    return TabularDetector.fit(synth_df, "logreg")


@pytest.fixture(scope="module")
def hybrid(synth_df, base):
    return HybridDetector.fit(synth_df, base)


def test_stacker_inputs_and_monotone_scores():
    x = stack_inputs(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.2, 0.0]))
    assert x.shape == (3, 3) and x[1, 0] == pytest.approx(0.0)
    assert x[:, 2].tolist() == [0.0, 1.0, 0.0] and np.isfinite(x).all()
    scores = apply_stacker(np.array([1.0, 2.0, 0.5]), -1.0, np.array([0.1, 0.9]), np.zeros(2))
    assert 0 < scores[0] < scores[1] < 1


def test_stacker_learns_that_motif_alerts_are_informative():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.05).astype(float)
    model = np.clip(0.3 * y + rng.random(4000) * 0.5, 0, 1)
    motif = np.where((y == 1) & (rng.random(4000) < 0.6), 0.8, 0.0)
    coef, _ = fit_stacker(y, model, motif)
    assert coef[0] > 0 and (coef[1] > 0 or coef[2] > 0)
    with pytest.raises(ConfigError, match="both classes"):
        fit_stacker(np.zeros(10), np.zeros(10), np.zeros(10))


def test_hybrid_detector_scores_saves_and_reloads(hybrid, synth_df, tmp_path):
    assert {"pr_auc", "f1"} <= set(hybrid.metrics)
    assert {"pr_auc", "f1"} <= set(hybrid.component_metrics["base"])
    assert {"pr_auc", "f1"} <= set(hybrid.component_metrics["motifs"])
    assert 0.0 <= hybrid.threshold <= 1.0
    info = hybrid.describe()
    assert info["model_type"] == "hybrid" and info["base"]["kind"] == "logreg"
    assert set(info["stacker"]) == {"model_logit", "motif_score", "motif_fired"}

    rows = synth_df.iloc[-400:]
    context = synth_df.iloc[:-400]
    scores = hybrid.score(rows, context)
    assert scores.index.equals(rows.index) and scores.between(0, 1).all()
    assert set(hybrid.predict(rows, context).columns) == {"score", "alert"}
    assert hybrid.score(rows.iloc[:0]).empty

    hybrid.save(tmp_path / "h")
    assert (tmp_path / "h" / "base" / "config.json").exists()
    assert json.loads((tmp_path / "h" / "config.json").read_text())["model_type"] == "hybrid"
    again = load_detector(tmp_path / "h")
    assert isinstance(again, HybridDetector)
    assert np.allclose(again.score(rows, context), scores)
    assert again.component_metrics == hybrid.component_metrics


def test_hybrid_needs_its_parts_and_labels(hybrid, synth_df, base, tmp_path):
    hybrid.save(tmp_path / "h")
    (tmp_path / "h" / "motifs" / "config.json").unlink()
    (tmp_path / "h" / "motifs").rmdir()
    with pytest.raises(ModelError, match="motifs"):
        load_detector(tmp_path / "h")
    unlabelled = synth_df.assign(label=np.nan)
    with pytest.raises(ConfigError, match="labelled"):
        HybridDetector.fit(unlabelled, base)


def test_cli_trains_a_hybrid_on_a_saved_model(tmp_path, synth_df, base):
    data = tmp_path / "t.csv"
    synth_df.to_csv(data)
    base.save(tmp_path / "base")
    runner = CliRunner()
    res = runner.invoke(
        main, ["train", str(data), "--model", "hybrid", "--out", str(tmp_path / "h")]
    )
    assert res.exit_code != 0 and "--base" in res.output
    res = runner.invoke(
        main,
        [
            "train",
            str(data),
            "--model",
            "hybrid",
            "--base",
            str(tmp_path / "base"),
            "--out",
            str(tmp_path / "h"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert isinstance(load_detector(tmp_path / "h"), HybridDetector)


def test_benchmark_runs_hybrid_models_and_reuses_scores(synth_df):
    report = run_benchmark(synth_df, models=("logreg", "hybrid-logreg"), seeds=(0, 1))
    res = report.results
    assert sorted(res["model"].unique()) == ["hybrid-logreg", "logreg"]
    assert len(res[res["model"] == "hybrid-logreg"]) == 2
    assert {"coef", "intercept"} <= set(report.fitted["hybrid-logreg"])
    with pytest.raises(ConfigError, match="hybrid-<model>"):
        run_benchmark(synth_df, models=("hybrid-patterns",))
