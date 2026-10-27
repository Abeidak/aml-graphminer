import json

import pandas as pd
import pytest

from aml_graphminer.evaluation import ALL_MODELS, run_benchmark
from aml_graphminer.evaluation.elliptic import run_elliptic_benchmark
from aml_graphminer.exceptions import ConfigError, OptionalDependencyError
from aml_graphminer.synthetic import generate_transactions, make_elliptic_like
from aml_graphminer.tracking import Tracker


class Recorder(Tracker):
    def __init__(self):
        self.params, self.metrics = {}, []

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics, step=None):
        self.metrics.append(dict(metrics))


@pytest.fixture(scope="module")
def report(synth_df):
    return run_benchmark(
        synth_df, models=("patterns", "rf", "logreg"), seeds=(0, 1), dataset="synth"
    )


def test_report_contains_one_row_per_model_and_seed(report):
    counts = report.results.groupby("model").size().to_dict()
    assert counts == {"patterns": 1, "rf": 2, "logreg": 2}
    assert {"pr_auc", "f1", "precision", "recall", "recall_at_k", "run_seconds"} <= set(
        report.results.columns
    )
    assert (report.results["dataset"] == "synth").all()
    assert report.results["pr_auc"].between(0, 1).all()


def test_models_beat_chance_on_the_synthetic_benchmark(report, synth_df):
    prevalence = synth_df["label"].mean()
    best = report.results.groupby("model")["pr_auc"].mean()
    assert (best > 3 * prevalence).all()


def test_summary_markdown_and_per_pattern_recall(report):
    assert "rf" in report.to_markdown() and report.summary().shape[0] == 3
    assert report.per_pattern is not None and "recall_any" in report.per_pattern.columns
    assert report.config["split_sizes"]["train"] > report.config["split_sizes"]["val"]


def test_save_writes_all_artifacts(report, tmp_path):
    out = report.save(tmp_path / "run")
    names = {p.name for p in out.iterdir()}
    assert {
        "results.csv",
        "summary.csv",
        "summary.md",
        "config.json",
        "per_pattern_recall.csv",
    } <= names
    assert json.loads((out / "config.json").read_text())["models"] == ["patterns", "rf", "logreg"]
    assert len(pd.read_csv(out / "results.csv")) == 5


def test_tracker_receives_params_and_metrics(synth_df):
    rec = Recorder()
    run_benchmark(synth_df, models=("hgb",), tracker=rec)
    assert rec.params["dataset"] == "dataset" and rec.params["n_transactions"] == len(synth_df)
    assert any("hgb/pr_auc" in m for m in rec.metrics)


def test_node_features_can_be_disabled(synth_df):
    r = run_benchmark(synth_df, models=("hgb",), tabular_node_features=False)
    assert r.config["tabular_node_features"] is False


def test_validation_errors(tiny_df):
    with pytest.raises(ConfigError, match="unknown model"):
        run_benchmark(tiny_df, models=("transformer",))
    with pytest.raises(ConfigError, match="fully labelled"):
        run_benchmark(tiny_df.assign(label=float("nan")), models=("rf",))
    no_test_positive = tiny_df.copy()
    no_test_positive.loc[no_test_positive.index[int(0.8 * len(tiny_df)) :], "label"] = 0.0
    with pytest.raises(ConfigError, match="no laundering transaction"):
        run_benchmark(no_test_positive, models=("rf",))


def test_gnn_models_need_the_gnn_extra(synth_df):
    from aml_graphminer.utils import has_module

    assert "gcn" in ALL_MODELS
    if has_module("torch_geometric"):
        pytest.skip("the gnn extra is installed")
    with pytest.raises(OptionalDependencyError, match="aml-graphminer\\[gnn\\]"):
        run_benchmark(synth_df, models=("gcn",))


def test_elliptic_benchmark_baselines():
    data = make_elliptic_like(2500, n_features=12, seed=3)
    rep = run_elliptic_benchmark(data, models=("rf", "logreg"), seeds=(0, 1))
    assert set(rep.results["model"]) == {"rf", "logreg"} and len(rep.results) == 4
    assert (rep.results["pr_auc"] > 0.3).all()
    assert rep.config["split_sizes"]["test"] > 0
    with pytest.raises(ConfigError, match="unknown model"):
        run_elliptic_benchmark(data, models=("nope",))


def test_elliptic_benchmark_needs_positives_in_every_split():
    data = make_elliptic_like(400, n_features=5, illicit_rate=0.01, seed=1)
    with pytest.raises(ConfigError, match="no illicit node"):
        run_elliptic_benchmark(data, models=("rf",))


def test_generated_data_can_be_benchmarked_end_to_end():
    df = generate_transactions(300, 1500, illicit_ratio=0.05, seed=21)
    assert len(run_benchmark(df, models=("hgb",)).results) == 1
