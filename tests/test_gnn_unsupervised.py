"""Unsupervised graph auto-encoder: training without labels, scoring, persistence, interfaces.

Tiny networks and 1-2 epochs: these tests check plumbing and invariants, not accuracy.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from aml_graphminer.detectors.bundle import load_detector  # noqa: E402
from aml_graphminer.detectors.gnn import (  # noqa: E402
    AutoEncoderDetector,
    EdgeAutoEncoder,
    GNNConfig,
    run_gae_experiment,
)
from aml_graphminer.detectors.gnn.unsupervised import (  # noqa: E402
    DEFAULT_ALPHA,
    fit_autoencoder,
    transaction_errors,
)
from aml_graphminer.evaluation import run_benchmark  # noqa: E402
from aml_graphminer.exceptions import ConfigError, ModelError  # noqa: E402
from aml_graphminer.features import FeaturePipeline  # noqa: E402

pytestmark = pytest.mark.gnn


def tiny(kind="sage", **over):
    base = dict(
        kind=kind,
        hidden=16,
        layers=2,
        num_neighbors=[5, 3],
        batch_size=512,
        epochs=2,
        device="cpu",
        gat_heads=2,
    )
    base.update(over)
    return GNNConfig(**base)


@pytest.fixture(scope="module")
def detector(synth_df):
    return AutoEncoderDetector.fit(synth_df, tiny(), seed=0)


def test_model_shapes():
    model = EdgeAutoEncoder("sage", in_node=4, in_edge=3, hidden=8)
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    h = model.encode(x, edge_index)
    pairs = torch.tensor([[0, 1], [1, 2]])
    attr, surprise = model.errors(h, pairs, torch.randn(2, 3))
    assert h.shape == (5, 8) and attr.shape == (2,) and surprise.shape == (2,)
    assert (attr >= 0).all() and (surprise >= 0).all()


def test_scores_are_percentiles_and_reproducible(detector, synth_df):
    sample = synth_df.iloc[3000:3300]
    scores = detector.score(sample)
    assert scores.index.equals(sample.index) and scores.between(0, 1).all()
    assert np.allclose(scores, detector.score(sample))
    shuffled = sample.sample(frac=1.0, random_state=2)
    again = detector.score(shuffled).reindex(sample.index)
    assert np.allclose(scores, again)
    pred = detector.predict(sample)
    assert list(pred.columns) == ["score", "alert"]
    assert detector.score(sample.iloc[:0]).empty


def test_describe_threshold_and_metrics(detector):
    info = detector.describe()
    assert info["model_type"] == "gnn_unsupervised" and info["parameters"] > 0
    assert info["alpha"] == DEFAULT_ALPHA and 0 < detector.threshold < 1
    assert "pr_auc" in detector.metrics  # synth_df is labelled: metrics are reported


def test_trained_without_any_label(synth_df):
    unlabelled = synth_df.assign(label=np.nan)
    det = AutoEncoderDetector.fit(unlabelled, tiny(epochs=1), contamination=0.02)
    assert det.threshold == pytest.approx(0.98) and det.metrics == {}
    assert det.score(unlabelled.iloc[:50]).between(0, 1).all()


def test_save_load_roundtrip(detector, synth_df, tmp_path):
    detector.save(tmp_path / "gae")
    sample = synth_df.iloc[4000:4200]
    for restored in (AutoEncoderDetector.load(tmp_path / "gae"), load_detector(tmp_path / "gae")):
        assert isinstance(restored, AutoEncoderDetector)
        assert np.allclose(restored.score(sample), detector.score(sample))
        assert restored.threshold == pytest.approx(detector.threshold)
    with pytest.raises(ModelError, match="does not contain"):
        AutoEncoderDetector.load(tmp_path)


def test_full_inference_mode_and_gat_encoder(synth_df, synth_masks):
    pipe = FeaturePipeline().fit(synth_df, synth_masks["train"])
    feats = pipe.transform_splits(synth_df, synth_masks)
    out = run_gae_experiment(
        synth_df, feats, synth_masks, tiny("gat", epochs=1, inference="full"), seed=1
    )
    assert out["scores_test"].shape == (synth_masks["test"].sum(),)
    trained = out["trained"]
    ids, attr, surprise = transaction_errors(
        trained.model,
        feats["test"],
        np.ones(len(synth_df), dtype=bool),
        synth_masks["test"],
        trained.config,
        torch.device("cpu"),
    )
    assert np.array_equal(ids, np.flatnonzero(synth_masks["test"]))
    assert attr.shape == surprise.shape == ids.shape


def test_invalid_settings(synth_df, synth_masks):
    pipe = FeaturePipeline().fit(synth_df, synth_masks["train"])
    feats = pipe.transform_splits(synth_df, synth_masks)
    with pytest.raises(ConfigError, match="alpha"):
        fit_autoencoder(feats, synth_masks, tiny(), alpha=2.0)
    with pytest.raises(ConfigError, match="trim"):
        fit_autoencoder(feats, synth_masks, tiny(), trim=0.7)
    with pytest.raises(ConfigError, match="contamination"):
        AutoEncoderDetector.fit(synth_df, tiny(), contamination=0.0)
    empty = np.zeros(len(synth_df), dtype=bool)
    with pytest.raises(ConfigError, match="empty"):
        transaction_errors(
            EdgeAutoEncoder("sage", feats["train"].node.shape[1], feats["train"].edge.shape[1], 8),
            feats["train"],
            synth_masks["train"],
            empty,
            tiny(),
            torch.device("cpu"),
        )


def test_benchmark_and_explanation(synth_df, detector):
    report = run_benchmark(synth_df, models=("gae",), gnn_config=tiny(epochs=1))
    assert list(report.results["model"]) == ["gae"]
    assert report.results["pr_auc"].between(0, 1).all()

    from aml_graphminer.explainers import OcclusionExplainer, TransactionIndex, detector_score_fn

    tx = int(synth_df.index[3100])
    explanation = OcclusionExplainer(
        detector_score_fn(detector), hops=1, max_edges=20, max_occlusions=4
    ).explain(synth_df, tx, TransactionIndex(synth_df))
    assert explanation.edges[0].tx_id == tx and 0 <= explanation.score <= 1


def test_cli_trains_an_unsupervised_model_on_unlabelled_data(synth_df, tmp_path):
    from click.testing import CliRunner

    from aml_graphminer.cli import main

    path = tmp_path / "unlabelled.csv"
    synth_df.drop(columns=["label"]).iloc[:3000].to_csv(path, index=False)
    runner = CliRunner()
    res = runner.invoke(
        main,
        ["train", str(path), "--model", "gae", "--epochs", "1", "--out", str(tmp_path / "m")],
    )
    assert res.exit_code == 0, res.output
    assert '"model_type": "gnn_unsupervised"' in res.output
    scored = runner.invoke(main, ["score", str(tmp_path / "m"), str(path), "--top", "3"])
    assert scored.exit_code == 0, scored.output
    # a supervised GNN still refuses unlabelled data
    res = runner.invoke(
        main, ["train", str(path), "--model", "sage", "--epochs", "1", "--out", str(tmp_path / "s")]
    )
    assert res.exit_code != 0 and "labelled" in res.output


def test_scores_of_new_data_with_context(detector, synth_df):
    target = synth_df.iloc[4500:4600]
    alone = detector.score(target)
    with_context = detector.score(target, context=synth_df.iloc[3500:4500])
    assert isinstance(with_context, pd.Series) and not np.allclose(alone, with_context)
