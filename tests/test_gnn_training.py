"""End-to-end GNN training, checkpoints, inference, node classification, explainers.

These tests need torch + torch_geometric and are skipped otherwise. They use tiny networks
and 1-3 epochs: they check *plumbing and invariants*, not accuracy.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from aml_graphminer.detectors.gnn import (  # noqa: E402
    GNNConfig,
    GNNDetector,
    fit_edge_gnn,
    load_checkpoint,
    run_gnn_experiment,
    run_node_experiment,
    save_checkpoint,
    score_transactions,
)
from aml_graphminer.detectors.gnn.data import (  # noqa: E402
    build_loaders,
    make_link_loader,
    resolve_device,
    split_graph_rows,
)
from aml_graphminer.detectors.gnn.trainer import predict_edges, score_splits  # noqa: E402
from aml_graphminer.evaluation import run_benchmark  # noqa: E402
from aml_graphminer.exceptions import ConfigError, ModelError  # noqa: E402
from aml_graphminer.features import FeaturePipeline  # noqa: E402

pytestmark = pytest.mark.gnn


def tiny_config(kind="sage", **over):
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
def prepared(synth_df, synth_masks):
    pipeline = FeaturePipeline().fit(synth_df, synth_masks["train"])
    return pipeline, pipeline.transform_splits(synth_df, synth_masks)


@pytest.fixture(scope="module")
def experiment(synth_df, synth_masks, prepared):
    pipeline, features = prepared
    return run_gnn_experiment(
        "sage",
        synth_df,
        features,
        synth_masks,
        tiny_config(),
        seed=0,
        pipeline_stats=pipeline.to_dict(),
    )


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        with pytest.raises(ConfigError, match="CUDA"):
            resolve_device("cuda")


def test_split_graph_rows_are_nested(synth_masks):
    rows = split_graph_rows(synth_masks)
    assert (rows["train"] <= rows["val"]).all() and (rows["val"] <= rows["test"]).all()
    assert rows["test"].all()


def test_loader_serves_exactly_the_requested_transactions(synth_df, synth_masks, prepared):
    _, features = prepared
    cfg = tiny_config()
    rows = split_graph_rows(synth_masks)
    loader = make_link_loader(features["val"], rows["val"], synth_masks["val"], cfg)
    served = torch.cat([b.edge_label for b in loader]).sort().values.numpy()
    assert np.array_equal(served, np.flatnonzero(synth_masks["val"]))
    with pytest.raises(ConfigError, match="empty"):
        make_link_loader(features["val"], rows["val"], np.zeros(len(synth_df), dtype=bool), cfg)
    assert set(build_loaders(features, synth_masks, cfg)) == {"train", "val", "test"}


def test_fit_returns_a_model_with_history(synth_df, synth_masks, prepared):
    pipeline, features = prepared
    trained = fit_edge_gnn(
        features,
        synth_df["label"].to_numpy(),
        synth_masks,
        tiny_config(epochs=2),
        pipeline_stats=pipeline.to_dict(),
    )
    assert len(trained.history) == 2 and trained.best_epoch in (1, 2)
    assert {"train_loss", "val_pr_auc"} <= set(trained.history[0])
    assert (
        trained.in_node == features["train"].node.shape[1]
        and trained.in_edge == features["train"].edge.shape[1]
    )
    assert not any(p.requires_grad is False for p in trained.model.parameters())


def test_run_experiment_outputs_are_aligned_with_the_splits(experiment, synth_masks):
    assert experiment["scores_val"].shape == (synth_masks["val"].sum(),)
    assert experiment["scores_test"].shape == (synth_masks["test"].sum(),)
    assert ((experiment["scores_test"] >= 0) & (experiment["scores_test"] <= 1)).all()
    trained = experiment["trained"]
    assert 0 < trained.threshold < 1 and "pr_auc" in trained.metrics


@pytest.mark.parametrize("kind", ["gcn", "gat"])
def test_other_architectures_train(kind, synth_df, synth_masks, prepared):
    pipeline, features = prepared
    out = run_gnn_experiment(
        kind,
        synth_df,
        features,
        synth_masks,
        tiny_config(epochs=1),
        pipeline_stats=pipeline.to_dict(),
    )
    assert (
        out["scores_test"].shape[0] == synth_masks["test"].sum()
        and out["trained"].config.kind == kind
    )


def test_training_is_reproducible_for_a_fixed_seed(synth_df, synth_masks, prepared):
    _, features = prepared
    cfg = tiny_config(epochs=1)
    a = run_gnn_experiment("sage", synth_df, features, synth_masks, cfg, seed=3)
    b = run_gnn_experiment("sage", synth_df, features, synth_masks, cfg, seed=3)
    assert np.allclose(a["scores_test"], b["scores_test"], atol=1e-5)


def test_focal_loss_and_early_stopping_options(synth_df, synth_masks, prepared):
    _, features = prepared
    out = run_gnn_experiment(
        "sage", synth_df, features, synth_masks, tiny_config(loss="focal", epochs=4, patience=1)
    )
    assert len(out["trained"].history) <= 4


def test_checkpoint_roundtrip_preserves_scores(experiment, synth_df, tmp_path):
    trained = experiment["trained"]
    save_checkpoint(trained, tmp_path / "ckpt")
    back = load_checkpoint(tmp_path / "ckpt")
    assert back.threshold == pytest.approx(trained.threshold) and back.config == trained.config
    sample = synth_df.iloc[3000:3300]
    assert np.allclose(
        score_transactions(trained, sample), score_transactions(back, sample), atol=1e-5
    )


def test_load_checkpoint_errors(tmp_path):
    with pytest.raises(ModelError, match="does not contain"):
        load_checkpoint(tmp_path)


def test_scoring_is_independent_of_row_order_and_uses_context(experiment, synth_df):
    trained = experiment["trained"]
    target = synth_df.iloc[3000:3200]
    scores = score_transactions(trained, target)
    assert scores.shape == (200,) and ((scores >= 0) & (scores <= 1)).all()
    shuffled = target.sample(frac=1.0, random_state=1)
    again = pd.Series(score_transactions(trained, shuffled), index=shuffled.index).reindex(
        target.index
    )
    assert np.allclose(scores, again.to_numpy(), atol=1e-5)
    with_context = score_transactions(trained, target, context=synth_df.iloc[2000:3000])
    assert with_context.shape == (200,) and not np.allclose(scores, with_context)
    assert score_transactions(trained, target.iloc[:0]).shape == (0,)


def test_scoring_requires_feature_statistics(experiment, synth_df):
    from dataclasses import replace

    stripped = replace(experiment["trained"], stats={})
    with pytest.raises(ModelError, match="feature statistics"):
        score_transactions(stripped, synth_df.head(10))


def test_predict_edges_sorts_by_transaction_id(experiment, synth_df, synth_masks, prepared):
    _, features = prepared
    trained = experiment["trained"]
    loaders = build_loaders(features, synth_masks, trained.config, ("val",))
    ef = torch.from_numpy(features["val"].edge)
    ids, probs = predict_edges(trained.model, loaders["val"], ef, torch.device("cpu"))
    assert np.array_equal(ids, np.flatnonzero(synth_masks["val"])) and probs.shape == ids.shape
    again = score_splits(trained, features, synth_masks, ("val",), device="cpu")
    assert np.allclose(again["val"], probs, atol=1e-5)


def test_full_neighbourhood_inference_is_exact_and_batch_independent(
    synth_df, synth_masks, prepared
):
    pipeline, features = prepared
    out = run_gnn_experiment(
        "gcn",
        synth_df,
        features,
        synth_masks,
        tiny_config("gcn", epochs=1, inference="full"),
        pipeline_stats=pipeline.to_dict(),
    )
    trained = out["trained"]
    assert out["scores_test"].shape == (synth_masks["test"].sum(),)
    target = synth_df.iloc[3000:3200]
    whole = score_transactions(trained, target)
    # with full neighbourhoods a transaction's score does not depend on how it is batched
    from aml_graphminer.detectors.gnn.trainer import predict_edges_full

    feats = FeaturePipeline.from_dict(trained.stats).transform(target)
    graph = np.ones(len(target), dtype=bool)
    first_half = np.zeros(len(target), dtype=bool)
    first_half[:100] = True
    ids, probs = predict_edges_full(trained.model, feats, graph, first_half, torch.device("cpu"))
    assert np.array_equal(ids, np.arange(100))
    assert np.allclose(probs, whole[:100], atol=1e-5)
    with pytest.raises(ConfigError, match="empty"):
        predict_edges_full(trained.model, feats, graph, ~graph, torch.device("cpu"))


def test_scoring_does_not_disturb_the_global_random_stream(experiment, synth_df):
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    score_transactions(experiment["trained"], synth_df.iloc[3000:3100])
    assert torch.equal(torch.rand(3), expected)


def test_detector_fit_predict_save_load(synth_df, tmp_path):
    detector = GNNDetector.fit(synth_df, tiny_config(epochs=1))
    sample = synth_df.iloc[3000:3200]
    pred = detector.predict(sample)
    assert list(pred.columns) == ["score", "alert"] and pred.index.equals(sample.index)
    assert detector.top(sample, 5).shape[0] == 5 and "score" in detector.top(sample, 5)
    assert detector.describe()["parameters"] > 0
    detector.save(tmp_path / "d")
    from aml_graphminer.detectors.bundle import load_detector

    for restored in (GNNDetector.load(tmp_path / "d"), load_detector(tmp_path / "d")):
        assert np.allclose(restored.score(sample), detector.score(sample), atol=1e-5)
    with pytest.raises(ConfigError, match="labelled"):
        GNNDetector.fit(synth_df.assign(label=np.nan), tiny_config(epochs=1))


def test_lightning_trainer(synth_df, synth_masks, prepared):
    pytest.importorskip("lightning")
    from aml_graphminer.tracking import Tracker

    class Recorder(Tracker):
        def __init__(self) -> None:
            self.steps: list[tuple[int | None, dict]] = []

        def log_metrics(self, metrics, step=None):
            self.steps.append((step, dict(metrics)))

    pipeline, features = prepared
    recorder = Recorder()
    out = run_gnn_experiment(
        "sage",
        synth_df,
        features,
        synth_masks,
        tiny_config(trainer="lightning", epochs=2),
        pipeline_stats=pipeline.to_dict(),
        tracker=recorder,
    )
    assert out["scores_test"].shape[0] == synth_masks["test"].sum()
    # one entry per epoch, like the PyTorch loop: history, tracker steps, best epoch among them
    history = out["trained"].history
    assert [e["epoch"] for e in history] == [1.0, 2.0]
    assert all(e["train_loss"] > 0 and 0 <= e["val_pr_auc"] <= 1 for e in history)
    per_epoch = [step for step, m in recorder.steps if "sage/train_loss" in m]
    assert per_epoch == [1, 2] and out["best_epoch"] in (1, 2)


def test_benchmark_runs_gnns(synth_df):
    report = run_benchmark(
        synth_df, models=("rf", "sage"), seeds=(0,), gnn_config=tiny_config(epochs=1)
    )
    assert set(report.results["model"]) == {"rf", "sage"}
    assert report.results["pr_auc"].between(0, 1).all()


def test_node_classification_on_an_elliptic_like_graph():
    from aml_graphminer.synthetic import make_elliptic_like

    data = make_elliptic_like(1200, n_features=10, seed=0)
    result = run_node_experiment("gcn", data, tiny_config("gcn", epochs=5), seed=0)
    assert result.scores.shape == (data.num_nodes,) and 0 < result.threshold < 1
    assert result.metrics["pr_auc"] > data.summary()["illicit_ratio_among_labelled"]
    assert result.best_epoch >= 1


def test_explainers_on_a_trained_detector(synth_df):
    from aml_graphminer.explainers import (
        AttentionExplainer,
        Explanation,
        GNNExplainerAdapter,
        OcclusionExplainer,
        TransactionIndex,
        detector_score_fn,
    )

    detector = GNNDetector.fit(synth_df, tiny_config("gat", epochs=1))
    index = TransactionIndex(synth_df)
    tx = int(synth_df.index[3100])

    occ = OcclusionExplainer(
        detector_score_fn(detector), hops=2, max_edges=30, max_occlusions=6
    ).explain(synth_df, tx, index)
    assert isinstance(occ, Explanation) and occ.edges[0].tx_id == tx

    att = AttentionExplainer(detector.trained, hops=2, max_edges=30).explain(synth_df, tx, index)
    assert (
        att.method == "attention"
        and att.edges
        and all(0 <= e.importance <= 1.0 + 1e-6 for e in att.edges)
    )

    pytest.importorskip("torch_geometric.explain")
    gnn_exp = GNNExplainerAdapter(detector.trained, hops=2, max_edges=30, epochs=10).explain(
        synth_df, tx, index
    )
    assert gnn_exp.method == "gnnexplainer" and len(gnn_exp.edges) == len(att.edges)
    assert 0.0 <= gnn_exp.score <= 1.0


def test_captum_and_pgexplainer(synth_df):
    pytest.importorskip("captum")
    from aml_graphminer.explainers import (
        CaptumExplainerAdapter,
        PGExplainerAdapter,
        TransactionIndex,
        available_methods,
        explain_transaction,
    )

    detector = GNNDetector.fit(synth_df, tiny_config("sage", epochs=1))
    index = TransactionIndex(synth_df)
    tx = int(synth_df.index[3100])

    captum = CaptumExplainerAdapter(detector.trained, hops=2, max_edges=30).explain(
        synth_df, tx, index
    )
    assert captum.method == "captum" and 0.0 <= captum.score <= 1.0
    names = [f.name for f in captum.features]
    assert len(names) == 10 and any(n.split(".")[0] in ("tx", "sender", "receiver") for n in names)
    assert "Most influential feature" in captum.summary
    with pytest.raises(ConfigError, match="attribution"):
        CaptumExplainerAdapter(detector.trained, attribution="LRP")

    pg = PGExplainerAdapter(detector.trained, hops=2, max_edges=30, epochs=3)
    with pytest.raises(ModelError, match="not fitted"):
        pg.explain(synth_df, tx, index)
    with pytest.raises(ConfigError, match="at least one"):
        pg.fit(synth_df, [], index)
    explained = pg.fit(synth_df, synth_df.index[3100:3104], index).explain(synth_df, tx, index)
    assert explained.method == "pgexplainer" and len(explained.edges) == len(captum.edges)
    assert all(0.0 <= e.importance <= 1.0 for e in explained.edges)
    assert "trained on 4" in explained.summary

    assert {"occlusion", "gnnexplainer", "pgexplainer", "captum"} <= set(
        available_methods(detector)
    )
    for method in ("captum", "pgexplainer", "gnnexplainer"):
        exp = explain_transaction(detector, synth_df, tx, method, max_edges=25, index=index)
        assert exp.method == method
        assert any(e.role == "target" and e.tx_id == tx for e in exp.edges)


def test_every_gnn_explanation_reports_the_measured_neighbourhood_effect(synth_df):
    """An edge mask is only meaningful if the neighbourhood changes the score: report by how much."""
    pytest.importorskip("captum")
    from aml_graphminer.explainers import TransactionIndex, explain_transaction
    from aml_graphminer.explainers.gnn_explainers import _local_view, neighbourhood_effect

    detector = GNNDetector.fit(synth_df, tiny_config("gat", epochs=1))
    index = TransactionIndex(synth_df)
    tx = int(synth_df.index[3100])
    view = _local_view(detector.trained, synth_df, tx, index, 2, 25)
    full, alone = neighbourhood_effect(detector.trained.model.eval(), view)
    assert 0.0 <= full <= 1.0 and 0.0 <= alone <= 1.0
    for method in ("gnnexplainer", "pgexplainer", "captum", "attention"):
        exp = explain_transaction(detector, synth_df, tx, method, max_edges=25, index=index)
        assert "measured neighbourhood effect" in exp.summary
        assert f"{full:.3f}" in f"{exp.score:.3f}" or abs(exp.score - full) < 1e-6


def test_explainers_can_run_in_any_order_on_one_model(synth_df):
    """GNNExplainer leaves `_edge_mask` registered as a parameter on the PyG layers, which used
    to detach the masks of the next explainers (Captum crashed, PGExplainer stopped learning)."""
    pytest.importorskip("captum")
    from aml_graphminer.explainers import (
        CaptumExplainerAdapter,
        GNNExplainerAdapter,
        PGExplainerAdapter,
        TransactionIndex,
    )

    detector = GNNDetector.fit(synth_df, tiny_config("gat", epochs=1))
    index = TransactionIndex(synth_df)
    tx = int(synth_df.index[3100])
    GNNExplainerAdapter(detector.trained, hops=2, max_edges=25, epochs=5).explain(
        synth_df, tx, index
    )
    assert not any("_edge_mask" in m._parameters for m in detector.trained.model.modules())
    CaptumExplainerAdapter(detector.trained, hops=2, max_edges=25).explain(synth_df, tx, index)
    pg = PGExplainerAdapter(detector.trained, hops=2, max_edges=25, epochs=2).fit(
        synth_df, [tx], index
    )
    grads = [p.grad for p in pg._explainer.algorithm.mlp.parameters()]
    assert all(g is not None for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0


def test_cli_trains_and_serves_a_gnn(tmp_path):
    from click.testing import CliRunner

    from aml_graphminer.cli import main

    runner = CliRunner()
    gen = runner.invoke(
        main,
        [
            "generate",
            str(tmp_path / "d"),
            "--accounts",
            "700",
            "--background",
            "5000",
            "--illicit-ratio",
            "0.04",
            "--seed",
            "3",
        ],
    )
    assert gen.exit_code == 0
    data = str(tmp_path / "d" / "Synth-Small_Trans.csv")
    res = runner.invoke(
        main, ["train", data, "--model", "sage", "--epochs", "1", "--out", str(tmp_path / "m")]
    )
    assert res.exit_code == 0, res.output
    scored = runner.invoke(main, ["score", str(tmp_path / "m"), data, "--top", "2"])
    assert scored.exit_code == 0, scored.output


def test_gnnexplainer_reports_the_account_features_it_keeps(synth_df):
    """The specification asks for the edges *and features* behind each GNN alert."""
    from aml_graphminer.explainers import GNNExplainerAdapter, TransactionIndex

    detector = GNNDetector.fit(synth_df, tiny_config("sage", epochs=1))
    tx = int(synth_df.index[3100])
    exp = GNNExplainerAdapter(detector.trained, max_edges=25, epochs=20, top_features=6).explain(
        synth_df, tx, TransactionIndex(synth_df)
    )
    assert 0 < len(exp.features) <= 6
    assert all(f.name.split(".")[0] in ("sender", "receiver") for f in exp.features)
    assert all(0.0 <= f.importance <= 1.0 for f in exp.features)  # mask weights
    weights = [f.importance for f in exp.features]
    assert weights == sorted(weights, reverse=True)
    assert "Account features kept by the mask" in exp.summary
    assert exp.to_dict()["features"][0]["name"] == exp.features[0].name  # exported in JSON
