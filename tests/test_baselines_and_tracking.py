import numpy as np
import pytest

from aml_graphminer.detectors.baselines import BASELINE_KINDS, TabularBaseline
from aml_graphminer.exceptions import ConfigError, ModelError, OptionalDependencyError
from aml_graphminer.tracking import MLflowTracker, Tracker, get_tracker
from aml_graphminer.utils import has_module


@pytest.fixture(scope="module")
def xy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] > 1.0).astype(int)
    return X, y


@pytest.mark.parametrize("kind", [k for k in BASELINE_KINDS if k != "xgb"])
def test_baselines_fit_predict_save_load(kind, xy, tmp_path):
    X, y = xy
    model = TabularBaseline(kind, seed=0).fit(X, y, [f"f{i}" for i in range(5)])
    proba = model.predict_proba(X)
    assert proba.shape == (400,) and ((proba >= 0) & (proba <= 1)).all()
    assert proba[y == 1].mean() > proba[y == 0].mean() + 0.2
    path = model.save(tmp_path / "m" / "b.joblib")
    loaded = TabularBaseline.load(path)
    assert np.allclose(loaded.predict_proba(X), proba)
    assert loaded.columns == model.columns


def test_feature_importance_names(xy):
    X, y = xy
    cols = [f"f{i}" for i in range(5)]
    rf = TabularBaseline("rf").fit(X, y, cols)
    imp = rf.feature_importance()
    assert set(imp) == set(cols) and max(imp, key=imp.get) in ("f0", "f1")
    assert set(TabularBaseline("logreg").fit(X, y, cols).feature_importance()) == set(cols)
    assert TabularBaseline("hgb").fit(X, y).feature_importance() == {}


def test_baseline_errors(xy):
    X, _ = xy
    with pytest.raises(ConfigError, match="unknown baseline"):
        TabularBaseline("svm")
    with pytest.raises(ModelError, match="single class"):
        TabularBaseline("rf").fit(X, np.zeros(len(X)))
    with pytest.raises(ModelError, match="not fitted"):
        TabularBaseline("rf").predict_proba(X)
    with pytest.raises(ModelError, match="not fitted"):
        TabularBaseline("rf").feature_importance()


def test_xgb_baseline_or_install_hint(xy):
    X, y = xy
    if has_module("xgboost"):
        assert TabularBaseline("xgb", n_estimators=20).fit(X, y).predict_proba(X).shape == (400,)
    else:
        with pytest.raises(OptionalDependencyError, match="xgboost"):
            TabularBaseline("xgb").fit(X, y)


def test_xgb_is_saved_in_the_portable_json_format(xy, tmp_path):
    """XGBoost pickles are not portable across platforms: the model goes to a JSON sidecar."""
    pytest.importorskip("xgboost")
    import joblib

    X, y = xy
    model = TabularBaseline("xgb", n_estimators=20).fit(X, y)
    path = model.save(tmp_path / "baseline.joblib")
    assert (tmp_path / "baseline.joblib.xgb.json").exists()
    assert joblib.load(path)["model"] is None
    np.testing.assert_allclose(TabularBaseline.load(path).predict_proba(X), model.predict_proba(X))


def test_noop_tracker_and_context_manager(tmp_path):
    with Tracker() as t:
        t.log_params({"a": 1})
        t.log_metrics({"m": 0.5}, step=1)
        t.log_artifact(tmp_path)
    assert isinstance(get_tracker(), Tracker) and not isinstance(get_tracker("none"), MLflowTracker)


def test_unknown_tracking_backend():
    with pytest.raises(ConfigError):
        get_tracker("wandb")


def test_mlflow_tracker_when_installed(tmp_path):
    pytest.importorskip("mlflow")
    # the recommended SQLite backend ...
    db = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    with get_tracker("mlflow", experiment="t", tracking_uri=db) as t:
        t.log_params({"a": 1})
        t.log_metrics({"m": 0.5, "nan": float("nan")})
    # ... and a plain directory store, which recent MLflow versions only accept on request
    with get_tracker("mlflow", experiment="t", tracking_uri=(tmp_path / "runs").as_uri()) as t:
        t.log_metrics({"m": 0.25}, step=1)


def test_xgboost_baseline_builds_its_defaults(xy, monkeypatch):
    """The XGBoost branch is exercised with a stand-in module (xgboost is an optional extra)."""
    import sys
    import types

    from sklearn.ensemble import HistGradientBoostingClassifier

    captured = {}

    class FakeXGB(HistGradientBoostingClassifier):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(max_iter=20)

    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(XGBClassifier=FakeXGB))
    X, y = xy
    model = TabularBaseline("xgb", seed=3, max_depth=2).fit(X, y)
    assert captured["max_depth"] == 2 and captured["random_state"] == 3
    assert captured["scale_pos_weight"] == pytest.approx((len(y) - y.sum()) / y.sum())
    assert model.predict_proba(X).shape == (len(X),)
