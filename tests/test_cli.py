import json

import pytest
from click.testing import CliRunner

from aml_graphminer import __version__
from aml_graphminer.cli import main


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "generate",
            str(root / "data"),
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
    assert result.exit_code == 0, result.output
    return root


def run(*args):
    return CliRunner().invoke(main, [str(a) for a in args])


def test_version_and_help():
    assert __version__ in run("--version").output
    out = run("--help").output
    for command in (
        "generate",
        "inspect",
        "scan",
        "train",
        "benchmark",
        "score",
        "explain",
        "serve",
        "export-hub",
        "elliptic",
        "init-config",
    ):
        assert command in out


def test_generate_writes_ibm_files(workspace):
    assert (workspace / "data" / "Synth-Small_Trans.csv").exists()
    assert (workspace / "data" / "Synth-Small_Patterns.txt").exists()


def test_inspect_prints_json_summary(workspace):
    res = run("inspect", workspace / "data" / "Synth-Small_Trans.csv")
    assert res.exit_code == 0
    summary = json.loads(res.output)
    assert summary["illicit"] > 0 and "patterns" in summary


def test_scan_writes_alerts_and_reports_metrics(workspace):
    out = workspace / "scan"
    res = run("scan", workspace / "data" / "Synth-Small_Trans.csv", "--out", out, "--top", 3)
    assert res.exit_code == 0, res.output
    assert "alert(s)" in res.output and "Recall per ground-truth motif" in res.output
    assert (out / "alerts.csv").exists() and (out / "scores.csv").exists()
    alerts = json.loads((out / "alerts.json").read_text())
    assert alerts and "evidence" in alerts[0]


def test_scan_with_calibration_and_config(workspace, tmp_path):
    cfg = tmp_path / "p.yaml"
    cfg.write_text("patterns:\n  fan_in:\n    min_counterparties: 6\n")
    res = run("scan", workspace / "data" / "Synth-Small_Trans.csv", "--config", cfg, "--calibrate")
    assert res.exit_code == 0, res.output


def test_train_score_explain_export_cycle(workspace):
    data = workspace / "data" / "Synth-Small_Trans.csv"
    for kind in ("rf", "patterns"):
        res = run("train", data, "--model", kind, "--out", workspace / "models" / kind)
        assert res.exit_code == 0, res.output
        assert (workspace / "models" / kind / "config.json").exists()

    scored = run(
        "score", workspace / "models" / "rf", data, "--out", workspace / "scores.csv", "--top", 2
    )
    assert scored.exit_code == 0, scored.output
    assert "alert(s)" in scored.output and (workspace / "scores.csv").exists()

    exp = run(
        "explain",
        workspace / "models" / "patterns",
        data,
        "--tx",
        10,
        "--out",
        workspace / "e.json",
    )
    assert exp.exit_code == 0, exp.output
    assert json.loads((workspace / "e.json").read_text())["method"] == "patterns"

    occ = run(
        "explain",
        workspace / "models" / "rf",
        data,
        "--tx",
        10,
        "--method",
        "occlusion",
        "--max-edges",
        30,
    )
    assert occ.exit_code == 0 and "occlusion" in occ.output

    hub = run("export-hub", workspace / "models" / "rf", workspace / "hub", "--repo-id", "user/rf")
    assert hub.exit_code == 0
    assert "user/rf" in (workspace / "hub" / "README.md").read_text()


def test_explain_errors(workspace):
    data = workspace / "data" / "Synth-Small_Trans.csv"
    res = run("explain", workspace / "models" / "rf", data, "--tx", 10_000_000)
    assert res.exit_code != 0 and "not found" in res.output
    res = run("explain", workspace / "models" / "rf", data, "--tx", 1, "--method", "patterns")
    assert res.exit_code != 0 and "motif" in res.output
    res = run("explain", workspace / "models" / "rf", data, "--tx", 1, "--method", "gnnexplainer")
    assert res.exit_code != 0 and "GNN" in res.output


def test_benchmark_command(workspace):
    out = workspace / "bench"
    res = run(
        "benchmark",
        workspace / "data" / "Synth-Small_Trans.csv",
        "--models",
        "patterns",
        "--models",
        "logreg",
        "--seeds",
        0,
        "--out",
        out,
    )
    assert res.exit_code == 0, res.output
    assert "logreg" in res.output and (out / "summary.md").exists()


def test_benchmark_gnn_without_torch_is_a_clean_error(workspace):
    from aml_graphminer.utils import has_module

    if has_module("torch_geometric"):
        pytest.skip("gnn extra installed")
    res = run("benchmark", workspace / "data" / "Synth-Small_Trans.csv", "--models", "gcn")
    assert res.exit_code != 0
    assert "aml-graphminer[gnn]" in str(res.exception)


def test_elliptic_command(tmp_path):
    from aml_graphminer.synthetic import make_elliptic_like, write_elliptic_files

    write_elliptic_files(make_elliptic_like(2000, n_features=8, seed=1), tmp_path / "ell")
    res = run("elliptic", tmp_path / "ell", "--models", "rf", "--out", tmp_path / "ell_out")
    assert res.exit_code == 0, res.output
    assert (tmp_path / "ell_out" / "results.csv").exists()


def test_init_config(tmp_path):
    target = tmp_path / "amlgm.yaml"
    assert run("init-config", target).exit_code == 0 and target.exists()
    again = run("init-config", target)
    assert again.exit_code != 0 and "already exists" in again.output
    assert run("init-config", target, "--force").exit_code == 0


def test_unlabelled_data_cannot_be_used_for_training(tmp_path):
    from aml_graphminer.ingestion import write_ibm_csv
    from aml_graphminer.synthetic import generate_transactions

    df = generate_transactions(100, 300, seed=1)
    path = write_ibm_csv(df, tmp_path / "U_Trans.csv", with_label=False)
    res = run("train", path, "--model", "rf", "--out", tmp_path / "m")
    assert res.exit_code != 0 and "labelled" in res.output
    assert run("inspect", path).exit_code == 0


def test_invalid_model_choice():
    res = run("train", "x.csv", "--model", "transformer", "--out", "o")
    assert res.exit_code != 0


# --------------------------------------------------------------------------- newer commands
def test_new_commands_are_listed():
    out = run("--help").output
    for command in ("eda", "export-neo4j", "download"):
        assert command in out


def test_eda_command(workspace):
    out = workspace / "eda"
    res = run("eda", workspace / "data" / "Synth-Small_Trans.csv", "--out", out, "--no-figures")
    assert res.exit_code == 0, res.output
    assert (out / "EDA.md").exists() and (out / "summary.json").exists()


def test_export_neo4j_command(workspace, monkeypatch):
    data = workspace / "data" / "Synth-Small_Trans.csv"
    if not (workspace / "models" / "rf" / "config.json").exists():
        assert (
            run("train", data, "--model", "rf", "--out", workspace / "models" / "rf").exit_code == 0
        )
    out = workspace / "neo4j"
    res = run("export-neo4j", data, out, "--model", workspace / "models" / "rf")
    assert res.exit_code == 0, res.output
    import pandas as pd

    assert pd.read_csv(out / "transactions.csv")["score"].notna().all()
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    res = run("export-neo4j", data, out, "--no-scan", "--push")
    assert res.exit_code != 0 and "password" in res.output

    import sys
    import types

    pushed = {}

    def fake_push(df, uri, user, password, **kwargs):
        pushed.update(uri=uri, user=user, password=password, n=len(df))
        return {"accounts": 3, "transactions": len(df)}

    monkeypatch.setattr("aml_graphminer.graph.neo4j.push_to_neo4j", fake_push)
    monkeypatch.setitem(sys.modules, "neo4j", types.SimpleNamespace())
    res = run(
        "export-neo4j", data, out, "--push", "--password", "secret", "--uri", "bolt://db:7687"
    )
    assert res.exit_code == 0, res.output
    assert pushed["uri"] == "bolt://db:7687" and pushed["password"] == "secret"


def test_download_command(tmp_path, monkeypatch):
    calls = {}

    def fake_ibm(directory, *, sizes, groups, force):
        calls["ibm"] = (directory, sizes, groups, force)
        return [tmp_path / "HI-Small_Trans.csv"]

    def fake_elliptic(directory, *, force):
        calls["elliptic"] = (directory, force)
        return [tmp_path / "elliptic_txs_features.csv"]

    monkeypatch.setattr("aml_graphminer.datasets.download_ibm", fake_ibm)
    monkeypatch.setattr("aml_graphminer.datasets.download_elliptic", fake_elliptic)
    res = run("download", "ibm", "--out", tmp_path, "--size", "small", "--group", "HI")
    assert res.exit_code == 0, res.output
    assert calls["ibm"][1:] == (("small",), ("HI",), False) and "1 file(s)" in res.output
    assert run("download", "elliptic", "--out", tmp_path, "--force").exit_code == 0
    assert calls["elliptic"][1] is True
    assert run("download", "mnist").exit_code != 0


def test_serve_command(workspace, monkeypatch):
    import sys
    import types

    launched = {}
    fake = types.SimpleNamespace(run=lambda app, host, port: launched.update(app=app, port=port))
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    res = run("serve")
    assert res.exit_code != 0 and "--demo" in res.output
    pytest.importorskip("fastapi")  # the rest needs the api extra

    data = workspace / "data" / "Synth-Small_Trans.csv"
    if not (workspace / "models" / "patterns" / "config.json").exists():
        assert (
            run(
                "train", data, "--model", "patterns", "--out", workspace / "models" / "patterns"
            ).exit_code
            == 0
        )
    res = run("serve", workspace / "models" / "patterns", "--port", 8123, "--data", data)
    assert res.exit_code == 0, res.output
    assert launched["port"] == 8123 and "http://127.0.0.1:8123/ui/" in res.output

    monkeypatch.setattr("aml_graphminer.cli._demo_model", lambda: workspace / "models" / "patterns")
    res = run("serve", "--demo", "--host", "0.0.0.0")
    assert res.exit_code == 0, res.output
    assert "http://127.0.0.1:8000/ui/" in res.output


def test_demo_model_is_trained_once(tmp_path, monkeypatch):
    import tempfile

    from aml_graphminer.cli import _demo_model
    from aml_graphminer.detectors.bundle import load_detector

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    first = _demo_model()
    assert (first / "config.json").exists() and load_detector(first).describe()["kind"] == "hgb"
    stamp = (first / "config.json").stat().st_mtime_ns
    assert _demo_model() == first and (first / "config.json").stat().st_mtime_ns == stamp


def test_explain_top_alerts_writes_a_case_file(tmp_path, synth_df):
    data = tmp_path / "t.csv"
    synth_df.to_csv(data)
    assert run("train", data, "--model", "patterns", "--out", tmp_path / "m").exit_code == 0
    res = run("explain", tmp_path / "m", data, "--top", 3, "--out-dir", tmp_path / "case")
    assert res.exit_code == 0, res.output
    files = {p.name for p in (tmp_path / "case").iterdir()}
    assert {"index.csv", "README.md"} <= files
    assert len([f for f in files if f.endswith(".json")]) == res.output.count("tx ")
    assert run("explain", tmp_path / "m", data).exit_code != 0  # neither --tx nor --top
    assert run("explain", tmp_path / "m", data, "--tx", 1, "--top", 2).exit_code != 0
