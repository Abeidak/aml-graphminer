"""The scripts of ``benchmarks/`` and ``scripts/`` run end to end on synthetic data."""

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scripts():
    for folder in ("benchmarks", "scripts"):
        sys.path.insert(0, str(ROOT / folder))
    try:
        yield {
            name: importlib.import_module(name)
            for name in ("run_ibm", "run_elliptic", "make_report", "py2nb")
        }
    finally:
        for folder in ("benchmarks", "scripts"):
            sys.path.remove(str(ROOT / folder))


@pytest.fixture(scope="module")
def results(scripts, tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    args = ["--synthetic", "--models", "patterns", "logreg", "--seeds", "0", "1", "--out", str(out)]
    assert scripts["run_ibm"].main(args) == 0
    args = ["--synthetic", "--models", "logreg", "--seeds", "0", "--out", str(out)]
    assert scripts["run_elliptic"].main(args) == 0
    return out


def test_run_scripts_write_results_and_environment(results):
    for name in ("synthetic-smoke", "elliptic-synthetic-smoke"):
        folder = results / name
        assert {"results.csv", "summary.md", "config.json", "environment.json"} <= {
            p.name for p in folder.iterdir()
        }
    env = json.loads((results / "synthetic-smoke" / "environment.json").read_text())
    assert env["aml_graphminer"] and "python" in env


def test_make_report_labels_synthetic_runs_and_writes_latex(scripts, results):
    assert scripts["make_report"].main([str(results)]) == 0
    report = (results / "REPORT.md").read_text(encoding="utf-8")
    assert "Synthetic stand-in data" in report and "| logreg |" in report
    assert "Per-motif recall" in report
    tex = (results / "results.tex").read_text(encoding="utf-8")
    assert tex.count("\\midrule") >= 2 and "\\bottomrule" in tex and "$\\pm$" in tex


def test_make_report_needs_results(scripts, tmp_path):
    with pytest.raises(SystemExit):
        scripts["make_report"].main([str(tmp_path)])


def test_make_report_merges_runs_of_the_same_dataset(scripts, results, tmp_path):
    """GNN runs made on a GPU (notebook 04) are merged into the CPU table of the same dataset."""
    import pandas as pd

    gpu = tmp_path / "gpu"
    (gpu / "synthetic-smoke").mkdir(parents=True)
    cpu = pd.read_csv(results / "synthetic-smoke" / "results.csv")
    gnn = cpu[cpu["model"] == "logreg"].assign(model="gcn")
    gnn.to_csv(gpu / "synthetic-smoke" / "results.csv", index=False)
    (gpu / "synthetic-smoke" / "environment.json").write_text('{"gpu": "test"}', encoding="utf-8")
    out = tmp_path / "report"
    out.mkdir()
    assert scripts["make_report"].main([str(out), str(results), str(gpu)]) == 0
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    section = report.split("## synthetic-smoke")[1]
    assert "| gcn |" in section and "| logreg |" in section and "Merged from" in section
    assert '"gpu": "test"' in report
    assert "gcn" in (out / "results.tex").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        scripts["make_report"].main([str(out), str(tmp_path / "missing")])


def test_run_ibm_requires_data_dir(scripts):
    with pytest.raises(SystemExit):
        scripts["run_ibm"].main(["--models", "rf"])


def test_cell_formatting(scripts):
    cell = scripts["make_report"].cell
    assert cell(0.5, 0.125) == "0.500 ± 0.125"
    assert cell(0.5, float("nan")) == "0.500"
    assert cell(float("nan"), 0.1) == ""


def test_py2nb_parses_percent_files_and_executes(scripts, tmp_path):
    py2nb = scripts["py2nb"]
    source = tmp_path / "demo.py"
    source.write_text(
        '"""header"""\n# %% [markdown]\n# # Title\n# text\n\n# %%\n# %pip install -q thing\n'
        "import matplotlib.pyplot as plt\nx = 2\nprint('hi')\nplt.plot([1, 2])\nx * 21\n",
        encoding="utf-8",
    )
    assert py2nb.main([str(source), "--out", str(tmp_path / "nb")]) == 0
    plain = json.loads((tmp_path / "nb" / "demo.ipynb").read_text())
    assert [c["cell_type"] for c in plain["cells"]] == ["markdown", "code"]
    assert plain["cells"][0]["source"] == ["# Title\n", "text"]
    assert plain["cells"][1]["source"][0] == "%pip install -q thing\n"
    assert plain["cells"][1]["outputs"] == []

    assert py2nb.main([str(source), "--out", str(tmp_path / "nb"), "--execute"]) == 0
    done = json.loads((tmp_path / "nb" / "demo.ipynb").read_text())
    kinds = [o["output_type"] for o in done["cells"][1]["outputs"]]
    assert kinds == ["stream", "display_data", "execute_result"]
    assert done["cells"][1]["outputs"][2]["data"]["text/plain"] == ["42"]


def test_py2nb_reports_the_failing_cell(scripts, tmp_path):
    cells = scripts["py2nb"].parse_percent("# %%\nraise ValueError('boom')\n")
    with pytest.raises(RuntimeError, match="cell 0 failed"):
        scripts["py2nb"].execute_cells(cells, tmp_path)
