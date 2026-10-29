"""Jupyter plugin: rich display of the results and the %amlgm / %aml_scan / %aml_explain magics."""

import builtins

import numpy as np
import pandas as pd
import pytest

from aml_graphminer import __version__
from aml_graphminer.detectors.bundle import MotifDetector
from aml_graphminer.detectors.patterns import PatternScanner
from aml_graphminer.detectors.patterns.base import Alert
from aml_graphminer.detectors.patterns.scanner import ScanResult
from aml_graphminer.evaluation.benchmark import run_benchmark
from aml_graphminer.explainers.base import (
    EdgeImportance,
    Explanation,
    FeatureImportance,
    NodeImportance,
)
from aml_graphminer.jupyter import (
    alert_html,
    benchmark_html,
    explanation_html,
    graph_svg,
    scan_html,
)
from aml_graphminer.jupyter.display import ALERT, NEUTRAL, POSITIVE


@pytest.fixture(scope="module")
def scan(tiny_df):
    return PatternScanner().scan(tiny_df)


def _explanation(account="A"):
    return Explanation(
        tx_id=7,
        method="occlusion",
        score=0.9,
        threshold=0.5,
        edges=[
            EdgeImportance(7, account, "B", 1000.0, "2024-01-01T10:00", 0.0, "target"),
            EdgeImportance(8, "B", "C", 250.5, "2024-01-01T11:00", 0.4),
            EdgeImportance(9, "C", "B", 99.0, "2024-01-01T12:00", -0.2),
            EdgeImportance(10, "C", "C", 10.0, "2024-01-01T13:00", 1e-5),  # self-transfer
            EdgeImportance(11, "B", "C", 75.0, "2024-01-01T14:00", 0.1),  # parallel to tx 8
        ],
        nodes=[NodeImportance("B", 0.5, "receiver")],
        features=[
            FeatureImportance("amount", 1000.0, 0.3),
            FeatureImportance("currency", "Euro", -1e-4),
        ],
        patterns=[
            {
                "pattern": "fan_in",
                "score": 0.8,
                "evidence": {"hub": "B", "n_counterparties": 5, "accounts": ["x", "y"]},
            }
        ],
        summary="Score 0.9 <because of B>",
    )


# ----------------------------------------------------------------------------- rich display
def test_a_scan_shows_the_alerts_per_motif_and_the_strongest_ones(scan):
    html = scan._repr_html_()
    assert html == scan_html(scan)
    assert f"{len(scan):,} alert(s)" in html
    for pattern in scan.summary()["alerts_by_pattern"]:
        assert pattern in html
    assert html.count("<tr>") == 1 + min(10, len(scan))  # header + the 10 strongest
    assert scan_html(scan, top=3).count("<tr>") == 4
    strongest = max(scan.weight(a) for a in scan.alerts)
    assert html.index(f"{strongest:.3f}") < html.index("</tbody>")


def test_an_empty_scan_says_so():
    html = ScanResult(alerts=[], tx_ids=np.arange(5))._repr_html_()
    assert "0 alert(s)" in html and "No motif found" in html


def test_an_alert_shows_its_evidence(scan):
    alert = scan.alerts[0]
    html = alert._repr_html_()
    assert html == alert_html(alert)
    assert alert.pattern in html
    for key in alert.evidence:
        assert key in html


def test_an_explanation_shows_verdict_graph_tables_and_motifs():
    exp = _explanation()
    html = exp._repr_html_()
    assert html == explanation_html(exp)
    assert "Transaction 7 — occlusion" in html
    assert "threshold 0.500" in html and f"color:{ALERT}'>alert" in html
    assert html.count("<title>tx ") == 5  # one arrow per transaction, with its tooltip
    assert html.count("<circle") == 3  # accounts A, B, C
    assert f'stroke="{ALERT}"' in html and f'stroke="{POSITIVE}"' in html
    assert f'stroke="{NEUTRAL}"' in html  # tx 9 lowers the score, tx 10 is ~0
    assert "Transactions that matter most" in html and "Attributes" in html
    assert "0.000" in html  # an importance that rounds to zero is shown as zero
    assert "fan_in" in html and "n_counterparties 5" in html
    assert "Score 0.9 &lt;because of B&gt;" in html


def test_an_explanation_below_the_threshold_is_not_an_alert():
    exp = _explanation()
    exp.score = 0.2
    assert "no alert" in exp._repr_html_()


def test_values_from_the_data_are_escaped_never_executed():
    script = "<script>alert(1)</script>"
    html = _explanation(account=script)._repr_html_()
    assert script not in html and "&lt;script&gt;" in html
    alert = Alert(
        pattern="fan_in",
        score=0.7,
        accounts=["<b>x</b>", "y"],
        tx_ids=[1, 2],
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-02"),
        total_amount=10.0,
        evidence={"<k>": "<v>"},
    )
    for html in (
        alert._repr_html_(),
        ScanResult(alerts=[alert], tx_ids=np.arange(3))._repr_html_(),
    ):
        assert "<b>x</b>" not in html and "&lt;b&gt;x&lt;/b&gt;" in html
    assert "&lt;k&gt;" in alert._repr_html_()


def test_graph_layouts_for_small_and_disconnected_explanations():
    assert "Empty explanation" in graph_svg(Explanation(tx_id=1, method="patterns"))
    loop = Explanation(1, "patterns", edges=[EdgeImportance(1, "A", "A", 5.0, "t", 1.0, "target")])
    assert graph_svg(loop).count("<circle") == 1
    parts = Explanation(
        1,
        "patterns",
        edges=[
            EdgeImportance(1, "A", "B", 5.0, "t", 1.0, "target"),
            EdgeImportance(2, "C", "D", 5.0, "t", 0.5, "pattern"),
        ],
    )
    svg = graph_svg(parts)
    assert svg.startswith("<svg") and svg.count("<circle") == 4


def test_a_benchmark_report_shows_its_summary_table(synth_df):
    report = run_benchmark(synth_df, models=("patterns",), seeds=(0,))
    html = report._repr_html_()
    assert html == benchmark_html(report)
    assert "1 model(s)" in html and "pr_auc" in html and "patterns" in html


# ----------------------------------------------------------------------------- magics
@pytest.fixture(scope="module")
def shell(tmp_path_factory):
    pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell
    from traitlets.config import Config

    config = Config()
    config.HistoryManager.enabled = False
    with pytest.MonkeyPatch.context() as patch:  # the profile goes to a temporary folder
        patch.setenv("IPYTHONDIR", str(tmp_path_factory.mktemp("ipython")))
        ipython = InteractiveShell(config=config)
    ipython.run_line_magic("load_ext", "aml_graphminer")
    yield ipython
    for name in ("__IPYTHON__", "display"):  # added to builtins by IPython
        builtins.__dict__.pop(name, None)


@pytest.fixture(scope="module")
def data_file(shell, tmp_path_factory):
    folder = tmp_path_factory.mktemp("magic") / "my data"  # a space: quoting must work
    shell.run_line_magic("amlgm", f'generate "{folder}" --accounts 250 --background 1200 --seed 7')
    return folder / "Synth-Small_Trans.csv"


def test_load_ext_registers_the_magics(shell):
    assert {"amlgm", "aml_scan", "aml_explain"} <= set(shell.magics_manager.magics["line"])


def test_amlgm_runs_the_command_line_inside_the_kernel(shell, data_file, capsys):
    shell.run_line_magic("amlgm", "--version")
    assert f"amlgm, version {__version__}" in capsys.readouterr().out
    shell.run_line_magic("amlgm", f'scan "{data_file}" --top 3')
    assert "alert(s) on" in capsys.readouterr().out


def test_amlgm_errors_are_short_messages(shell, tmp_path):
    from IPython.core.error import UsageError

    with pytest.raises(UsageError, match="does not exist"):
        shell.run_line_magic("amlgm", "scan no-such-file.csv")
    with pytest.raises(UsageError, match="No such command"):
        shell.run_line_magic("amlgm", "frobnicate")
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(UsageError, match="amlgm: "):  # a library error (invalid data)
        shell.run_line_magic("amlgm", f'inspect "{bad}"')


def test_aml_scan_takes_a_dataframe_of_the_notebook_or_a_file(shell, tiny_df, data_file, tmp_path):
    from IPython.core.error import UsageError

    shell.user_ns["df"] = tiny_df
    result = shell.run_line_magic("aml_scan", "df")
    assert isinstance(result, ScanResult)
    assert len(result) == len(PatternScanner().scan(tiny_df))
    config = tmp_path / "amlgm.yaml"
    shell.run_line_magic("amlgm", f'init-config "{config}"')
    from_file = shell.run_line_magic("aml_scan", f'"{data_file}" --config "{config}"')
    assert len(from_file) > 0
    with pytest.raises(UsageError, match="neither a DataFrame"):
        shell.run_line_magic("aml_scan", "missing")
    with pytest.raises(UsageError):  # the positional argument is required
        shell.run_line_magic("aml_scan", "")


def test_aml_explain_with_a_detector_of_the_notebook_or_a_model_folder(shell, tiny_df, tmp_path):
    from IPython.core.error import UsageError

    motifs = MotifDetector.fit(tiny_df, fractions=(0.6, 0.2, 0.2))
    shell.user_ns.update(df=tiny_df, motifs=motifs)
    tx = int(motifs.score(tiny_df).idxmax())
    exp = shell.run_line_magic("aml_explain", f"motifs df {tx}")
    assert isinstance(exp, Explanation) and exp.tx_id == tx and exp.method == "patterns"
    folder = motifs.save(tmp_path / "motifs")
    again = shell.run_line_magic(
        "aml_explain", f'"{folder}" df {tx} --method occlusion --hops 1 --max-edges 10'
    )
    assert again.method == "occlusion" and again.edges
    with pytest.raises(UsageError, match="not found"):
        shell.run_line_magic("aml_explain", "motifs df 99999999")
    with pytest.raises(UsageError, match="neither a detector"):
        shell.run_line_magic("aml_explain", f"nothing df {tx}")
    with pytest.raises(UsageError, match="needs a supervised GNN detector"):
        shell.run_line_magic("aml_explain", f"motifs df {tx} --method gnnexplainer")
