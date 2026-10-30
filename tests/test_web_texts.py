"""French texts of the web interface (``api/static/fr.js``), run with Node.js on real API answers.

The API answers in English: codes (motif, method, evidence key, attribute ...) and figures, next
to English summaries. ``fr.js`` words them in French for the interface. These tests give it what
the investigation service really returns and check that every code has a French text and that no
English reaches the analyst. Skipped without Node.js (the CI runners have it).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

import aml_graphminer
from aml_graphminer.api.investigation import MAX_PAGE, InvestigationService
from aml_graphminer.detectors.baselines import BASELINE_KINDS
from aml_graphminer.detectors.bundle import MotifDetector, TabularDetector
from aml_graphminer.detectors.gnn.config import GNN_KINDS
from aml_graphminer.detectors.hybrid import HybridDetector
from aml_graphminer.detectors.patterns import PatternConfig, PatternScanner
from aml_graphminer.detectors.patterns.scanner import DETECTOR_REGISTRY
from aml_graphminer.exceptions import AMLGraphMinerError
from aml_graphminer.explainers import PATTERN_DESCRIPTIONS, Explanation
from aml_graphminer.explainers.base import EdgeImportance, FeatureImportance
from aml_graphminer.explainers.dispatch import EXPLAIN_METHODS
from aml_graphminer.explainers.occlusion import ATTRIBUTE_COLUMNS
from aml_graphminer.features.network import BETWEENNESS_COLUMN, NODE_COLUMNS, PAGERANK_COLUMN
from aml_graphminer.features.pipeline import BASE_EDGE_COLUMNS, FeatureConfig
from aml_graphminer.features.temporal import temporal_feature_names
from aml_graphminer.synthetic.generator import CURRENCIES, PAYMENT_FORMATS

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
FR_JS = Path(aml_graphminer.__file__).parent / "api" / "static" / "fr.js"
THIN = chr(0x202F)  # narrow no-break space: the French thousands separator
NBSP = chr(0xA0)  # no-break space between a number and its unit ("24 h")
# FR.<name>(...args) for every [name, ...args]; a name that is not a function returns its value
RUNNER = """
const fs = require("fs");
const FR = require(process.argv[1]);
const calls = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = calls.map(([name, ...args]) => (typeof FR[name] === "function" ? FR[name](...args) : FR[name]));
fs.writeFileSync(process.argv[3], JSON.stringify(out));
"""
# words of the English summaries and codes of the API that must not reach the interface
ENGLISH = re.compile(
    r"\b(the|and|with|without|of|to|from|is|would|this|which|money|accounts?|amounts?|sends?|"
    r"receives?|neighbou?r\w*|raises|lowers|structural|unknown|missing|expected|seed|synthetic|"
    r"threshold|transfers?|split|hub|most|influential|features?|sender|receiver|relay|got|file|"
    r"columns?|found|larger|exceed)\b"
    r"|\b[a-z]+_[a-z_]+\b",  # a raw snake_case code (evidence key, attribute ...)
    re.IGNORECASE,
)


def fr(*calls: list[Any]) -> list[Any]:
    """Results of ``FR.<name>(*args)`` for every ``[name, *args]``, computed by Node.js."""
    with tempfile.TemporaryDirectory() as tmp:
        source, target = Path(tmp) / "calls.json", Path(tmp) / "out.json"
        source.write_text(json.dumps(calls), encoding="utf-8")
        subprocess.run(
            [str(NODE), "-e", RUNNER, str(FR_JS), str(source), str(target)],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return list(json.loads(target.read_text(encoding="utf-8")))


def english(text: str) -> list[str]:
    return [m.group(0) for m in ENGLISH.finditer(text)]


@pytest.fixture(scope="module")
def service() -> InvestigationService:
    config = PatternConfig()  # every motif detector, relay (opt-in) included
    config = config.model_copy(update={"relay": config.relay.model_copy(update={"enabled": True})})
    return InvestigationService(MotifDetector(PatternScanner(config)))


@pytest.fixture(scope="module")
def dataset(service: InvestigationService, synth_df) -> dict[str, Any]:
    return service.add(synth_df, "synthetic")


@pytest.fixture(scope="module")
def alerts(service: InvestigationService, dataset) -> dict[str, dict[str, Any]]:
    """The strongest alert of every motif (the synthetic data contains all of them)."""
    best: dict[str, dict[str, Any]] = {}
    for alert in service.alerts(dataset["id"], limit=MAX_PAGE)["items"]:
        best.setdefault(alert["pattern"], alert)
    assert set(best) == set(service.scanner.detectors)
    return best


def test_every_code_sent_by_the_server_has_a_french_text(service, dataset, alerts, synth_df):
    (codes,) = fr(["codes"])
    assert set(DETECTOR_REGISTRY) | set(PATTERN_DESCRIPTIONS) <= set(codes["patterns"])
    assert set(EXPLAIN_METHODS) <= set(codes["methods"])
    model_types = {MotifDetector.model_type, TabularDetector.model_type, HybridDetector.model_type}
    assert model_types | {"gnn", "gnn_unsupervised"} <= set(codes["models"])
    assert set(BASELINE_KINDS) | set(GNN_KINDS) <= set(codes["kinds"])
    assert set(PAYMENT_FORMATS) <= set(codes["paymentFormats"])
    assert set(CURRENCIES) <= set(codes["currencies"])
    evidence = {
        key
        for item in service.alerts(dataset["id"], limit=MAX_PAGE)["items"]
        for key in item["evidence"]
    }
    assert evidence <= set(codes["evidence"]), evidence - set(codes["evidence"])
    described = set(service.detector.describe()) | set(
        TabularDetector.fit(synth_df, "logreg").describe()
    )
    assert described - {"model_type", "metrics"} <= set(codes["modelFields"])


def test_alerts_are_described_in_french(service, dataset, alerts):
    items = list(alerts.values())
    details = [service.alert(dataset["id"], a["alert_id"]) for a in items]
    results = fr(
        *(["alertSentence", a] for a in items),
        *(["evidenceItems", d["evidence"]] for d in details),
        *(["patternLabel", a["pattern"]] for a in items),
    )
    sentences, evidence, labels = (
        results[: len(items)],
        results[len(items) : -len(items)],
        results[-len(items) :],
    )
    for alert, sentence, label in zip(items, sentences, labels):
        assert sentence.startswith(f"{label} : ") and not english(sentence), sentence
        counts = [
            v for v in alert["evidence"].values() if isinstance(v, int) and not isinstance(v, bool)
        ]
        assert all(f"{v:,}".replace(",", THIN) in sentence for v in counts), sentence
    for detail, rows in zip(details, evidence):
        assert len(rows) == len([v for v in detail["evidence"].values() if not isinstance(v, dict)])
        assert not [row for row in rows if english(" ".join(row))], rows
    (unlabelled,) = fr(["alertSentence", items[0], {"label": False}])
    assert sentences[0] == f"{labels[0]} : {unlabelled}"  # next to a link that names the motif
    # the alerts of a transaction carry their evidence too, so the interface can describe them
    tx = details[0]["tx_ids"][0]
    assert all(a["evidence"] for a in service.transaction(dataset["id"], tx)["alerts"])
    assert set(service.model()["patterns"]) == set(service.scanner.detectors)


def test_explanation_summaries_are_french(service, dataset, alerts, synth_df):
    inside = service.alert(dataset["id"], alerts["fan_in"]["alert_id"])["tx_ids"][0]
    outside = next(
        int(t) for t in synth_df.index if not service.transaction(dataset["id"], int(t))["alerts"]
    )
    tabular = InvestigationService(TabularDetector.fit(synth_df, "hgb"))
    table = tabular.add(synth_df, "synthetic")
    top = tabular.transactions(table["id"], limit=1)["items"][0]["tx_id"]
    occlusion = tabular.explain(table["id"], top, method="occlusion")
    explained = [
        service.explain(dataset["id"], inside, method="patterns"),
        service.explain(dataset["id"], outside, method="patterns"),
        occlusion,
    ]
    in_motif, no_motif, occluded, strongest = fr(
        *(["explanationSummary", e] for e in explained),
        ["patternLabel", explained[0]["patterns"][0]["pattern"]],
    )
    assert in_motif.startswith("Fait partie de ") and f". {strongest} : " in in_motif
    assert not english(in_motif), in_motif
    assert no_motif == "Aucun motif structurel ne contient cette transaction."
    neighbours = occlusion["details"]["neighbours"]
    assert f"calculé avec {neighbours} transactions voisines" in occluded and not english(occluded)
    named = re.search(r"neighbour is transaction (\d+)", occlusion["summary"])
    assert (
        named is None or f"la n° {named.group(1)} " in occluded
    )  # the same neighbour as in English


def _gnn(method: str, **details: Any) -> dict[str, Any]:
    """An explanation of a GNN method as the API serialises it (no PyTorch needed)."""
    edges = [
        EdgeImportance(1, "A", "B", 900.0, "2022-09-01 10:00:00", 0.9, "target"),
        EdgeImportance(2, "C", "A", 450.0, "2022-09-01 09:00:00", 0.42, "neighbour"),
        EdgeImportance(3, "D", "A", 120.0, "2022-09-01 08:00:00", -0.07, "neighbour"),
    ]
    features = [
        FeatureImportance("tx.s_cnt_24h", 1.73, 0.21),
        FeatureImportance("sender.uniq_out_rate", -0.4, -0.05),
        FeatureImportance("receiver.pagerank", 2.1, 0.03),
    ]
    return Explanation(1, method, 0.81, 0.5, edges, features=features, details=details).to_dict()


def test_gnn_explanation_summaries_are_french():
    effect = {"neighbours": 2, "score_without_neighbours": 0.31, "neighbourhood_effect": 0.5}
    explainer, captum, uniform, attention = fr(
        ["explanationSummary", _gnn("gnnexplainer", **effect)],
        ["explanationSummary", _gnn("captum", **effect)],
        ["explanationSummary", _gnn("pgexplainer", uniform_mask=0.5, trained_on=16, **effect)],
        ["explanationSummary", _gnn("attention", **effect)],
    )
    note = "Sans ses 2 transactions voisines, le score serait de 0,310 (effet mesuré du voisinage : +0,500)."
    assert all(text.endswith(note) for text in (explainer, captum, uniform, attention))
    assert attention.startswith(
        "Score 0,810 ; transaction voisine la plus influente : n° 2 (C → A, importance 0,420)."
    )
    kept = f"Attributs de compte retenus par le masque : envois de l'émetteur sur 24{NBSP}h, "
    assert kept in explainer
    assert "(valeur standardisée 1,73) ; il augmente la probabilité de 0,210." in captum
    assert (
        "tous les masques ≈ 0,50" in uniform and "entraîné sur 16 voisinages d'alertes" in uniform
    )
    assert not english(" ".join([explainer, captum, uniform, attention]))


def test_a_tiny_effect_is_not_written_as_zero():
    edges = [
        EdgeImportance(1, "A", "B", 900.0, "2022-09-01 10:00:00", 0.0, "target"),
        EdgeImportance(2, "C", "A", 450.0, "2022-09-01 09:00:00", 0.0002, "neighbour"),
    ]
    occlusion = Explanation(1, "occlusion", 1.0, 0.5, edges, details={"neighbours": 1}).to_dict()
    captum = _gnn("captum")
    captum["features"][0]["importance"] = -0.0001
    occluded, attributed = fr(["explanationSummary", occlusion], ["explanationSummary", captum])
    assert occluded.endswith(" : elle augmente le score de moins de 0,001.")  # not "de 0,000"
    assert "il diminue la probabilité de moins de 0,001." in attributed


def test_attributes_of_the_models_have_french_names():
    names = [
        *ATTRIBUTE_COLUMNS,
        *(
            f"{side}.{c}"
            for side in ("sender", "receiver")
            for c in (*NODE_COLUMNS, PAGERANK_COLUMN, BETWEENNESS_COLUMN)
        ),
        *(
            f"tx.{c}"
            for c in (*BASE_EDGE_COLUMNS, *temporal_feature_names(FeatureConfig().windows))
        ),
        *(f"tx.fmt_{p}" for p in PAYMENT_FORMATS),
    ]
    labels = fr(*(["featureLabel", n] for n in names))
    untranslated = [n for n, label in zip(names, labels) if label == n or english(label)]
    assert not untranslated
    values = fr(
        ["featureValue", {"name": "amount", "value": 1234.5}],
        ["featureValue", {"name": "transaction_type", "value": "Credit Card"}],
        ["featureValue", {"name": "currency", "value": "US Dollar"}],
        ["currencyCode", "Swiss Franc"],
    )
    assert values == [f"1{THIN}234,50", "Carte de crédit", "dollar américain", "CHF"]


def test_error_messages_of_the_api_are_french(service, dataset, synth_df, tmp_path):
    canonical = synth_df[["timestamp", "sender", "receiver", "amount"]].head(20)
    no_amount = canonical.drop(columns=["amount"]).to_csv(index=False).encode()
    bad_label = canonical.assign(label=2).to_csv(index=False).encode()
    small = InvestigationService(service.detector, max_transactions=10)
    failures = [
        lambda: service.account(dataset["id"], "nobody"),
        lambda: service.transaction(dataset["id"], 10**9),
        lambda: service.alert(dataset["id"], 10**6),
        lambda: service.overview("0123456789ab"),
        lambda: service.add(synth_df.iloc[:0], "empty"),
        lambda: small.add(synth_df, "too big"),
        lambda: service.add_upload(b"x", "data.xlsx"),
        lambda: service.add_upload(no_amount, "t.csv", schema="canonical"),
        lambda: service.add_upload(b"a,b\n1,2\n", "t.csv"),
        lambda: service.add_upload(b"1,2,3\n4,5,6\n", "t.csv", schema="ibm"),
        lambda: service.add_upload(bad_label, "t.csv", schema="canonical"),
    ]
    messages = []
    for fail in failures:
        with pytest.raises(AMLGraphMinerError) as info:
            fail()
        messages.append(str(info.value))
    messages.append("file larger than 20 MB")  # raised by the upload route (app.py)
    translated = fr(
        *(["errorMessage", 422, {"error": "invalid_request", "detail": m}] for m in messages)
    )
    for message, text in zip(messages, translated):
        # column names stay as written in the file (timestamp, sender, receiver, amount, label)
        prose = re.sub(r"\[[^\]]*\]|timestamp, sender, receiver, amount|label", "", text)
        assert text != message and not english(prose), (message, text)
    validation = {"detail": [{"loc": ["query", "min_score"], "msg": "Input should be <= 1"}]}
    other, invalid, down, key = fr(
        ["errorMessage", 422, {"error": "invalid_request", "detail": "something new"}],
        ["errorMessage", 422, validation],
        ["errorMessage", 502, "<html>Bad Gateway</html>"],
        ["errorMessage", 401, {"error": "unauthorized", "detail": "missing or invalid API key"}],
    )
    assert other == "something new" and invalid == "Paramètre invalide : min_score."
    assert key.startswith("Clé d'accès manquante ou invalide")
    assert down.startswith("Le serveur n'a pas pu répondre (HTTP 502)")


def test_demo_datasets_get_a_french_name(service, tmp_path, synth_df):
    demo = service.add_demo(accounts=200, background=1000, seed=3)
    path = tmp_path / "my transactions.csv"
    synth_df.head(50).to_csv(path)
    file = service.add_file(path)
    try:
        assert demo["origin"] == {"type": "demo", "seed": 3} and file["origin"] == {"type": "file"}
        assert fr(["datasetName", demo], ["datasetName", file]) == [
            "Démo synthétique (graine 3)",
            "my transactions.csv",
        ]
    finally:
        service.remove(demo["id"])
        service.remove(file["id"])
