import pytest

from aml_graphminer.api import ScoringService
from aml_graphminer.api.service import MAX_TRANSACTIONS, error_payload
from aml_graphminer.detectors.bundle import MotifDetector, TabularDetector
from aml_graphminer.exceptions import ConfigError, ModelError

FIELDS = (
    "timestamp",
    "sender",
    "receiver",
    "amount",
    "currency",
    "transaction_type",
    "amount_received",
    "currency_received",
    "sender_bank",
    "receiver_bank",
)


def to_records(df):
    out = []
    for rec in df.to_dict("records"):
        item = {k: rec[k] for k in FIELDS}
        item["timestamp"] = str(item["timestamp"])
        out.append(item)
    return out


@pytest.fixture(scope="module")
def services(synth_df):
    return {
        "rf": ScoringService(TabularDetector.fit(synth_df, "rf")),
        "patterns": ScoringService(MotifDetector.fit(synth_df)),
    }


@pytest.fixture(scope="module")
def records(synth_df):
    return to_records(synth_df.iloc[3000:3200])


def test_health(services):
    health = services["rf"].health()
    assert health["status"] == "ok" and health["model"]["model_type"] == "tabular"


@pytest.mark.parametrize("name", ["rf", "patterns"])
def test_score_returns_one_result_per_posted_row_in_order(name, services, records):
    out = services[name].score(records)
    assert out["n_transactions"] == 200
    assert [r["row"] for r in out["results"]] == list(range(200))
    assert out["n_alerts"] == sum(r["alert"] for r in out["results"])
    assert all(0 <= r["score"] <= 1 for r in out["results"])


def test_score_matches_direct_detector_call(services, records, synth_df):
    direct = services["rf"].detector.score(synth_df.iloc[3000:3200])
    api = services["rf"].score(records)["results"]
    assert [round(r["score"], 6) for r in api] == [round(float(v), 6) for v in direct.to_numpy()]


def test_threshold_override_and_context(services, records, synth_df):
    svc = services["rf"]
    strict = svc.score(records, threshold=0.999)
    lax = svc.score(records, threshold=0.0)
    assert (
        strict["threshold"] == 0.999
        and lax["n_alerts"] == 200
        and strict["n_alerts"] <= lax["n_alerts"]
    )
    with_ctx = svc.score(records, context=to_records(synth_df.iloc[2500:3000]))
    assert with_ctx["n_transactions"] == 200


def test_rows_are_reported_in_posting_order_even_when_unsorted(services, records):
    reversed_records = list(reversed(records[:20]))
    out = services["rf"].score(reversed_records)
    assert [r["row"] for r in out["results"]] == list(range(20))
    first_posted = out["results"][0]
    assert first_posted["tx_id"] == 19  # the last posted row is the earliest transaction


def test_explain_by_row(services, records):
    for name, method in (("rf", "occlusion"), ("patterns", "patterns")):
        exp = services[name].explain(records, row=5)
        assert exp["method"] == method and exp["edges"] is not None
        auto = services[name].explain(records, row=5, method="auto")
        assert auto["method"] == method


def test_explain_validation(services, records):
    with pytest.raises(ConfigError, match="outside"):
        services["rf"].explain(records, row=10_000)
    with pytest.raises(ConfigError, match="motif detector"):
        services["rf"].explain(records, row=1, method="patterns")
    with pytest.raises(ConfigError, match="unknown explanation"):
        services["rf"].explain(records, row=1, method="magic")


def test_scan_returns_alert_evidence(services, records):
    out = services["rf"].scan(records, top=5)
    assert out["n_alerts"] >= len(out["alerts"]) and len(out["alerts"]) <= 5
    motif = services["patterns"].scan(records, top=5)
    assert "alerts" in motif and len(motif["alerts"]) <= 5


def test_input_validation(services):
    with pytest.raises(ConfigError, match="no transactions"):
        services["rf"].score([])
    with pytest.raises(ConfigError, match="too many"):
        services["rf"]._frame([{}] * (MAX_TRANSACTIONS + 1))
    with pytest.raises(ValueError):  # pydantic.ValidationError subclasses ValueError
        services["rf"].score([{"timestamp": "bad", "sender": "a", "receiver": "b", "amount": 1}])


def test_ibm_style_records_are_accepted(services):
    row = {
        "Timestamp": "2022/09/01 10:00",
        "From Bank": "11",
        "Account": "A",
        "To Bank": "12",
        "Account.1": "B",
        "Amount Received": 5.0,
        "Receiving Currency": "US Dollar",
        "Amount Paid": 5.0,
        "Payment Currency": "US Dollar",
        "Payment Format": "ACH",
    }
    out = services["rf"].score([row])
    assert out["n_transactions"] == 1


def test_from_dir(tmp_path, synth_df):
    TabularDetector.fit(synth_df, "logreg").save(tmp_path / "m")
    svc = ScoringService.from_dir(tmp_path / "m")
    assert svc.model_dir == str(tmp_path / "m") and svc.health()["model"]["kind"] == "logreg"
    with pytest.raises(ModelError):
        ScoringService.from_dir(tmp_path / "missing")


def test_error_payload_kinds():
    assert error_payload(ConfigError("x"))["error"] == "invalid_request"
    assert error_payload(RuntimeError("boom")) == {"error": "internal_error", "detail": "boom"}
