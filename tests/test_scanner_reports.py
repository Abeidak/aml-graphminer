"""ScanResult reporting helpers, cycle-search budgets and the pydantic API schemas."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from aml_graphminer.api.schemas import (
    ExplainRequest,
    ScanRequest,
    ScoreRequest,
    ScoreResponse,
    TransactionIn,
)
from aml_graphminer.detectors.patterns import CycleDetector, PatternConfig, PatternScanner
from helpers import make_tx
from test_patterns import background, labelled


def test_empty_scan_result_has_a_typed_empty_table():
    df = make_tx(background(60))
    result = PatternScanner(only=["cycle"]).scan(df)
    assert len(result) == 0
    frame = result.to_frame()
    assert frame.empty and {"pattern", "score", "tx_ids"} <= set(frame.columns)
    scores, top = result.transaction_scores()
    assert not scores.any() and (top == "").all()
    assert result.summary()["n_alerts"] == 0


def test_scan_result_reports(tmp_path):
    cycle = labelled(
        [(60, "A", "B", 5000.0), (180, "B", "C", 4900.0), (360, "C", "A", 4800.0)], "cycle"
    )
    df = make_tx(background(80) + cycle)
    result = PatternScanner(only=["cycle"]).scan(df)
    assert len(result) == 1

    accounts = result.account_scores()
    assert set(accounts) == {"A", "B", "C"} and all(0 < v <= 1 for v in accounts.values())

    summary = result.summary()
    assert summary["alerts_by_pattern"] == {"cycle": 1}
    assert summary["n_flagged_transactions"] == 3
    assert summary["n_transactions"] == len(df)

    out = tmp_path / "alerts.json"
    text = result.to_json(str(out))
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(text)
    assert json.loads(text)["alerts"][0]["pattern"] == "cycle"
    assert "\n" not in result.to_json(indent=None)


def test_calibrated_weights_change_the_reported_scores():
    cycle = labelled(
        [(60, "A", "B", 5000.0), (180, "B", "C", 4900.0), (360, "C", "A", 4800.0)], "cycle"
    )
    df = make_tx(cycle)
    scanner = PatternScanner(only=["cycle"])
    raw = scanner.scan(df)
    scanner.calibration = {"cycle": 0.5}
    half = scanner.scan(df)
    assert half.to_frame()["score"].iloc[0] == pytest.approx(0.5 * raw.to_frame()["score"].iloc[0])
    assert half.calibration == {"cycle": 0.5}


def _two_cycles():
    rows = []
    for tag, base in (("x", 0), ("y", 1000)):
        a, b, c = (f"{tag}{i}" for i in range(3))
        rows += [
            (base + 10, a, b, 5000.0),
            (base + 70, b, c, 4900.0),
            (base + 130, c, a, 4800.0),
        ]
    return make_tx(rows)


def test_cycle_search_stops_at_max_cycles_and_flags_truncation():
    df = _two_cycles()
    full = CycleDetector()
    assert len(full.detect_frame(df)) == 2 and full.truncated is False

    capped = CycleDetector(PatternConfig().cycle.model_copy(update={"max_cycles": 1}))
    assert len(capped.detect_frame(df)) == 1 and capped.truncated is True


def test_cycle_search_respects_the_expansion_budget():
    df = _two_cycles()
    tight = CycleDetector(PatternConfig().cycle.model_copy(update={"max_expansions": 2}))
    alerts = tight.detect_frame(df)
    assert tight.truncated is True and len(alerts) < 2


def test_disabled_cycle_detector_and_short_inputs_return_nothing():
    off = CycleDetector(PatternConfig.from_dict({"cycle": {"enabled": False}}).cycle)
    assert off.detect_frame(_two_cycles()) == []
    assert CycleDetector().detect_frame(make_tx([(0, "A", "B", 10.0), (5, "B", "C", 10.0)])) == []


def test_transaction_in_schema_defaults_and_validation():
    tx = TransactionIn(timestamp="2022-09-01 10:00:00", sender="a", receiver="b", amount=10)
    assert tx.currency == "US Dollar" and tx.amount_received is None
    with pytest.raises(ValidationError):
        TransactionIn(timestamp="2022-09-01", sender="a", receiver="b", amount=-1)
    extra = TransactionIn(timestamp="t", sender="a", receiver="b", amount=1, note="kept")
    assert extra.model_dump()["note"] == "kept"


def test_request_schemas_enforce_bounds():
    assert ScoreRequest(transactions=[{"a": 1}], threshold=0.5).context is None
    with pytest.raises(ValidationError):
        ScoreRequest(transactions=[])
    with pytest.raises(ValidationError):
        ScoreRequest(transactions=[{"a": 1}], threshold=1.5)
    assert ExplainRequest(transactions=[{"a": 1}], row=0).max_edges == 80
    for bad in ({"row": -1}, {"row": 0, "hops": 9}, {"row": 0, "max_edges": 1}):
        with pytest.raises(ValidationError):
            ExplainRequest(transactions=[{"a": 1}], **bad)
    assert ScanRequest(transactions=[{"a": 1}]).top == 20
    with pytest.raises(ValidationError):
        ScanRequest(transactions=[{"a": 1}], top=0)


def test_score_response_roundtrip():
    payload = {
        "threshold": 0.5,
        "n_transactions": 1,
        "n_alerts": 0,
        "results": [{"row": 0, "tx_id": 3, "score": float(np.float32(0.25)), "alert": False}],
    }
    assert ScoreResponse.model_validate(payload).model_dump() == payload
