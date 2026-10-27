"""The code snippets of the documentation keep working (they are copied from docs/*.md)."""

import numpy as np
import pandas as pd
import pytest

import aml_graphminer as aml
from aml_graphminer.api.service import ScoringService
from aml_graphminer.detectors.bundle import MotifDetector, TabularDetector
from aml_graphminer.detectors.patterns import (
    DETECTOR_REGISTRY,
    PatternDetector,
    PatternScanner,
    register_detector,
)
from aml_graphminer.exceptions import ConfigError
from aml_graphminer.ingestion.adapters import Adapter


def test_extending_a_custom_motif_detector():
    class VeryLargePayment(PatternDetector):
        name = "very_large_payment"

        def detect(self, ctx):
            rows = np.flatnonzero(ctx.paid > 1_000_000)
            return [
                self.make_alert(
                    ctx,
                    self.name,
                    np.array([r]),
                    [int(ctx.src[r])],
                    1.0,
                    {"amount": float(ctx.paid[r])},
                )
                for r in rows
            ]

    register_detector("very_large_payment", lambda config: VeryLargePayment())
    try:
        df = pd.DataFrame(
            {
                "timestamp": ["2022-09-01 10:00", "2022-09-01 11:00"],
                "sender": ["a", "a"],
                "receiver": ["b", "c"],
                "amount": [2_000_000.0, 10.0],
            }
        )
        df = aml.read_transactions(df)
        result = PatternScanner(only=["very_large_payment"]).scan(df)
        assert len(result) == 1
        assert result.alerts[0].evidence == {"amount": 2_000_000.0}
    finally:
        DETECTOR_REGISTRY.pop("very_large_payment", None)


def test_extending_a_custom_input_adapter():
    class MyBankAdapter(Adapter):
        name = "mybank"

        def matches(self, columns):
            return "booking_date" in columns and "iban_from" in columns

        def to_frame(self, raw):
            return pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(raw["booking_date"]),
                    "sender": raw["iban_from"],
                    "receiver": raw["iban_to"],
                    "amount": raw["amount_eur"],
                    "currency": "EUR",
                }
            )

    raw = pd.DataFrame(
        {
            "booking_date": ["2023-01-02", "2023-01-01"],
            "iban_from": ["FR1", "FR2"],
            "iban_to": ["FR2", "FR3"],
            "amount_eur": [10.0, 20.0],
        }
    )
    df = aml.read_transactions(raw, schema=MyBankAdapter())
    assert list(df["sender"]) == ["FR2", "FR1"]  # sorted by time
    assert (df["currency"] == "EUR").all() and df["label"].isna().all()


def test_column_mapping_example_from_the_data_page():
    raw = pd.DataFrame(
        {
            "booking_date": ["2023-01-01 10:00"],
            "debtor_iban": ["FR1"],
            "creditor_iban": ["FR2"],
            "amt_eur": [12.5],
        }
    )
    df = aml.read_transactions(
        raw,
        schema={
            "timestamp": "booking_date",
            "sender": "debtor_iban",
            "receiver": "creditor_iban",
            "amount": "amt_eur",
        },
    )
    assert df["amount"].iloc[0] == 12.5


def test_rest_documentation_example_minimal_payload(synth_df):
    """The curl example of docs/rest_api.md posts only the four mandatory fields."""
    service = ScoringService(MotifDetector.fit(synth_df))
    answer = service.score(
        [
            {
                "timestamp": "2022-09-01 10:05:00",
                "sender": "70_ACC_A",
                "receiver": "12_ACC_C",
                "amount": 9500,
            },
            {
                "timestamp": "2022-09-01 10:20:00",
                "sender": "70_ACC_A",
                "receiver": "12_ACC_D",
                "amount": 9400,
            },
        ]
    )
    assert answer["n_transactions"] == 2
    assert [r["row"] for r in answer["results"]] == [0, 1]
    assert set(answer) == {"threshold", "n_transactions", "n_alerts", "results"}
    assert service.health()["status"] == "ok"


def test_quickstart_snippet(synth_df):
    detector = TabularDetector.fit(synth_df, "rf", params={"n_estimators": 30})
    assert detector.baseline.model.n_estimators == 30
    scores = detector.score(synth_df)
    tx_id = int(scores.idxmax())
    from aml_graphminer.explainers import OcclusionExplainer, detector_score_fn

    explanation = OcclusionExplainer(detector_score_fn(detector), hops=2).explain(synth_df, tx_id)
    assert explanation.tx_id == tx_id and "Transaction" in explanation.to_markdown()


def test_documented_pattern_config_example():
    config = aml.PatternConfig.from_dict({"fan_in": {"min_counterparties": 8}})
    assert config.fan_in.min_counterparties == 8
    scanner = PatternScanner(config, only=["fan_in", "cycle"])
    assert list(scanner.detectors) == ["fan_in", "cycle"]
    with pytest.raises(ConfigError, match="duration"):
        aml.PatternConfig.from_dict({"fan_in": {"window": "24x"}})
