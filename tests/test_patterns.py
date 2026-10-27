"""Structural-motif detectors on hand-built scenarios (ground truth known by construction)."""

import numpy as np
import pytest

from aml_graphminer.detectors.patterns import (
    DETECTOR_REGISTRY,
    CycleDetector,
    FanInDetector,
    FanOutDetector,
    GatherScatterDetector,
    PatternConfig,
    PatternScanner,
    ScatterGatherDetector,
    SmurfingDetector,
    alert_precision_table,
    evaluate_scan,
    pattern_recall_table,
    register_detector,
)
from aml_graphminer.detectors.patterns.base import (
    PatternDetector,
    TxArrays,
    group_slices,
    sliding_windows,
)
from aml_graphminer.exceptions import ConfigError
from helpers import make_tx


def background(n=400, accounts=60, days=6, seed=0, prefix="bg"):
    """Sparse legitimate traffic from a pool of payers to a disjoint pool of payees.

    Payers never receive, so the background contains no directed cycle by construction.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        s, r = rng.integers(0, accounts, 2)
        rows.append(
            (
                float(rng.uniform(0, days * 24 * 60)),
                f"{prefix}_payer{s}",
                f"{prefix}_payee{r}",
                float(rng.lognormal(5, 0.8)),
            )
        )
    return rows


def labelled(rows, tag):
    return [(m, s, r, a, 1.0, tag) for m, s, r, a in rows]


def scan_one(detector, df):
    return detector.detect_frame(df)


def covered(alerts, df, pattern):
    ids = {t for a in alerts for t in a.tx_ids}
    truth = set(df.index[df["pattern"] == pattern])
    return len(ids & truth) / max(len(truth), 1)


def test_fan_out_detected():
    t0 = 3 * 24 * 60
    motif = labelled([(t0 + 10 * i, "hub", f"mule{i}", 800.0 + i) for i in range(8)], "fan_out")
    df = make_tx(background() + motif)
    alerts = scan_one(FanOutDetector(), df)
    assert covered(alerts, df, "fan_out") == 1.0
    top = max(alerts, key=lambda a: a.score)
    assert top.pattern == "fan_out" and "hub" in top.accounts
    assert 0 < top.score <= 1


def test_fan_in_detected():
    t0 = 3 * 24 * 60
    motif = labelled([(t0 + 10 * i, f"src{i}", "collector", 900.0 + i) for i in range(8)], "fan_in")
    df = make_tx(background() + motif)
    assert covered(scan_one(FanInDetector(), df), df, "fan_in") == 1.0


def test_background_alone_raises_no_fan_alert():
    df = make_tx(background(300, accounts=300))
    assert scan_one(FanOutDetector(), df) == [] and scan_one(FanInDetector(), df) == []


def test_busy_hub_with_steady_traffic_is_not_a_burst():
    rows = background() + [(i * 20.0, "shop", f"cust{i % 40}", 25.0) for i in range(6 * 24 * 3)]
    df = make_tx(rows)
    assert not [a for a in scan_one(FanInDetector(), df) if "shop" in a.accounts]


def test_smurfing_structuring_below_threshold():
    t0 = 2 * 24 * 60
    motif = labelled(
        [(t0 + 30 * i, "smurf", f"dest{i}", 9_000.0 + 90 * i) for i in range(7)], "smurfing"
    )
    df = make_tx(background() + motif)
    alerts = scan_one(SmurfingDetector(), df)
    assert covered(alerts, df, "smurfing") >= 0.85
    assert any(a.evidence.get("mode") == "structuring" for a in alerts)


def test_smurfing_split_of_a_large_inflow():
    t0 = 2 * 24 * 60
    rows = [(t0, "origin", "splitter", 52_000.0)] + [
        (t0 + 60 + 20 * i, "splitter", f"p{i}", 10_400.0) for i in range(5)
    ]
    tagged = [(m, s, r, a, 1.0, "smurfing") for m, s, r, a in rows]
    df = make_tx(background() + tagged)
    alerts = scan_one(SmurfingDetector(), df)
    assert any(a.evidence.get("mode") == "split" for a in alerts)


def test_cycle_detected_and_amount_conservation_enforced():
    t0 = 60
    good = labelled(
        [(t0, "A", "B", 5000.0), (t0 + 120, "B", "C", 4900.0), (t0 + 300, "C", "A", 4800.0)],
        "cycle",
    )
    df = make_tx(background(200) + good)
    alerts = scan_one(CycleDetector(), df)
    assert covered(alerts, df, "cycle") == 1.0
    assert alerts[0].evidence["length"] == 3

    # the money shrinks too much along the way → not a laundering loop
    leaky = make_tx(
        background(200)
        + [(t0, "A", "B", 5000.0), (t0 + 120, "B", "C", 2000.0), (t0 + 300, "C", "A", 500.0)]
    )
    assert scan_one(CycleDetector(), leaky) == []


def test_cycle_must_be_time_respecting():
    backwards = make_tx([(300, "A", "B", 5000.0), (120, "B", "C", 4900.0), (10, "C", "A", 4800.0)])
    assert scan_one(CycleDetector(), backwards) == []


def test_cycle_ignores_amounts_in_different_currencies_by_default():
    df = make_tx([(0, "A", "B", 5000.0), (60, "B", "C", 4900.0), (120, "C", "A", 4800.0)])
    df.loc[1, "currency"] = "Euro"
    df.loc[1, "currency_received"] = "Euro"
    assert scan_one(CycleDetector(), df) == []
    config = PatternConfig.from_dict({"cycle": {"cross_currency": True}}).cycle
    assert len(scan_one(CycleDetector(config), df)) == 1


def test_scatter_gather_detected():
    t0 = 100
    rows = [(t0 + 5 * i, "source", f"mid{i}", 2_000.0) for i in range(4)] + [
        (t0 + 300 + 5 * i, f"mid{i}", "sink", 1_950.0) for i in range(4)
    ]
    df = make_tx(background() + labelled(rows, "scatter_gather"))
    alerts = scan_one(ScatterGatherDetector(), df)
    assert covered(alerts, df, "scatter_gather") == 1.0
    assert alerts[0].evidence["n_intermediaries"] == 4


def test_gather_scatter_detected():
    t0 = 100
    rows = [(t0 + 8 * i, f"s{i}", "hub", 2_000.0) for i in range(4)] + [
        (t0 + 240 + 8 * i, "hub", f"t{i}", 1_950.0) for i in range(4)
    ]
    df = make_tx(background() + labelled(rows, "gather_scatter"))
    alerts = scan_one(GatherScatterDetector(), df)
    assert covered(alerts, df, "gather_scatter") == 1.0


def test_gather_scatter_requires_conserved_volume():
    rows = [(100 + 8 * i, f"s{i}", "hub", 2_000.0) for i in range(4)] + [
        (400 + 8 * i, "hub", f"t{i}", 50.0) for i in range(4)
    ]
    df = make_tx(background() + rows)
    assert scan_one(GatherScatterDetector(), df) == []


def test_scanner_runs_all_detectors_and_scores_transactions(synth_df):
    scanner = PatternScanner()
    result = scanner.scan(synth_df)
    assert set(result.timings) == set(DETECTOR_REGISTRY)
    scores, _ = result.transaction_scores()
    assert scores.shape == (len(synth_df),) and ((scores >= 0) & (scores <= 1)).all()
    metrics = evaluate_scan(synth_df, result)
    assert metrics["roc_auc"] > 0.85 and metrics["recall"] > 0.7
    table = pattern_recall_table(synth_df, result)
    assert set(table["pattern"]) == set(synth_df.loc[synth_df["label"] == 1, "pattern"])
    assert (table["recall_any"] >= table["recall_same"]).all()
    prec = alert_precision_table(synth_df, result)
    assert {"n_alerts", "tx_precision"} <= set(prec.columns)
    frame = result.to_frame()
    assert frame["score"].is_monotonic_decreasing
    assert len(frame) == len(result)


def test_scanner_only_and_disabled_sections():
    with pytest.raises(ConfigError, match="unknown detector"):
        PatternScanner(only=["nope"])
    scanner = PatternScanner(only=["cycle"])
    assert list(scanner.detectors) == ["cycle"]
    config = PatternConfig.from_dict({"smurfing": {"enabled": False}})
    assert "smurfing" not in PatternScanner(config).detectors
    assert "smurfing" not in config.enabled_names()


def test_calibration_scales_alert_scores(synth_df, synth_masks):
    scanner = PatternScanner()
    val = synth_df[synth_masks["val"]]
    calibration = scanner.fit_calibration(val)
    assert set(calibration) == set(DETECTOR_REGISTRY)
    assert all(0 < v <= 1 for v in calibration.values())
    result = scanner.scan(val)
    raw_max = max((a.score for a in result.alerts), default=0.0)
    assert max((result.weight(a) for a in result.alerts), default=0.0) <= raw_max + 1e-12
    with pytest.raises(ConfigError, match="labelled"):
        scanner.fit_calibration(val.assign(label=np.nan))


def test_register_custom_detector():
    class EveryBigTransaction(PatternDetector):
        name = "big"

        def detect(self, ctx):
            rows = np.flatnonzero(ctx.paid > 1e6)
            return [
                self.make_alert(
                    ctx,
                    "big",
                    np.array([r]),
                    [int(ctx.src[r])],
                    1.0,
                    {"amount": float(ctx.paid[r])},
                )
                for r in rows
            ]

    register_detector("big", lambda config: EveryBigTransaction())
    try:
        df = make_tx([(0, "a", "b", 2e6), (1, "a", "b", 10.0)])
        result = PatternScanner(only=["big"]).scan(df)
        assert len(result) == 1 and result.alerts[0].pattern == "big"
    finally:
        DETECTOR_REGISTRY.pop("big", None)


def test_alert_is_json_serialisable(synth_df):
    import json

    alerts = PatternScanner().scan(synth_df).alerts
    payload = json.dumps([a.to_dict() for a in alerts[:20]])
    assert "pattern" in payload


def test_tx_arrays_requires_sorted_table(synth_df):
    arrays = TxArrays.from_frame(synth_df)
    assert len(arrays) == len(synth_df)
    assert arrays.currency_code("US Dollar") >= 0 and arrays.currency_code("Martian") == -1
    with pytest.raises(ValueError, match="sorted"):
        TxArrays.from_frame(synth_df.sample(frac=1.0, random_state=1))


def test_group_slices_and_sliding_windows():
    keys = np.array([2, 0, 2, 1, 0, 2])
    groups = dict(group_slices(keys))
    assert {k: v.tolist() for k, v in groups.items()} == {0: [1, 4], 1: [3], 2: [0, 2, 5]}
    assert [k for k, _ in group_slices(keys, candidates=np.array([1]))] == [1]
    assert group_slices(np.zeros(0, dtype=int)) == []
    ts = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    others = np.array([1, 2, 3, 4, 5, 6])
    assert sliding_windows(ts, others, window=3.0, min_distinct=3) == [(0, 3), (3, 6)]
    assert sliding_windows(ts, others, window=3.0, min_distinct=4) == []
    assert sliding_windows(ts, others, window=3.0, min_distinct=2, accept=lambda i, j: i >= 3) == [
        (3, 6)
    ]
