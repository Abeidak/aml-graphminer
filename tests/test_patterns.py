"""Structural-motif detectors on hand-built scenarios (ground truth known by construction)."""

import numpy as np
import pandas as pd
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
    exchange_rates,
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


def test_cycle_rejects_hops_whose_exchange_rate_is_unknown_by_default():
    # no cross-currency transaction reveals the dollar / euro rate: the hop cannot be checked
    df = make_tx([(0, "A", "B", 5000.0), (60, "B", "C", 4900.0), (120, "C", "A", 4800.0)])
    df.loc[1, "currency"] = "Euro"
    df.loc[1, "currency_received"] = "Euro"
    assert scan_one(CycleDetector(), df) == []
    config = PatternConfig.from_dict({"cycle": {"cross_currency": True}}).cycle
    assert len(scan_one(CycleDetector(config), df)) == 1


def _cycle_through_euros(forwarded_eur: float):
    """A -> B in dollars, B -> C in euros, C -> A in dollars; X -> Y reveals the rate 0.85."""
    df = make_tx(
        [
            (0, "A", "B", 5000.0, 1.0, "cycle"),
            (60, "B", "C", forwarded_eur, 1.0, "cycle"),
            (120, "C", "A", 4800.0, 1.0, "cycle"),
            (500, "X", "Y", 1000.0),
        ]
    )
    df.loc[1, ["currency", "currency_received"]] = "Euro"
    df.loc[3, ["currency_received", "amount_received"]] = ["Euro", 850.0]
    return df


def test_cycle_follows_money_that_changes_currency():
    df = _cycle_through_euros(4900.0 * 0.85)  # B forwards 98 % of what it got, in euros
    scanner = PatternScanner(only=["cycle"])
    alerts = scanner.scan(df).alerts
    assert len(alerts) == 1 and alerts[0].evidence["length"] == 3
    # amounts reported in the reference currency (US dollars)
    assert alerts[0].evidence["amount_retained"] == pytest.approx(4800 / 5000)
    # without conversion the euro hop cannot be compared and the cycle is lost
    off = PatternConfig.from_dict({"currency": {"convert": False}})
    assert PatternScanner(off, only=["cycle"]).scan(df).alerts == []
    # conversion also enforces conservation across currencies
    leaky = _cycle_through_euros(2000.0 * 0.85)
    assert scanner.scan(leaky).alerts == []


def test_exchange_rates_are_inferred_transitively_and_can_be_given():
    names = np.array(["US Dollar", "Euro", "UK Pound", "Yen"], dtype=object)
    # USD -> EUR at 0.85 (three times, one rounded), EUR -> GBP at 0.9; the yen is never exchanged
    paid = np.array([100.0, 200.0, 300.0, 100.0, 50.0])
    recv = np.array([85.0, 170.0, 255.1, 90.0, 50.0])
    cur_paid = np.array([0, 0, 0, 1, 3])
    cur_recv = np.array([1, 1, 1, 2, 3])
    rates = exchange_rates(paid, recv, cur_paid, cur_recv, names, reference="US Dollar")
    assert rates[0] == 1.0
    assert rates[1] == pytest.approx(1 / 0.85)
    assert rates[2] == pytest.approx(1 / (0.85 * 0.9))
    assert np.isnan(rates[3])
    given = exchange_rates(paid, recv, cur_paid, cur_recv, names, known={"Yen": 0.007})
    assert given[3] == pytest.approx(0.007)
    # unknown reference: the most used payment currency
    other = exchange_rates(paid, recv, cur_paid, cur_recv, names, reference="Peso")
    assert other[0] == 1.0
    assert exchange_rates(paid, recv, cur_paid, cur_recv, names[:0]).size == 0


def test_alert_amounts_are_converted_to_the_reference_currency():
    df = _cycle_through_euros(4900.0 * 0.85)
    ctx = TxArrays.from_frame(df)
    assert ctx.paid_c[1] == pytest.approx(4900.0)
    received, forwarded, comparable = ctx.hop_amounts(np.array([0, 1]), np.array([1, 2]))
    assert comparable.all()
    assert forwarded / received == pytest.approx([0.98, 4800 / 4900])
    off = TxArrays.from_frame(
        df, PatternConfig.from_dict({"currency": {"convert": False}}).currency
    )
    assert np.isnan(off.fx).all() and off.paid_c[1] == pytest.approx(4900.0 * 0.85)
    assert not off.hop_amounts(np.array([0]), np.array([1]))[2].any()


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
    assert set(result.timings) == set(PatternConfig().enabled_names())  # relay is opt-in
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
    assert set(calibration) == set(PatternConfig().enabled_names())
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


def test_scan_with_history_sees_motifs_that_started_before_the_slice():
    rows = [
        (0, "A", "B", 5000.0, 1.0, "cycle"),
        (60, "B", "C", 4900.0, 1.0, "cycle"),
        (120, "C", "A", 4800.0, 1.0, "cycle"),
        (10, "P", "Q", 100.0),
        (20, "Q", "R", 100.0),
        (30, "R", "P", 100.0, 1.0, "cycle"),  # a cycle entirely in the history
    ]
    df = make_tx(rows)
    slice_ = df[df["timestamp"] >= df["timestamp"].iloc[0] + pd.Timedelta(minutes=100)]
    history = df[~df.index.isin(slice_.index)]
    scanner = PatternScanner(only=["cycle"])
    assert scanner.scan(slice_).alerts == []  # the slice alone holds one hop of the loop
    result = scanner.scan(slice_, history)
    assert len(result.alerts) == 1 and len(result.alerts[0].tx_ids) == 3
    assert list(result.tx_ids) == list(slice_.index)  # only the slice is scored
    scores, _ = result.transaction_scores()
    assert scores.tolist() == [scores.max()] and scores.max() > 0
    # calibration only reads the labels of the slice (history labels are hidden here)
    hidden = history.assign(label=np.nan)
    assert scanner.fit_calibration(slice_, history=hidden)["cycle"] == pytest.approx(2 / 3)


def test_scan_can_be_restricted_to_some_payment_formats():
    df = make_tx([(0, "A", "B", 5000.0), (60, "B", "C", 4900.0), (120, "C", "A", 4800.0)])
    assert len(PatternScanner(only=["cycle"]).scan(df).alerts) == 1
    wire_only = PatternConfig.from_dict({"transaction_types": ["Wire"]})
    result = PatternScanner(wire_only, only=["cycle"]).scan(df)
    assert result.alerts == [] and len(result.tx_ids) == 3  # every transaction is still scored
    assert PatternConfig.from_dict({"lookback": None}).lookback is None
    with pytest.raises(ConfigError):
        PatternConfig.from_dict({"lookback": "soon"})


def test_fan_minimum_span_and_amount_skip_quick_small_bursts():
    t0 = 3 * 24 * 60
    burst = [(t0 + i, f"src{i}", "payroll", 300.0) for i in range(8)]  # 8 payments in 8 minutes
    slow = labelled([(t0 + 600 * i, f"m{i}", "collector", 9_000.0) for i in range(6)], "fan_in")
    df = make_tx(background() + burst + slow)
    wide = PatternConfig.from_dict({"fan_in": {"window": "72h", "min_burst_ratio": 0}}).fan_in
    hubs = {a.evidence["hub"] for a in scan_one(FanInDetector(wide), df)}
    assert {"payroll", "collector"} <= hubs
    cfg = PatternConfig.from_dict(
        {
            "fan_in": {
                "window": "72h",
                "min_span": "24h",
                "min_mean_amount": 5000,
                "min_burst_ratio": 0,
            }
        }
    ).fan_in
    hubs = {a.evidence["hub"] for a in scan_one(FanInDetector(cfg), df)}
    assert hubs == {"collector"}
    with pytest.raises(ConfigError):
        PatternConfig.from_dict({"fan_in": {"min_span": "later"}})


def test_gather_scatter_may_pay_out_before_it_is_paid():
    t0 = 3000
    rows = [(t0 - 300 + 8 * i, "hub", f"t{i}", 1_950.0) for i in range(3)] + [
        (t0 + 8 * i, f"s{i}", "hub", 2_000.0) for i in range(3)
    ]
    df = make_tx(background() + labelled(rows, "gather_scatter"))
    assert covered(scan_one(GatherScatterDetector(), df), df, "gather_scatter") < 1.0
    cfg = PatternConfig.from_dict({"gather_scatter": {"scatter_lead": "12h"}}).gather_scatter
    assert covered(scan_one(GatherScatterDetector(cfg), df), df, "gather_scatter") == 1.0


def test_scanner_exclusions_keep_rates_and_skip_internal_transfers():
    rows = [(0, "A", "B", 5000.0), (60, "B", "C", 4900.0), (120, "C", "A", 4800.0)]
    df = make_tx(rows + [(500, "X", "Y", 1000.0)])
    df.loc[1, ["currency", "currency_received", "amount", "amount_received"]] = [
        "Euro",
        "Euro",
        4900.0 * 0.85,
        4900.0 * 0.85,
    ]
    df.loc[3, ["currency_received", "amount_received"]] = ["Euro", 850.0]  # reveals the rate
    cfg = PatternConfig.from_dict({"exclude": ["cross_currency", "self_transfers"]})
    # the only cross-currency row is excluded, yet its rate still converts the euro hop
    assert len(PatternScanner(cfg, only=["cycle"]).scan(df).alerts) == 1
    df.loc[:2, ["sender_bank", "receiver_bank"]] = "001"
    same_bank = PatternConfig.from_dict({"exclude": ["same_bank"]})
    assert PatternScanner(same_bank, only=["cycle"]).scan(df).alerts == []
    with pytest.raises(ConfigError):
        PatternConfig.from_dict({"exclude": ["weekends"]})


def test_noisy_or_combines_the_alerts_of_a_transaction():
    from aml_graphminer.detectors.patterns.base import Alert
    from aml_graphminer.detectors.patterns.scanner import ScanResult

    ts = pd.Timestamp("2022-09-01")
    alerts = [Alert("a", 0.5, ["x"], [1, 2], ts, ts, 1.0), Alert("b", 0.5, ["x"], [2], ts, ts, 1.0)]
    best, _ = ScanResult(alerts, np.array([1, 2, 3])).transaction_scores()
    both, _ = ScanResult(alerts, np.array([1, 2, 3]), combine="noisy_or").transaction_scores()
    assert best.tolist() == [0.5, 0.5, 0.0]
    assert both == pytest.approx([0.5, 0.75, 0.0])


def test_calibration_by_score_bands_reorders_alerts_and_survives_saving(synth_df, tmp_path):
    from aml_graphminer.detectors.bundle import MotifDetector, load_detector

    one = PatternScanner()
    one.fit_calibration(synth_df)
    assert one.bin_calibration == {}
    cfg = PatternConfig.from_dict({"calibration_bins": 3})
    banded = PatternScanner(cfg)
    banded.fit_calibration(synth_df)
    assert banded.bin_calibration  # at least one detector has enough alerts for 3 bands
    for edges, precision in banded.bin_calibration.values():
        assert len(precision) == len(edges) + 1 and all(0 < p < 1 for p in precision)
        assert edges == sorted(edges)
    result = banded.scan(synth_df)
    for alert in result.alerts[:50]:
        if alert.pattern in result.bins:
            edges, precision = result.bins[alert.pattern]
            assert result.weight(alert) <= max(precision)
    det = MotifDetector.fit(synth_df, cfg)
    det.save(tmp_path / "m")
    again = load_detector(tmp_path / "m")
    assert again.scanner.bin_calibration == det.scanner.bin_calibration
    assert np.allclose(again.score(synth_df.iloc[:500]), det.score(synth_df.iloc[:500]))
    with pytest.raises(ConfigError):
        PatternConfig.from_dict({"calibration_bins": 0})


def test_relay_finds_money_forwarded_in_another_currency():
    from aml_graphminer.detectors.patterns import RelayDetector

    rows = [
        (0, "A", "B", 5000.0, 1.0, "stack"),
        (600, "B", "C", 4850.0 * 0.85, 1.0, "stack"),  # 97 % forwarded, in euros, 10 h later
        (700, "X", "Y", 1000.0),  # reveals the dollar / euro rate
        (900, "D", "E", 3000.0),
        (950, "E", "F", 1000.0),  # forwards a third only: not a relay
    ]
    df = make_tx(background(200) + rows)
    n = len(df)
    euro = df.index[(df["sender"] == "B") & (df["receiver"] == "C")]
    df.loc[euro, ["currency", "currency_received"]] = "Euro"
    fx = df.index[(df["sender"] == "X")]
    df.loc[fx, ["currency_received", "amount_received"]] = ["Euro", 850.0]
    assert len(df) == n
    cfg = PatternConfig.from_dict({"relay": {"enabled": True}}).relay
    alerts = RelayDetector(cfg).detect_frame(df)
    assert len(alerts) == 1 and alerts[0].accounts == ["A", "B", "C"]
    assert alerts[0].evidence["forwarded_share"] == pytest.approx(0.97)
    same = PatternConfig.from_dict({"relay": {"enabled": True, "currency_change": False}}).relay
    assert len(RelayDetector(same).detect_frame(df)) >= 1
    assert RelayDetector().detect_frame(df) == []  # opt-in
