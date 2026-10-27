"""Risk-community detector on hand-built scenarios (ground truth known by construction)."""

import numpy as np
import pytest

from aml_graphminer.detectors.patterns import (
    CommunityConfig,
    CommunityDetector,
    PatternConfig,
    PatternScanner,
    pattern_recall_table,
)
from aml_graphminer.exceptions import ConfigError
from aml_graphminer.explainers import PATTERN_DESCRIPTIONS, explain_with_patterns
from helpers import make_tx

DAY = 24 * 60  # minutes


def ring(prefix, start, amount=5000.0, size=4, step=90.0):
    """``size`` fresh accounts passing (almost) the same amount around a loop."""
    rows = []
    for i in range(size):
        rows.append((start + i * step, f"{prefix}{i}", f"{prefix}{(i + 1) % size}", amount))
        amount *= 0.98
    return rows


def scatter_gather(prefix, start, k=4, amount=3000.0):
    rows = [(start + i, f"{prefix}_src", f"{prefix}_m{i}", amount) for i in range(k)]
    rows += [(start + 120 + i, f"{prefix}_m{i}", f"{prefix}_sink", amount * 0.97) for i in range(k)]
    return rows


def spread_background(n=300, accounts=40, days=10, seed=0):
    """Accounts that both send and receive but over the whole period (not a burst)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        s, r = rng.choice(accounts, 2, replace=False)
        rows.append((float(rng.uniform(0, days * DAY)), f"bg{s}", f"bg{r}", 200.0))
    return rows


def detect(rows, **cfg):
    return CommunityDetector(CommunityConfig(**cfg)).detect_frame(make_tx(rows))


def test_laundering_ring_is_a_risk_community():
    alerts = detect(ring("r", 2 * DAY))
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.pattern == "community" and sorted(alert.accounts) == ["r0", "r1", "r2", "r3"]
    ev = alert.evidence
    assert ev["has_cycle"] and ev["n_relays"] == 4 and ev["n_internal_tx"] == 4
    assert ev["internal_share"] == pytest.approx(1.0)
    assert 0.3 < alert.score <= 1.0 and len(alert.tx_ids) == 4


def test_scatter_gather_group_includes_source_and_sink():
    alerts = detect(scatter_gather("sg", DAY))
    assert len(alerts) == 1
    members = set(alerts[0].accounts)
    assert {"sg_src", "sg_sink"} <= members and len(members) == 6
    assert alerts[0].evidence["n_relays"] == 4 and not alerts[0].evidence["has_cycle"]


def test_activity_spread_over_the_period_is_not_a_community_alert():
    assert detect(spread_background()) == []


def test_slow_ring_exceeds_max_span():
    slow = ring("s", 0, step=2 * DAY)  # 6 days around the loop
    assert detect(slow) == []
    assert len(detect(slow, max_span="7d")) == 1


def test_internal_share_and_relay_requirements():
    rows = ring("r", DAY)
    # busy hubs (a bank and an exchange) with many other customers: ignored by the detector
    rows += [(DAY + 3 * i, f"c{i}", "exchange", 30.0) for i in range(60)]
    rows += [(DAY + 3 * i, "bank", f"d{i}", 30.0) for i in range(60)]
    # the ring members receive from / pay to the hubs much more than they circulate internally,
    # so most of their outflow leaves the group: no alert
    rows += [(DAY + 10 + i, "bank", f"r{i}", 40_000.0) for i in range(4)]
    rows += [(DAY + 20 + i, f"r{i}", "exchange", 40_000.0) for i in range(4)]
    assert detect(rows) == []
    assert len(detect(rows, min_internal_share=0.05)) == 1
    # a group without enough relays is not reported either
    assert detect(ring("q", DAY), min_relays=5) == []


def test_hubs_are_ignored():
    rows = ring("r", DAY)
    # 'shop' is touched by every ring member and many customers: a hub, it must not merge
    rows += [(DAY + 5 * i, f"c{i}", "shop", 30.0) for i in range(60)]
    rows += [(DAY + 300 + i, "shop", f"r{i % 4}", 30.0) for i in range(4)]
    alerts = detect(rows)
    assert len(alerts) == 1 and "shop" not in alerts[0].accounts


def test_large_components_are_split_with_louvain():
    rows = ring("a", DAY, size=5) + ring("b", DAY + 30, size=5)
    rows.append((DAY + 200, "a0", "b0", 4800.0))  # a single bridge links the two rings
    alerts = detect(rows, max_size=6)
    groups = sorted(sorted(a.accounts) for a in alerts)
    assert groups == [[f"a{i}" for i in range(5)], [f"b{i}" for i in range(5)]]
    assert detect(rows, max_size=6, min_size=6) == []


def test_disabled_or_empty_inputs_return_nothing():
    assert (
        CommunityDetector(CommunityConfig(enabled=False)).detect_frame(make_tx(ring("r", 0))) == []
    )
    assert detect([(0, "a", "b", 10.0)]) == []


def test_invalid_configuration_is_rejected():
    with pytest.raises(ConfigError):
        PatternConfig.from_dict({"community": {"max_span": "soon"}})
    with pytest.raises(ConfigError):
        PatternConfig.from_dict({"community": {"min_flow_through": 0}})


def test_scanner_reports_and_explains_communities():
    rows = [(m, s, r, a, 1.0, "cycle") for m, s, r, a in ring("r", DAY)] + spread_background()
    df = make_tx(rows)
    result = PatternScanner(only=["community"]).scan(df)
    assert {a.pattern for a in result.alerts} == {"community"}
    table = pattern_recall_table(df, result)
    assert table.set_index("pattern").loc["cycle", "recall_any"] == pytest.approx(1.0)
    tx = int(df.index[df["label"] == 1.0][0])
    explanation = explain_with_patterns(df, tx, result)
    assert "community" in explanation.summary and "community" in PATTERN_DESCRIPTIONS
    assert result.summary()["alerts_by_pattern"] == {"community": 1}
