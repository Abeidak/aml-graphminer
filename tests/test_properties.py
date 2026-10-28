"""Property-based tests (Hypothesis) of the motif detectors, the scanner and the graph converters."""

import networkx as nx
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aml_graphminer.detectors.patterns import CycleDetector, PatternScanner, exchange_rates
from aml_graphminer.graph.convert import arrays_to_nx, nx_to_arrays
from helpers import make_tx

SETTINGS = settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])

# random tables: few accounts (so motifs do appear), distinct minutes (so the order is unique)
tables = st.lists(
    st.tuples(
        st.integers(0, 7),
        st.integers(0, 7),
        st.floats(10, 20_000, allow_nan=False),
    ),
    min_size=3,
    max_size=60,
).map(
    lambda rows: [
        (minute * 37, f"acc{s}", f"acc{r if r != s else (r + 1) % 8}", round(a, 2))
        for minute, (s, r, a) in enumerate(rows)
    ]
)


def _scores_by_content(df):
    scores, _ = PatternScanner().scan(df).transaction_scores()
    keys = zip(df["timestamp"], df["sender"], df["receiver"], df["amount"])
    return dict(zip(keys, np.round(scores, 12)))


@SETTINGS
@given(rows=tables, seed=st.integers(0, 2**16))
def test_scanner_ignores_the_order_of_the_input_rows(rows, seed):
    df = make_tx(rows)
    shuffled = make_tx([rows[i] for i in np.random.default_rng(seed).permutation(len(rows))])
    assert _scores_by_content(df) == _scores_by_content(shuffled)


@SETTINGS
@given(rows=tables)
def test_scan_scores_are_probabilities_and_alerts_cite_real_transactions(rows):
    df = make_tx(rows)
    result = PatternScanner().scan(df)
    scores, _ = result.transaction_scores()
    assert len(scores) == len(df) and ((scores >= 0) & (scores <= 1)).all()
    ids = set(df.index.tolist())
    for alert in result.alerts:
        assert set(alert.tx_ids) <= ids and 0 <= alert.score <= 1
        assert alert.start <= alert.end


@SETTINGS
@given(rows=tables, cut=st.integers(1, 59))
def test_scan_with_history_only_scores_and_reports_the_slice(rows, cut):
    df = make_tx(rows)
    cut = min(cut, len(df) - 1)
    part, history = df.iloc[cut:], df.iloc[:cut]
    result = PatternScanner().scan(part, history)
    assert list(result.tx_ids) == list(part.index)
    own = set(part.index.tolist())
    assert all(own & set(a.tx_ids) for a in result.alerts)
    # an empty history changes nothing
    alone = PatternScanner().scan(part, history.iloc[:0])
    assert [a.tx_ids for a in alone.alerts] == [
        a.tx_ids for a in PatternScanner().scan(part).alerts
    ]


@SETTINGS
@given(
    length=st.integers(3, 6),
    first=st.floats(1_000, 50_000),
    kept=st.lists(st.floats(0.85, 1.0), min_size=6, max_size=6),
    gaps=st.lists(st.integers(1, 600), min_size=6, max_size=6),
)
def test_a_planted_time_respecting_cycle_is_always_found(length, first, kept, gaps):
    rows, amount, minute = [], first, 0
    for k in range(length):
        rows.append((minute, f"c{k}", f"c{(k + 1) % length}", round(amount, 2)))
        amount *= kept[k]
        minute += gaps[k]
    background = [(m * 7 + 3, f"payer{m % 5}", f"payee{m % 4}", 100.0) for m in range(40)]
    alerts = CycleDetector().detect_frame(make_tx(rows + background))
    cycles = [a for a in alerts if len(a.tx_ids) == length]
    assert cycles and set(cycles[0].accounts) == {f"c{k}" for k in range(length)}


@SETTINGS
@given(
    rates=st.lists(st.floats(0.001, 1000.0), min_size=2, max_size=6),
    data=st.data(),
)
def test_exchange_rates_recover_any_consistent_set_of_rates(rates, data):
    k = len(rates) + 1
    true = np.array([1.0, *rates])  # currency 0 is the reference
    names = np.array([f"C{i}" for i in range(k)], dtype=object)
    # a random spanning tree of exchanges: every currency is linked to an earlier one
    links = [(data.draw(st.integers(0, i - 1)), i) for i in range(1, k)]
    paid, recv, cp, cr = [], [], [], []
    for a, b in links:
        for amount in (100.0, 2_500.0):
            a_, b_ = (a, b) if data.draw(st.booleans()) else (b, a)
            paid.append(amount)
            recv.append(amount * true[a_] / true[b_])
            cp.append(a_)
            cr.append(b_)
    got = exchange_rates(
        np.array(paid), np.array(recv), np.array(cp), np.array(cr), names, reference="C0"
    )
    assert got == pytest.approx(true, rel=1e-6)


@SETTINGS
@given(
    edges=st.lists(
        st.tuples(st.integers(0, 12), st.integers(0, 12)).filter(lambda e: e[0] != e[1]),
        max_size=40,
        unique=True,
    )
)
def test_array_conversion_round_trips_any_directed_graph(edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(13))
    graph.add_edges_from(edges)
    arrays = nx_to_arrays(graph)
    back = arrays_to_nx(arrays["edge_index"], node_names=arrays["node_names"])
    assert set(back.nodes) == set(graph.nodes)
    assert set(back.edges) == set(graph.edges)
