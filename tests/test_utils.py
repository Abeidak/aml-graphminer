import numpy as np
import pandas as pd
import pytest

from aml_graphminer.exceptions import OptionalDependencyError
from aml_graphminer.utils import (
    epoch_seconds,
    has_module,
    parse_duration,
    require,
    set_seed,
    shannon_entropy,
)


def test_epoch_seconds_is_resolution_independent():
    s_ns = pd.Series(pd.to_datetime(["1970-01-01 00:00:10", "1970-01-02 00:00:00"]))
    s_s = s_ns.astype("datetime64[s]")
    assert epoch_seconds(s_ns).tolist() == [10.0, 86400.0]
    assert epoch_seconds(s_s).tolist() == [10.0, 86400.0]


def test_epoch_seconds_accepts_datetime_index():
    idx = pd.DatetimeIndex(["1970-01-01 00:01:00"])
    assert epoch_seconds(idx).tolist() == [60.0]


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("30s", 30.0),
        ("90min", 5400.0),
        ("1h", 3600.0),
        ("24h", 86400.0),
        ("7d", 604800.0),
        (12, 12.0),
        (1.5, 1.5),
    ],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


def test_parse_duration_timedelta():
    assert parse_duration(pd.Timedelta(hours=2)) == 7200.0


def test_set_seed_makes_numpy_deterministic():
    set_seed(5)
    a = np.random.rand(3)
    set_seed(5)
    assert np.allclose(a, np.random.rand(3))


def test_require_returns_module_and_raises_with_hint():
    assert require("json", "none").dumps({"a": 1}) == '{"a": 1}'
    with pytest.raises(OptionalDependencyError) as info:
        require("definitely_not_installed_pkg", "gnn", "a feature")
    message = str(info.value)
    assert "definitely_not_installed_pkg" in message
    assert "aml-graphminer[gnn]" in message
    assert "a feature" in message


def test_has_module():
    assert has_module("numpy")
    assert not has_module("definitely_not_installed_pkg")


def test_shannon_entropy():
    assert shannon_entropy(np.array([1.0, 1.0])) == pytest.approx(np.log(2))
    assert shannon_entropy(np.array([5.0])) == 0.0
    assert shannon_entropy(np.array([0.0, 0.0])) == 0.0
    assert shannon_entropy(np.array([1.0, 0.0, 1.0])) == pytest.approx(np.log(2))
