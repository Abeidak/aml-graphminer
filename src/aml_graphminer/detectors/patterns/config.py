"""YAML-friendly configuration of the structural-motif detectors (Pydantic models)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...exceptions import ConfigError
from ...utils import parse_duration


def _check_duration(value: str) -> str:
    try:
        seconds = parse_duration(value)
    except Exception as exc:
        raise ValueError(f"invalid duration {value!r} (use e.g. '30min', '24h', '7d')") from exc
    if seconds <= 0:
        raise ValueError("duration must be positive")
    return value


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class FanConfig(_Section):
    """Fan-in / fan-out: many distinct counterparties within a short window."""

    min_counterparties: int = Field(default=5, ge=2)
    window: str = "24h"
    min_total_amount: float = Field(default=0.0, ge=0)
    # The burst must be at least this many times the hub's own average rate, otherwise busy
    # merchants / payment hubs would be flagged all the time.
    min_burst_ratio: float = Field(default=3.0, ge=0)

    _w = field_validator("window")(_check_duration)


class SmurfingConfig(_Section):
    """Structuring (many sub-threshold payments) and split (one inflow chopped into pieces)."""

    structuring: bool = True
    split: bool = True
    threshold: float = Field(default=10_000.0, gt=0)
    band: float = Field(default=0.25, gt=0, lt=1)
    currency: str | None = "US Dollar"
    min_transactions: int = Field(default=5, ge=2)
    window: str = "24h"
    # share of the account's payments inside the window that must sit just below the threshold
    min_band_share: float = Field(default=0.5, ge=0, le=1)
    min_parts: int = Field(default=4, ge=2)
    split_window: str = "24h"
    split_tolerance: float = Field(default=0.15, ge=0, lt=1)
    max_part_fraction: float = Field(default=0.5, gt=0, le=1)

    _w = field_validator("window", "split_window")(_check_duration)


class CycleConfig(_Section):
    """Time-respecting directed cycles with (almost) conserved amounts."""

    min_length: int = Field(default=3, ge=2)
    max_length: int = Field(default=6, ge=3)
    max_span: str = "7d"
    min_hop_ratio: float = Field(default=0.8, gt=0, le=1)
    max_hop_ratio: float = Field(default=1.02, ge=1)
    # Amounts in two currencies are compared after conversion (``currency`` section); a hop whose
    # rates are unknown is rejected, unless this is True (accepted without any amount check).
    cross_currency: bool = False
    max_cycles: int = Field(default=100_000, ge=1)
    max_branching: int = Field(default=100, ge=1)
    max_expansions: int = Field(default=3_000_000, ge=1_000)

    _w = field_validator("max_span")(_check_duration)


class ScatterGatherConfig(_Section):
    """``source -> k intermediaries -> sink``."""

    min_intermediaries: int = Field(default=3, ge=2)
    hop_window: str = "24h"
    max_span: str = "3d"
    min_hop_ratio: float = Field(default=0.8, gt=0, le=1)
    max_hop_ratio: float = Field(default=1.02, ge=1)
    max_hub_degree: int = Field(default=30, ge=2)

    _w = field_validator("hop_window", "max_span")(_check_duration)


class GatherScatterConfig(_Section):
    """``k sources -> hub -> k' targets`` with conserved volume."""

    min_in: int = Field(default=3, ge=2)
    min_out: int = Field(default=3, ge=2)
    gather_window: str = "24h"
    scatter_window: str = "24h"
    min_conservation: float = Field(default=0.7, gt=0, le=1)
    max_conservation: float = Field(default=1.05, ge=1)
    min_burst_ratio: float = Field(default=3.0, ge=0)
    # share of the hub's payments during the episode that belong to the gather/scatter pattern
    min_activity_share: float = Field(default=0.5, ge=0, le=1)

    _w = field_validator("gather_window", "scatter_window")(_check_duration)


class CommunityConfig(_Section):
    """Risk communities: small groups organised around pass-through (relay) accounts.

    A *relay* forwards most of what it receives (``min_flow_through``); hubs with more than
    ``max_member_degree`` transactions are ignored. Groups are the connected components of the
    transactions that touch a relay (split with Louvain when larger than ``max_size``). A group
    with at least ``min_relays`` relays is an alert when most of its members' outflow stays
    inside it (``min_internal_share``) and its internal transactions fit in ``max_span``.
    """

    min_size: int = Field(default=3, ge=2)
    max_size: int = Field(default=40, ge=3)
    min_flow_through: float = Field(default=0.75, gt=0, le=1)
    max_member_degree: int = Field(default=40, ge=2)
    min_relays: int = Field(default=1, ge=1)
    min_internal_tx: int = Field(default=3, ge=2)
    min_internal_share: float = Field(default=0.6, ge=0, le=1)
    max_span: str = "3d"
    resolution: float = Field(default=1.0, gt=0)
    seed: int = 0

    _w = field_validator("max_span")(_check_duration)


class CurrencyConfig(BaseModel):
    """Conversion of the amounts to one reference currency before they are compared.

    Laundering chains often change currency between hops (an account receives dollars and
    forwards euros): without conversion the amount checks of the cycle, scatter-gather and
    gather-scatter detectors cannot follow them. The rates are inferred from the cross-currency
    transactions of the scanned table (``rates`` overrides them, in units of ``reference`` per
    unit of each currency).
    """

    model_config = ConfigDict(extra="forbid")
    convert: bool = True
    reference: str | None = "US Dollar"
    rates: dict[str, float] | None = None


class PatternConfig(BaseModel):
    """Full detector configuration; every section can be disabled or tuned."""

    model_config = ConfigDict(extra="forbid")

    fan_in: FanConfig = FanConfig()
    fan_out: FanConfig = FanConfig()
    smurfing: SmurfingConfig = SmurfingConfig()
    cycle: CycleConfig = CycleConfig()
    scatter_gather: ScatterGatherConfig = ScatterGatherConfig()
    gather_scatter: GatherScatterConfig = GatherScatterConfig()
    community: CommunityConfig = CommunityConfig()
    currency: CurrencyConfig = CurrencyConfig()
    # When a time slice is scanned (benchmark validation / test), the transactions of this
    # period before it are given as unlabelled context so that motifs straddling the start of
    # the slice are seen whole; None scans the slice alone.
    lookback: str | None = "4d"
    # Only transactions of these payment formats are searched for motifs (None: all). In the
    # IBM AML simulator every laundering motif is made of ACH transfers; elsewhere leave None.
    transaction_types: list[str] | None = None

    @field_validator("lookback")
    @classmethod
    def _check_lookback(cls, value: str | None) -> str | None:
        return None if value is None else _check_duration(value)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PatternConfig:
        try:
            return cls.model_validate(data or {})
        except Exception as exc:
            raise ConfigError(f"invalid pattern configuration: {exc}") from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> PatternConfig:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"configuration file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "patterns" in data and isinstance(data["patterns"], dict):  # allow a shared file
            data = data["patterns"]
        return cls.from_dict(data)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False)

    def enabled_names(self) -> list[str]:
        return [
            name
            for name in type(self).model_fields
            if isinstance(getattr(self, name), _Section) and getattr(self, name).enabled
        ]
