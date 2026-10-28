"""Deterministic detectors: fan-in/out, smurfing, cycles, (gather|scatter), risk communities."""

from .base import (
    Alert,
    PatternDetector,
    TxArrays,
    exchange_rates,
    group_slices,
    sliding_windows,
)
from .community import CommunityDetector
from .config import (
    CommunityConfig,
    CurrencyConfig,
    CycleConfig,
    FanConfig,
    GatherScatterConfig,
    PatternConfig,
    ScatterGatherConfig,
    SmurfingConfig,
)
from .cycles import CycleDetector
from .evaluation import alert_precision_table, evaluate_scan, pattern_recall_table
from .fan import FanInDetector, FanOutDetector
from .layering import GatherScatterDetector, ScatterGatherDetector
from .scanner import (
    DETECTOR_REGISTRY,
    PatternScanner,
    ScanResult,
    lookback_history,
    register_detector,
)
from .smurfing import SmurfingDetector

__all__ = [
    "DETECTOR_REGISTRY",
    "Alert",
    "CommunityConfig",
    "CommunityDetector",
    "CurrencyConfig",
    "CycleConfig",
    "CycleDetector",
    "FanConfig",
    "FanInDetector",
    "FanOutDetector",
    "GatherScatterConfig",
    "GatherScatterDetector",
    "PatternConfig",
    "PatternDetector",
    "PatternScanner",
    "ScanResult",
    "ScatterGatherConfig",
    "ScatterGatherDetector",
    "SmurfingConfig",
    "SmurfingDetector",
    "TxArrays",
    "alert_precision_table",
    "evaluate_scan",
    "exchange_rates",
    "group_slices",
    "lookback_history",
    "pattern_recall_table",
    "register_detector",
    "sliding_windows",
]
