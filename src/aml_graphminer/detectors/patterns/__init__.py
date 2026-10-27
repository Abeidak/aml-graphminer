"""Deterministic detectors: fan-in/out, smurfing, cycles, (gather|scatter), risk communities."""

from .base import Alert, PatternDetector, TxArrays, group_slices, sliding_windows
from .community import CommunityDetector
from .config import (
    CommunityConfig,
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
from .scanner import DETECTOR_REGISTRY, PatternScanner, ScanResult, register_detector
from .smurfing import SmurfingDetector

__all__ = [
    "DETECTOR_REGISTRY",
    "Alert",
    "CommunityConfig",
    "CommunityDetector",
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
    "group_slices",
    "pattern_recall_table",
    "register_detector",
    "sliding_windows",
]
