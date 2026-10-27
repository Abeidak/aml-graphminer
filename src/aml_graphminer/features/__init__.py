"""Layer 2 — feature engineering (temporal windows, network metrics, categorical encoding)."""

from .network import NODE_COLUMNS, approx_betweenness, index_accounts, node_features, pagerank
from .pipeline import FeatureConfig, FeaturePipeline, FeatureSet
from .temporal import (
    WindowStats,
    previous_gap,
    temporal_feature_names,
    temporal_features,
    trailing_window_stats,
)

__all__ = [
    "NODE_COLUMNS",
    "FeatureConfig",
    "FeaturePipeline",
    "FeatureSet",
    "WindowStats",
    "approx_betweenness",
    "index_accounts",
    "node_features",
    "pagerank",
    "previous_gap",
    "temporal_feature_names",
    "temporal_features",
    "trailing_window_stats",
]
