"""Fit / transform feature pipeline with JSON-serialisable statistics (Layer 2).

The pipeline generalises ``fit_stats`` / ``build_features`` of the original Colab notebooks:

* statistics (standardisation constants, payment-format vocabulary, column order) are
  fitted on the **training period only** and saved in the model configuration;
* at inference the *same* statistics are reloaded, so user data is transformed exactly like
  the training data;
* temporal features are causal (trailing windows) and node aggregates are computed from an
  explicit ``node_rows`` mask (train rows at training time, all provided rows at inference).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ConfigError, ModelError
from .network import BETWEENNESS_COLUMN, PAGERANK_COLUMN, index_accounts, node_features
from .temporal import temporal_features

STATS_VERSION = 1
BASE_EDGE_COLUMNS = ("log_paid", "log_recv", "same_cur", "same_bank", "h_sin", "h_cos")
_STANDARDIZE_BASE = ("log_paid", "log_recv")


@dataclass
class FeatureConfig:
    """Feature-engineering options (saved with every trained model)."""

    windows: tuple[str, ...] = ("1h", "24h", "7d", "all")
    temporal: bool = True
    pagerank: bool = True
    betweenness: bool = False
    betweenness_k: int = 100

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["windows"] = list(self.windows)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureConfig:
        data = dict(data)
        if "windows" in data:
            data["windows"] = tuple(data["windows"])
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown FeatureConfig field(s): {sorted(unknown)}")
        return cls(**data)


@dataclass
class FeatureSet:
    """Numeric features ready for models.

    Attributes
    ----------
    edge:
        ``(n_tx, n_edge_features)`` float32 — one row per transaction.
    node:
        ``(n_accounts, n_node_features)`` float32 — one row per account.
    src, dst:
        Integer account ids of sender / receiver for every transaction.
    accounts:
        ``accounts[i]`` is the identifier of account ``i``.
    """

    edge: np.ndarray
    node: np.ndarray
    edge_cols: list[str]
    node_cols: list[str]
    src: np.ndarray
    dst: np.ndarray
    accounts: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.node.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge.shape[0])

    def tabular(self, rows: np.ndarray | None = None, with_nodes: bool = True) -> np.ndarray:
        """Flat matrix for tabular baselines: ``[edge | node[src] | node[dst]]``."""
        parts = [self.edge]
        if with_nodes:
            parts += [self.node[self.src], self.node[self.dst]]
        mat = np.hstack(parts)
        return mat if rows is None else mat[rows]

    def tabular_columns(self, with_nodes: bool = True) -> list[str]:
        cols = list(self.edge_cols)
        if with_nodes:
            cols += [f"src_{c}" for c in self.node_cols] + [f"dst_{c}" for c in self.node_cols]
        return cols

    def edge_frame(self, index: pd.Index | None = None) -> pd.DataFrame:
        return pd.DataFrame(self.edge, columns=self.edge_cols, index=index)

    def node_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.node, columns=self.node_cols, index=pd.Index(self.accounts))


def _raw_edge_frame(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    base = pd.DataFrame(
        {
            "log_paid": np.log1p(df["amount"].to_numpy(dtype="float64")),
            "log_recv": np.log1p(df["amount_received"].to_numpy(dtype="float64")),
            "same_cur": (df["currency"] == df["currency_received"]).to_numpy(dtype="float64"),
            "same_bank": (
                (df["sender_bank"] == df["receiver_bank"]) & (df["sender_bank"] != "")
            ).to_numpy(dtype="float64"),
            "h_sin": np.sin(2 * np.pi * hour.to_numpy(dtype="float64") / 24.0),
            "h_cos": np.cos(2 * np.pi * hour.to_numpy(dtype="float64") / 24.0),
        },
        index=df.index,
    )
    return base


class FeaturePipeline:
    """Fit on a training period, transform any transaction table.

    Example
    -------
    >>> pipe = FeaturePipeline().fit(df, rows=train_mask)         # doctest: +SKIP
    >>> fs = pipe.transform(df, node_rows=train_mask)             # doctest: +SKIP
    >>> X = fs.tabular()                                          # doctest: +SKIP
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.stats: dict[str, Any] | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, rows: np.ndarray | None = None) -> FeaturePipeline:
        """Learn normalisation constants and the payment-format vocabulary from ``rows``."""
        rows = np.ones(len(df), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
        if rows.sum() == 0:
            raise ConfigError("cannot fit the feature pipeline on zero rows")
        src, dst, n, _ = index_accounts(df)
        formats = sorted(df.loc[rows, "transaction_type"].astype(str).unique().tolist())
        edge_raw = self._edge_matrix(df, src, dst, formats)
        std_cols = [c for c in edge_raw.columns if self._is_standardised(c)]
        fit_part = edge_raw.loc[rows, std_cols]
        node_df = self._node_matrix(df, src, dst, n, rows)
        active = np.zeros(n, dtype=bool)
        active[src[rows]] = True
        active[dst[rows]] = True
        node_fit = node_df.loc[active] if active.any() else node_df
        self.stats = {
            "version": STATS_VERSION,
            "config": self.config.to_dict(),
            "formats": formats,
            "edge_cols": list(edge_raw.columns),
            "edge_standardised": std_cols,
            "edge_mean": {c: float(fit_part[c].mean()) for c in std_cols},
            "edge_std": {c: float(fit_part[c].std(ddof=0)) + 1e-6 for c in std_cols},
            "node_cols": list(node_df.columns),
            "node_mean": [float(v) for v in node_fit.mean()],
            "node_std": [float(v) + 1e-6 for v in node_fit.std(ddof=0)],
        }
        return self

    # ------------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame, node_rows: np.ndarray | None = None) -> FeatureSet:
        """Build edge and node features with the fitted statistics.

        Parameters
        ----------
        node_rows:
            Boolean mask of the transactions node aggregates are computed from. ``None`` →
            all rows (inference); pass the training mask when training/evaluating.
        """
        stats = self._require_fitted()
        src, dst, n, names = index_accounts(df)
        edge_raw = self._edge_matrix(df, src, dst, stats["formats"])
        edge_raw = edge_raw.reindex(columns=stats["edge_cols"], fill_value=0.0)
        for c in stats["edge_standardised"]:
            edge_raw[c] = (edge_raw[c] - stats["edge_mean"][c]) / stats["edge_std"][c]
        node_df = self._node_matrix(df, src, dst, n, node_rows)
        node_df = node_df.reindex(columns=stats["node_cols"], fill_value=0.0)
        node_arr = (
            node_df.to_numpy(dtype="float64") - np.asarray(stats["node_mean"])
        ) / np.asarray(stats["node_std"])
        return FeatureSet(
            edge=edge_raw.to_numpy(dtype="float32"),
            node=node_arr.astype("float32"),
            edge_cols=list(stats["edge_cols"]),
            node_cols=list(stats["node_cols"]),
            src=src,
            dst=dst,
            accounts=names,
        )

    def transform_splits(
        self, df: pd.DataFrame, masks: dict[str, np.ndarray]
    ) -> dict[str, FeatureSet]:
        """One :class:`FeatureSet` per split, mirroring what the GNN loaders see.

        Node aggregates are computed from the *graph rows* of each split: ``train`` → train,
        ``val`` → train + val, ``test`` → train + val + test (no label is ever used). Edge
        features are causal and therefore shared.
        """
        graph_rows = {
            "train": masks["train"],
            "val": masks["train"] | masks["val"],
            "test": masks["train"] | masks["val"] | masks["test"],
        }
        return {
            name: self.transform(df, node_rows=graph_rows[name])
            for name in ("train", "val", "test")
            if name in masks
        }

    def fit_transform(self, df: pd.DataFrame, rows: np.ndarray | None = None) -> FeatureSet:
        """Fit on ``rows`` then transform ``df`` using ``rows`` for node aggregates."""
        self.fit(df, rows)
        return self.transform(df, node_rows=rows)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return dict(self._require_fitted())

    @classmethod
    def from_dict(cls, stats: dict[str, Any]) -> FeaturePipeline:
        if stats.get("version") != STATS_VERSION:
            raise ModelError(
                f"unsupported feature-statistics version {stats.get('version')!r} "
                f"(this release reads version {STATS_VERSION})"
            )
        pipe = cls(FeatureConfig.from_dict(stats["config"]))
        pipe.stats = dict(stats)
        return pipe

    # ------------------------------------------------------------------ internals
    def _require_fitted(self) -> dict[str, Any]:
        if self.stats is None:
            raise ModelError("FeaturePipeline is not fitted: call fit() first")
        return self.stats

    @staticmethod
    def _is_standardised(col: str) -> bool:
        """Binary / bounded columns stay as they are, everything continuous is standardised."""
        return not (col in ("same_cur", "same_bank", "h_sin", "h_cos") or col.startswith("fmt_"))

    def _edge_matrix(
        self, df: pd.DataFrame, src: np.ndarray, dst: np.ndarray, formats: list[str]
    ) -> pd.DataFrame:
        parts = [_raw_edge_frame(df)]
        fmt = df["transaction_type"].astype(str)
        dummies = pd.DataFrame(
            {f"fmt_{f}": (fmt == f).to_numpy(dtype="float64") for f in formats}, index=df.index
        )
        parts.append(dummies)
        if self.config.temporal:
            parts.append(temporal_features(df, src, dst, self.config.windows))
        return pd.concat(parts, axis=1)

    def _node_matrix(
        self,
        df: pd.DataFrame,
        src: np.ndarray,
        dst: np.ndarray,
        n: int,
        rows: np.ndarray | None,
    ) -> pd.DataFrame:
        frame = node_features(
            df,
            src,
            dst,
            n,
            rows,
            use_pagerank=self.config.pagerank,
            use_betweenness=self.config.betweenness,
            betweenness_k=self.config.betweenness_k,
        )
        order = [c for c in frame.columns if c not in (PAGERANK_COLUMN, BETWEENNESS_COLUMN)]
        if self.config.pagerank:
            order.append(PAGERANK_COLUMN)
        if self.config.betweenness:
            order.append(BETWEENNESS_COLUMN)
        return frame[order]


__all__ = [
    "BASE_EDGE_COLUMNS",
    "FeatureConfig",
    "FeaturePipeline",
    "FeatureSet",
]
