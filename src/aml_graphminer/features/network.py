"""Account-level (node) features: degrees, volumes, counterparties, entropy, PageRank (Layer 2)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from ..utils import epoch_seconds

# Count / volume features are expressed as *rates per day* over the period covered by the
# transactions the aggregates are computed from. Rates stay comparable when that period grows
# (training graph = 6 days, test graph = 10 days, user history = 3 days ...), which totals do not.
NODE_COLUMNS: tuple[str, ...] = (
    "out_rate",
    "in_rate",
    "out_amt_mean",
    "in_amt_mean",
    "out_vol_rate",
    "in_vol_rate",
    "uniq_out_rate",
    "uniq_in_rate",
    "io_ratio",
    "ent_out",
    "ent_in",
    "velocity",
)
PAGERANK_COLUMN = "pagerank"
BETWEENNESS_COLUMN = "betweenness"


def index_accounts(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Map every account to an integer id (same convention as the original notebooks).

    Returns ``(src, dst, n_accounts, names)`` where ``names[i]`` is the account with id ``i``.
    """
    codes, uniques = pd.factorize(pd.concat([df["sender"], df["receiver"]], ignore_index=True))
    n_edges = len(df)
    return (
        np.asarray(codes[:n_edges], dtype="int64"),
        np.asarray(codes[n_edges:], dtype="int64"),
        len(uniques),
        np.asarray(uniques, dtype=object),
    )


def pagerank(
    src: np.ndarray,
    dst: np.ndarray,
    n: int,
    weights: np.ndarray | None = None,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    """PageRank by sparse power iteration (scales to millions of edges).

    Values are rescaled so that their mean is 1 (``n * pr``) which keeps them comparable
    across graph sizes.
    """
    if n == 0:
        return np.zeros(0)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype="float64")
    adj = sparse.coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()
    out = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.zeros(n)
    np.divide(1.0, out, out=inv, where=out > 0)
    transition_t = (sparse.diags(inv) @ adj).T.tocsr()
    dangling = out == 0
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        nxt = alpha * (transition_t @ x + x[dangling].sum() / n) + (1 - alpha) / n
        done = np.abs(nxt - x).sum() < tol
        x = nxt
        if done:
            break
    return x * n


def approx_betweenness(
    src: np.ndarray, dst: np.ndarray, n: int, k: int = 100, seed: int = 0
) -> np.ndarray:
    """Sampled betweenness centrality (``k`` pivots) on the aggregated directed graph."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(zip(src.tolist(), dst.tolist()))
    values = nx.betweenness_centrality(graph, k=min(k, n), seed=seed, normalized=True)
    return np.array([values[i] for i in range(n)], dtype="float64")


def _entropy_by_node(node: np.ndarray, other: np.ndarray, amount: np.ndarray, n: int) -> np.ndarray:
    """Shannon entropy of the amount distribution over counterparties, per node."""
    pairs = pd.DataFrame({"n": node, "o": other, "a": amount}).groupby(["n", "o"])["a"].sum()
    total = pairs.groupby(level=0).transform("sum")
    p = (pairs / total.where(total > 0, 1.0)).to_numpy()
    contrib = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    ent = pd.Series(contrib, index=pairs.index).groupby(level=0).sum()
    out = np.zeros(n)
    out[ent.index.to_numpy()] = ent.to_numpy()
    return out


def node_features(
    df: pd.DataFrame,
    src: np.ndarray,
    dst: np.ndarray,
    n: int,
    rows: np.ndarray | None = None,
    *,
    use_pagerank: bool = True,
    use_betweenness: bool = False,
    betweenness_k: int = 100,
) -> pd.DataFrame:
    """Aggregate account-level features from the transactions flagged by ``rows``.

    ``rows`` (boolean mask) selects the transactions the aggregates are computed from:
    the *training period* when fitting a model (no leakage), or all provided
    transactions at inference time.
    """
    if rows is None:
        rows = np.ones(len(df), dtype=bool)
    rows = np.asarray(rows, dtype=bool)
    s, d = src[rows], dst[rows]
    paid = df["amount"].to_numpy(dtype="float64")[rows]
    recv = df["amount_received"].to_numpy(dtype="float64")[rows]
    ts = epoch_seconds(df["timestamp"])[rows]

    feats = pd.DataFrame(0.0, index=np.arange(n), columns=list(NODE_COLUMNS))
    if len(s):
        period_days = max((float(ts.max()) - float(ts.min())) / 86400.0, 1.0)
        out = (
            pd.DataFrame({"s": s, "d": d, "paid": paid})
            .groupby("s")
            .agg(
                out_deg=("paid", "size"),
                out_amt_mean=("paid", "mean"),
                out_vol=("paid", "sum"),
                uniq_out=("d", "nunique"),
            )
        )
        inc = (
            pd.DataFrame({"s": s, "d": d, "recv": recv})
            .groupby("d")
            .agg(
                in_deg=("recv", "size"),
                in_amt_mean=("recv", "mean"),
                in_vol=("recv", "sum"),
                uniq_in=("s", "nunique"),
            )
        )
        out_deg = np.zeros(n)
        in_deg = np.zeros(n)
        out_deg[out.index] = out["out_deg"].to_numpy()
        in_deg[inc.index] = inc["in_deg"].to_numpy()
        out_vol = np.zeros(n)
        in_vol = np.zeros(n)
        out_vol[out.index] = out["out_vol"].to_numpy()
        in_vol[inc.index] = inc["in_vol"].to_numpy()
        feats["out_rate"] = out_deg / period_days
        feats["in_rate"] = in_deg / period_days
        feats["out_vol_rate"] = out_vol / period_days
        feats["in_vol_rate"] = in_vol / period_days
        feats.loc[out.index, "out_amt_mean"] = out["out_amt_mean"].to_numpy()
        feats.loc[inc.index, "in_amt_mean"] = inc["in_amt_mean"].to_numpy()
        feats.loc[out.index, "uniq_out_rate"] = out["uniq_out"].to_numpy() / period_days
        feats.loc[inc.index, "uniq_in_rate"] = inc["uniq_in"].to_numpy() / period_days

        feats["ent_out"] = _entropy_by_node(s, d, paid, n)
        feats["ent_in"] = _entropy_by_node(d, s, recv, n)

        times = (
            pd.concat(
                [pd.DataFrame({"n": s, "t": ts}), pd.DataFrame({"n": d, "t": ts})],
                ignore_index=True,
            )
            .groupby("n")["t"]
            .agg(["min", "max"])
        )
        span_days = np.maximum((times["max"] - times["min"]).to_numpy() / 86400.0, 1.0)
        velocity = np.zeros(n)
        idx = times.index.to_numpy()
        velocity[idx] = (out_deg[idx] + in_deg[idx]) / span_days
        feats["velocity"] = velocity
        feats["io_ratio"] = np.log((in_vol + 1.0) / (out_vol + 1.0))

    log_cols = [
        "out_rate",
        "in_rate",
        "out_amt_mean",
        "in_amt_mean",
        "out_vol_rate",
        "in_vol_rate",
        "uniq_out_rate",
        "uniq_in_rate",
        "velocity",
    ]
    feats[log_cols] = np.log1p(feats[log_cols])

    if use_pagerank:
        feats[PAGERANK_COLUMN] = np.log1p(pagerank(s, d, n)) if len(s) else 0.0
    if use_betweenness:
        feats[BETWEENNESS_COLUMN] = approx_betweenness(s, d, n, k=betweenness_k) if len(s) else 0.0
    return feats
