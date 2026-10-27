"""Neo4j export of the transaction graph (Layer 3): property graph for exploration in Cypher.

* :func:`export_neo4j` writes ``accounts.csv``, ``transactions.csv``, ``import.cypher`` (the
  ``LOAD CSV`` script) and ``queries.cypher`` (ready-made investigation queries: cycles, fan-in,
  fan-out, the most suspicious transfers ...). It needs no extra dependency.
* :func:`push_to_neo4j` loads the same graph into a running database through the official driver
  (``pip install 'aml-graphminer[neo4j]'``).

Graph model: ``(:Account {id, bank, n_sent, n_received, amount_sent, amount_received,
max_score, n_alerts})-[:TRANSFER {tx_id, timestamp, amount, currency, amount_received,
currency_received, payment_format, label, pattern, score, motif}]->(:Account)``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ConfigError
from ..utils import require

ACCOUNTS_FILE = "accounts.csv"
TRANSACTIONS_FILE = "transactions.csv"
IMPORT_FILE = "import.cypher"
QUERIES_FILE = "queries.cypher"

IMPORT_CYPHER = """// AML-GraphMiner -> Neo4j import (Neo4j 5.23 or newer).
// 1. Copy accounts.csv and transactions.csv into the import directory of the database
//    (docker compose --profile neo4j: the ./neo4j-import folder).
// 2. Run:  cypher-shell -u neo4j -p <password> -f import.cypher
//    (in the Neo4j Browser, run each statement prefixed with :auto)
CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE;
CREATE INDEX transfer_tx IF NOT EXISTS FOR ()-[t:TRANSFER]-() ON (t.tx_id);

LOAD CSV WITH HEADERS FROM 'file:///accounts.csv' AS row
CALL (row) {
  MERGE (a:Account {id: row.account_id})
  SET a.bank = row.bank,
      a.n_sent = toInteger(row.n_sent),
      a.n_received = toInteger(row.n_received),
      a.amount_sent = toFloat(row.amount_sent),
      a.amount_received = toFloat(row.amount_received),
      a.max_score = toFloat(row.max_score),
      a.n_alerts = toInteger(row.n_alerts)
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///transactions.csv' AS row
CALL (row) {
  MATCH (s:Account {id: row.sender})
  MATCH (r:Account {id: row.receiver})
  MERGE (s)-[t:TRANSFER {tx_id: toInteger(row.tx_id)}]->(r)
  SET t.timestamp = datetime(row.timestamp),
      t.amount = toFloat(row.amount),
      t.currency = row.currency,
      t.amount_received = toFloat(row.amount_received),
      t.currency_received = row.currency_received,
      t.payment_format = row.payment_format,
      t.label = toInteger(row.label),
      t.pattern = row.pattern,
      t.score = toFloat(row.score),
      t.motif = row.motif
} IN TRANSACTIONS OF 10000 ROWS;
"""

QUERIES_CYPHER = """// AML-GraphMiner — investigation queries for Neo4j (run them one by one).

// 1. Round-tripping found by the AML-GraphMiner cycle detector (exported with --scan)
MATCH (s:Account)-[t:TRANSFER {motif: 'cycle'}]->(r:Account)
RETURN s, t, r LIMIT 200;

// 1b. Time-respecting cycles of 3 to 6 transfers through ONE account (replace the identifier),
//     closed within 7 days, every account forwarding 80-102 % of what it received in the same
//     currency (the rule of the cycle detector). Anchoring on one account keeps the search bounded.
MATCH p = (a:Account {id: 'ACCOUNT_A'})-[ts:TRANSFER*3..6]->(a)
WHERE all(i IN range(0, size(ts) - 2) WHERE ts[i].timestamp < ts[i + 1].timestamp
          AND ts[i].currency_received = ts[i + 1].currency
          AND 0.8 * ts[i].amount_received <= ts[i + 1].amount <= 1.02 * ts[i].amount_received)
  AND duration.inSeconds(ts[0].timestamp, ts[-1].timestamp).seconds <= 7 * 24 * 3600
RETURN p, [t IN ts | t.label] AS labels LIMIT 25;

// 2. Fan-in: accounts paid by at least 5 distinct senders on the same day
MATCH (s:Account)-[t:TRANSFER]->(hub:Account)
WITH hub, date(t.timestamp) AS day, count(DISTINCT s) AS senders, sum(t.amount) AS total
WHERE senders >= 5
RETURN hub.id AS account, day, senders, total ORDER BY senders DESC LIMIT 25;

// 3. Fan-out: accounts paying at least 5 distinct receivers on the same day
MATCH (hub:Account)-[t:TRANSFER]->(r:Account)
WITH hub, date(t.timestamp) AS day, count(DISTINCT r) AS receivers, sum(t.amount) AS total
WHERE receivers >= 5
RETURN hub.id AS account, day, receivers, total ORDER BY receivers DESC LIMIT 25;

// 4. Structuring: many payments just below 10 000 by the same account within a day
MATCH (a:Account)-[t:TRANSFER]->()
WHERE 7500 <= t.amount < 10000
WITH a, date(t.timestamp) AS day, count(t) AS n
WHERE n >= 5
RETURN a.id AS account, day, n ORDER BY n DESC LIMIT 25;

// 5. The most suspicious transfers according to the model, with their neighbourhood
MATCH (s:Account)-[t:TRANSFER]->(r:Account)
WHERE t.score IS NOT NULL
WITH s, t, r ORDER BY t.score DESC LIMIT 10
MATCH p = (s)-[:TRANSFER]-(:Account)
RETURN t.tx_id, t.score, s.id, r.id, collect(p)[..20] AS neighbourhood;

// 6. Accounts involved in structural-motif alerts, by risk
MATCH (a:Account) WHERE a.n_alerts > 0
RETURN a.id, a.n_alerts, a.max_score, a.amount_sent, a.amount_received
ORDER BY a.n_alerts DESC, a.max_score DESC LIMIT 25;

// 7. Shortest money path between two accounts (replace the identifiers)
MATCH p = shortestPath((a:Account {id: 'ACCOUNT_A'})-[:TRANSFER*..8]->(b:Account {id: 'ACCOUNT_B'}))
RETURN p;
"""


def neo4j_frames(
    df: pd.DataFrame,
    *,
    scores: np.ndarray | pd.Series | None = None,
    scan: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(accounts, transactions)`` tables of the property graph (one row per node / edge).

    Parameters
    ----------
    scores:
        Optional model score of every transaction (aligned with ``df``).
    scan:
        Optional :class:`~aml_graphminer.detectors.patterns.ScanResult` of ``df``: the strongest
        motif of every transaction and the number of alerts of every account are added.
    """
    n = len(df)
    score = np.full(n, np.nan) if scores is None else np.asarray(scores, dtype="float64")
    if len(score) != n:
        raise ConfigError(f"scores has {len(score)} values for {n} transactions")
    motif = np.full(n, "", dtype=object)
    alerts_per_account: dict[str, int] = {}
    if scan is not None:
        _, motif = scan.transaction_scores()
        for alert in scan.alerts:
            for account in alert.accounts:
                alerts_per_account[account] = alerts_per_account.get(account, 0) + 1
    label = df["label"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    transactions = pd.DataFrame(
        {
            "tx_id": df.index.to_numpy(),
            "sender": df["sender"].to_numpy(),
            "receiver": df["receiver"].to_numpy(),
            "timestamp": df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(),
            "amount": df["amount"].to_numpy(),
            "currency": df["currency"].to_numpy(),
            "amount_received": df["amount_received"].to_numpy(),
            "currency_received": df["currency_received"].to_numpy(),
            "payment_format": df["transaction_type"].to_numpy(),
            "label": label.to_numpy(),
            "pattern": df["pattern"].to_numpy(),
            "score": score,
            "motif": np.asarray(motif, dtype=object),
        }
    )
    sent = transactions.groupby("sender").agg(
        n_sent=("amount", "size"), amount_sent=("amount", "sum")
    )
    received = transactions.groupby("receiver").agg(
        n_received=("amount_received", "size"), amount_received=("amount_received", "sum")
    )
    sender_score = transactions.groupby("sender")["score"].max()
    receiver_score = transactions.groupby("receiver")["score"].max()
    accounts = sent.join(received, how="outer").fillna(0.0)
    accounts.index.name = "account_id"
    accounts["max_score"] = pd.concat([sender_score, receiver_score], axis=1).max(axis=1)
    banks = pd.concat(
        [
            pd.Series(df["sender_bank"].to_numpy(), index=df["sender"].to_numpy()),
            pd.Series(df["receiver_bank"].to_numpy(), index=df["receiver"].to_numpy()),
        ]
    )
    accounts["bank"] = banks[~banks.index.duplicated()].reindex(accounts.index).fillna("")
    accounts["n_alerts"] = [alerts_per_account.get(str(a), 0) for a in accounts.index]
    accounts = accounts.reset_index()
    accounts["n_sent"] = accounts["n_sent"].astype("int64")
    accounts["n_received"] = accounts["n_received"].astype("int64")
    columns = [
        "account_id",
        "bank",
        "n_sent",
        "n_received",
        "amount_sent",
        "amount_received",
        "max_score",
        "n_alerts",
    ]
    return accounts[columns], transactions


def export_neo4j(
    df: pd.DataFrame,
    directory: str | Path,
    *,
    scores: np.ndarray | pd.Series | None = None,
    scan: Any = None,
) -> dict[str, Path]:
    """Write the CSV files and Cypher scripts of the Neo4j import into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    accounts, transactions = neo4j_frames(df, scores=scores, scan=scan)
    paths = {
        "accounts": directory / ACCOUNTS_FILE,
        "transactions": directory / TRANSACTIONS_FILE,
        "import": directory / IMPORT_FILE,
        "queries": directory / QUERIES_FILE,
    }
    accounts.to_csv(paths["accounts"], index=False)
    transactions.to_csv(paths["transactions"], index=False)
    paths["import"].write_text(IMPORT_CYPHER, encoding="utf-8")
    paths["queries"].write_text(QUERIES_CYPHER, encoding="utf-8")
    return paths


def _batches(frame: pd.DataFrame, size: int) -> Iterator[list[dict[str, Any]]]:
    clean = frame.astype(object).where(frame.notna(), None)
    records = clean.to_dict("records")
    for start in range(0, len(records), size):
        yield [{str(k): v for k, v in rec.items()} for rec in records[start : start + size]]


_MERGE_ACCOUNTS = "UNWIND $rows AS row MERGE (a:Account {id: row.account_id}) SET a += row"
_MERGE_TRANSFERS = (
    "UNWIND $rows AS row "
    "MATCH (s:Account {id: row.sender}) MATCH (r:Account {id: row.receiver}) "
    "MERGE (s)-[t:TRANSFER {tx_id: row.tx_id}]->(r) "
    "SET t += row, t.timestamp = datetime(row.timestamp)"
)


def push_to_neo4j(
    df: pd.DataFrame,
    uri: str,
    user: str,
    password: str,
    *,
    database: str = "neo4j",
    batch_size: int = 5000,
    scores: np.ndarray | pd.Series | None = None,
    scan: Any = None,
) -> dict[str, int]:
    """Load the graph into a running Neo4j database (idempotent: nodes and edges are merged)."""
    if batch_size < 1:
        raise ConfigError("batch_size must be >= 1")
    neo4j = require("neo4j", "neo4j", "pushing the graph to Neo4j")
    accounts, transactions = neo4j_frames(df, scores=scores, scan=scan)
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            session.run(
                "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE"
            )
            for rows in _batches(accounts, batch_size):
                session.run(_MERGE_ACCOUNTS, rows=rows)
            for rows in _batches(transactions, batch_size):
                session.run(_MERGE_TRANSFERS, rows=rows)
    finally:
        driver.close()
    return {"accounts": len(accounts), "transactions": len(transactions)}


__all__ = [
    "ACCOUNTS_FILE",
    "IMPORT_FILE",
    "QUERIES_FILE",
    "TRANSACTIONS_FILE",
    "export_neo4j",
    "neo4j_frames",
    "push_to_neo4j",
]
