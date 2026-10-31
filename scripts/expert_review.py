"""Blind review of the alerts by an anti-money-laundering expert: build the grid, then score it.

    python scripts/expert_review.py build data/HI-Small-test_Trans.csv --model runs/hgb-HI-Small \\
        --config configs/patterns.yaml --out results/expert-review
    python scripts/expert_review.py score results/expert-review/grille_expert.csv \\
        results/expert-review/cle_de_correction.csv

``build`` mixes three groups in a shuffled order, so the expert cannot tell them apart: motif
alerts (the strongest of each motif, in turn), transactions flagged by the model (highest scores,
outside those alerts) and control transactions the model did not flag (drawn at random). The
expert judges every line twice: first from the facts alone (accounts, amounts, dates, payment
format, the evidence of a motif alert), then after reading the explanation of the tool (score,
summary sentence, most influential attributes), which measures what the explanations add. The
texts are worded in French by the web interface's ``fr.js`` when Node.js is installed.

``build`` writes ``grille_expert.csv`` (for the expert: semicolon-separated UTF-8 that opens in a
French Excel), ``cle_de_correction.csv`` (group and ground truth, kept apart) and
``CONSIGNES.md``; ``score`` reads the filled grid back.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

import aml_graphminer
from aml_graphminer.config import load_config
from aml_graphminer.detectors.bundle import load_detector
from aml_graphminer.detectors.patterns import PatternScanner
from aml_graphminer.explainers import TransactionIndex, describe_alert, explain_transaction
from aml_graphminer.ingestion import load_dataset

FR_JS = Path(aml_graphminer.__file__).parent / "api" / "static" / "fr.js"
RUNNER = """
const fs = require("fs");
const FR = require(process.argv[1]);
const calls = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = calls.map(([name, ...args]) => FR[name](...args));
fs.writeFileSync(process.argv[3], JSON.stringify(out));
"""
GRID = "grille_expert.csv"
KEY = "cle_de_correction.csv"
JUDGEMENTS = ("suspect", "non suspect", "incertain")
TOP_ATTRIBUTES = 3
INSTRUCTIONS = """# Consignes pour l'expert

Merci de relire ces {n} éléments tirés des données IBM AML (transactions simulées, aucune donnée
réelle de client). Certains sont des alertes de l'outil, d'autres des transactions qu'il n'a pas
signalées ; l'ordre est mélangé et rien n'indique à quel groupe appartient une ligne.

Le fichier `{grid}` s'ouvre dans Excel. Pour chaque ligne, en deux temps :

1. Lisez les colonnes de faits (de `element` à `faits`) **sans regarder** la colonne
   `explication_de_l_outil`, puis remplissez **jugement** : `suspect` (vous feriez un examen
   approfondi ou une déclaration de soupçon), `non suspect` ou `incertain`.
2. Lisez ensuite `explication_de_l_outil`, puis remplissez **jugement_apres_explication** (même
   choix) et **utilite_explication**, de 1 (l'explication ne vous aide pas à décider) à 5 (elle
   suffit à décider).

La colonne **commentaire** est libre : ce qui vous a convaincu, ce qui manque, ce qui est faux.
Comptez environ une heure. C'est votre jugement de spécialiste qui est mesuré : ne cherchez pas la
réponse ailleurs. Renvoyez le fichier rempli ; le fichier `{key}` ne doit pas vous être transmis
avant la fin.
"""


def _french(calls: list[list[Any]]) -> list[Any] | None:
    """French texts of the web interface (fr.js run by Node.js); None without Node.js."""
    node = shutil.which("node")
    if node is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        source, target = Path(tmp) / "in.json", Path(tmp) / "out.json"
        source.write_text(json.dumps(calls, default=str), encoding="utf-8")
        subprocess.run(
            [node, "-e", RUNNER, str(FR_JS), str(source), str(target)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        return list(json.loads(target.read_text(encoding="utf-8")))


def _accounts(accounts: list[str], shown: int = 6) -> str:
    more = len(accounts) - shown
    return ", ".join(accounts[:shown]) + (f" (+{more})" if more > 0 else "")


def _num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    df = load_dataset(args.data)
    detector = load_detector(args.model)
    result = PatternScanner(load_config(args.config).pattern_config()).scan(df)
    by_tx = detector.score(df).reindex(df.index)

    # A: the strongest alert of each motif, in turn, until --alerts
    queues: dict[str, list[tuple[int, Any]]] = {}
    for i, alert in sorted(enumerate(result.alerts), key=lambda ia: -result.weight(ia[1])):
        queues.setdefault(alert.pattern, []).append((i, alert))
    alerts: list[tuple[int, Any]] = []
    while len(alerts) < args.alerts and any(queues.values()):
        for queue in queues.values():
            if queue and len(alerts) < args.alerts:
                alerts.append(queue.pop(0))
    in_alerts = {t for _, a in alerts for t in a.tx_ids}

    # B: highest model scores outside those alerts; C: random controls under the threshold
    flagged = by_tx[(by_tx >= detector.threshold) & ~by_tx.index.isin(in_alerts)]
    chosen_flagged = [
        int(t) for t in flagged.sort_values(ascending=False).index[: args.transactions]
    ]
    quiet = [int(t) for t in by_tx[by_tx < detector.threshold].index]
    chosen_controls = rng.sample(quiet, min(args.controls, len(quiet)))

    index = TransactionIndex(df)
    explained = {
        tx: explain_transaction(detector, df, tx, args.method, index=index).to_dict()
        for tx in [*chosen_flagged, *chosen_controls]
    }
    tops = {
        tx: sorted(e["features"], key=lambda f: -abs(f["importance"]))[:TOP_ATTRIBUTES]
        for tx, e in explained.items()
    }
    # every French text in one Node.js run: alert sentences and labels, explanation summaries,
    # attribute names, payment formats and currencies of the transactions
    calls: list[list[Any]] = [["alertSentence", a.to_dict()] for _, a in alerts]
    calls += [["patternLabel", a.pattern] for _, a in alerts]
    calls += [["explanationSummary", e] for e in explained.values()]
    calls += [["featureLabel", f["name"]] for tx in explained for f in tops[tx]]
    calls += [["paymentFormat", str(df.loc[tx, "transaction_type"])] for tx in explained]
    calls += [["currencyCode", str(df.loc[tx, "currency"])] for tx in explained]
    french = _french(calls)
    texts = iter(french) if french else None

    def say(fallback: str) -> str:
        return str(next(texts)) if texts else fallback

    sentences = [say(describe_alert(a)) for _, a in alerts]
    labels = [say(a.pattern) for _, a in alerts]
    summaries = {tx: say(e["summary"]) for tx, e in explained.items()}
    names = {tx: [say(f["name"]) for f in tops[tx]] for tx in explained}
    formats = {tx: say(str(df.loc[tx, "transaction_type"])) for tx in explained}
    currencies = {tx: say(str(df.loc[tx, "currency"])) for tx in explained}

    items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for (i, alert), sentence, label in zip(alerts, sentences, labels):
        rows = df.loc[df.index.intersection(alert.tx_ids)]
        formats_seen = ", ".join(sorted(set(rows["transaction_type"].astype(str))))
        items.append(
            (
                {
                    "element": f"Alerte de motif ({label})",
                    "reference": f"alerte n° {i}",
                    "date_ou_periode": f"{alert.start} → {alert.end}",
                    "comptes": _accounts(alert.accounts),
                    "montant": _num(alert.total_amount),
                    "format_de_paiement": formats_seen,
                    "faits": sentence,
                    "explication_de_l_outil": f"score calibré de l'alerte {_num(result.weight(alert), 3)}",
                },
                {
                    "groupe": "A - alerte de motif",
                    "verite": f"{int(rows['label'].fillna(0).sum())}/{len(rows)} transactions de blanchiment",
                    "typologie_reelle": ", ".join(sorted(set(rows["pattern"].astype(str)) - {""})),
                },
            )
        )
    for tx in explained:
        row = df.loc[tx]
        attributes = "; ".join(
            f"{name} ({f['importance']:+.3f})".replace(".", ",")
            for name, f in zip(names[tx], tops[tx])
        )
        tool = (
            f"score du modèle sur toute la table {_num(float(by_tx[tx]), 3)}. "
            f"Explication sur le voisinage local : {summaries[tx]}"
        )
        if attributes:
            tool += f" Attributs les plus influents : {attributes}."
        items.append(
            (
                {
                    "element": "Transaction",
                    "reference": f"transaction n° {tx}",
                    "date_ou_periode": str(row["timestamp"]),
                    "comptes": f"{row['sender']} → {row['receiver']}",
                    "montant": f"{_num(float(row['amount']))} {currencies[tx]}",
                    "format_de_paiement": formats[tx],
                    "faits": "",
                    "explication_de_l_outil": tool,
                },
                {
                    "groupe": "B - signalée par le modèle"
                    if tx in chosen_flagged
                    else "C - témoin",
                    "verite": "blanchiment" if row["label"] == 1 else "légitime",
                    "typologie_reelle": str(row.get("pattern", "")),
                },
            )
        )
    rng.shuffle(items)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame([{"numero": n, **shown} for n, (shown, _) in enumerate(items, start=1)])
    grid = grid.assign(
        jugement="", jugement_apres_explication="", utilite_explication="", commentaire=""
    )
    key = pd.DataFrame([{"numero": n, **hidden} for n, (_, hidden) in enumerate(items, start=1)])
    grid.to_csv(out / GRID, sep=";", index=False, encoding="utf-8-sig")
    key.to_csv(out / KEY, sep=";", index=False, encoding="utf-8-sig")
    (out / "CONSIGNES.md").write_text(
        INSTRUCTIONS.format(n=len(items), grid=GRID, key=KEY), encoding="utf-8"
    )
    language = "français (fr.js)" if french else "anglais (Node.js absent)"
    print(
        f"{len(items)} lignes ({len(alerts)} alertes, {len(chosen_flagged)} signalées, "
        f"{len(chosen_controls)} témoins), textes en {language} → {out}"
    )


def score(args: argparse.Namespace) -> None:
    grid = pd.read_csv(args.grid, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    key = pd.read_csv(args.key, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    both = grid.merge(key, on="numero")
    for column in ("jugement", "jugement_apres_explication"):
        both[column] = both[column].str.strip().str.lower()
    filled = both[both["jugement"].isin(JUDGEMENTS)]
    print(f"{len(filled)} lignes jugées sur {len(both)}")
    for group, part in filled.groupby("groupe"):
        before = (part["jugement"] == "suspect").mean()
        after = (part["jugement_apres_explication"] == "suspect").mean()
        print(
            f"  {group} : {len(part)} jugées ; suspectes {before:.0%} avant, {after:.0%} après l'explication"
        )
    tx = filled[filled["element"] == "Transaction"]
    if len(tx):
        truth = tx["verite"] == "blanchiment"
        for column, moment in (("jugement", "avant"), ("jugement_apres_explication", "après")):
            agree = ((tx[column] == "suspect") == truth).mean()
            print(f"  accord avec la vérité terrain {moment} l'explication : {agree:.0%}")
    useful = pd.to_numeric(filled["utilite_explication"], errors="coerce").dropna()
    if len(useful):
        print(
            f"  utilité moyenne des explications : {_num(useful.mean(), 1)} / 5 ({len(useful)} notes)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build the grid, the answer key and the instructions")
    b.add_argument("data")
    b.add_argument("--model", required=True, help="a model folder (amlgm train)")
    b.add_argument("--config", default=None, help="motif configuration (YAML)")
    b.add_argument("--method", default="auto", help="explanation method (captum for a GNN)")
    b.add_argument("--alerts", type=int, default=30)
    b.add_argument("--transactions", type=int, default=10)
    b.add_argument("--controls", type=int, default=10)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="results/expert-review")
    s = sub.add_parser("score", help="read the filled grid back")
    s.add_argument("grid")
    s.add_argument("key")
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args)
    else:
        score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
