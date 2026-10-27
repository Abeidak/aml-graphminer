# Web investigation interface

`amlgm serve` serves, next to the REST API, a single-page web application for analysts at
`http://HOST:PORT/ui/`. It needs only the `api` extra (no JavaScript build, no CDN: it works
offline and inside the Docker image).

```bash
amlgm serve --demo --open-browser                       # nothing to prepare: demo detector + synthetic data
amlgm serve runs/hgb --data data/Synth-Small_Trans.csv  # your detector, a file analysed at start-up
docker compose -f docker/docker-compose.demo.yml up --build
```

## Workflow

1. **Load data** — upload a CSV / JSON / Parquet file (IBM AML or canonical columns; format
   detected automatically or forced) or generate a synthetic demo dataset. Every dataset is
   *analysed* on arrival: all structural-motif detectors run on it and the loaded detector scores
   every transaction.
2. **Overview** — key figures (transactions, accounts, period, motif alerts, transactions flagged
   by the model), transactions per day with the laundering share, alerts per motif, the score
   distribution with the decision threshold, amounts by class, payment formats, and the detection
   metrics of the model and of the motif scanner when the file is labelled.
3. **Alerts (motifs)** — the motif and risk-community alerts, best first, filtered by motif and
   score. Selecting one shows its description, the numeric evidence of the detector, the
   sub-graph of its transactions and the list of transactions.
4. **Transactions** — every transaction with its model score, alert flag, strongest motif and,
   when known, its true label; filter by account, keep only the model's alerts, sort by score,
   amount or date. Selecting one shows its details, the motif alerts that contain it and **its
   explanation** with the method of your choice (pattern evidence, occlusion and, for a GNN,
   GNNExplainer, PGExplainer, Captum or attention): a summary sentence, the explanation graph
   (the explained transaction in red, edges that raise the score in blue, thickness and node
   size proportional to the importance), the most influential neighbouring transactions and the
   most influential attributes.
5. **Accounts** — the profile of any account (click a node of any graph or an account link):
   activity sent and received, counterparties, first and last activity, highest score, motif
   alerts it belongs to, its neighbourhood graph and its most suspicious transactions.
6. **Export** — all alerts or all scored transactions as CSV, for the case file.

The graphs are interactive: drag a node, drag the background to pan, use the mouse wheel or the
`+` / `−` buttons to zoom, `⟲` to reset the view; hover a node or an edge for its details.

## Architecture

```text
browser (api/static: index.html, app.js, style.css — vanilla JS, SVG)
   │  fetch /api/...
   ▼
FastAPI (api/app.py) ── InvestigationService (api/investigation.py, framework-free, unit-tested)
                           ├── PatternScanner (alerts)      ├── detector.score (model scores)
                           ├── eda_summary (dashboard)      └── explain_transaction (explanations)
```

The JSON API behind the interface is documented in {doc}`rest_api` (and interactively at
`/docs`). Values coming from the data are always inserted as text in the page, never as HTML.

## Limits

* Analysed tables are kept in memory: at most 5 investigations of 2 million transactions each.
  For a whole IBM *Medium* or *Large* file use `amlgm score` / `amlgm scan` and load a time slice
  in the interface.
* There are no user accounts and no authentication: run it on your workstation (the default
  `127.0.0.1` binding) or behind an authenticating reverse proxy, and never expose real customer
  data on a public address.
* Metrics shown on a labelled file are computed on the whole file; if the detector was trained on
  part of it they are optimistic — use `amlgm benchmark` for an evaluation on a temporal split.
