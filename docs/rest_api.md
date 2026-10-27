# REST API

`amlgm serve runs/hgb --port 8000` loads one saved detector and exposes it with FastAPI (`api`
extra). Interactive documentation is generated at `/docs` (Swagger UI), `/redoc` and
`/openapi.json` (OpenAPI 3.1); the web investigation interface is served at `/ui/` (see
{doc}`web_interface`). The logic lives in two framework-free classes —
`aml_graphminer.api.service.ScoringService` (stateless scoring) and
`aml_graphminer.api.investigation.InvestigationService` (investigations) — that can be embedded
in any other framework.

## Stateless scoring

| method | path | body | answer |
|---|---|---|---|
| GET | `/health` | | status, library version, loaded model, available explanation methods |
| POST | `/score` | `{"transactions": [...], "context": [...], "threshold": 0.5}` | a score and an alert flag per transaction |
| POST | `/explain` | `{"transactions": [...], "row": 0, "method": "auto", "hops": 2}` | an `Explanation` as JSON |
| POST | `/scan` | `{"transactions": [...], "top": 20}` | structural-motif alerts of the batch |

Transactions are JSON objects with the canonical fields (`timestamp, sender, receiver, amount`,
optionally `currency, transaction_type, amount_received, currency_received, sender_bank,
receiver_bank`) or the IBM column names.

```bash
curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "transactions": [
    {"timestamp": "2022-09-01 10:05:00", "sender": "70_ACC_A", "receiver": "12_ACC_C", "amount": 9500},
    {"timestamp": "2022-09-01 10:20:00", "sender": "70_ACC_A", "receiver": "12_ACC_D", "amount": 9400}
  ]}'
```

Answer (values are illustrative):

```json
{"threshold": 0.93, "n_transactions": 2, "n_alerts": 0,
 "results": [{"row": 0, "tx_id": 0, "score": 0.02, "alert": false},
             {"row": 1, "tx_id": 1, "score": 0.04, "alert": false}]}
```

* `row` is the position of the transaction in the request; `tx_id` is its position after the
  time sort. `/explain` takes the `row` to explain.
* `context` holds earlier transactions that complete the graph without being scored — send the
  recent history of the accounts involved, otherwise window and neighbourhood features see an
  empty past and scores will be lower than in batch evaluation.
* At most 50 000 transactions per request (`MAX_TRANSACTIONS`).
* Library errors (invalid data, unknown method, row out of range) return HTTP 422 with
  `{"error": "invalid_request", "detail": "..."}`.
* `/explain` supports every method that applies to the served detector (`patterns`,
  `occlusion`, and for a supervised GNN `gnnexplainer`, `pgexplainer`, `captum`, `attention`);
  `/health` lists them in `explain_methods`.

## Investigations (back end of the web interface)

An *investigation* is a transaction table loaded in memory and **analysed** once (all motif
detectors are run and the detector scores every transaction). At most 5 investigations and
2 million transactions each are kept; score bigger files with `amlgm score`.

| method | path | answer |
|---|---|---|
| GET | `/api/model` | loaded detector, threshold, explanation methods, motif descriptions |
| GET | `/api/datasets` | the investigations in memory |
| POST | `/api/datasets/demo` | analyse a synthetic dataset (`{"accounts": 1500, "background": 15000, "illicit_ratio": 0.02, "seed": 0}`) |
| POST | `/api/datasets/upload` | analyse an uploaded file (multipart: `file`, `schema` = `auto`, `ibm` or `canonical`) |
| GET | `/api/datasets/{id}` | dashboard: statistics (EDA), alerts per motif, score histogram, metrics when labelled |
| DELETE | `/api/datasets/{id}` | drop an investigation |
| GET | `/api/datasets/{id}/alerts` | motif alerts, best first (`pattern`, `min_score`, `limit`, `offset`) |
| GET | `/api/datasets/{id}/alerts/{alert_id}` | evidence, transactions and sub-graph of one alert |
| GET | `/api/datasets/{id}/transactions` | scored transactions (`sort` = `score`/`amount`/`timestamp`, `descending`, `account`, `only_alerts`, `pattern`, `limit`, `offset`) |
| GET | `/api/datasets/{id}/transactions/{tx}` | one transaction and the alerts that contain it |
| GET | `/api/datasets/{id}/transactions/{tx}/explain` | explanation + graph (`method`, `hops`, `max_edges`) |
| GET | `/api/datasets/{id}/accounts/{account}` | account profile: activity, counterparties, risk, alerts, neighbourhood graph |
| GET | `/api/datasets/{id}/export/alerts.csv` | all alerts (CSV download) |
| GET | `/api/datasets/{id}/export/scores.csv` | every transaction with its score, alert flag and strongest motif |

Unknown datasets, transactions, alerts or accounts return HTTP 404 with
`{"error": "not_found", "detail": "..."}`.

```bash
curl -s -X POST localhost:8000/api/datasets/demo -H 'content-type: application/json' -d '{"seed": 3}'
curl -s -F file=@data/Synth-Small_Trans.csv -F schema=ibm localhost:8000/api/datasets/upload
```

## Security notes

The API has **no authentication, rate limiting or TLS**. Put it behind your own gateway, never
expose it directly to the internet, and treat the posted transactions as personal data (do not
log request bodies). The model directory is loaded with `joblib`/`torch.load` — only serve
models you trained or whose provenance you trust.
