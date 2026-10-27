# Deployment

## Docker

`docker/Dockerfile` builds a CPU image (multi-stage, non-root) with the core package, the REST
API, the web interface and the visualisation extras. Its default command serves the web
interface on a synthetic demo, so `docker run` alone gives a working demonstration.
`--build-arg WITH_GNN=1` adds CPU PyTorch, PyTorch Geometric (+ `pyg-lib`), Lightning and Captum
to serve or train GNN detectors. GPU training images should start from an NVIDIA/PyTorch base image
instead.

```bash
docker build -f docker/Dockerfile -t aml-graphminer .
docker run --rm -p 127.0.0.1:8000:8000 aml-graphminer                          # demo: http://127.0.0.1:8000/ui/
docker run --rm -v "$PWD/data:/data" aml-graphminer amlgm scan /data/HI-Small_Trans.csv --out /data/alerts
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/runs:/models:ro" aml-graphminer amlgm serve /models/hgb --host 0.0.0.0
docker build -f docker/Dockerfile -t aml-graphminer:gnn --build-arg WITH_GNN=1 .    # + PyTorch / PyG (CPU)
```

Three compose files, all bound to `127.0.0.1` only (the API has no authentication, see
{doc}`rest_api`), with a read-only root filesystem and no capabilities:

```bash
docker compose -f docker/docker-compose.demo.yml up --build                   # web interface + synthetic demo
MODEL_DIR=./runs/hgb docker compose -f docker/docker-compose.yml up --build   # your trained model
NEO4J_PASSWORD=change-me docker compose -f docker/docker-compose.neo4j.yml up -d   # Neo4j, see below
```

The container runs as an unprivileged user and contains no data or models; mount them. For
experiment tracking run `mlflow ui` (or a server of your choice) on the host and use
`--mlflow` / `tracking.uri`; see {doc}`configuration`.

## Cloud deployment

The image is a standard HTTP service on port 8000 (`GET /health` for health checks), so any
container platform works (Render, Fly.io, Railway, a university VM ...). Use the demo command for a
public showcase; for real data add authentication in front (reverse proxy, VPN) — the API has
none — and never upload real customer data to a public instance.

## Neo4j

`amlgm export-neo4j DATA OUT_DIR` writes the property graph
`(:Account)-[:TRANSFER]->(:Account)` as `accounts.csv` / `transactions.csv`, an `import.cypher`
script (`LOAD CSV ... IN TRANSACTIONS`) and `queries.cypher` with ready-made investigation
queries (time-respecting cycles, fan-in, fan-out, structuring, the most suspicious transfers and
their neighbourhood, shortest money path). `--model runs/hgb` adds the model score of every
transfer and `--scan` (default) the strongest motif and the number of alerts of every account.

```bash
NEO4J_PASSWORD=change-me docker compose -f docker/docker-compose.neo4j.yml up -d
amlgm export-neo4j data/HI-Small_Trans.csv neo4j-import --model runs/hgb --nrows 200000
docker compose -f docker/docker-compose.neo4j.yml exec neo4j \
    cypher-shell -u neo4j -p change-me -f /import/import.cypher
# or load through the driver (neo4j extra):
amlgm export-neo4j data/HI-Small_Trans.csv neo4j-import --push --password change-me
```

Then open the Neo4j Browser on `http://127.0.0.1:7474` and run the queries of `queries.cypher`.

## Hugging Face Hub

```bash
amlgm export-hub runs/hgb hub/aml-hgb --repo-id my-org/aml-hgb      # writes the repo + model card
amlgm export-hub runs/hgb hub/aml-hgb --repo-id my-org/aml-hgb --push --token $HF_TOKEN
```

The generated `README.md` (model card) states the model type, decision threshold and test metrics,
the intended use, and repeats the limitations: trained on the data you provided, not validated
for production use. Review it before pushing, especially if the training data are not public.
Repositories are created **private** by default.

Loading from the Hub: `huggingface_hub.snapshot_download("my-org/aml-hgb")` then
`load_detector(path)`.

## Serving checklist

1. Train with a temporal split that mimics deployment; freeze the threshold.
2. Version the model directory (it contains the feature statistics) together with the code
   version (`GET /health` reports the library version).
3. Send `context` (recent history) with each scoring request.
4. Monitor the share of alerts per day and score drift; retrain when they move.
5. Keep a human in the loop: scores prioritise investigations, they are not decisions.
