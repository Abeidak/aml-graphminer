"""HTTP layer; skipped when FastAPI / httpx are not installed (they are in CI)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from aml_graphminer.api import create_app
from aml_graphminer.api.service import ScoringService
from aml_graphminer.detectors.bundle import TabularDetector
from test_api_service import to_records


@pytest.fixture(scope="module")
def client(synth_df):
    return TestClient(create_app(ScoringService(TabularDetector.fit(synth_df, "rf"))))


def test_health_and_openapi(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "/score" in client.get("/openapi.json").json()["paths"]


def test_score_endpoint(client, synth_df):
    body = {"transactions": to_records(synth_df.iloc[3000:3050])}
    res = client.post("/score", json=body)
    assert res.status_code == 200 and res.json()["n_transactions"] == 50


def test_validation_errors_are_422(client):
    assert client.post("/score", json={"transactions": []}).status_code == 422
    res = client.post(
        "/score",
        json={"transactions": [{"timestamp": "bad", "sender": "a", "receiver": "b", "amount": 1}]},
    )
    assert res.status_code == 422 and res.json()["error"] == "invalid_request"


def test_explain_and_scan_endpoints(client, synth_df):
    records = to_records(synth_df.iloc[3000:3100])
    assert client.post("/explain", json={"transactions": records, "row": 3}).status_code == 200
    assert client.post("/scan", json={"transactions": records, "top": 3}).status_code == 200


# --------------------------------------------------------------------------- web interface
def test_web_interface_is_served(client):
    root = client.get("/", follow_redirects=False)
    assert root.status_code in (302, 307) and root.headers["location"].endswith("/ui/")
    page = client.get("/ui/")
    assert page.status_code == 200 and "AML-GraphMiner" in page.text and "app.js" in page.text
    assert page.text.index("fr.js") < page.text.index("app.js")  # texts loaded first
    for asset, kind in (("fr.js", "javascript"), ("app.js", "javascript"), ("style.css", "css")):
        res = client.get(f"/ui/{asset}")
        assert res.status_code == 200 and kind in res.headers["content-type"]
    assert client.get("/health").json()["explain_methods"] == ["occlusion"]


def test_investigation_workflow_over_http(client):
    model = client.get("/api/model").json()
    assert model["model"]["model_type"] == "tabular" and model["explain_methods"]
    created = client.post(
        "/api/datasets/demo", json={"accounts": 300, "background": 2500, "illicit_ratio": 0.04}
    )
    assert created.status_code == 200, created.text
    ds = created.json()["id"]
    assert ds in [d["id"] for d in client.get("/api/datasets").json()]
    overview = client.get(f"/api/datasets/{ds}").json()
    assert overview["dataset"]["id"] == ds and overview["metrics"] is not None

    alerts = client.get(f"/api/datasets/{ds}/alerts", params={"limit": 3}).json()
    assert alerts["total"] > 0 and len(alerts["items"]) <= 3
    alert = client.get(f"/api/datasets/{ds}/alerts/{alerts['items'][0]['alert_id']}").json()
    assert alert["graph"]["edges"]

    txs = client.get(
        f"/api/datasets/{ds}/transactions",
        params={"limit": 5, "sort": "amount", "descending": False},
    ).json()
    assert len(txs["items"]) == 5
    tx = txs["items"][0]["tx_id"]
    assert client.get(f"/api/datasets/{ds}/transactions/{tx}").json()["tx_id"] == tx
    exp = client.get(
        f"/api/datasets/{ds}/transactions/{tx}/explain", params={"hops": 1, "max_edges": 20}
    ).json()
    assert exp["method"] == "occlusion" and exp["graph"]["nodes"]
    account = client.get(f"/api/datasets/{ds}/accounts/{txs['items'][0]['sender']}").json()
    assert account["n_sent"] >= 1

    csv = client.get(f"/api/datasets/{ds}/export/scores.csv")
    assert csv.status_code == 200 and "text/csv" in csv.headers["content-type"]
    assert "attachment" in csv.headers["content-disposition"]
    assert client.get(f"/api/datasets/{ds}/export/alerts.csv").text.startswith("alert_id,")
    assert client.delete(f"/api/datasets/{ds}").json()["status"] == "deleted"


def test_investigation_errors(client):
    assert client.get("/api/datasets/nope").status_code == 404
    assert client.get("/api/datasets/nope/alerts").json()["error"] == "not_found"
    ds = client.post("/api/datasets/demo", json={"accounts": 200, "background": 1000}).json()["id"]
    assert client.get(f"/api/datasets/{ds}/transactions/99999999").status_code == 404
    assert (
        client.get(f"/api/datasets/{ds}/transactions", params={"sort": "sender"}).status_code == 422
    )
    assert client.get(f"/api/datasets/{ds}/accounts/unknown-account").status_code == 404
    res = client.get(f"/api/datasets/{ds}/transactions/0/explain", params={"method": "magic"})
    assert res.status_code == 422 and "unknown explanation method" in res.json()["detail"]
    assert client.post("/api/datasets/demo", json={"accounts": 1}).status_code == 422


def test_upload_endpoint(client, synth_df, tmp_path):
    pytest.importorskip("python_multipart")
    path = tmp_path / "upload_Trans.csv"
    synth_df.iloc[:600].to_csv(path, index=False)
    with path.open("rb") as handle:
        res = client.post(
            "/api/datasets/upload",
            files={"file": ("upload_Trans.csv", handle, "text/csv")},
            data={"schema": "canonical"},
        )
    assert res.status_code == 200, res.text
    assert res.json()["name"] == "upload_Trans.csv" and res.json()["transactions"] == 600
    bad = client.post(
        "/api/datasets/upload", files={"file": ("x.docx", b"hello", "application/octet-stream")}
    )
    assert bad.status_code == 422


def test_app_options_preload_datasets(synth_df, tmp_path):
    path = tmp_path / "pre_Trans.csv"
    synth_df.iloc[:500].to_csv(path, index=False)
    service = ScoringService(TabularDetector.fit(synth_df, "logreg"))
    app = create_app(service, datasets=[path], demo=True)
    names = [d["name"] for d in TestClient(app).get("/api/datasets").json()]
    assert "pre_Trans.csv" in names and any(n.startswith("synthetic demo") for n in names)
