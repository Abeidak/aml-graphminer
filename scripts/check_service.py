#!/usr/bin/env python
"""Acceptance check of a running AML-GraphMiner service (REST API + investigation API + web UI).

    python scripts/check_service.py https://aml-graphminer-demo.onrender.com
    python scripts/check_service.py http://127.0.0.1:8000 --upload-limit-mb 1 --report out.json

Every check calls the service as a client would and verifies the *content* of the answer, not
only its status code. The transactions sent are synthetic (generated locally); the datasets the
check creates on the server are deleted at the end. Exit status 1 when a check fails.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

import aml_graphminer as aml
from aml_graphminer.ingestion import write_ibm_csv

GROUPS = {
    "api": "R1 — API REST, OpenAPI 3.1",
    "scoring": "R3 — Scoring for existing systems",
    "explain": "R4 — Explanations exported as JSON",
    "investigation": "R2 — Investigation API",
    "ui": "R2 — Web interface",
    "robustness": "Robustness and errors",
}
EXPECTED_PATHS = {
    "/health",
    "/score",
    "/explain",
    "/scan",
    "/api/model",
    "/api/datasets",
    "/api/datasets/demo",
    "/api/datasets/{dataset_id}",
    "/api/datasets/{dataset_id}/alerts",
    "/api/datasets/{dataset_id}/alerts/{alert_id}",
    "/api/datasets/{dataset_id}/transactions",
    "/api/datasets/{dataset_id}/transactions/{tx_id}",
    "/api/datasets/{dataset_id}/transactions/{tx_id}/explain",
    "/api/datasets/{dataset_id}/accounts/{account}",
    "/api/datasets/{dataset_id}/export/alerts.csv",
    "/api/datasets/{dataset_id}/export/scores.csv",
}


class CheckFailed(AssertionError):
    pass


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


@dataclass
class Result:
    group: str
    name: str
    ok: bool
    seconds: float
    detail: str = ""


@dataclass
class Checker:
    base: str
    timeout: float = 240.0
    api_key: str | None = None
    results: list[Result] = field(default_factory=list)
    created: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        headers = {"X-API-Key": self.api_key} if self.api_key else None
        self.client = httpx.Client(
            base_url=self.base.rstrip("/"), timeout=self.timeout, headers=headers
        )

    def run(self, group: str, name: str, fn: Callable[[], str | None]) -> Any:
        t0 = time.perf_counter()
        try:
            detail = fn() or ""
            ok = True
        except Exception as exc:  # report every failure, keep checking
            detail, ok = f"{type(exc).__name__}: {exc}"[:300], False
        self.results.append(Result(group, name, ok, time.perf_counter() - t0, detail))
        mark = "PASS" if ok else "FAIL"
        print(
            f"[{mark}] {GROUPS[group].split(' — ')[0]} {name} ({self.results[-1].seconds:.1f}s) {detail}",
            flush=True,
        )
        return ok

    def get(self, path: str, **kw: Any) -> httpx.Response:
        return self.client.get(path, **kw)

    def post(self, path: str, **kw: Any) -> httpx.Response:
        return self.client.post(path, **kw)

    def json(self, response: httpx.Response, status: int = 200) -> Any:
        ensure(
            response.status_code == status,
            f"{response.request.method} {response.request.url.path} -> {response.status_code}"
            f" (expected {status}): {response.text[:200]}",
        )
        return response.json()


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [
        "timestamp",
        "sender",
        "receiver",
        "amount",
        "currency",
        "amount_received",
        "currency_received",
        "transaction_type",
        "sender_bank",
        "receiver_bank",
    ]
    out = df[cols].copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.to_dict("records")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument(
        "--upload-limit-mb",
        type=float,
        default=None,
        help="the server's --max-upload-mb: send a larger file and expect a refusal",
    )
    parser.add_argument("--report", default=None, help="write the results as JSON")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("AMLGM_API_KEY"),
        help="key of a server started with one (default: $AMLGM_API_KEY)",
    )
    args = parser.parse_args(argv)
    c = Checker(args.base_url, api_key=args.api_key)
    df = aml.generate_transactions(n_accounts=250, n_background=1800, illicit_ratio=0.04, seed=11)
    sample = records(df)
    state: dict[str, Any] = {}

    # wake the service up (free hosts sleep)
    for _ in range(30):
        try:
            if c.get("/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(5)

    # ------------------------------------------------------------------ R1 API + OpenAPI
    def health() -> str:
        h = c.json(c.get("/health"))
        ensure(h["status"] == "ok", "status is not ok")
        ensure(h["model"].get("model_type"), "no model type")
        state["methods"] = h["explain_methods"]
        return f"model {h['model'].get('model_type')}/{h['model'].get('kind', '')}, methods {h['explain_methods']}"

    def openapi() -> str:
        spec = c.json(c.get("/openapi.json"))
        ensure(str(spec["openapi"]).startswith("3.1"), f"OpenAPI {spec['openapi']}")
        missing = EXPECTED_PATHS - set(spec["paths"])
        ensure(not missing, f"undocumented routes {sorted(missing)}")
        return f"OpenAPI {spec['openapi']}, {len(spec['paths'])} routes documented"

    def docs() -> str:
        for path in ("/docs", "/redoc"):
            r = c.get(path)
            ensure(r.status_code == 200 and "openapi.json" in r.text, f"{path} -> {r.status_code}")
        return "/docs (Swagger UI) and /redoc served"

    c.run("api", "GET /health", health)
    c.run("api", "GET /openapi.json", openapi)
    c.run("api", "interactive documentation", docs)

    # ------------------------------------------------------------------ R3 scoring
    def score() -> str:
        out = c.json(c.post("/score", json={"transactions": sample}))
        ensure(out["n_transactions"] == len(sample), "wrong count")
        scores = [r["score"] for r in out["results"]]
        ensure(all(0.0 <= s <= 1.0 for s in scores), "score outside [0, 1]")
        ensure([r["row"] for r in out["results"]] == list(range(len(sample))), "rows out of order")
        ensure(sum(r["alert"] for r in out["results"]) == out["n_alerts"], "n_alerts inconsistent")
        state["top_row"] = max(range(len(scores)), key=scores.__getitem__)
        return f"{len(sample)} transactions scored, {out['n_alerts']} alerts, threshold {out['threshold']:.4g}"

    def score_context() -> str:
        out = c.json(c.post("/score", json={"transactions": sample[-50:], "context": sample[:-50]}))
        ensure(out["n_transactions"] == 50, "context transactions were scored")
        return "50 transactions scored with 1 750 earlier ones as context"

    def score_ibm() -> str:
        ibm = [
            {
                "Timestamp": "2022/09/01 10:00",
                "From Bank": "11",
                "Account": "A",
                "To Bank": "12",
                "Account.1": "B",
                "Amount Received": 9500.0,
                "Receiving Currency": "US Dollar",
                "Amount Paid": 9500.0,
                "Payment Currency": "US Dollar",
                "Payment Format": "ACH",
            }
        ]
        out = c.json(c.post("/score", json={"transactions": ibm}))
        ensure(out["n_transactions"] == 1, "IBM record not scored")
        return "IBM AML column names accepted"

    def scan() -> str:
        out = c.json(c.post("/scan", json={"transactions": sample, "top": 20}))
        alerts = out.get("alerts", [])
        ensure(alerts, "no motif alert on a table with planted motifs")
        a = alerts[0]
        ensure({"pattern", "accounts", "tx_ids", "evidence"} <= set(a), f"alert keys {sorted(a)}")
        return f"{out.get('summary', {}).get('n_alerts', len(alerts))} motif alerts, first: {a['pattern']}"

    c.run("scoring", "POST /score (canonical records)", score)
    c.run("scoring", "POST /score with context", score_context)
    c.run("scoring", "POST /score (IBM AML records)", score_ibm)
    c.run("scoring", "POST /scan (motifs with evidence)", scan)

    # ------------------------------------------------------------------ R4 explanations
    for method in ["auto", *state.get("methods", [])]:

        def explain(method: str = method) -> str:
            payload = {
                "transactions": sample,
                "row": state.get("top_row", 0),
                "method": method,
                "max_edges": 60,
            }
            exp = c.json(c.post("/explain", json=payload))
            ensure({"tx_id", "method", "edges", "nodes", "features", "summary"} <= set(exp), "keys")
            ensure(exp["summary"], "empty summary")
            json.loads(json.dumps(exp))  # JSON export round trip
            neigh = [e for e in exp["edges"] if e.get("role") != "target"]
            ensure(exp["edges"] or exp["patterns"], "no edge and no pattern in the explanation")
            return (
                f"{exp['method']}: {len(neigh)} neighbouring transactions, {len(exp['nodes'])}"
                f" accounts, {len(exp['features'])} features"
            )

        c.run("explain", f"POST /explain method={method}", explain)

    # ------------------------------------------------------------------ R2 investigation
    def model() -> str:
        m = c.json(c.get("/api/model"))
        ensure(m["explain_methods"] and m["patterns"], "no methods / pattern descriptions")
        return f"{len(m['patterns'])} motif descriptions"

    def demo() -> str:
        d = c.json(
            c.post(
                "/api/datasets/demo",
                json={"accounts": 300, "background": 2500, "illicit_ratio": 0.04, "seed": 3},
            )
        )
        state["ds"] = d["id"]
        c.created.append(d["id"])
        ensure(d["transactions"] > 0 and d["n_alerts"] > 0, "empty demo")
        ensure(d["origin"] == {"type": "demo", "seed": 3}, f"origin {d.get('origin')}")
        listed = {x["id"] for x in c.json(c.get("/api/datasets"))}
        ensure(d["id"] in listed, "demo not listed")
        return (
            f"{d['transactions']} transactions, {d['n_alerts']} alerts, {d['analysis_seconds']} s"
        )

    def overview() -> str:
        o = c.json(c.get(f"/api/datasets/{state['ds']}"))
        ensure(
            {"dataset", "eda", "alerts_by_pattern", "score_histogram", "metrics"} <= set(o), "keys"
        )
        ensure(sum(o["score_histogram"]["counts"]) == o["dataset"]["transactions"], "histogram")
        return f"motifs {o['alerts_by_pattern']}, metrics {'yes' if o['metrics'] else 'no'}"

    def alerts() -> str:
        a = c.json(c.get(f"/api/datasets/{state['ds']}/alerts", params={"limit": 5}))
        ensure(a["total"] > 0 and len(a["items"]) <= 5, "pagination")
        ensure(all(x["evidence"] for x in a["items"]), "alerts listed without their evidence")
        pattern = a["items"][0]["pattern"]
        f = c.json(
            c.get(f"/api/datasets/{state['ds']}/alerts", params={"pattern": pattern, "limit": 500})
        )
        ensure(all(x["pattern"] == pattern for x in f["items"]), "pattern filter")
        m = c.json(
            c.get(f"/api/datasets/{state['ds']}/alerts", params={"min_score": 0.3, "limit": 500})
        )
        ensure(all(x["score"] >= 0.3 for x in m["items"]), "min_score filter")
        p2 = c.json(c.get(f"/api/datasets/{state['ds']}/alerts", params={"limit": 5, "offset": 5}))
        ensure(
            not {x["alert_id"] for x in p2["items"]} & {x["alert_id"] for x in a["items"]},
            "pages overlap",
        )
        state["alert_id"] = a["items"][0]["alert_id"]
        return f"{a['total']} alerts; pattern, min_score and pagination filters consistent"

    def alert() -> str:
        d = c.json(c.get(f"/api/datasets/{state['ds']}/alerts/{state['alert_id']}"))
        ids = {n["id"] for n in d["graph"]["nodes"]}
        ensure(d["transactions"] and d["graph"]["edges"], "no transactions / sub-graph")
        ensure(
            all(e["source"] in ids and e["target"] in ids for e in d["graph"]["edges"]),
            "dangling edge",
        )
        return f"{d['pattern']}: {len(d['transactions'])} transactions, sub-graph of {len(ids)} accounts"

    def transactions() -> str:
        base = f"/api/datasets/{state['ds']}/transactions"
        s = c.json(c.get(base, params={"limit": 50}))["items"]
        ensure(all(a["score"] >= b["score"] for a, b in itertools.pairwise(s)), "score order")
        am = c.json(c.get(base, params={"sort": "amount", "descending": False, "limit": 50}))[
            "items"
        ]
        ensure(all(a["amount"] <= b["amount"] for a, b in itertools.pairwise(am)), "amount order")
        acc = s[0]["sender"]
        ac = c.json(c.get(base, params={"account": acc, "limit": 500}))["items"]
        ensure(ac and all(acc in (r["sender"], r["receiver"]) for r in ac), "account filter")
        al = c.json(c.get(base, params={"only_alerts": True, "limit": 500}))["items"]
        ensure(all(r["alert"] for r in al), "only_alerts filter")
        state["tx"], state["account"] = s[0]["tx_id"], acc
        return "sorting (score, amount), account and alert filters consistent"

    def transaction() -> str:
        t = c.json(c.get(f"/api/datasets/{state['ds']}/transactions/{state['tx']}"))
        ensure(json.dumps(t).count(str(state["tx"])) > 0, "transaction id missing")
        return "transaction detail with its alerts"

    c.run("investigation", "GET /api/model", model)
    c.run("investigation", "POST /api/datasets/demo", demo)
    if "ds" in state:
        c.run("investigation", "GET /api/datasets/{id} (dashboard)", overview)
        c.run("investigation", "GET alerts (filters, pages)", alerts)
        c.run("investigation", "GET alerts/{id} (evidence, sub-graph)", alert)
        c.run("investigation", "GET transactions (sort, filters)", transactions)
        c.run("investigation", "GET transactions/{tx}", transaction)
        for method in ["auto", *state.get("methods", [])]:

            def inv_explain(method: str = method) -> str:
                r = c.json(
                    c.get(
                        f"/api/datasets/{state['ds']}/transactions/{state['tx']}/explain",
                        params={"method": method},
                    )
                )
                exp, graph = r, r["graph"]  # the explanation, plus the graph to draw
                ensure(exp["summary"] and graph["nodes"], "empty explanation / graph")
                ensure(isinstance(exp.get("details"), dict), "no details (figures of the summary)")
                json.loads(json.dumps(r))
                return f"{exp['method']}: graph of {len(graph['nodes'])} accounts, {len(graph['edges'])} transactions"

            c.run("explain", f"GET transactions/{{tx}}/explain method={method}", inv_explain)

        def account() -> str:
            a = c.json(c.get(f"/api/datasets/{state['ds']}/accounts/{state['account']}"))
            ensure(a["n_sent"] + a["n_received"] > 0 and a["graph"]["nodes"], "empty profile")
            return f"{a['n_sent']} sent, {a['n_received']} received, {len(a['alerts'])} alerts"

        def exports() -> str:
            out = []
            for kind in ("alerts", "scores"):
                r = c.get(f"/api/datasets/{state['ds']}/export/{kind}.csv")
                ensure(
                    r.status_code == 200
                    and "attachment" in r.headers.get("content-disposition", ""),
                    f"{kind}.csv not a download",
                )
                table = pd.read_csv(io.StringIO(r.text))
                ensure(len(table) > 0, f"empty {kind}.csv")
                out.append(f"{kind}.csv {len(table)} rows")
            return ", ".join(out)

        c.run("investigation", "GET accounts/{account} (profile)", account)
        c.run("investigation", "CSV exports", exports)

    def upload(kind: str) -> Callable[[], str]:
        def run() -> str:
            if kind == "canonical":
                content = df.to_csv(index=False).encode()
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    content = write_ibm_csv(df, Path(tmp) / "ibm.csv").read_bytes()
            r = c.post(
                "/api/datasets/upload",
                files={"file": (f"{kind}.csv", content, "text/csv")},
                data={"schema": "auto"},
            )
            d = c.json(r)
            c.created.append(d["id"])
            ensure(
                d["transactions"] == len(df),
                f"{d['transactions']} transactions instead of {len(df)}",
            )
            return f"{kind} CSV ({len(content) // 1024} kB): {d['transactions']} transactions, {d['n_alerts']} alerts"

        return run

    c.run("investigation", "upload canonical CSV", upload("canonical"))
    c.run("investigation", "upload IBM AML CSV", upload("ibm"))

    # ------------------------------------------------------------------ web interface (static)
    def ui() -> str:
        r = c.get("/", follow_redirects=True)
        ensure(r.status_code == 200 and str(r.url).rstrip("/").endswith("/ui"), f"/ -> {r.url}")
        ensure("AML-GraphMiner" in r.text, "page title missing")
        # resources loaded from another site (the SVG namespace URL is a name, not a download)
        external = re.findall(r"""(?:src|href)\s*=\s*["']https?://[^"']+""", r.text)
        ensure(not external, f"external resources {external}")
        for asset in ("fr.js", "app.js", "style.css"):
            a = c.get(f"/ui/{asset}")
            ensure(a.status_code == 200 and len(a.text) > 1000, f"{asset} -> {a.status_code}")
        return "page, scripts and style served; no external resource (works offline)"

    c.run("ui", "GET / -> /ui/ (assets)", ui)

    # ------------------------------------------------------------------ robustness
    def errors() -> str:
        ensure(
            c.post("/score", json={"transactions": []}).status_code == 422, "empty list accepted"
        )
        bad = dict(sample[0], amount_received=-5)
        r = c.post("/score", json={"transactions": [bad]})
        ensure(r.status_code == 422 and "error" in r.json(), f"negative amount -> {r.status_code}")
        r = c.post("/explain", json={"transactions": sample[:5], "row": 0, "method": "telepathy"})
        ensure(r.status_code == 422, f"unknown method -> {r.status_code}")
        ensure(c.get("/api/datasets/unknown-id").status_code == 404, "unknown dataset not 404")
        if "ds" in state:
            ensure(
                c.get(f"/api/datasets/{state['ds']}/transactions/987654321").status_code == 404,
                "unknown transaction not 404",
            )
        return "422 on invalid input (with an error message), 404 on unknown ids"

    def too_big() -> str:
        size = int((args.upload_limit_mb + 0.5) * 1024 * 1024)
        content = b"timestamp,sender,receiver,amount\n" + b"2022-09-01 10:00:00,a,b,1\n" * (
            size // 26
        )
        r = c.post("/api/datasets/upload", files={"file": ("big.csv", content, "text/csv")})
        ensure(r.status_code == 422 and "MB" in r.text, f"oversized upload -> {r.status_code}")
        return (
            f"{len(content) // (1024 * 1024)} MB file refused (limit {args.upload_limit_mb:g} MB)"
        )

    c.run("robustness", "invalid requests", errors)
    if args.upload_limit_mb is not None:
        c.run("robustness", "upload size limit", too_big)

    def cleanup() -> str:
        for ds in c.created:
            ensure(c.json(c.client.delete(f"/api/datasets/{ds}"))["status"] == "deleted", ds)
            ensure(c.get(f"/api/datasets/{ds}").status_code == 404, f"{ds} still there")
        return f"{len(c.created)} datasets created by the check deleted"

    c.run("investigation", "DELETE /api/datasets/{id}", cleanup)

    passed = sum(r.ok for r in c.results)
    print(f"\n{passed}/{len(c.results)} checks passed against {args.base_url}")
    slow = sorted(c.results, key=lambda r: -r.seconds)[:3]
    print("slowest:", ", ".join(f"{r.name} {r.seconds:.1f}s" for r in slow))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "base_url": args.base_url,
                    "passed": passed,
                    "total": len(c.results),
                    "results": [r.__dict__ for r in c.results],
                },
                fh,
                indent=1,
            )
    return 0 if passed == len(c.results) else 1


if __name__ == "__main__":
    sys.exit(main())
