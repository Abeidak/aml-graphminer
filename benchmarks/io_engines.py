"""Pandas or Polars to read the IBM AML files: time and peak memory of a full ingestion.

    python benchmarks/io_engines.py data/HI-Small_Trans.csv data/LI-Small_Trans.csv --repeats 2

Every measure runs in a fresh Python process (a warm cache or a grown heap of a previous run
would bias it): ``load_dataset`` (read + canonical schema + validation + laundering patterns)
with ``engine="pandas"`` then ``engine="polars"``. Peak memory = the peak resident set of that
process; the memory of the interpreter and the imports, measured before reading, is reported
too so that the cost of the data itself can be read. Both engines must give the same table
(same rows, same total amount, same laundering count), which the script checks. Results go to
``benchmarks/results_parts/io/`` (JSON + Markdown).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).parent / "results_parts" / "io"


def _peak_bytes() -> int:
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak is None:  # pragma: no cover - Linux / macOS
        import resource

        scale = 1 if sys.platform == "darwin" else 1024
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    return int(peak)


def child(path: str, engine: str) -> None:
    """One measure, printed as one JSON line (runs in its own process)."""
    import psutil

    from aml_graphminer.ingestion import load_dataset

    baseline = psutil.Process().memory_info().rss
    start = time.perf_counter()
    df = load_dataset(path, engine=engine)
    seconds = time.perf_counter() - start
    print(
        json.dumps(
            {
                "engine": engine,
                "file": Path(path).name,
                "rows": len(df),
                "seconds": round(seconds, 2),
                "peak_mb": round(_peak_bytes() / 2**20),
                "baseline_mb": round(baseline / 2**20),
                "amount_sum": round(float(df["amount"].sum()), 2),
                "illicit": int(df["label"].sum()),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    results = []
    for path in args.files:
        for engine in ("pandas", "polars"):
            for run in range(args.repeats):
                done = subprocess.run(
                    [sys.executable, __file__, "--child", path, engine],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                row = json.loads(done.stdout.strip().splitlines()[-1])
                row["run"] = run + 1
                results.append(row)
                print(row, flush=True)
    # same table whatever the engine
    for path in args.files:
        name = Path(path).name
        keys = {(r["rows"], r["amount_sum"], r["illicit"]) for r in results if r["file"] == name}
        if len(keys) != 1:
            raise SystemExit(f"{name}: the two engines disagree: {keys}")
    OUT.mkdir(parents=True, exist_ok=True)
    machine = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "processor": platform.processor(),
    }
    (OUT / "io_engines.json").write_text(
        json.dumps({"machine": machine, "results": results}, indent=1), encoding="utf-8"
    )
    lines = [
        "# Reading the IBM AML files: pandas or polars",
        "",
        f"`python benchmarks/io_engines.py ...` on {machine['os']} (Python {machine['python']});",
        "`load_dataset` = read + canonical schema + validation + laundering patterns; best of",
        f"{args.repeats} runs, each in a fresh process. Both engines give the same table.",
        "",
        "| file | rows | engine | seconds | peak memory (MB) | of which interpreter + imports |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for path in args.files:
        name = Path(path).name
        for engine in ("pandas", "polars"):
            runs = [r for r in results if r["file"] == name and r["engine"] == engine]
            best = min(runs, key=lambda r: r["seconds"])
            lines.append(
                f"| {name} | {best['rows']:,} | {engine} | {best['seconds']:.1f} | "
                f"{max(r['peak_mb'] for r in runs):,} | {best['baseline_mb']:,} |"
            )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        main()
