#!/usr/bin/env python
"""Benchmark motifs, tabular baselines and GNNs on the IBM AML datasets.

Examples
--------
Real data (the ``*_Patterns.txt`` files next to the CSVs give the per-motif breakdown)::

    python benchmarks/run_ibm.py --data-dir ~/data/ibm-aml --datasets HI-Small LI-Small \
        --models patterns rf hgb gcn sage gat gae --seeds 0 1 2 --out benchmarks/results

Pipeline smoke test without any download (synthetic stand-in, **not a result**)::

    python benchmarks/run_ibm.py --synthetic --models patterns logreg rf --seeds 0 1 --out /tmp/smoke

Then build the report with ``python benchmarks/make_report.py benchmarks/results``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import aml_graphminer as aml
from aml_graphminer.config import load_config
from aml_graphminer.evaluation import ALL_MODELS, GNN_KINDS, UNSUPERVISED_KINDS, run_benchmark
from aml_graphminer.ingestion import load_dataset


def environment() -> dict[str, str]:
    """Versions that matter for reproducibility, stored next to every result."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "aml_graphminer": aml.__version__,
    }
    for module in ("numpy", "pandas", "sklearn", "torch", "torch_geometric"):
        try:
            info[module] = __import__(module).__version__
        except ImportError:
            info[module] = "not installed"
    try:
        import torch

        info["cuda"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, help="directory with <name>_Trans.csv files")
    parser.add_argument("--datasets", nargs="+", default=["HI-Small", "LI-Small"])
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic generator")
    parser.add_argument(
        "--models", nargs="+", default=["patterns", "rf", "hgb"], choices=ALL_MODELS
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--config", type=Path, default=None, help="AML-GraphMiner YAML configuration"
    )
    parser.add_argument("--nrows", type=int, default=None, help="read only the first N rows")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args(argv)

    cfg = load_config(str(args.config) if args.config else None)
    neural = (*GNN_KINDS, *UNSUPERVISED_KINDS)
    gnn_config = cfg.gnn_config() if any(m in neural for m in args.models) else None

    jobs: list[tuple[str, object]] = []
    if args.synthetic:
        frame = aml.generate_transactions(
            n_accounts=1200, n_background=12000, days=10, illicit_ratio=0.012, seed=5
        )
        jobs.append(("synthetic-smoke", frame))
    else:
        if args.data_dir is None:
            parser.error("--data-dir is required unless --synthetic is given")
        for name in args.datasets:
            path = args.data_dir / f"{name}_Trans.csv"
            if not path.exists():
                parser.error(f"{path} not found (download the IBM AML files first)")
            jobs.append((name, path))

    for name, source in jobs:
        print(f"\n=== {name} ===", flush=True)
        frame = (
            source
            if not isinstance(source, Path)
            # load_dataset also attaches the sibling *_Patterns.txt (per-motif recall)
            else load_dataset(source, schema="ibm", nrows=args.nrows)
        )
        started = time.perf_counter()
        report = run_benchmark(
            frame,
            models=args.models,
            seeds=args.seeds,
            dataset=name,
            fractions=cfg.benchmark.fractions,
            feature_config=cfg.feature_config(),
            pattern_config=cfg.pattern_config(),
            gnn_config=gnn_config,
        )
        target = report.save(args.out / name)
        (target / "environment.json").write_text(
            json.dumps(
                {**environment(), "wall_seconds": round(time.perf_counter() - started, 1)}, indent=2
            ),
            encoding="utf-8",
        )
        print(report.to_markdown())
        print(f"saved to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
