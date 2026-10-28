#!/usr/bin/env python
"""Benchmark tabular baselines and GNNs on the Elliptic Bitcoin dataset (node classification).

The three CSV files of the Kaggle dataset (``elliptic_txs_features.csv``,
``elliptic_txs_classes.csv``, ``elliptic_txs_edgelist.csv``) must sit in ``--data-dir``::

    python benchmarks/run_elliptic.py --data-dir ~/data/elliptic --models rf hgb gcn sage gat \
        --seeds 0 1 2 --out benchmarks/results

``--synthetic`` runs the pipeline on a small Elliptic-shaped stand-in (**not a result**).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from run_ibm import environment  # same folder

from aml_graphminer.config import load_config
from aml_graphminer.evaluation.benchmark import GNN_KINDS
from aml_graphminer.evaluation.elliptic import ELLIPTIC_MODELS, run_elliptic_benchmark
from aml_graphminer.exceptions import IngestionError
from aml_graphminer.ingestion import read_elliptic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--models", nargs="+", default=["rf", "hgb"], choices=ELLIPTIC_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args(argv)

    if args.synthetic:
        from aml_graphminer.synthetic.elliptic_like import make_elliptic_like

        data = make_elliptic_like(seed=0)
        name = "elliptic-synthetic-smoke"
    else:
        if args.data_dir is None:
            parser.error("--data-dir is required unless --synthetic is given")
        try:
            data = read_elliptic(args.data_dir)
        except IngestionError as exc:
            parser.error(f"{exc} (download the Elliptic files first: amlgm download elliptic)")
        name = "elliptic"
    print(json.dumps(data.summary(), indent=2))

    cfg = load_config(str(args.config) if args.config else None)
    gnn_config = cfg.gnn_config() if any(m in GNN_KINDS for m in args.models) else None
    started = time.perf_counter()
    report = run_elliptic_benchmark(
        data, args.models, args.seeds, dataset=name, gnn_config=gnn_config
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
