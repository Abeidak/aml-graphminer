"""Hybrid detector on IBM AML: Kaggle-trained models (runs/, seed 0) + tuned motif scanner.

No model is retrained: the stacker (3 coefficients) and the motif calibration are fitted on the
validation slice, metrics are on the test slice. The base model alone and the motifs alone are
evaluated on exactly the same scores (component_metrics).

    python benchmarks/run_hybrid.py benchmarks/results_parts/hybrid/hybrid.json HI-Small:rf,hgb,sage LI-Small:rf,hgb,sage

Run from the repository root (IBM files in data/, saved models in runs/<model>-<dataset>); each
hybrid is also saved to runs/hybrid-<model>-<dataset>.
"""

from __future__ import annotations

import json
import sys
import time

from aml_graphminer.detectors.bundle import MotifDetector, load_detector
from aml_graphminer.detectors.hybrid import HybridDetector
from aml_graphminer.detectors.patterns import PatternConfig
from aml_graphminer.ingestion import load_dataset

out_path = sys.argv[1]
config = PatternConfig.from_yaml("configs/patterns.yaml")
report: dict = {}
for spec in sys.argv[2:]:
    name, bases = spec.split(":")
    t0 = time.time()
    df = load_dataset(f"data/{name}_Trans.csv")
    motifs = MotifDetector.fit(df, config)
    print(
        f"{name}: loaded + motifs calibrated in {time.time() - t0:.0f}s; motifs test PR-AUC "
        f"{motifs.metrics.get('pr_auc')}",
        flush=True,
    )
    report[name] = {"motifs_alone": motifs.metrics, "hybrids": {}}
    for base_name in bases.split(","):
        t0 = time.time()
        base = load_detector(f"runs/{base_name}-{name}")
        hybrid = HybridDetector.fit(df, base, motifs=motifs)
        hybrid.save(f"runs/hybrid-{base_name}-{name}")
        row = {
            "hybrid": hybrid.metrics,
            "components": hybrid.component_metrics,
            "stacker": hybrid.describe()["stacker"],
            "seconds": round(time.time() - t0),
        }
        report[name]["hybrids"][base_name] = row
        print(
            f"  {base_name}: base {row['components']['base']['pr_auc']:.4f} | motifs "
            f"{row['components']['motifs']['pr_auc']:.4f} | hybrid {row['hybrid']['pr_auc']:.4f} "
            f"({row['seconds']}s) stacker {row['stacker']}",
            flush=True,
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
print("ALL DONE", flush=True)
