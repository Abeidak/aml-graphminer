# Quick start

Everything below runs on **synthetic data** shipped with the package, so nothing needs to be
downloaded. The executed notebook `notebooks/01_quickstart.ipynb` contains the same steps with
outputs and figures.

## From Python

```python
import aml_graphminer as aml
from aml_graphminer.detectors.bundle import TabularDetector

# 1. data in the canonical schema (read_transactions("HI-Small_Trans.csv") for the real files)
df = aml.generate_transactions(n_accounts=900, n_background=7000, days=10, illicit_ratio=0.02)

# 2. structural motifs: no training, every alert has evidence
result = aml.PatternScanner().scan(df)
print(result.to_frame().head())

# 3. a learned detector on leak-free features (temporal split 60/20/20)
detector = TabularDetector.fit(df, "hgb")
print(detector.metrics)  # test metrics, threshold chosen on validation
detector.save("runs/hgb")

# 4. why was this transaction flagged?
from aml_graphminer.explainers import OcclusionExplainer, detector_score_fn

tx_id = int(detector.score(df).idxmax())
explanation = OcclusionExplainer(detector_score_fn(detector), hops=2).explain(df, tx_id)
print(explanation.to_markdown())
```

## From the command line

```bash
amlgm generate data/ --accounts 1500 --seed 0       # writes data/Synth-Small_Trans.csv (+ Patterns)
amlgm inspect data/Synth-Small_Trans.csv            # validation + dataset summary
amlgm scan    data/Synth-Small_Trans.csv --out alerts/        # motif alerts as CSV and JSON
amlgm train   data/Synth-Small_Trans.csv --model hgb --out runs/hgb
amlgm score   runs/hgb data/Synth-Small_Trans.csv --out scores.csv
amlgm explain runs/hgb data/Synth-Small_Trans.csv --tx 1234 --plot why.png --out why.json
amlgm benchmark data/Synth-Small_Trans.csv --models patterns --models rf --models hgb --seeds 0 --seeds 1
amlgm serve   runs/hgb --port 8000                  # REST API (needs the `api` extra)
```

On the real IBM files replace the path by `HI-Small_Trans.csv`; the sibling
`HI-Small_Patterns.txt` is detected automatically and fills the `pattern` column.
`amlgm --help` and `amlgm <command> --help` list every option; see {doc}`cli`.

## What to read next

* {doc}`concepts` — the schema, the temporal protocol and why it matters.
* {doc}`patterns` — what each motif detector looks for and how to tune it.
* {doc}`evaluation` — how to benchmark properly and what to report.
