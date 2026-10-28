# Contributing

Thank you for helping. This is research software: correctness of the evaluation protocol matters
more than raw features.

## Set up

```bash
git clone https://github.com/Abeidak/aml-graphminer && cd aml-graphminer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"            # add ",gnn" (+ the pyg-lib wheel, see docs) to work on the GNN code
```

## Checks (the CI runs the same)

```bash
ruff check . && ruff format --check .
mypy src
pytest                                  # tests needing an optional extra are skipped when it is missing
pytest --cov=aml_graphminer --cov-config=.coveragerc-core    # core gate: 90 % without PyTorch
node --check src/aml_graphminer/api/static/app.js            # the web interface script
sphinx-build -b html -n docs docs/_build/html                # documentation, nitpicky
```

Coverage has two gates: the torch-free **core** must stay ≥ 90 % (`.coveragerc-core` omits the
modules that cannot run without PyTorch / FastAPI), and the **full** installation, where the GNN
tests run, ≥ 90 % (`pyproject.toml`, the threshold of the project specification).

The web interface (`src/aml_graphminer/api/static/`) is plain HTML / CSS / JavaScript without a
build step or external dependency; insert data into the page with `textContent` only, never
`innerHTML`. Its back end (`api/investigation.py`) is framework-free and unit-tested.

## Rules that protect the science

1. **No leakage.** Anything computed from data (means, vocabularies, aggregates, thresholds) is
   fitted on the training split; thresholds are chosen on validation; the test split is only
   scored. Add a test when you touch feature code (see `tests/test_pipeline.py`).
2. **Time is causal.** Features of a transaction use only earlier transactions.
3. **Never present synthetic-data numbers as results.** Synthetic runs are labelled as such by the
   report generator.
4. **Seeds.** Stochastic code takes a `seed`; benchmarks use several.
5. **Honest errors.** Raise the exceptions of `aml_graphminer.exceptions` with a message that says
   what to do; never swallow a failure to keep a notebook green.

## Code style

Type-annotated, NumPy-style docstrings, no global state, heavy imports inside functions or behind
`aml_graphminer.utils.require` so that the core stays importable without optional extras. Public
behaviour changes need a line in `CHANGELOG.md` and, if user-visible, in `docs/`.

## Notebooks

The `.ipynb` files are generated from `notebooks/src/*.py` (percent format) so diffs stay
reviewable:

```bash
python scripts/py2nb.py notebooks/src/01_quickstart.py notebooks/src/02_ibm_aml_benchmark.py \
       notebooks/src/03_explainability.py --out notebooks --execute --workdir /tmp/nbwork
python scripts/py2nb.py notebooks/src/04_gnn_training_colab.py --out notebooks   # needs GPU + data to execute
```

The CI re-executes `01`–`03` with Jupyter and fails when a notebook no longer matches its source.
Notebook `04` has to be run on Colab by hand; commit it with its outputs.

## Adding a detector, adapter or explainer

See `docs/extending.md`. A new motif detector needs: a config section (if tunable), tests on
hand-built scenarios with known ground truth, and an entry in `docs/patterns.md`.

## Pull requests

Small, focused, with tests. The template lists the checklist. For security issues see
`SECURITY.md`.
