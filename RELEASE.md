# Release checklist

Work through this list top to bottom for every release; steps marked ⚠ cannot be done by the
CI and must be checked by a person.

## Once, before the first public release

- [ ] ⚠ Replace every `OWNER` (`grep -rn OWNER .`) with the real GitHub organisation / user.
- [ ] ⚠ `CITATION.cff` and `pyproject.toml`: real author names, e-mail addresses, ORCID iDs and
      the release date; remove the `TODO` markers.
- [ ] ⚠ Confirm the licence of the generated documentation and of any text copied from the
      project specification.
- [ ] ⚠ Run notebook `04_gnn_training_colab` on a Colab GPU; commit it with its outputs. Push the
      trained checkpoints to the Hugging Face Hub (`amlgm export-hub ... --push`, deliverable L6).
- [ ] ⚠ Download the real IBM AML and Elliptic files (`amlgm download ibm`, `amlgm download
      elliptic` with a Kaggle token), run `amlgm eda`, `benchmarks/run_ibm.py` and
      `benchmarks/run_elliptic.py` (3 seeds), then `benchmarks/make_report.py`. Check that
      `amlgm -v inspect HI-Small_Trans.csv` reports `attached N/M laundering-pattern transactions`
      with N close to M (validates the `*_Patterns.txt` parser on the real files).
- [ ] ⚠ Open the web interface on a slice of the real data (`amlgm serve runs/<model>`), check
      the alerts and explanations with a domain expert, take the screenshots for the paper.
- [ ] ⚠ Tune the motif thresholds on the real validation split (`configs/patterns.yaml`) and
      document the values used in the paper.
- [ ] CI is green on `main`: lint, core (3 Pythons), extras, **gnn**, notebooks, docs, package, docker.
      Fix whatever the GNN/Docker/docs jobs reveal — they could not be run when this version was written.
- [ ] Read the documentation build log (`sphinx-build -n`) and fix warnings.
- [ ] Create the PyPI project, configure *trusted publishing* for `release.yml` (environment `pypi`).
- [ ] Zenodo: enable the GitHub integration so each release gets a DOI; add the DOI to
      `CITATION.cff`, the README and the paper (Software Impacts requires an archived version).
- [ ] Read the Docs: import the repository (`.readthedocs.yaml` is provided).

## Every release

- [ ] Update `__version__` in `src/aml_graphminer/__init__.py`.
- [ ] Move the `Unreleased` items of `CHANGELOG.md` under `## [X.Y.Z] - YYYY-MM-DD`
      (the release workflow publishes that section as release notes).
- [ ] Update `date-released` and `version` in `CITATION.cff`.
- [ ] `pytest`, `ruff`, `mypy` pass; notebooks re-executed (`scripts/py2nb.py ... --execute`).
- [ ] `git tag vX.Y.Z && git push origin vX.Y.Z` — the *Release* workflow builds, checks the tag
      against the version, publishes to PyPI and creates the GitHub release.
- [ ] Verify in a clean environment: `pip install aml-graphminer==X.Y.Z && amlgm --version`.
