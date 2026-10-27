## What and why

<!-- One paragraph. Link the issue if there is one. -->

## Checklist

- [ ] `ruff check . && ruff format --check . && mypy src` pass
- [ ] Tests added or updated; `pytest` passes locally (tests needing an extra may be skipped)
- [ ] No leakage: nothing uses validation/test data or labels when building features or thresholds
- [ ] Docs / `CHANGELOG.md` updated for user-visible changes
- [ ] If a notebook source changed: `python scripts/py2nb.py notebooks/src/*.py --execute` re-run
