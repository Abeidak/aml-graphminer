#!/usr/bin/env python
"""Turn ``# %%`` percent-format Python files into Jupyter notebooks (optionally executed).

Why this exists: the notebooks of the project are maintained as plain ``.py`` files
(``notebooks/src/*.py``) so that diffs stay reviewable. This script builds the ``.ipynb`` files
that are shipped with the repository.

* Without ``--execute`` the notebook has no outputs (what Colab users get on first open).
* With ``--execute`` every code cell is run in one shared namespace and its outputs (stdout,
  stderr, the value of a trailing expression, DataFrames as HTML, matplotlib figures as PNG)
  are stored in the notebook. This lightweight executor needs nothing but the standard
  library and matplotlib; the CI re-executes the shipped notebooks with ``jupyter nbconvert
  --execute`` (see ``.github/workflows/ci.yml``), which is the authoritative check.

Conventions of the source files
-------------------------------
``# %%``            starts a code cell
``# %% [markdown]``  starts a markdown cell; its lines are ``# `` prefixed
``# %pip ...``       in a code cell becomes the magic ``%pip ...`` (skipped when executing here)

Usage::

    python scripts/py2nb.py notebooks/src/01_quickstart.py --out notebooks --execute
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- parsing
def parse_percent(text: str) -> list[dict[str, Any]]:
    """Split a percent-format script into ``{"type", "source"}`` cells."""
    cells: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        if line.startswith("# %%"):
            if current is not None:
                cells.append(current)
            kind = "markdown" if "[markdown]" in line else "code"
            current = {"type": kind, "lines": []}
            continue
        if current is None:
            continue  # module docstring / header before the first cell
        current["lines"].append(line)
    if current is not None:
        cells.append(current)

    out = []
    for cell in cells:
        lines = list(cell["lines"])
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if cell["type"] == "markdown":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines]
        else:
            lines = [
                ln[2:] if ln.startswith(("# %pip", "# %matplotlib", "# !")) else ln for ln in lines
            ]
        if lines:
            out.append({"type": cell["type"], "source": "\n".join(lines)})
    return out


# --------------------------------------------------------------------------- notebook json
def _source(text: str) -> list[str]:
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def build_notebook(cells: list[dict[str, Any]], outputs: list[list[dict[str, Any]]] | None = None):
    nb_cells = []
    counter = 0
    for i, cell in enumerate(cells):
        if cell["type"] == "markdown":
            nb_cells.append(
                {"cell_type": "markdown", "metadata": {}, "source": _source(cell["source"])}
            )
            continue
        counter += 1
        nb_cells.append(
            {
                "cell_type": "code",
                "execution_count": counter if outputs is not None else None,
                "metadata": {},
                "outputs": outputs[i] if outputs is not None else [],
                "source": _source(cell["source"]),
            }
        )
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": platform.python_version()},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


# --------------------------------------------------------------------------- execution
def _strip_magics(source: str) -> str:
    return "\n".join("" if ln.lstrip().startswith(("%", "!")) else ln for ln in source.split("\n"))


def _figures_as_png() -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    outputs = []
    for num in plt.get_fignums():
        buffer = io.BytesIO()
        plt.figure(num).savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        outputs.append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "text/plain": ["<Figure>"],
                },
            }
        )
    plt.close("all")
    return outputs


def _value_output(value: Any, count: int) -> dict[str, Any]:
    data: dict[str, Any] = {"text/plain": _source(repr(value))}
    html = getattr(value, "_repr_html_", None)
    if callable(html):
        with contextlib.suppress(Exception):
            data["text/html"] = _source(html())
    return {"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": data}


def execute_cells(cells: list[dict[str, Any]], workdir: Path) -> list[list[dict[str, Any]]]:
    """Run the code cells in order; raise ``RuntimeError`` at the first failing cell."""
    import matplotlib

    matplotlib.use("Agg")
    namespace: dict[str, Any] = {"__name__": "__main__"}
    outputs: list[list[dict[str, Any]]] = []
    count = 0
    previous = Path.cwd()
    try:
        os.chdir(workdir)
        for index, cell in enumerate(cells):
            if cell["type"] != "code":
                outputs.append([])
                continue
            count += 1
            source = _strip_magics(cell["source"])
            stdout, stderr = io.StringIO(), io.StringIO()
            cell_outputs: list[dict[str, Any]] = []
            tree = ast.parse(source)
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.Expression(tree.body.pop().value)
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(tree, f"<cell {index}>", "exec"), namespace)
                    value = (
                        eval(compile(last_expr, f"<cell {index}>", "eval"), namespace)
                        if last_expr is not None
                        else None
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"cell {index} failed:\n{cell['source']}\n\n{traceback.format_exc()}"
                ) from exc
            for name, stream in (("stdout", stdout), ("stderr", stderr)):
                if stream.getvalue():
                    cell_outputs.append(
                        {"output_type": "stream", "name": name, "text": _source(stream.getvalue())}
                    )
            cell_outputs.extend(_figures_as_png())
            if last_expr is not None and value is not None:
                cell_outputs.append(_value_output(value, count))
            outputs.append(cell_outputs)
    finally:
        os.chdir(previous)
    return outputs


# --------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("sources", nargs="+", type=Path, help="percent-format .py files")
    parser.add_argument("--out", type=Path, default=Path("notebooks"), help="output directory")
    parser.add_argument("--execute", action="store_true", help="run the cells and store outputs")
    parser.add_argument(
        "--workdir", type=Path, default=None, help="working directory when executing"
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        cells = parse_percent(source.read_text(encoding="utf-8"))
        outputs = None
        if args.execute:
            work = (args.workdir or args.out).resolve()
            work.mkdir(parents=True, exist_ok=True)
            outputs = execute_cells(cells, work)
        notebook = build_notebook(cells, outputs)
        target = args.out / f"{source.stem}.ipynb"
        target.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"{source} -> {target} ({len(cells)} cells{', executed' if args.execute else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
