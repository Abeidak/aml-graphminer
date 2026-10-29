"""Jupyter plugin: notebooks show the results and run the toolkit through magics.

* **Rich display** — a :class:`~aml_graphminer.detectors.patterns.ScanResult`, an
  :class:`~aml_graphminer.detectors.patterns.Alert`, an
  :class:`~aml_graphminer.explainers.Explanation` or a
  :class:`~aml_graphminer.evaluation.BenchmarkReport` left at the end of a cell is shown as
  tables and an interactive explanation graph (:mod:`aml_graphminer.jupyter.display`). Nothing
  to load: it works in Jupyter, JupyterLab, VS Code and Colab.
* **Magics** — ``%load_ext aml_graphminer`` adds ``%amlgm`` (any command of the command line,
  run inside the kernel), ``%aml_scan`` and ``%aml_explain``
  (:mod:`aml_graphminer.jupyter.magics`, needs IPython).
"""

from __future__ import annotations

from typing import Any

from .display import alert_html, benchmark_html, explanation_html, graph_svg, scan_html


def load_ipython_extension(ipython: Any) -> None:
    """Called by ``%load_ext aml_graphminer`` (or ``%load_ext aml_graphminer.jupyter``)."""
    from .magics import AmlMagics

    ipython.register_magics(AmlMagics)


__all__ = [
    "alert_html",
    "benchmark_html",
    "explanation_html",
    "graph_svg",
    "load_ipython_extension",
    "scan_html",
]
