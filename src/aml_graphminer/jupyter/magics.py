"""IPython magics of the Jupyter plugin, registered by ``%load_ext aml_graphminer``.

* ``%amlgm COMMAND ...`` — any command of the ``amlgm`` command line, run inside the kernel;
* ``%aml_scan DATA [--config FILE]`` — motif alerts of a DataFrame of the notebook or a file;
* ``%aml_explain MODEL DATA TX [--method M] [--hops H] [--max-edges N]`` — explanation of a
  transaction by a detector of the notebook or a model folder.

``%aml_scan?`` and ``%aml_explain?`` print the full help in a notebook; see the Jupyter plugin
page of the documentation for examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import pandas as pd
from IPython.core.error import UsageError
from IPython.core.magic import Magics, line_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments
from IPython.utils.process import arg_split

from ..exceptions import AMLGraphMinerError

METHODS = ["auto", "patterns", "occlusion", "gnnexplainer", "pgexplainer", "captum", "attention"]


def _split(line: str) -> list[str]:
    # quotes are removed ("my file.csv"); on Windows backslashes of paths are kept
    return arg_split(line, posix=True)


@magics_class
class AmlMagics(Magics):
    """``%amlgm`` (the command line inside the kernel), ``%aml_scan`` and ``%aml_explain``."""

    def _lookup(self, name: str) -> Any:
        return None if self.shell is None else self.shell.user_ns.get(name)

    def _frame(self, name: str) -> pd.DataFrame:
        """A DataFrame of the notebook (by its name) or a transaction file."""
        value = self._lookup(name)
        if isinstance(value, pd.DataFrame):
            return value
        if Path(name).is_file():
            from ..ingestion import load_dataset

            return load_dataset(name)
        raise UsageError(f"{name!r} is neither a DataFrame of the notebook nor a transaction file")

    def _detector(self, name: str) -> Any:
        """A detector of the notebook (by its name) or a model folder."""
        value = self._lookup(name)
        if value is not None and callable(getattr(value, "score", None)):
            return value
        if Path(name).is_dir():
            from ..detectors.bundle import load_detector

            return load_detector(name)
        raise UsageError(f"{name!r} is neither a detector of the notebook nor a model folder")

    @line_magic
    def amlgm(self, line: str) -> None:
        """Run a command of the ``amlgm`` command line inside the kernel.

        ``%amlgm scan data.csv --top 5``, ``%amlgm train data.csv --model hgb --out runs/hgb``,
        ``%amlgm --help``: the commands and options of the terminal, run by the interpreter of
        the notebook (no PATH to set up, the version installed in the kernel). IPython expands
        ``{variable}`` before the command runs.
        """
        from ..cli import main

        try:
            main.main(args=_split(line), prog_name="amlgm", standalone_mode=False)
        except click.exceptions.Abort:
            raise UsageError("amlgm: aborted") from None
        except click.ClickException as exc:
            raise UsageError(f"amlgm: {exc.format_message()}") from None
        except AMLGraphMinerError as exc:
            raise UsageError(f"amlgm: {exc}") from None

    @magic_arguments()
    @argument("data", help="DataFrame of the notebook (its name) or a transaction file")
    @argument("-c", "--config", default=None, help="YAML configuration (amlgm init-config)")
    @line_magic
    def aml_scan(self, line: str) -> Any:
        """Scan transactions for laundering motifs and show the alerts: ``%aml_scan df``.

        Returns the :class:`~aml_graphminer.detectors.patterns.ScanResult`
        (``result = %aml_scan df``).
        """
        args = self.aml_scan.parser.parse_args(_split(line))
        from ..config import load_config
        from ..detectors.patterns import PatternScanner

        df = self._frame(args.data)
        try:
            return PatternScanner(load_config(args.config).pattern_config()).scan(df)
        except AMLGraphMinerError as exc:
            raise UsageError(str(exc)) from None

    @magic_arguments()
    @argument("model", help="detector of the notebook (its name) or a model folder")
    @argument("data", help="DataFrame of the notebook (its name) or a transaction file")
    @argument("tx", type=int, help="id of the transaction to explain")
    @argument("-m", "--method", default="auto", choices=METHODS, help="explanation method")
    @argument("--hops", type=int, default=None, help="neighbourhood radius")
    @argument("--max-edges", type=int, default=80, help="largest neighbourhood considered")
    @line_magic
    def aml_explain(self, line: str) -> Any:
        """Explain why a transaction was (not) flagged: ``%aml_explain runs/hgb df 1234``.

        Shows the verdict, the explanation graph and the transactions and attributes that
        matter; returns the :class:`~aml_graphminer.explainers.Explanation`.
        """
        args = self.aml_explain.parser.parse_args(_split(line))
        from ..explainers import explain_transaction

        df = self._frame(args.data)
        if args.tx not in df.index:
            raise UsageError(
                f"transaction {args.tx} not found (valid ids: {df.index.min()}..{df.index.max()})"
            )
        detector = self._detector(args.model)
        try:
            return explain_transaction(
                detector, df, args.tx, args.method, hops=args.hops, max_edges=args.max_edges
            )
        except AMLGraphMinerError as exc:
            raise UsageError(str(exc)) from None


__all__ = ["AmlMagics"]
