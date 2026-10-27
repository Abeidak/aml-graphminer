"""Elliptic Bitcoin dataset adapter (CDC §5.3: ``elliptic_adapter``).

In Elliptic (Weber et al., 2019) the *transactions are the nodes*, so the adapter returns a
node-classification dataset (:class:`EllipticData`) rather than a table of transfers. The
implementation lives in :mod:`aml_graphminer.ingestion.elliptic`.

Example
-------
>>> from aml_graphminer.ingestion.elliptic_adapter import read_elliptic
>>> data = read_elliptic("data/elliptic")       # doctest: +SKIP
>>> data.summary()                              # doctest: +SKIP
"""

from __future__ import annotations

from .elliptic import ILLICIT, LICIT, UNKNOWN, EllipticData, read_elliptic

__all__ = ["ILLICIT", "LICIT", "UNKNOWN", "EllipticData", "read_elliptic"]
