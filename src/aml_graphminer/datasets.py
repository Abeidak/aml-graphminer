"""Download the reference datasets of the project specification from Kaggle.

* **IBM Transactions for Anti-Money Laundering** (Altman et al., NeurIPS 2023) — Kaggle
  ``ealtman2019/ibm-transactions-for-anti-money-laundering-aml``; only the requested files are
  fetched (HI/LI x Small/Medium/Large, ``*_Trans.csv`` + ``*_Patterns.txt``).
* **Elliptic Bitcoin** (Weber et al., 2019) — Kaggle ``ellipticco/elliptic-data-set``.

A Kaggle account and an API token are required (Kaggle → Settings → API → *Create New Token*,
saved as ``~/.kaggle/kaggle.json`` or exported as ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``) and the
``data`` extra (``pip install 'aml-graphminer[data]'``). Check the licence of each dataset before
redistributing anything derived from it. ``amlgm download ibm`` / ``amlgm download elliptic`` call
these functions.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .exceptions import ConfigError, IngestionError
from .utils import logger, require

IBM_DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
ELLIPTIC_DATASET = "ellipticco/elliptic-data-set"
IBM_SIZES = ("small", "medium", "large")
IBM_GROUPS = ("HI", "LI")
ELLIPTIC_FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
)


def ibm_file_names(
    sizes: Iterable[str] = ("small",), groups: Iterable[str] = IBM_GROUPS
) -> list[str]:
    """``HI-Small_Trans.csv``, ``HI-Small_Patterns.txt``, ... for the requested sizes / groups."""
    names = []
    for size in sizes:
        if size not in IBM_SIZES:
            raise ConfigError(f"unknown IBM AML size {size!r}; choose among {IBM_SIZES}")
        for group in groups:
            if group not in IBM_GROUPS:
                raise ConfigError(f"unknown IBM AML group {group!r}; choose among {IBM_GROUPS}")
            stem = f"{group}-{size.capitalize()}"
            names += [f"{stem}_Trans.csv", f"{stem}_Patterns.txt"]
    return names


CREDENTIALS_HELP = (
    "Kaggle credentials not found or rejected: create an API token on kaggle.com (Settings → "
    "API → Create New Token) and save it as ~/.kaggle/kaggle.json (Windows: "
    "C:\\Users\\<you>\\.kaggle\\kaggle.json), or set KAGGLE_USERNAME and KAGGLE_KEY"
)


def kaggle_api() -> Any:  # pragma: no cover - needs the kaggle package and credentials
    """Authenticated Kaggle API client (clear error messages when it cannot be created)."""
    try:
        kaggle = require("kaggle", "data", "downloading the Kaggle datasets")
    except OSError as exc:  # kaggle < 2 authenticates at import time
        raise ConfigError(f"{CREDENTIALS_HELP} ({exc})") from exc
    api = kaggle.api
    try:
        api.authenticate()
    except (Exception, SystemExit) as exc:  # kaggle may call sys.exit() on missing credentials
        raise ConfigError(f"{CREDENTIALS_HELP} ({exc})") from None
    return api


def _unzip_next_to(path: Path) -> None:
    """Kaggle may deliver ``name.csv.zip`` / ``name.zip``: extract and remove the archive."""
    for archive in (path.with_name(path.name + ".zip"), path.with_suffix(".zip")):
        if archive.exists() and zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(path.parent)
            archive.unlink()


def download_ibm(
    directory: str | Path = "data",
    *,
    sizes: Iterable[str] = ("small",),
    groups: Iterable[str] = IBM_GROUPS,
    force: bool = False,
    api: Any = None,
) -> list[Path]:
    """Download the IBM AML files into ``directory`` (existing files are kept unless ``force``)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = ibm_file_names(sizes, groups)
    api = api or kaggle_api()
    paths = []
    for name in names:
        target = directory / name
        if target.exists() and not force:
            logger.info("%s already present", target)
        else:
            logger.info("downloading %s from %s", name, IBM_DATASET)
            api.dataset_download_file(IBM_DATASET, name, path=str(directory), force=force)
            _unzip_next_to(target)
        if not target.exists():
            raise IngestionError(f"download of {name} failed (not found in {directory})")
        paths.append(target)
    return paths


def download_elliptic(
    directory: str | Path = "data/elliptic", *, force: bool = False, api: Any = None
) -> list[Path]:
    """Download and unzip the Elliptic Bitcoin dataset into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def present() -> list[Path]:
        found = []
        for name in ELLIPTIC_FILES:
            hits = sorted(directory.rglob(name))
            if hits:
                found.append(hits[0])
        return found

    if force or len(present()) < len(ELLIPTIC_FILES):
        api = api or kaggle_api()
        logger.info("downloading %s", ELLIPTIC_DATASET)
        api.dataset_download_files(ELLIPTIC_DATASET, path=str(directory), force=force, unzip=True)
    files = present()
    if len(files) < len(ELLIPTIC_FILES):
        raise IngestionError(f"the Elliptic files were not found under {directory} after download")
    return files


__all__ = [
    "ELLIPTIC_DATASET",
    "IBM_DATASET",
    "download_elliptic",
    "download_ibm",
    "ibm_file_names",
    "kaggle_api",
]
