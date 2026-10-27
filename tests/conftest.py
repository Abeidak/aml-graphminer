"""Shared fixtures (small synthetic datasets are built once per session)."""

from __future__ import annotations

import warnings

import pytest

from aml_graphminer.evaluation.splits import temporal_split
from aml_graphminer.synthetic import generate_transactions

warnings.filterwarnings("ignore", category=DeprecationWarning)


def pytest_configure(config):  # pragma: no cover - only with real pytest
    config.addinivalue_line("markers", "gnn: needs torch + torch_geometric")
    config.addinivalue_line("markers", "slow: slower end-to-end tests")


@pytest.fixture(scope="session")
def synth_df():
    """~6k transactions with every motif type (seeded, deterministic)."""
    return generate_transactions(n_accounts=700, n_background=5000, illicit_ratio=0.04, seed=3)


@pytest.fixture(scope="session")
def tiny_df():
    """~1.5k transactions (fast tests)."""
    return generate_transactions(n_accounts=250, n_background=1200, illicit_ratio=0.05, seed=7)


@pytest.fixture(scope="session")
def synth_masks(synth_df):
    return temporal_split(synth_df)
