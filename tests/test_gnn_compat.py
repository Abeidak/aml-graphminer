"""Early, readable failure when PyG has no neighbour-sampling backend (no torch needed)."""

import sys
import types

import pytest

from aml_graphminer.detectors.gnn._compat import SAMPLER_HELP, check_sampler_backend
from aml_graphminer.exceptions import AMLGraphMinerError


def _fake_pyg_typing(monkeypatch, **flags):
    package = types.ModuleType("torch_geometric")
    typing_module = types.ModuleType("torch_geometric.typing")
    for name, value in flags.items():
        setattr(typing_module, name, value)
    package.typing = typing_module
    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.typing", typing_module)


def test_missing_backends_raise_an_actionable_error(monkeypatch):
    _fake_pyg_typing(monkeypatch, WITH_PYG_LIB=False, WITH_TORCH_SPARSE=False)
    with pytest.raises(AMLGraphMinerError, match="pyg-lib") as err:
        check_sampler_backend()
    assert "data.pyg.org" in str(err.value) and str(err.value) == SAMPLER_HELP


@pytest.mark.parametrize(
    "flags",
    [
        {"WITH_PYG_LIB": True, "WITH_TORCH_SPARSE": False},
        {"WITH_PYG_LIB": False, "WITH_TORCH_SPARSE": True},
        {},  # a PyG version that does not expose the flags is never blocked
        {"WITH_PYG_LIB": False},  # only one flag exposed: not enough to conclude
    ],
)
def test_any_available_backend_passes(monkeypatch, flags):
    _fake_pyg_typing(monkeypatch, **flags)
    check_sampler_backend()
