"""Torch-free compatibility checks for the GNN stack (importable without PyTorch)."""

from __future__ import annotations

from ...exceptions import AMLGraphMinerError
from ...utils import require

SAMPLER_HELP = (
    "PyTorch Geometric's neighbour sampler needs the compiled 'pyg-lib' (recommended) or "
    "'torch-sparse' package, which are not installed by 'pip install aml-graphminer[gnn]'. "
    "Install the wheel matching your PyTorch / CUDA build, e.g. "
    "pip install pyg-lib -f https://data.pyg.org/whl/torch-<torch version>+<cpu|cu121>.html "
    "(see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)."
)


def check_sampler_backend() -> None:
    """Fail early (and clearly) when PyG cannot sample neighbours.

    PyG raises ``ImportError: 'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'``
    at the *first batch*, after features have been built. The flags checked here are exposed
    by ``torch_geometric.typing``; when a PyG version does not expose them nothing is blocked.
    """
    typing_mod = require("torch_geometric.typing", "gnn", "graph neural networks")
    flags = [getattr(typing_mod, name, None) for name in ("WITH_PYG_LIB", "WITH_TORCH_SPARSE")]
    if all(flag is False for flag in flags):
        raise AMLGraphMinerError(SAMPLER_HELP)


__all__ = ["SAMPLER_HELP", "check_sampler_backend"]
