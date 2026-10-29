"""Sphinx configuration of the AML-GraphMiner documentation."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aml_graphminer

project = "AML-GraphMiner"
author = "Ahmed Ahmed Mahmoud Abeideck, Saade Jaavar Boubacar"
copyright = f"{date.today().year}, {author}"
release = version = aml_graphminer.__version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "tasklist"]
myst_heading_anchors = 3

# autodoc: the heavy optional dependencies are not needed to build the documentation
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "lightning",
    "fastapi",
    "mlflow",
    "huggingface_hub",
    "xgboost",
    "polars",
    "pyarrow",
]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest", None),
}

# References that cannot be resolved in nitpicky mode (-n): mocked optional libraries, pydantic
# constraint metadata rendered in signatures, and short names of postponed annotations.
nitpick_ignore_regex = [
    (r"py:.*", r"(torch|torch_geometric|lightning|fastapi|mlflow|captum|neo4j)(\..*)?"),
    (r"py:.*", r"annotated_types\..*"),
    (r"py:.*", r"(gt|ge|lt|le|min_length|max_length)=.*"),
    (r"py:.*", r"(Path|Tensor|Data|HeteroData|ScoreFn|LossFn|DetectorFactory|Iterator|Any)"),
    (r"py:.*", r"(EdgeGNN|NodeGNN|TrainedGNN|GNNConfig|BenchmarkReport|FeatureSet|ScanResult)"),
    (r"py:.*", r"(FeaturePipeline|TransactionIndex|OcclusionExplainer|Explanation)(\..*)?"),
    (r"py:.*", r"aml_graphminer\..*"),
    (r"py:.*", r"(np|pd|nx)\..*"),
    # short names used in docstrings of the module that defines them
    (
        r"py:.*",
        r"(Adapter|SchemaError|CANONICAL_COLUMNS|PATTERN_TYPES|EllipticData|run_gnn_experiment"
        r"|EXPLAIN_METHODS|DEFAULT_TRACKING_URI|app|frame|n_matched)",
    ),
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"AML-GraphMiner {release}"
