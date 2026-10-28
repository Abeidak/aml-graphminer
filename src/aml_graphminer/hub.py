"""Hugging Face Hub export: model repository layout + generated model card.

``export_model`` needs nothing but the standard library; ``push_to_hub`` needs the ``hub``
extra (``pip install 'aml-graphminer[hub]'``) and an access token.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from . import __version__
from .detectors.bundle import read_bundle_config
from .exceptions import ModelError
from .utils import require

_METRIC_ORDER = ("pr_auc", "f1", "precision", "recall", "recall_at_k")


def render_model_card(config: dict[str, Any], repo_id: str | None = None) -> str:
    """Markdown model card (with Hub YAML metadata) generated from a saved detector's config."""
    model_type = str(config.get("model_type") or "gnn")
    gnn = config.get("gnn") or {}
    kind = gnn.get("kind") or config.get("kind") or model_type
    metrics = config.get("metrics") or {}
    shown = [(m, metrics[m]) for m in _METRIC_ORDER if isinstance(metrics.get(m), (int, float))]
    rows = "\n".join(f"| {name} | {float(value):.4f} |" for name, value in shown)
    metric_block = (
        f"Held-out temporal test split (threshold selected on validation):\n\n| metric | value |\n|---|---:|\n{rows}\n"
        if rows
        else "No evaluation metrics were stored with this model.\n"
    )
    title = repo_id or f"aml-graphminer-{kind}"
    threshold = config.get("threshold", "n/a")
    return f"""---
library_name: aml-graphminer
license: mit
tags:
  - anti-money-laundering
  - graph-neural-networks
  - fraud-detection
  - {model_type}
---

# {title}

Detector of suspicious financial transactions produced with
[AML-GraphMiner](https://github.com/Abeidak/aml-graphminer) v{config.get("library_version", __version__)}.

- **Model type**: `{model_type}` ({kind})
- **Decision threshold** (F1-optimal on validation): `{threshold}`
- **Input**: a table of transactions (timestamp, sender, receiver, amount, currency, payment format ...)
- **Output**: a laundering probability per transaction

## Evaluation

{metric_block}
## Usage

```python
from aml_graphminer.detectors.bundle import load_detector
from aml_graphminer.ingestion import load_dataset

detector = load_detector("path/to/this/repository")
transactions = load_dataset("transactions.csv")
alerts = detector.predict(transactions)          # DataFrame: score, alert
```

## Intended use and limitations

- Research and teaching on graph-based AML detection; **not** a compliance product.
- Trained on a *synthetic* benchmark (e.g. IBM AML) unless stated otherwise: scores on real
  banking data may differ substantially. Re-calibrate the threshold on your own labelled data.
- Laundering is extremely rare (~0.1 % of transactions): always review PR-AUC and recall@k, not accuracy.
- Alerts should be reviewed by a human analyst; use the explanation tools (`amlgm explain`).
"""


def export_model(model_dir: str | Path, out_dir: str | Path, repo_id: str | None = None) -> Path:
    """Copy a saved detector into ``out_dir`` and add ``README.md`` (the model card)."""
    src, dst = Path(model_dir), Path(out_dir)
    config = read_bundle_config(src)
    if src.resolve() == dst.resolve():
        raise ModelError("the export directory must differ from the model directory")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst / item.name)
    (dst / "README.md").write_text(render_model_card(config, repo_id), encoding="utf-8")
    return dst


def push_to_hub(
    directory: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = True,
    commit_message: str = "Upload AML-GraphMiner detector",
) -> str:  # pragma: no cover - needs network + huggingface_hub
    """Create (if needed) the model repository and upload ``directory``; returns its URL."""
    hf = require("huggingface_hub", "hub", "pushing to the Hugging Face Hub")
    api = hf.HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(directory),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


__all__ = ["export_model", "push_to_hub", "render_model_card"]
