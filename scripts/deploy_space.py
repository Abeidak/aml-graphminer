#!/usr/bin/env python
"""Deploy the REST API + web interface demo to a Hugging Face Space (Docker, free CPU).

    hf auth login                     # once, in your own terminal (token with "write" scope)
    python scripts/deploy_space.py --repo-id YOUR_ACCOUNT/aml-graphminer-demo
    python scripts/deploy_space.py --repo-id YOUR_ACCOUNT/aml-graphminer-demo --dry-run   # stage only

The Space builds ``Dockerfile`` (the package + its ``api`` and ``viz`` extras) and runs
``amlgm serve --demo`` on port 7860: a detector is trained on SYNTHETIC data at start-up, the
interface is at ``https://<account>-aml-graphminer-demo.hf.space/ui/`` and the OpenAPI docs at
``/docs``. The API has no authentication: never load real customer data into a public Space.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOCKERFILE = """\
# Hugging Face Space: AML-GraphMiner REST API + web interface on synthetic demo data.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 \\
    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl
# Spaces run the container as user 1000
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/src
COPY --chown=user pyproject.toml README.md LICENSE ./
COPY --chown=user src ./src
RUN pip install ".[api,viz]"
USER user
WORKDIR /home/user
EXPOSE 7860
CMD ["amlgm", "serve", "--demo", "--host", "0.0.0.0", "--port", "7860", "--max-upload-mb", "{max_upload_mb}"]
"""

CARD = """\
---
title: AML-GraphMiner demo
emoji: 🕵️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
license: mit
short_description: Explainable money-laundering detection on transaction graphs
---

# AML-GraphMiner — live demo

REST API and web investigation interface of
[AML-GraphMiner]({repo_url}), served by `amlgm serve --demo`:
a gradient-boosting detector is trained at start-up on **synthetic** transactions, whose alerts,
sub-graphs and explanations can be explored in the browser.

* web interface: `/ui/`
* REST API documentation (OpenAPI 3.1): `/docs` — `POST /score`, `POST /explain`, `POST /scan`,
  `GET /health`

The demo has no authentication and keeps uploaded files in memory (at most {max_upload_mb} MB
each): do not upload real customer data. Research software — not a compliance product.
"""


def stage(destination: Path, max_upload_mb: int = 20, repo_url: str = "") -> Path:
    """Write the Space files (Dockerfile, card, package sources) into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "Dockerfile").write_text(
        DOCKERFILE.format(max_upload_mb=max_upload_mb), encoding="utf-8"
    )
    (destination / "README.md").write_text(
        CARD.format(
            max_upload_mb=max_upload_mb,
            repo_url=repo_url or "https://github.com/Abeidak/aml-graphminer",
        ),
        encoding="utf-8",
    )
    for name in ("pyproject.toml", "LICENSE"):
        shutil.copy2(ROOT / name, destination / name)
    shutil.copytree(
        ROOT / "src",
        destination / "src",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    return destination


def deploy(repo_id: str, folder: Path, private: bool) -> str:  # pragma: no cover - network
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy AML-GraphMiner demo",
    )
    return f"https://huggingface.co/spaces/{repo_id}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", required=True, help="ACCOUNT/SPACE_NAME on Hugging Face")
    parser.add_argument("--private", action="store_true", help="create a private Space")
    parser.add_argument("--max-upload-mb", type=int, default=20)
    parser.add_argument("--repo-url", default="", help="link shown on the Space card")
    parser.add_argument("--dry-run", action="store_true", help="stage the files and stop")
    parser.add_argument(
        "--out", type=Path, default=None, help="staging folder (default: temporary)"
    )
    args = parser.parse_args(argv)
    folder = stage(
        args.out or Path(tempfile.mkdtemp(prefix="amlgm-space-")), args.max_upload_mb, args.repo_url
    )
    print(f"staged in {folder}")
    if args.dry_run:
        return 0
    print("deployed:", deploy(args.repo_id, folder, args.private))  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
