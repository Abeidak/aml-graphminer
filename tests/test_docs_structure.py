"""Documentation hygiene that does not need Sphinx: links, snippets and CLI names stay valid."""

import ast
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").glob("*.md"))
TOP_LEVEL = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "RELEASE.md"]
FENCE = re.compile(r"^```(\w+)?\n(.*?)^```", re.S | re.M)


def test_every_page_is_in_a_toctree_and_every_entry_exists():
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    entries = []
    for block in re.findall(r"```\{toctree\}\n(.*?)```", index, re.S):
        entries += [
            ln.strip() for ln in block.splitlines() if ln.strip() and not ln.startswith(":")
        ]
    assert entries, "index.md has no toctree"
    for name in entries:
        assert (ROOT / "docs" / f"{name}.md").exists(), f"toctree entry without page: {name}"
    pages = {p.stem for p in DOCS} - {"index"}
    assert pages == set(entries), f"pages not in the toctree: {pages - set(entries)}"


def test_doc_roles_point_to_existing_pages():
    pages = {p.stem for p in DOCS}
    for path in DOCS:
        for target in re.findall(r"\{doc\}`([^`]+)`", path.read_text(encoding="utf-8")):
            assert target in pages, f"{path.name}: unknown page {target!r}"


@pytest.mark.parametrize("path", [*DOCS, *TOP_LEVEL], ids=lambda p: p.name)
def test_python_snippets_are_valid_python(path):
    text = path.read_text(encoding="utf-8")
    for lang, body in FENCE.findall(text):
        if lang == "python":
            ast.parse(body)


@pytest.mark.parametrize("path", [*DOCS, *TOP_LEVEL], ids=lambda p: p.name)
def test_documented_cli_commands_exist(path):
    from aml_graphminer.cli import main

    commands = set(main.commands)
    for lang, body in FENCE.findall(path.read_text(encoding="utf-8")):
        if lang not in ("bash", "sh", "shell"):
            continue
        for match in re.finditer(r"(?m)^\s*(?:\$ )?amlgm(?: -v+)?\s+([a-z][a-z0-9-]*)", body):
            assert match.group(1) in commands, f"{path.name}: unknown command {match.group(1)!r}"


def test_documented_cli_options_exist():
    """Every ``--option`` written next to an amlgm command in the docs is a real option of it."""
    from aml_graphminer.cli import main

    problems = []
    for path in [*DOCS, *TOP_LEVEL]:
        for lang, body in FENCE.findall(path.read_text(encoding="utf-8")):
            if lang not in ("bash", "sh", "shell"):
                continue
            joined = re.sub(r"\\\n\s*", " ", body)
            for line in joined.splitlines():
                m = re.match(r"\s*(?:\$ )?amlgm(?: -v+)?\s+([a-z][a-z0-9-]*)(.*)", line)
                if not m or m.group(1) not in main.commands:
                    continue
                known = {o for p in main.commands[m.group(1)].params for o in p.opts}
                known |= {o for p in main.commands[m.group(1)].params for o in p.secondary_opts}
                for option in re.findall(r"(?<!\w)(--[a-z][a-z0-9-]*)", m.group(2)):
                    if option not in known:
                        problems.append(f"{path.name}: amlgm {m.group(1)} {option}")
    assert not problems, problems


def test_default_yaml_files_are_valid_configurations():
    from aml_graphminer.config import load_config

    for name in ("default.yaml", "patterns.yaml"):
        load_config(ROOT / "configs" / name)
        yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_workflow_files_are_valid_yaml():
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "jobs" in data, path.name
    compose = yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8"))
    assert "api" in compose["services"]


def test_changelog_has_a_section_for_the_current_version():
    from aml_graphminer import __version__

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{__version__}]" in changelog


def test_no_unreplaced_placeholders_outside_the_documented_ones():
    """Only the placeholders listed in RELEASE.md may remain; anything else is a leftover."""
    allowed = {"OWNER", "TODO"}
    checked = [ROOT / "pyproject.toml", ROOT / "CITATION.cff", *TOP_LEVEL, *DOCS]
    for path in checked:
        text = path.read_text(encoding="utf-8")
        for token in re.findall(r"\b(?:FIXME|XXX|CHANGEME|lorem ipsum)\b", text, flags=re.I):
            raise AssertionError(f"{path.name}: leftover marker {token!r}")
    release = (ROOT / "RELEASE.md").read_text(encoding="utf-8")
    assert all(token in release for token in allowed)
