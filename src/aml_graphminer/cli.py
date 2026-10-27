"""Command-line interface: ``amlgm`` (generate, inspect, scan, train, benchmark, score, explain, serve ...)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd

from . import __version__
from .config import default_config_yaml, load_config
from .exceptions import AMLGraphMinerError

SCHEMAS = click.Choice(["auto", "ibm", "canonical"])
GNN_MODELS = ("gcn", "sage", "gat", "gae")
MODEL_CHOICES = ("patterns", "rf", "hgb", "xgb", "logreg", *GNN_MODELS)


def _echo_json(payload: Any) -> None:
    click.echo(json.dumps(payload, indent=2, default=str, ensure_ascii=False))


def _load(data: str, schema: str, nrows: int | None, patterns_file: str | None) -> pd.DataFrame:
    from .ingestion import load_dataset

    return load_dataset(data, schema=schema, nrows=nrows, patterns_file=patterns_file or True)


def _require_labels(df: pd.DataFrame, what: str) -> None:
    if df["label"].isna().any():
        raise click.ClickException(
            f"{what} needs fully labelled data (column 'label' / 'Is Laundering')"
        )


data_options = [
    click.option("--schema", type=SCHEMAS, default="auto", show_default=True, help="Input layout."),
    click.option("--nrows", type=int, default=None, help="Read only the first N rows."),
    click.option(
        "--patterns-file",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="IBM *_Patterns.txt (default: the sibling file of an IBM *_Trans.csv).",
    ),
]


def _with_data_options(func: Any) -> Any:
    for option in reversed(data_options):
        func = option(func)
    return func


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="amlgm")
@click.option("-v", "--verbose", count=True, help="-v: info, -vv: debug.")
def main(verbose: int) -> None:
    """AML-GraphMiner — graph-based anti-money-laundering detection toolkit."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# --------------------------------------------------------------------------- data
@main.command()
@click.argument("out_dir", type=click.Path(file_okay=False))
@click.option(
    "--name", default="Synth-Small", show_default=True, help="File prefix (<name>_Trans.csv)."
)
@click.option("--accounts", type=int, default=1500, show_default=True)
@click.option(
    "--background", type=int, default=15000, show_default=True, help="Legitimate transactions."
)
@click.option("--days", type=float, default=10.0, show_default=True)
@click.option("--illicit-ratio", type=float, default=0.01, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def generate(
    out_dir: str,
    name: str,
    accounts: int,
    background: int,
    days: float,
    illicit_ratio: float,
    seed: int,
) -> None:
    """Generate an IBM-AML-like synthetic dataset (Trans.csv + Patterns.txt)."""
    from .synthetic import generate_transactions, write_ibm_files

    df = generate_transactions(
        accounts, background, days=days, illicit_ratio=illicit_ratio, seed=seed
    )
    trans, patterns = write_ibm_files(df, out_dir, name)
    click.echo(f"wrote {trans} and {patterns}")
    click.echo(f"{len(df)} transactions, {int(df['label'].sum())} laundering")


@main.command()
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@_with_data_options
def inspect(data: str, schema: str, nrows: int | None, patterns_file: str | None) -> None:
    """Validate a dataset and print its summary."""
    from .ingestion import summarize

    _echo_json(summarize(_load(data, schema, nrows, patterns_file)))


@main.command("init-config")
@click.argument("path", type=click.Path(dir_okay=False), default="amlgm.yaml")
@click.option("--force", is_flag=True, help="Overwrite an existing file.")
def init_config(path: str, force: bool) -> None:
    """Write a fully-populated default configuration file."""
    target = Path(path)
    if target.exists() and not force:
        raise click.ClickException(f"{path} already exists (use --force)")
    target.write_text(default_config_yaml(), encoding="utf-8")
    click.echo(f"wrote {path}")


# --------------------------------------------------------------------------- detection
@main.command()
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option(
    "--calibrate/--no-calibrate",
    default=False,
    help="Calibrate pattern precision on the validation slice (needs labels).",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory for alerts.csv / alerts.json.",
)
@click.option("--top", type=int, default=10, show_default=True)
@_with_data_options
def scan(
    data: str,
    config_path: str | None,
    calibrate: bool,
    out: str | None,
    top: int,
    schema: str,
    nrows: int | None,
    patterns_file: str | None,
) -> None:
    """Scan for laundering motifs (fan-in/out, smurfing, cycles, scatter-gather, gather-scatter)."""
    from .detectors.patterns import (
        PatternScanner,
        alert_precision_table,
        evaluate_scan,
        pattern_recall_table,
    )
    from .evaluation.splits import temporal_split

    cfg = load_config(config_path)
    df = _load(data, schema, nrows, patterns_file)
    scanner = PatternScanner(cfg.pattern_config())
    labelled = df["label"].notna().all()
    if calibrate:
        _require_labels(df, "--calibrate")
        scanner.fit_calibration(df[temporal_split(df)["val"]])
    result = scanner.scan(df)
    frame = result.to_frame()
    click.echo(f"{len(result)} alert(s) on {len(df)} transactions")
    if len(frame):
        view = frame.drop(columns=["accounts", "tx_ids"]).head(top)
        click.echo(view.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if labelled and df["label"].sum() > 0:
        click.echo("\nTransaction-level metrics:")
        _echo_json({k: round(float(v), 4) for k, v in evaluate_scan(df, result).items()})
        click.echo("\nPer-detector precision:")
        click.echo(alert_precision_table(df, result).to_string(index=False))
        if (df["pattern"] != "").any():
            click.echo("\nRecall per ground-truth motif:")
            click.echo(pattern_recall_table(df, result).to_string(index=False))
    if out:
        directory = Path(out)
        directory.mkdir(parents=True, exist_ok=True)
        flat = (
            frame.assign(
                accounts=frame["accounts"].map(lambda a: ";".join(a)),
                tx_ids=frame["tx_ids"].map(lambda t: ";".join(map(str, t))),
            )
            if len(frame)
            else frame
        )
        flat.to_csv(directory / "alerts.csv", index=False)
        (directory / "alerts.json").write_text(
            json.dumps([a.to_dict() for a in result.alerts], indent=2), encoding="utf-8"
        )
        scores, top_pattern = result.transaction_scores()
        pd.DataFrame({"tx_id": df.index, "score": scores, "pattern": top_pattern}).to_csv(
            directory / "scores.csv", index=False
        )
        click.echo(f"\nwrote alerts and scores to {directory}")


@main.command()
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", "kind", type=click.Choice(MODEL_CHOICES), default="rf", show_default=True)
@click.option(
    "--out",
    type=click.Path(file_okay=False),
    required=True,
    help="Directory where the detector is saved.",
)
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--epochs", type=int, default=None, help="GNN epochs (overrides the configuration).")
@click.option(
    "--trainer", type=click.Choice(["torch", "lightning"]), default=None, help="GNN training loop."
)
@click.option("--mlflow", "use_mlflow", is_flag=True, help="Track the run with MLflow.")
@_with_data_options
def train(
    data: str,
    kind: str,
    out: str,
    config_path: str | None,
    seed: int,
    epochs: int | None,
    trainer: str | None,
    use_mlflow: bool,
    schema: str,
    nrows: int | None,
    patterns_file: str | None,
) -> None:
    """Train a detector (motifs, RF/HGB/XGB/logreg, GCN/SAGE/GAT or the unsupervised GAE) and save it.

    ``--model gae`` is the unsupervised graph auto-encoder: it needs no label (labels, when
    present, only choose the alert threshold); its encoder is ``gnn.kind`` of the configuration.
    """
    from .detectors.bundle import MotifDetector, TabularDetector

    cfg = load_config(config_path)
    if use_mlflow:
        cfg.tracking.backend = "mlflow"
    df = _load(data, schema, nrows, patterns_file)
    if kind == "patterns":
        detector: Any = MotifDetector.fit(
            df if df["label"].notna().all() else None, cfg.pattern_config()
        )
    elif kind in GNN_MODELS:
        from .detectors.gnn import AutoEncoderDetector, GNNConfig, GNNDetector

        overrides = dict(cfg.gnn) if kind == "gae" else {**cfg.gnn, "kind": kind}
        if epochs is not None:
            overrides["epochs"] = epochs
        if trainer is not None:
            overrides["trainer"] = trainer
        if kind != "gae":
            _require_labels(df, "training")
        with cfg.tracker(f"train-{kind}") as tracker:
            fit = AutoEncoderDetector.fit if kind == "gae" else GNNDetector.fit
            detector = fit(
                df,
                GNNConfig.from_dict(overrides),
                feature_config=cfg.feature_config(),
                seed=seed,
                tracker=tracker,
            )
    else:
        _require_labels(df, "training")
        detector = TabularDetector.fit(
            df,
            kind,
            fractions=cfg.benchmark.fractions,
            feature_config=cfg.feature_config(),
            seed=seed,
            with_nodes=cfg.benchmark.tabular_node_features,
        )
    detector.save(out)
    click.echo(f"saved {kind} detector to {out}")
    _echo_json(detector.describe())


@main.command()
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--models", multiple=True, help="Models to run (repeat the option); default: configuration."
)
@click.option(
    "--seeds", multiple=True, type=int, help="Seeds (repeat the option); default: configuration."
)
@click.option(
    "--out",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory for results.csv / summary.md / config.json.",
)
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option(
    "--dataset-name", default=None, help="Label stored in the results (default: file name)."
)
@click.option("--mlflow", "use_mlflow", is_flag=True, help="Track the run with MLflow.")
@_with_data_options
def benchmark(
    data: str,
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    out: str | None,
    config_path: str | None,
    dataset_name: str | None,
    use_mlflow: bool,
    schema: str,
    nrows: int | None,
    patterns_file: str | None,
) -> None:
    """Compare motifs, tabular baselines and GNNs on a temporal split (the CDC benchmark)."""
    from .evaluation import run_benchmark

    cfg = load_config(config_path)
    if use_mlflow:
        cfg.tracking.backend = "mlflow"
    df = _load(data, schema, nrows, patterns_file)
    _require_labels(df, "the benchmark")
    chosen = list(models) or cfg.benchmark.models
    with cfg.tracker("benchmark") as tracker:
        report = run_benchmark(
            df,
            models=chosen,
            seeds=list(seeds) or cfg.benchmark.seeds,
            dataset=dataset_name or Path(data).stem,
            fractions=cfg.benchmark.fractions,
            feature_config=cfg.feature_config(),
            pattern_config=cfg.pattern_config(),
            gnn_config=cfg.gnn_config() if any(m in GNN_MODELS for m in chosen) else None,
            tabular_node_features=cfg.benchmark.tabular_node_features,
            tracker=tracker,
        )
        click.echo(report.to_markdown())
        if report.per_pattern is not None:
            click.echo("\nRecall per ground-truth motif (pattern scanner, test slice):")
            click.echo(report.per_pattern.to_string(index=False))
        if out:
            directory = report.save(out)
            tracker.log_artifact(directory / "summary.md")
            click.echo(f"\nresults written to {directory}")


@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--models", multiple=True, help="Models to run (repeat the option).")
@click.option("--seeds", multiple=True, type=int)
@click.option("--out", type=click.Path(file_okay=False), default=None)
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None)
def elliptic(
    directory: str,
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    out: str | None,
    config_path: str | None,
) -> None:
    """Benchmark on the Elliptic Bitcoin dataset (node classification)."""
    from .evaluation.elliptic import run_elliptic_benchmark
    from .ingestion import read_elliptic

    cfg = load_config(config_path)
    data = read_elliptic(directory)
    click.echo(json.dumps(data.summary(), indent=2))
    chosen = list(models) or ["rf", "hgb"]
    report = run_elliptic_benchmark(
        data,
        chosen,
        list(seeds) or cfg.benchmark.seeds,
        gnn_config=cfg.gnn_config() if any(m in ("gcn", "sage", "gat") for m in chosen) else None,
    )
    click.echo(report.to_markdown())
    if out:
        report.save(out)


# --------------------------------------------------------------------------- use
@main.command()
@click.argument("model_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--context",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Extra history (not scored) that is part of the graph.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default=None,
    help="CSV with one score per transaction.",
)
@click.option("--top", type=int, default=20, show_default=True)
@_with_data_options
def score(
    model_dir: str,
    data: str,
    context: str | None,
    out: str | None,
    top: int,
    schema: str,
    nrows: int | None,
    patterns_file: str | None,
) -> None:
    """Score transactions with a saved detector."""
    from .detectors.bundle import load_detector

    detector = load_detector(model_dir)
    df = _load(data, schema, nrows, patterns_file)
    ctx = _load(context, schema, None, None) if context else None
    result = detector.predict(df, ctx)
    table = df[["timestamp", "sender", "receiver", "amount", "currency", "transaction_type"]].join(
        result
    )
    if df["label"].notna().all():
        table["label"] = df["label"]
    click.echo(
        f"{int(result['alert'].sum())} alert(s) on {len(df)} transactions (threshold {detector.threshold:.3g})"
    )
    click.echo(
        table.sort_values("score", ascending=False)
        .head(top)
        .to_string(float_format=lambda v: f"{v:.3f}")
    )
    if out:
        table.to_csv(out)
        click.echo(f"scores written to {out}")


@main.command()
@click.argument("model_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--tx",
    "tx_id",
    type=int,
    required=True,
    help="Transaction id (row position / index of the table).",
)
@click.option(
    "--method",
    type=click.Choice(
        ["auto", "patterns", "occlusion", "gnnexplainer", "pgexplainer", "captum", "attention"]
    ),
    default="auto",
    show_default=True,
    help="auto = pattern evidence for motif detectors, occlusion otherwise; "
    "gnnexplainer / pgexplainer / captum / attention need a GNN detector.",
)
@click.option(
    "--hops",
    type=int,
    default=None,
    help="Neighbourhood radius (default: 1 for motifs, 2 for models).",
)
@click.option("--max-edges", type=int, default=80, show_default=True)
@click.option(
    "--out", type=click.Path(dir_okay=False), default=None, help="JSON file for the explanation."
)
@click.option(
    "--plot",
    "plot_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="PNG image of the explanation graph.",
)
@_with_data_options
def explain(
    model_dir: str,
    data: str,
    tx_id: int,
    method: str,
    hops: int | None,
    max_edges: int,
    out: str | None,
    plot_path: str | None,
    schema: str,
    nrows: int | None,
    patterns_file: str | None,
) -> None:
    """Explain why a transaction was (not) flagged."""
    from .detectors.bundle import load_detector
    from .explainers import explain_transaction

    detector = load_detector(model_dir)
    df = _load(data, schema, nrows, patterns_file)
    if tx_id not in df.index:
        raise click.ClickException(
            f"transaction {tx_id} not found (valid ids: {df.index.min()}..{df.index.max()})"
        )
    try:
        explanation = explain_transaction(
            detector, df, tx_id, method, hops=hops, max_edges=max_edges
        )
    except AMLGraphMinerError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(explanation.to_markdown())
    if out:
        Path(out).write_text(explanation.to_json(), encoding="utf-8")
        click.echo(f"\nexplanation written to {out}")
    if plot_path:
        from .explainers import plot_explanation

        plot_explanation(explanation, path=plot_path)
        click.echo(f"figure written to {plot_path}")


def _demo_model() -> Path:
    """Train (once) a quick gradient-boosting detector on synthetic data for ``serve --demo``."""
    import tempfile

    from .detectors.bundle import TabularDetector
    from .synthetic import generate_transactions

    directory = Path(tempfile.gettempdir()) / f"aml-graphminer-demo-model-{__version__}"
    if not (directory / "config.json").exists():
        click.echo("training a demo detector on synthetic data (first run only)...")
        df = generate_transactions(1500, 15000, illicit_ratio=0.02, seed=42)
        TabularDetector.fit(df, "hgb").save(directory)
    return directory


@main.command()
@click.argument("model_dir", type=click.Path(exists=True, file_okay=False), required=False)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8000, show_default=True)
@click.option(
    "--data",
    "data_files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Transaction file analysed at start-up and shown in the web interface (repeatable).",
)
@click.option(
    "--demo",
    is_flag=True,
    help="Analyse a synthetic demo dataset at start-up; without MODEL_DIR a demo detector is trained.",
)
@click.option("--open-browser", is_flag=True, help="Open the web interface in the default browser.")
def serve(
    model_dir: str | None,
    host: str,
    port: int,
    data_files: tuple[str, ...],
    demo: bool,
    open_browser: bool,
) -> None:
    """Serve a detector: REST API + web investigation interface at http://HOST:PORT/ui/.

    Needs the ``api`` extra. ``amlgm serve --demo`` starts everything without any preparation.
    """
    if model_dir is None and not demo:
        raise click.ClickException(
            "give MODEL_DIR (a directory written by `amlgm train`) or --demo"
        )
    from .api.app import create_app
    from .utils import require

    uvicorn = require("uvicorn", "api", "the REST API")
    if model_dir is None:
        model_dir = str(_demo_model())
    app = create_app(model_dir, datasets=list(data_files), demo=demo)
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{shown}:{port}/ui/"
    click.echo(f"web interface: {url}")
    click.echo(f"API documentation: http://{shown}:{port}/docs")
    if open_browser:  # pragma: no cover - opens a browser
        import threading
        import webbrowser

        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=host, port=port)


@main.command()
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--out",
    type=click.Path(file_okay=False),
    required=True,
    help="Directory for EDA.md, summary.json and the figures.",
)
@click.option("--no-figures", is_flag=True, help="Statistics only (matplotlib not needed).")
@_with_data_options
def eda(
    data: str,
    out: str,
    no_figures: bool,
    schema: str,
    nrows: int | None,
    patterns_file: str | None,
) -> None:
    """Exploratory data analysis: statistics, figures and a Markdown report."""
    from .eda import eda_report

    df = _load(data, schema, nrows, patterns_file)
    report = eda_report(df, out, name=Path(data).name, figures=not no_figures)
    click.echo(f"wrote {report}")


@main.command("export-neo4j")
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_dir", type=click.Path(file_okay=False))
@click.option(
    "--model",
    "model_dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Saved detector whose scores are added to the transactions.",
)
@click.option("--scan/--no-scan", default=True, show_default=True, help="Add the motif alerts.")
@click.option("--push", is_flag=True, help="Also load the graph into a running Neo4j database.")
@click.option("--uri", default="bolt://localhost:7687", show_default=True)
@click.option("--user", default="neo4j", show_default=True)
@click.option("--password", envvar="NEO4J_PASSWORD", default=None, help="Or set NEO4J_PASSWORD.")
@click.option("--database", default="neo4j", show_default=True)
@_with_data_options
def export_neo4j(
    data: str,
    out_dir: str,
    model_dir: str | None,
    scan: bool,
    push: bool,
    uri: str,
    user: str,
    password: str | None,
    database: str,
    schema: str,
    nrows: int | None,
    patterns_file: str | None,
) -> None:
    """Export the transaction graph to Neo4j (CSV files + Cypher scripts, optionally pushed)."""
    from .graph.neo4j import export_neo4j as write_files
    from .graph.neo4j import push_to_neo4j

    df = _load(data, schema, nrows, patterns_file)
    scores = None
    if model_dir:
        from .detectors.bundle import load_detector

        scores = load_detector(model_dir).score(df).to_numpy()
    result = None
    if scan:
        from .detectors.patterns import PatternScanner

        result = PatternScanner().scan(df)
    paths = write_files(df, out_dir, scores=scores, scan=result)
    click.echo(f"wrote {', '.join(p.name for p in paths.values())} to {out_dir}")
    if push:
        if not password:
            raise click.ClickException("--push needs --password (or NEO4J_PASSWORD)")
        counts = push_to_neo4j(
            df, uri, user, password, database=database, scores=scores, scan=result
        )
        click.echo(f"loaded {counts['accounts']} accounts and {counts['transactions']} transfers")


@main.command()
@click.argument("dataset", type=click.Choice(["ibm", "elliptic"]))
@click.option(
    "--out", "out_dir", default="data", show_default=True, type=click.Path(file_okay=False)
)
@click.option(
    "--size",
    "sizes",
    multiple=True,
    type=click.Choice(["small", "medium", "large"]),
    default=("small",),
    show_default=True,
    help="IBM AML size (repeatable).",
)
@click.option(
    "--group",
    "groups",
    multiple=True,
    type=click.Choice(["HI", "LI"]),
    default=("HI", "LI"),
    show_default=True,
    help="IBM AML illicit-ratio group (repeatable).",
)
@click.option("--force", is_flag=True, help="Download again even if the files exist.")
def download(
    dataset: str, out_dir: str, sizes: tuple[str, ...], groups: tuple[str, ...], force: bool
) -> None:
    """Download the reference datasets from Kaggle (``data`` extra + a Kaggle API token)."""
    from .datasets import download_elliptic, download_ibm

    if dataset == "ibm":
        paths = download_ibm(out_dir, sizes=sizes, groups=groups, force=force)
    else:
        paths = download_elliptic(out_dir, force=force)
    for path in paths:
        click.echo(f"  {path}")
    click.echo(f"{len(paths)} file(s) ready in {out_dir}")


@main.command("export-hub")
@click.argument("model_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("out_dir", type=click.Path(file_okay=False))
@click.option("--repo-id", default=None, help="Hugging Face repo (user/name); needed to push.")
@click.option(
    "--push", is_flag=True, help="Upload to the Hugging Face Hub (needs the hub extra and a token)."
)
@click.option("--token", envvar="HF_TOKEN", default=None, help="Hub token (or set HF_TOKEN).")
def export_hub(
    model_dir: str, out_dir: str, repo_id: str | None, push: bool, token: str | None
) -> None:
    """Prepare (and optionally push) a model repository with a generated model card."""
    from .hub import export_model, push_to_hub

    target = export_model(model_dir, out_dir, repo_id=repo_id)
    click.echo(f"model repository prepared in {target}")
    if push:
        if not repo_id:
            raise click.ClickException("--push needs --repo-id")
        click.echo(f"pushed to {push_to_hub(target, repo_id, token=token)}")


def run() -> None:  # pragma: no cover - console entry
    """Entry point that turns library errors into clean messages."""
    try:
        main(standalone_mode=False)
    except AMLGraphMinerError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)
    except click.exceptions.Abort:
        sys.exit(130)
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)


__all__ = ["main", "run"]

if __name__ == "__main__":  # pragma: no cover
    run()
