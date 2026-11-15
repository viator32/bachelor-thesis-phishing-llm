from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import typer

from .experiment_runner import ExperimentRunner, load_config, train_tfidf
from .metrics import compute_metrics
from .mlx_server import MlxServerStartupError
from .model_adapter import MlxAdapter, mlx_api_key
from .personalization import write_personalization_template
from .storage import ResultStore
from .task_evaluation import run_task_evaluation
from .tfidf_training import configure_saved_classifier, train_colab_classifier

app = typer.Typer(help="Local LLM phishing research prototype")


# Resolve PROJECT_ROOT or the current directory.
def _root() -> Path:
    return Path(os.getenv("PROJECT_ROOT", ".")).resolve()


# Evaluate a base or adapted model on validation or test data and print the report.
@app.command("evaluate-adapter")
def evaluate_adapter(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    split: str = typer.Option("test", help="validation for setup; test for the frozen holdout"),
    limit: int | None = typer.Option(None, min=1, help="Maximum emails; internal holdouts require an even paired subset"),
    run_id: str | None = typer.Option(None, help="Reuse the ID to resume the same experiment"),
    personalization_annotations: Path | None = typer.Option(
        None, exists=True, readable=True, help="Independent P0/P1 annotation CSV"
    ),
    dataset_manifest: Path | None = typer.Option(
        None, exists=True, readable=True, help="Override the frozen dataset manifest, e.g. external evaluation"
    ),
) -> None:
    """Compare base or adapted models using phishing/origin or injection JSON schemas."""
    if split not in {"validation", "test"}:
        raise typer.BadParameter("split must be validation or test")
    result = run_task_evaluation(
        config, _root(), split, limit, run_id, typer.echo, personalization_annotations,
        dataset_manifest,
    )
    typer.echo(json.dumps(result, indent=2))


# Create a blank annotation CSV for task or prototype JSONL records.
@app.command("personalization-template")
def personalization_template(
    dataset: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
) -> None:
    """Create blank P0/P1 annotations without changing a dataset or contacting models."""
    count = write_personalization_template(dataset, output)
    typer.echo(f"Created {count} blank annotation rows: {output}")


# Run an experiment with optional CLI overrides and print its output paths.
@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    run_id: str | None = typer.Option(None, help="Reuse this ID to resume an interrupted run"),
    dataset: str | None = typer.Option(None, help="Override the configured dataset path"),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of emails"),
    resume: bool | None = typer.Option(None, "--resume/--no-resume"),
    config_id: str | None = typer.Option(None, help="Override the experiment identifier"),
    personalization_annotations: Path | None = typer.Option(
        None, exists=True, readable=True, help="Independent P0/P1 annotation CSV"
    ),
) -> None:
    """Execute one configured experiment."""
    overrides = {
        "dataset_path": dataset, "limit": limit, "resume": resume, "config_id": config_id,
        "personalization_annotations": str(personalization_annotations)
        if personalization_annotations else None,
    }
    experiment_config = load_config(
        config, **{key: value for key, value in overrides.items() if value is not None}
    )
    runner = ExperimentRunner(experiment_config, _root())

    actual_id, records, _, paths = runner.run(run_id)
    typer.echo(
        json.dumps({"run_id": actual_id, "records": len(records), "reports": paths}, indent=2)
    )


@app.command("configure-tfidf")
def configure_tfidf(
    name: str = typer.Option(..., help="Unique experiment name"),
    model: Path = typer.Option(..., help="Fitted TF-IDF pipeline under models/"),
    dataset: Path = typer.Option(..., help="Backend-format dataset under data/processed/"),
) -> None:
    """Create an experiment for an existing classifier without fitting it again."""
    typer.echo(json.dumps(configure_saved_classifier(_root(), name, model, dataset), indent=2))


# Import the saved Colab splits and train a separate named classifier.
@app.command("train-tfidf-colab")
def train_tfidf_colab(
    name: str = typer.Option(..., help="Unique name for this classifier and experiment"),
    source_dir: Path | None = typer.Option(None, help="Saved Colab run folder"),
    config: Path = typer.Option(Path("configs/baseline_training/tfidf.yaml")),
) -> None:
    """Train on saved joint-task splits and create a matching test experiment."""
    result = train_colab_classifier(_root(), name, source_dir, config)
    typer.echo(json.dumps(result, indent=2))


# Train a dataset that already uses the backend email schema.
@app.command("train-tfidf")
def train_tfidf_command(
    dataset: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(Path("models/tfidf_logreg.joblib")),
    split: str = typer.Option("train"),
) -> None:
    """Train the deterministic TF-IDF logistic-regression baseline."""
    count = train_tfidf(dataset, output, split=split)
    typer.echo(f"Trained on {count} records and saved model to {output}")


# Recompute and print metrics from a stored run.
@app.command("metrics")
def metrics_command(run_id: str) -> None:
    """Recompute metrics exclusively from stored results."""
    store = ResultStore(
        os.getenv("RESULTS_DIR", str(_root() / "results")), os.getenv("DATABASE_PATH") or None
    )
    records = store.load_run(run_id)
    if not records:
        raise typer.BadParameter(f"No stored records for {run_id}")
    typer.echo(json.dumps(compute_metrics(records), indent=2))


# Check server connectivity, report failure and release owned resources.
@app.command("check-mlx")
def check_mlx(
    model: str = typer.Option(os.getenv("MLX_MODEL", "Qwen3-4B-4bit")),
    base_url: str = typer.Option(os.getenv("MLX_BASE_URL", "http://localhost:8080")),
    auto_start: bool = typer.Option(
        False,
        "--auto-start",
        help="Spawn a raw `mlx_lm.server` if nothing is reachable at base-url. Leave off when "
        "using the oMLX app, which manages its own configured server.",
    ),
) -> None:
    """Check local MLX server connectivity (oMLX app on :8080 by default)."""
    adapter = MlxAdapter(
        model,
        base_url,
        api_key=mlx_api_key(),
        auto_start=auto_start,
    )
    try:
        typer.echo(json.dumps(adapter.health(), indent=2))
    except (MlxServerStartupError, httpx.HTTPError) as exc:
        typer.echo(f"MLX server unavailable: {exc}")
        raise typer.Exit(code=1) from None
    finally:
        adapter.close()


if __name__ == "__main__":
    app()
