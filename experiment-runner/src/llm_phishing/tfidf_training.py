"""Train the TF-IDF baseline on the saved splits of a joint Colab run."""

import json
import re
from pathlib import Path
from typing import Any

import yaml

from .dataset_importer import file_sha256, load_dataset
from .experiment_runner import train_tfidf
from .schemas import ExperimentConfig


def configure_saved_classifier(
    project_root: Path,
    name: str,
    model_path: Path,
    dataset_path: Path,
) -> dict[str, str]:
    """Create a test experiment for an existing fitted TF-IDF pipeline, without retraining."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Name must contain 1–64 letters, numbers, underscores or hyphens.")
    project_root = project_root.resolve()
    model = (project_root / model_path).resolve()
    dataset = (project_root / dataset_path).resolve()
    model.relative_to(project_root / "models")
    dataset.relative_to(project_root / "data/processed")
    if not model.is_file() or model.suffix != ".joblib":
        raise ValueError("Select a saved .joblib classifier from models/.")
    rows, _ = load_dataset(dataset)
    if not any(row.split == "test" and row.ground_truth is not None
               and row.ground_truth.is_phishing is not None for row in rows):
        raise ValueError("The evaluation dataset needs phishing-labeled test rows; injection labels are insufficient.")
    experiment_path = Path("configs/baselines") / f"tfidf_{name}.yaml"
    output = project_root / experiment_path
    if output.exists():
        raise ValueError("This experiment name already exists. Choose a new name.")
    config = ExperimentConfig(
        config_id=f"tfidf-{name}",
        method="tfidf_logreg",
        model=f"tfidf-{name}",
        tfidf_model_path=str(model.relative_to(project_root)),
        dataset_path=str(dataset.relative_to(project_root)),
        seed=42,
        split="test",
        languages=[],
        max_input_tokens=None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return {
        "config_path": str(experiment_path),
        "dataset_path": config.dataset_path,
        "model_path": config.tfidf_model_path,
    }


def prepare_colab_dataset(source_dir: Path, seed: int) -> tuple[list[dict], dict]:
    """Keep the saved emails and group assignments; never draw a new split."""
    run_config = json.loads((source_dir / "run_config.json").read_text(encoding="utf-8"))
    if (
        not isinstance(run_config, dict)
        or run_config.get("task") != "phishing_origin"
        or run_config.get("seed") != seed
    ):
        raise ValueError(f"Select a phishing_origin run trained with seed {seed}.")

    rows = []
    counts = {}
    seen_ids = set()
    group_splits = {}
    for split in ("train", "validation", "test"):
        path = source_dir / f"{split}.jsonl"
        if split == "validation" and not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            saved_rows = [json.loads(line) for line in stream if line.strip()]
        if not saved_rows:
            raise ValueError(f"{path.name} must contain emails.")
        counts[split] = len(saved_rows)
        for row in saved_rows:
            target = row["target"]
            if (
                not isinstance(target, dict)
                or type(target.get("is_phishing")) is not bool
                or target.get("generation_origin") not in {"human", "llm"}
                or not isinstance(row["input_text"], str)
                or not row["input_text"].strip()
            ):
                raise ValueError(
                    "Each email needs text, a boolean phishing label and human/llm origin."
                )
            instance_id, group_id = str(row["id"]), str(row["group_id"])
            if not instance_id.strip() or not group_id.strip():
                raise ValueError("Email and group IDs must not be empty.")
            if instance_id in seen_ids or group_splits.get(group_id, split) != split:
                raise ValueError("Duplicate email IDs or groups shared between splits.")
            seen_ids.add(instance_id)
            group_splits[group_id] = split
            rows.append(
                {
                    "instance_id": instance_id,
                    "base_id": group_id,
                    # Keep the complete saved input, including subject/sender/URL fields.
                    "text_body": row["input_text"],
                    "is_phishing": target["is_phishing"],
                    "generation_origin": "ai" if target["generation_origin"] == "llm" else "human",
                    "split": split,
                    "language": "en",
                    "source_dataset": "PhishFuzzer",
                }
            )
    if len({row["is_phishing"] for row in rows if row["split"] == "train"}) != 2:
        raise ValueError("Training requires both phishing and legitimate emails.")
    return rows, {"source_run": run_config, "counts": counts}


def train_colab_classifier(
    project_root: Path,
    name: str,
    source_dir: Path | None = None,
    config_path: Path = Path("configs/baseline_training/tfidf.yaml"),
) -> dict[str, Any]:
    """Import saved splits, fit on train only and publish a matching experiment config."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Name must contain 1–64 letters, numbers, underscores or hyphens.")
    project_root = project_root.resolve()
    settings = yaml.safe_load((project_root / config_path).read_text(encoding="utf-8"))
    seed = settings["seed"]
    source_dir = project_root / (source_dir or Path(settings["source_dir"]))
    rows, metadata = prepare_colab_dataset(source_dir, seed)

    dataset_path = Path("data/processed") / f"tfidf_{name}.jsonl"
    model_path = Path("models") / f"tfidf_{name}.joblib"
    experiment_path = Path("configs/baselines") / f"tfidf_{name}.yaml"
    metadata_path = model_path.with_suffix(".json")
    outputs = (dataset_path, model_path, experiment_path, metadata_path)
    if any((project_root / path).exists() for path in outputs):
        raise ValueError("This training name already exists. Choose a new name.")

    experiment = ExperimentConfig(
        config_id=f"tfidf-{name}",
        method="tfidf_logreg",
        model=f"tfidf-{name}",
        dataset_path=str(dataset_path),
        tfidf_model_path=str(model_path),
        seed=seed,
        max_input_tokens=settings["max_input_tokens"],
        split="test",
    )
    for path in outputs:
        (project_root / path).parent.mkdir(parents=True, exist_ok=True)
    (project_root / dataset_path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    trained = train_tfidf(
        project_root / dataset_path,
        project_root / model_path,
        seed=seed,
        max_input_tokens=experiment.max_input_tokens,
    )
    metadata.update(
        {
            "seed": seed,
            "trained_records": trained,
            "dataset_path": str(dataset_path),
            "dataset_sha256": file_sha256(project_root / dataset_path),
            "source_sha256": {
                path.name: file_sha256(path)
                for path in (
                    source_dir / filename
                    for filename in (
                        "run_config.json",
                        "train.jsonl",
                        "validation.jsonl",
                        "test.jsonl",
                    )
                )
                if path.exists()
            },
            "model_path": str(model_path),
            "config_path": str(experiment_path),
            "max_input_tokens": experiment.max_input_tokens,
        }
    )
    (project_root / metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (project_root / experiment_path).write_text(
        yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return metadata
