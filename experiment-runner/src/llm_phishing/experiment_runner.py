from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from .baselines import TfidfLogRegBaseline, rules_predict
from .comparison import component_result
from .dataset_importer import file_sha256, filter_records, load_dataset
from .feature_extractor import extract_features
from .metrics import compute_metrics
from .personalization import load_personalization_annotations
from .preprocessor import preprocess_email
from .reporting import write_report
from .run_identity import new_run_id, run_identity
from .sampling import label_counts, stratified_sample
from .schemas import (
    ComponentResult,
    ExperimentConfig,
    ExperimentRecord,
    ProcessedEmail,
    RawEmail,
)
from .storage import ResultStore, validate_run_id


# Read YAML and validate both config and overrides.
def load_config(path: str | Path, **overrides: Any) -> ExperimentConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(data)
    return ExperimentConfig.model_validate(config.model_dump() | overrides)


# Hash the canonical serialized experiment config.
def config_hash(config: ExperimentConfig) -> str:
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Read the current Git revision.
def code_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=2
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unversioned"


# Collect Python, platform and installed package versions.
def environment_versions() -> dict[str, str]:
    packages = (
        "llm-phishing-prototype",
        "pydantic",
        "scikit-learn",
        "httpx",
    )
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


class ExperimentRunner:
    """Execute the shared classification pipeline and persist reproducible batch results."""

    # Configure the baseline pipeline with an optional result store.
    def __init__(
        self,
        config: ExperimentConfig,
        project_root: str | Path = ".",
        store: ResultStore | None = None,
    ) -> None:
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.store = store or ResultStore(
            os.getenv("RESULTS_DIR", str(self.project_root / "results")),
            os.getenv("DATABASE_PATH") or None,
        )
        self.tfidf: TfidfLogRegBaseline | None = None

    # Load the configured TF-IDF baseline once and reuse it.
    def _get_tfidf(self) -> TfidfLogRegBaseline:
        if self.tfidf is None:
            model_path = self.project_root / self.config.tfidf_model_path
            self.tfidf = TfidfLogRegBaseline.load(model_path)
        return self.tfidf

    # Run the configured baseline without an LLM prompt.
    def classify_processed(
        self, email: ProcessedEmail
    ) -> tuple[dict[str, ComponentResult], ComponentResult, str]:
        started = time.perf_counter()
        components: dict[str, ComponentResult] = {}
        prompt_id = "not_applicable"

        baseline_name = self.config.baseline_component
        if baseline_name == "rules":
            components[baseline_name] = rules_predict(email)
        elif baseline_name == "tfidf":
            components[baseline_name] = self._get_tfidf().predict(email)

        final_result = components[baseline_name]
        final_result.measurements.total_duration_ms = (time.perf_counter() - started) * 1000
        for component in components.values():
            component.comparison = component_result(component, email.instance_id)
        final_result.comparison = component_result(final_result, email.instance_id)
        return components, final_result, prompt_id

    def _apply_personalization_annotations(
        self, raw_records: list[RawEmail], manifest: dict[str, Any]
    ) -> None:
        """Overlay reviewed annotations and record their source for evaluation."""
        if not self.config.personalization_annotations:
            return
        annotation_path = Path(self.config.personalization_annotations).expanduser()
        if not annotation_path.is_absolute():
            annotation_path = self.project_root / annotation_path
        annotations = load_personalization_annotations(
            annotation_path, {record.instance_id for record in raw_records}
        )
        manifest["personalization_annotations_sha256"] = file_sha256(annotation_path)
        manifest["personalization_annotations_path"] = str(annotation_path)
        for raw in raw_records:
            annotation = annotations.get(raw.instance_id)
            if not annotation or annotation["personalization_level"] is None:
                continue
            reviewed_level = annotation["personalization_level"]
            if raw.ground_truth:
                current_level = raw.ground_truth.personalization_level
                # A reviewed annotation may fill a missing label, but cannot silently replace one.
                if current_level not in ("unknown", reviewed_level):
                    raise ValueError(f"Conflicting personalization annotation: {raw.instance_id}")
                raw.ground_truth.personalization_level = reviewed_level
            raw.metadata.update(annotation)

    # Evaluate selected emails and persist results.
    def run(
        self,
        run_id: str | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[str, list[ExperimentRecord], dict[str, Any], dict[str, str]]:
        identity = run_identity(self.config.config_id, self.config.task, self.config.dataset_path,
                                self.config.split)
        run_id = validate_run_id(run_id if run_id is not None else new_run_id(self.config.config_id, identity))
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.is_absolute():
            dataset_path = self.project_root / dataset_path
        raw_records, manifest = load_dataset(dataset_path)
        if self.config.baseline_component == "tfidf":
            # A classifier change also changes the experiment, even with the same YAML path.
            manifest["tfidf_model_sha256"] = file_sha256(
                self.project_root / self.config.tfidf_model_path
            )
        self._apply_personalization_annotations(raw_records, manifest)
        selected = filter_records(raw_records, self.config.split)
        if self.config.languages:
            selected = [record for record in selected if record.language in self.config.languages]
        label = "is_phishing" if self.config.task == "phishing" else "has_prompt_injection"
        selected = [record for record in selected if record.ground_truth is not None
                    and getattr(record.ground_truth, label) is not None]
        if not selected:
            raise ValueError(f"No {self.config.task} reference labels in the selected dataset/split/languages.")
        sampling_rows = []
        for email in selected:
            truth = email.ground_truth
            origin = truth.generation_origin.value if truth else "unknown"
            target = {
                "is_phishing": truth.is_phishing if truth else None,
                "generation_origin": {"ai": "llm", "human": "human"}.get(origin),
                "contains_prompt_injection": truth.has_prompt_injection if truth else None,
            }
            sampling_rows.append({
                "id": email.instance_id, "email": email,
                "target": {key: value for key, value in target.items() if value is not None},
                "stratum": "|".join(str(value) for value in (
                    email.language, origin, email.generator_model,
                    truth.has_prompt_injection if truth else None,
                )),
            })
        sampled = stratified_sample(sampling_rows, self.config.limit, self.config.seed)
        selected = [row["email"] for row in sampled]
        manifest["selection"] = {
            "method": "label_and_metadata_stratified_rows", "seed": self.config.seed,
            "requested_maximum": self.config.limit,
            "available_labels": label_counts(sampling_rows), "selected_labels": label_counts(sampled),
            "selected_ids_sha256": hashlib.sha256(json.dumps(
                [row["id"] for row in sampled], ensure_ascii=False,
            ).encode()).hexdigest(),
        }
        total = len(selected)
        manifest["selected_count"] = total
        manifest["selected_split"] = self.config.split
        manifest["selected_languages"] = self.config.languages
        existing = self.store.load_run(run_id)
        if existing and not self.config.resume:
            raise ValueError("run_id already exists and resume is disabled; use a new run_id")
        if any(
            record.reproducibility.get("dataset_manifest", {}).get("personalization_annotations_sha256")
            != manifest.get("personalization_annotations_sha256")
            for record in existing
        ):
            raise ValueError("Personalization annotations changed; use a new run_id")
        if any(
            record.reproducibility.get("config_hash") != config_hash(self.config)
            or record.reproducibility.get("dataset_manifest") != manifest
            for record in existing
        ):
            raise ValueError(
                "Resume configuration or dataset changed; use a new run_id"
            )
        runtime_metadata: dict[str, Any] = {"backend": "not_applicable"}
        # These values describe the run and do not need recomputing for every email.
        reproducibility = {
            "run_identity": identity,
            "config_hash": config_hash(self.config),
            "config": self.config.model_dump(mode="json"),
            "dataset_manifest": manifest,
            "code_version": code_version(),
            "seed": self.config.seed,
            "runtime": runtime_metadata,
            "environment": environment_versions(),
        }
        if progress:
            progress(len(existing), total, "Starting evaluation")
        for index, raw in enumerate(selected, start=1):
            if should_stop and should_stop():
                break
            if self.config.resume and self.store.exists(run_id, raw.instance_id):
                if progress:
                    progress(index, total, f"skipped:{raw.instance_id}")
                continue
            processed = preprocess_email(raw, self.config.max_input_tokens)
            extract_features(processed)
            components, final, prompt_id = self.classify_processed(processed)
            record = ExperimentRecord(
                run_id=run_id,
                instance_id=raw.instance_id,
                config_id=self.config.config_id,
                # Baselines have no model; repurpose model_id to hold the method name so
                # every record still has a meaningful value to group/filter results by.
                model_id=self.config.method,
                prompt_id=prompt_id,
                source_dataset=raw.source_dataset,
                split=raw.split,
                base_id=raw.base_id,
                ground_truth=raw.ground_truth,
                raw_email=raw,
                processed_email=processed,
                components=components,
                final_result=final,
                reproducibility=reproducibility,
            )
            self.store.append(record)
            if progress:
                progress(index, total, raw.instance_id)

        records = self.store.load_run(run_id)
        metrics = compute_metrics(records)
        metrics["run_identity"] = identity
        metrics["run_status"] = (
            "stopped" if should_stop and should_stop() and len(records) < total else "completed"
        )
        metrics["selected_rows"] = total
        report_paths = write_report(records, metrics, self.store.results_dir / "reports" / run_id)
        return run_id, records, metrics, report_paths

# Preprocess one dataset split, train and save the baseline.
def train_tfidf(
    dataset_path: str | Path, output_path: str | Path, split: str = "train",
    seed: int = 42, max_input_tokens: int = 4096,
) -> int:
    records, _ = load_dataset(dataset_path)
    selected = filter_records(records, split=split)
    processed = [preprocess_email(record, max_input_tokens) for record in selected]
    for email in processed:
        extract_features(email)
    model = TfidfLogRegBaseline(seed=seed).train(processed)
    model.save(output_path)
    return len(processed)
