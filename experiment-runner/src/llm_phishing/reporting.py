from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from .comparison import component_result
from .dataset_importer import file_sha256
from .run_identity import record_identity
from .schemas import ExperimentRecord


# Extract run, model and backend identity from a record.
def _identity_columns(record: ExperimentRecord) -> dict[str, Any]:
    runtime = record.reproducibility.get("runtime") or {}
    details = _details(record)
    return {
        "run_id": record.run_id,
        **record_identity(record.model_dump(mode="json")),
        "instance_id": record.instance_id,
        "config_id": record.config_id,
        "model_id": record.model_id,
        "backend": runtime.get("backend"),
        "backend_model": details.get("backend_model"),
        "backend_request_id": details.get("backend_request_id"),
        "model_installed": (runtime.get("model") or {}).get("installed"),
        "model_call_attempted": details.get("model_call_attempted"),
        "model_response_received": details.get("model_response_received"),
        "method": record.final_result.method,
    }


# Extract dataset, split and pair identity from a record.
def _dataset_columns(record: ExperimentRecord) -> dict[str, Any]:
    return {
        "source_dataset": record.source_dataset,
        "split": record.split,
        "base_id": record.base_id,
        "language": record.raw_email.language,
        "generator_model": record.raw_email.generator_model,
        **{key: record.raw_email.metadata.get(key) for key in (
            "original_source", "attack_category", "position", "variant",
        )},
    }


# Extract prompt identity and sampling settings from a record.
def _request_columns(record: ExperimentRecord) -> dict[str, Any]:
    details = _details(record)
    return {
        "prompt_id": record.prompt_id,
        "prompt_instance_hash": details.get("prompt_instance_hash"),
        "prompt_characters": details.get("prompt_characters"),
        "raw_response_characters": details.get("raw_response_characters"),
        "constrained_json_requested": details.get("constrained_json_requested"),
        "response_format_fallback": details.get("response_format_fallback"),
        "request_temperature": details.get("request_temperature"),
        "request_seed": details.get("request_seed"),
        "request_max_tokens": details.get("request_max_tokens"),
        "finish_reason": details.get("finish_reason") or details.get("done_reason"),
    }


# Place ground truth beside each predicted field.
def _prediction_columns(record: ExperimentRecord) -> dict[str, Any]:
    truth = record.ground_truth
    prediction = component_result(record.final_result, record.instance_id)["prediction"]
    return {
        "true_is_phishing": truth.is_phishing if truth else None,
        "pred_is_phishing": prediction["is_phishing"],
        "true_phishing_type": truth.phishing_type.value if truth else None,
        "pred_phishing_type": prediction["phishing_type"],
        "true_generation_origin": truth.generation_origin.value if truth else None,
        "pred_generation_origin": prediction["generation_origin"],
        "true_personalization_level": record.raw_email.metadata.get(
            "personalization_level", truth.personalization_level if truth else None
        ),
        "personalization_annotation_note": record.raw_email.metadata.get("annotation_note", ""),
        "pred_personalization_level": prediction["personalization_level"],
        "true_prompt_injection": truth.has_prompt_injection if truth else None,
        "pred_prompt_injection": prediction["prompt_injection_detected"],
        "phishing_risk_score": prediction["phishing_risk_score"],
        "prompt_injection_risk_score": (
            prediction["prompt_injection_risk_score"]
        ),
    }


# Extract initial and repaired validation outcomes.
def _validation_columns(record: ExperimentRecord) -> dict[str, Any]:
    llm = record.components.get("llm")
    initial = llm.validation_initial if llm else record.final_result.validation_initial
    repaired = llm.validation_repaired if llm else None
    return {
        "json_syntax_initial_valid": initial.json_syntax_valid if initial else None,
        "schema_initial_valid": initial.schema_valid if initial else None,
        "score_sum_initial_valid": initial.score_sum_valid if initial else None,
        "semantic_initial_valid": initial.semantic_consistency_valid if initial else None,
        "json_initial_valid": initial.fully_valid if initial else None,
        "repair_attempted": repaired is not None,
        "repair_valid": repaired.fully_valid if repaired else None,
        "json_final_valid": (
            repaired.fully_valid
            if repaired
            else initial.fully_valid
            if initial
            else None
        ),
    }


# Extract reported timings and token counts, including repair calls.
def _measurement_columns(record: ExperimentRecord) -> dict[str, Any]:
    llm = record.components.get("llm")
    measurement = llm.measurements if llm else record.final_result.measurements
    repair = _details(record).get("repair_measurements") or {}
    return {
        "latency_ms": measurement.total_duration_ms,
        "inference_duration_ms": measurement.inference_duration_ms,
        "time_to_first_token_ms": measurement.time_to_first_token_ms,
        "prompt_duration_ms": measurement.prompt_duration_ms,
        "generation_duration_ms": measurement.generation_duration_ms,
        "model_load_ms": measurement.load_duration_ms,
        "tokens_per_second": measurement.tokens_per_second,
        "tokens_per_second_source": _details(record).get("tokens_per_second_source"),
        "prompt_tokens_per_second": measurement.prompt_tokens_per_second,
        "prompt_tokens": measurement.prompt_tokens,
        "output_tokens": measurement.output_tokens,
        "total_tokens": measurement.total_tokens,
        "cached_prompt_tokens": measurement.cached_prompt_tokens,
        "repair_latency_ms": repair.get("total_duration_ms"),
        "repair_prompt_tokens": repair.get("prompt_tokens"),
        "repair_output_tokens": repair.get("output_tokens"),
    }


# Prefer LLM component details over final-result details.
def _details(record: ExperimentRecord) -> dict[str, Any]:
    llm = record.components.get("llm")
    return llm.details if llm else record.final_result.details


# Flatten results into the shared dashboard and CSV layout.
def records_to_dataframe(records: list[ExperimentRecord]) -> pd.DataFrame:
    rows = [
        {
            **_identity_columns(record),
            **_request_columns(record),
            **_dataset_columns(record),
            **_prediction_columns(record),
            **_validation_columns(record),
            **_measurement_columns(record),
            "llm_status": record.components["llm"].status if "llm" in record.components else None,
            "status": record.final_result.status,
            "timestamp": record.timestamp,
            "raw_response": record.final_result.raw_response,
            "reproducibility_json": json.dumps(record.reproducibility, ensure_ascii=False),
            "metadata_json": json.dumps(record.raw_email.metadata, ensure_ascii=False),
            "error": (
                record.components["llm"].error
                if "llm" in record.components and record.components["llm"].error
                else record.final_result.error
            ),
        }
        for record in records
    ]
    return pd.DataFrame(rows)


# Write metrics JSON and prediction CSV into the output directory.
def write_report(
    records: list[ExperimentRecord], metrics: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    metrics_path = target / "metrics.json"
    csv_path = target / "predictions.csv"
    comparison_path = target / "comparison.json"
    comparison_path.write_text(
        json.dumps(metrics["comparison"], ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    comparison_predictions_path = target / "comparison_predictions.jsonl"
    with comparison_predictions_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(
                component_result(record.final_result, record.instance_id),
                ensure_ascii=False, allow_nan=False,
            ) + "\n")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    records_to_dataframe(records).to_csv(csv_path, index=False)
    return {"metrics": str(metrics_path), "predictions": str(csv_path),
            "comparison": str(comparison_path),
            "comparison_predictions": str(comparison_predictions_path)}


def metric_rows(value: Any, path: str = "") -> list[dict[str, Any]]:
    """Flatten every metric; keep matrices/lists as JSON cells and missing values empty."""
    if isinstance(value, dict) and value:
        return [row for key, child in value.items()
                for row in metric_rows(child, f"{path}.{key}" if path else str(key))]
    return [{"metric": path, "value": json.dumps(value, ensure_ascii=False)
             if isinstance(value, (dict, list)) else value}]


def write_run_archive(
    records: list[ExperimentRecord], metrics: dict[str, Any], destination: Path,
    task_dir: Path, note: str = "",
) -> None:
    """Export stored evidence without inference or reconstructing unavailable past settings."""
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        add_run_to_archive(records, metrics, archive, task_dir, note)


def add_run_to_archive(
    records: list[ExperimentRecord], metrics: dict[str, Any], archive: ZipFile,
    task_dir: Path, note: str = "", prefix: str = "",
) -> None:
    """Add one run's evidence to an open ZIP, optionally inside its own folder."""
    native_names = ("run_manifest.json", "predictions.jsonl", "report.json",
                    "comparison.json", "comparison_predictions.jsonl")
    is_task_run = any(r.reproducibility.get("native_task") for r in records)
    included = [name for name in native_names if is_task_run and (task_dir / name).is_file()]
    # Full records preserve email input, truth, raw output, errors and reproducibility.
    with archive.open(prefix + "records.jsonl", "w") as stream:
        for record in records:
            stream.write((record.model_dump_json() + "\n").encode("utf-8"))
    archive.writestr(prefix + "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    archive.writestr(prefix + "metrics.csv", pd.DataFrame(metric_rows(metrics)).to_csv(index=False))
    archive.writestr(prefix + "predictions.csv", records_to_dataframe(records).to_csv(index=False))
    archive.writestr(prefix + "notes.txt", note)
    for name in included:
        archive.write(task_dir / name, prefix + "task/" + name)
    # Export the current evaluator as evidence of how recomputed metrics were produced.
    code_paths = sorted(Path(__file__).parent.glob("*.py"))
    for path in code_paths:
        archive.write(path, prefix + "code/llm_phishing/" + path.name)
    project = Path(__file__).resolve().parents[2]
    for name in ("requirements.txt", "pyproject.toml"):
        if (project / name).is_file():
            archive.write(project / name, prefix + "code/" + name)
    archive.writestr(prefix + "export_manifest.json", json.dumps({
        "run_id": records[0].run_id, "records": len(records),
        "export_code_sha256": {path.name: file_sha256(path) for path in code_paths},
        "task_files_included": included,
        "task_files_missing": [name for name in native_names if name not in included]
        if is_task_run else [],
        "scope": "All persisted run details; model weights and original source datasets are not bundled.",
        "metrics_policy": "metrics.json is recomputed from stored records with the current code; task/report.json retains its saved native report.",
    }, indent=2))
