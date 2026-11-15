"""One comparison contract for full detectors, task adapters and deterministic baselines."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from .run_identity import record_identity
from .schemas import (
    ComponentResult,
    ExperimentRecord,
    GroundTruth,
    LLMPrediction,
    PartialPrediction,
    ProcessedEmail,
    RawEmail,
    TechnicalMeasurements,
    ValidationReport,
)

VERSION = "comparison_v1"
FIELDS = tuple(LLMPrediction.model_fields)
TASK_FIELDS = {
    "phishing_origin": ["is_phishing", "generation_origin"],
    "injection": ["prompt_injection_detected"],
}
TASK_KEYS = {
    "phishing": "is_phishing", "generation_origin": "generation_origin",
    "prompt_injection": "prompt_injection_detected",
}


def summary(values: list[float | int | None]) -> dict:
    clean = [v for v in values if v is not None]
    return {"n": len(clean), "mean": float(np.mean(clean)) if clean else None,
            "median": float(np.median(clean)) if clean else None,
            "p95": float(np.percentile(clean, 95)) if clean else None}


def prediction_result(
    instance_id: str, method: str, task: str, prediction: dict | None,
    supported: list[str], status: str, validation: dict, measurements: dict,
    error: str | None = None,
) -> dict:
    normalized = dict.fromkeys(FIELDS)
    if prediction is not None:
        for key in supported:
            if key in normalized:
                normalized[key] = prediction.get(key)
        if normalized["generation_origin"] == "llm":
            normalized["generation_origin"] = "ai"
    return {
        "schema_version": VERSION, "instance_id": instance_id, "method": method,
        "task": task, "status": status, "prediction": normalized,
        "supported_fields": supported, "validation": validation,
        "measurements": TechnicalMeasurements.model_validate(measurements).model_dump(),
        "error": error,
    }


def component_result(component, instance_id: str) -> dict:
    unsupported = set(component.details.get("unsupported_tasks", []))
    if "generation_origin" in unsupported:
        unsupported.add("generation_scores")
    if "prompt_injection_detected" in unsupported:
        unsupported.update(("injection_goals", "prompt_injection_risk_score"))
    if component.method == "tfidf_logreg":
        unsupported.add("evidence")
    report = component.validation_repaired or component.validation_initial
    initial = component.validation_initial
    programmatic = component.method in {"rules_only", "tfidf_logreg"}
    valid = report.fully_valid if report else programmatic and component.prediction is not None
    task = component.details.get("native_task", "full_email")
    return prediction_result(
        instance_id, component.method, task,
        component.prediction.model_dump(mode="json") if valid and component.prediction else None,
        TASK_FIELDS[task] if task in TASK_FIELDS else [field for field in FIELDS if field not in unsupported],
        component.status if valid else ("timeout" if component.status == "timeout" else "error"),
        {"source": "programmatic" if programmatic else "model_response",
         "json_syntax_valid": report.json_syntax_valid if report else bool(valid),
         "schema_valid": report.schema_valid if report else bool(valid),
         "semantic_consistency_valid": report.semantic_consistency_valid
         if report and task == "full_email" else None,
         "initial_json_syntax_valid": initial.json_syntax_valid if initial else bool(valid),
         "initial_schema_valid": initial.schema_valid if initial else bool(valid),
         "repair_attempted": component.validation_repaired is not None,
         "fully_valid": bool(valid)},
        component.measurements.model_dump(), component.error,
    )


def task_result(record: dict, task: str) -> dict:
    valid = record["schema_valid"]
    prediction = dict(record.get("prediction") or {}) if valid else None
    if prediction is not None and task == "injection":
        prediction["prompt_injection_detected"] = prediction.pop("contains_prompt_injection")
    usage = record.get("usage") or {}
    def milliseconds(key):
        return usage[key] * 1000 if usage.get(key) is not None else None
    return prediction_result(
        record["id"], "llm_base" if record.get("model_variant") == "base" else "llm_task_adapter",
        task, prediction, TASK_FIELDS[task],
        "success" if valid else ("timeout" if record["status"] == "timeout" else "error"),
        {"source": "model_response", "json_syntax_valid": record["json_syntax_valid"],
         "schema_valid": valid, "semantic_consistency_valid": None,
         "initial_json_syntax_valid": record["json_syntax_valid"],
         "initial_schema_valid": valid, "repair_attempted": False, "fully_valid": valid},
        {"total_duration_ms": record.get("latency_ms"),
         "prompt_tokens": usage.get("prompt_tokens"),
         "output_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens"),
         "tokens_per_second": usage.get("generation_tokens_per_second"),
         "prompt_tokens_per_second": usage.get("prompt_tokens_per_second"),
         "inference_duration_ms": milliseconds("total_time"),
         "time_to_first_token_ms": milliseconds("time_to_first_token"),
         "prompt_duration_ms": milliseconds("prompt_eval_duration"),
         "generation_duration_ms": milliseconds("generation_duration"),
         "load_duration_ms": milliseconds("model_load_duration")},
        record.get("error") or ("No schema-valid JSON model prediction" if not valid else None),
    )


def normalize_truth(truth: dict) -> dict:
    origin = truth.get("generation_origin")
    return {"is_phishing": truth.get("is_phishing"),
            "generation_origin": "ai" if origin == "llm" else None if origin == "unknown" else origin,
            "prompt_injection_detected": truth.get("has_prompt_injection", truth.get("contains_prompt_injection"))}


def classification(rows: list[tuple[dict, dict]], key: str) -> dict:
    origin = key == "generation_origin"
    labels = ["human", "ai", "unknown"] if origin else [False, True]
    scored_labels = labels[:2]
    labeled = [(result, truth) for result, truth in rows if truth.get(key) in scored_labels
               and truth.get(key) is not None]
    eligible = [(result, truth) for result, truth in labeled if key in result["supported_fields"]]
    valid = [(result, truth) for result, truth in eligible
             if result["validation"]["fully_valid"] and result["prediction"][key] is not None]
    out = {"status": "available" if valid else "not_available",
           "reason": None if valid else "No valid supported predictions with reference labels",
           "labeled_records": len(labeled), "eligible_records": len(eligible),
           "evaluated_records": len(valid), "invalid_outputs": len(eligible) - len(valid),
           "unsupported_records": len(labeled) - len(eligible),
           **dict.fromkeys(("accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc",
                            "false_positive_rate", "false_negative_rate", "tp", "tn", "fp", "fn",
                            "confusion_matrix", "end_to_end_accuracy")),
           "labels": labels, "per_class": {}}
    if eligible:
        out["end_to_end_accuracy"] = sum(r["prediction"][key] == t[key] for r, t in valid) / len(eligible)
    if not valid:
        return out
    truth = [t[key] for _, t in valid]
    pred = [r["prediction"][key] for r, _ in valid]
    matrix = confusion_matrix(truth, pred, labels=labels)
    p, r, f, support = precision_recall_fscore_support(
        truth, pred, labels=labels, zero_division=0,
    )
    out.update(accuracy=float(accuracy_score(truth, pred)),
               confusion_matrix=matrix.tolist(),
               mcc=float(matthews_corrcoef(truth, pred)) if len(set(truth + pred)) > 1 else 0.0,
               per_class={str(label): {"precision": float(p[i]), "recall": float(r[i]),
                                       "f1": float(f[i]), "support": int(support[i])}
                          for i, label in enumerate(labels)})
    if origin:
        out.update(precision=float(np.mean(p[:2])), recall=float(np.mean(r[:2])),
                   f1=float(np.mean(f[:2])))
        if all(label in truth for label in scored_labels):
            out["balanced_accuracy"] = out["recall"]
    else:
        tn, fp, fn, tp = (int(v) for v in matrix.ravel())
        out.update(tp=tp, tn=tn, fp=fp, fn=fn,
                   precision=float(precision_score(truth, pred, zero_division=0)),
                   recall=float(recall_score(truth, pred, zero_division=0)) if tp + fn else None,
                   f1=float(f1_score(truth, pred, zero_division=0)) if tp + fn and tn + fp else None,
                   false_positive_rate=fp / (tn + fp) if tn + fp else None,
                   false_negative_rate=fn / (tp + fn) if tp + fn else None,
                   balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2
                   if tp + fn and tn + fp else None)
    return out


def comparison_report(rows: list[tuple[dict, dict]], extras: dict | None = None) -> dict:
    results = [r for r, _ in rows]
    measurements = [r["measurements"] for r in results]
    model = [r for r in results if r["validation"]["source"] == "model_response"]
    extras = extras or {}
    n = len(results)
    paired = extras.get("paired_robustness", {})
    return {
        "schema_version": VERSION, "records": n,
        "context": {"run_id": None, "config_id": None, "model_id": None,
                    "prompt_id": None, "source_datasets": [], "splits": []},
        "successful_records": sum(r["status"] == "success" for r in results),
        "failed_records": sum(r["status"] != "success" for r in results),
        "classification": {task: classification(rows, key) for task, key in TASK_KEYS.items()},
        "validity": {
            "model_requests_evaluated": len(model),
            "json_validity_rate": sum(r["validation"]["initial_json_syntax_valid"] for r in model) / len(model) if model else None,
            "schema_validity_rate": sum(r["validation"]["initial_schema_valid"] for r in model) / len(model) if model else None,
            "final_schema_validity_rate": sum(r["validation"]["schema_valid"] for r in model) / len(model) if model else None,
            "repair_attempts": sum(r["validation"]["repair_attempted"] for r in model),
            "repair_successes": sum(r["validation"]["repair_attempted"] and r["validation"]["fully_valid"] for r in model),
            "result_validity_rate": sum(r["validation"]["fully_valid"] for r in results) / n if n else None,
            "note": "Model JSON validity includes failed requests; deterministic results are programmatic JSON, not LLM responses.",
        },
        "performance": {
            "latency_ms": summary([m["total_duration_ms"] for m in measurements]),
            "tokens_per_second": summary([m["tokens_per_second"] for m in measurements]),
            "prompt_tokens": summary([m["prompt_tokens"] for m in measurements]),
            "output_tokens": summary([m["output_tokens"] for m in measurements]),
            "throughput_records_per_min": extras.get("throughput_records_per_min", {}).get("rate"),
            "model_process_rss_mb": summary([m["process_rss_mb_after"] for m in measurements]),
            "server_peak_memory_gib": None,
            "memory_note": "Only owned model-process RSS is measured when available; server peak memory is not measured.",
        },
        "robustness": {
            "prompt_injection_attack_success_rate": paired.get("attack_success_rate_warning_suppression"),
            "eligible_attack_pairs": paired.get("warning_suppression_eligible_pairs", 0),
            "label_flip_rate": paired.get("label_flip_rate") if paired.get("paired_examples") else None,
            "benign_suspicious_wording_fpr": extras.get("control_group_fpr", {}).get("fpr"),
            "note": "ASR requires labeled clean/injected phishing pairs and a defined attack goal; injection recall is not ASR.",
        },
    }


def experiment_comparison(records, extras: dict | None = None) -> dict:
    report = comparison_report([
        (component_result(record.final_result, record.instance_id),
         normalize_truth(record.ground_truth.model_dump(mode="json") if record.ground_truth else {}))
        for record in records
    ], extras)
    if records:
        first = records[0]
        report["context"].update(
            **record_identity(first.model_dump(mode="json")),
            run_id=first.run_id, config_id=first.config_id, model_id=first.model_id,
            prompt_id=first.prompt_id,
            source_datasets=sorted({r.source_dataset for r in records}),
            splits=sorted({r.split for r in records}),
        )
    return report


def task_experiment_record(record: dict, row: dict, config, run_id: str, run_context: dict | None = None) -> ExperimentRecord:
    """Publish task runs through the existing dashboard/storage/reporting path."""
    normalized = task_result(record, config.task)
    truth = normalize_truth(row["target"])
    ground_truth = GroundTruth(
        is_phishing=truth["is_phishing"], phishing_type="unknown",
        generation_origin=truth["generation_origin"] or "unknown",
        has_prompt_injection=truth["prompt_injection_detected"],
        personalization_level=row.get("personalization_level")
        if row.get("personalization_level") is not None else "unknown",
    )
    email = RawEmail(
        instance_id=row["id"], text_body=row["input_text"], base_id=row["group_id"],
        split=row.get("split", "test"), language=row.get("language") or "en",
        source_dataset=row.get("source_dataset") or "internal_" + config.task,
        generator_model=row.get("generator_model"), ground_truth=ground_truth,
        metadata={**{key: value for key, value in row.items() if key not in {"input_text", "target"}},
                  "personalization_level": row.get("personalization_level"),
                  "annotation_note": row.get("annotation_note", "")},
    )
    prediction = PartialPrediction.model_validate(normalized["prediction"]) if record["schema_valid"] else None
    component = ComponentResult(
        method=normalized["method"], prediction=prediction, raw_response=record["raw_output"],
        measurements=TechnicalMeasurements.model_validate(normalized["measurements"]),
        validation_initial=ValidationReport(
            json_syntax_valid=record["json_syntax_valid"], schema_valid=record["schema_valid"],
            score_sum_valid=True, semantic_consistency_valid=record["schema_valid"],
        ),
        status=normalized["status"], error=record.get("error"), comparison=normalized,
        details={"native_task": config.task,
                 "backend_model": record.get("backend_model"),
                 "backend_request_id": record.get("backend_request_id"),
                 "response_metadata": record.get("response_metadata", {}),
                 "finish_reason": record.get("finish_reason"),
                 "request_temperature": config.temperature, "request_seed": config.seed,
                 "request_max_tokens": config.max_new_tokens,
                 "constrained_json_requested": config.constrained_json,
                 "raw_response_characters": len(record["raw_output"]),
                 "unsupported_tasks": [f for f in FIELDS if f not in TASK_FIELDS[config.task]],
                 "model_call_attempted": True, "model_response_received": record["status"] == "response"},
    )
    return ExperimentRecord(
        run_id=run_id, instance_id=email.instance_id, config_id=config.config_id,
        model_id=config.model_id, prompt_id="task_native_v1:" + config.task,
        timestamp=record.get("timestamp") or datetime.now(UTC).isoformat(),
        source_dataset=email.source_dataset, split=email.split, base_id=email.base_id,
        ground_truth=ground_truth, raw_email=email,
        processed_email=ProcessedEmail(
            instance_id=email.instance_id, sender_name="", sender_address="", reply_to="",
            subject="", text=email.text_body, date="", language=email.language,
            source_dataset=email.source_dataset, split=email.split, base_id=email.base_id,
            ground_truth=ground_truth,
        ),
        final_result=component, components={"llm": component},
        reproducibility={**(run_context or {}), "output_contract": VERSION, "native_task": config.task,
                         "model_variant": config.model_variant,
                         "input_policy": "full_saved_input"},
    )
