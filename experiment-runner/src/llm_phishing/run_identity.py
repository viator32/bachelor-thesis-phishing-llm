"""Readable run identity shared by the API, CLI and saved exports."""

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path


def run_identity(config_id: str, task: str, dataset_path: str, split: str) -> dict:
    name = Path(dataset_path).stem
    if "ephishllm_paired_injection" in name.lower():
        dataset, evaluation_task = "E-PhishLLM paired injection v1", "phishing"
    elif "ephishllm" in name.lower():
        dataset, evaluation_task = "E-PhishLLM", "phishing"
    elif "crossmodel" in name.lower():
        dataset = "Cross-Model"
        evaluation_task = "phishing" if task == "phishing" else "generation_origin"
    elif "llmail" in name.lower():
        dataset, evaluation_task = "LLMail-Inject", "injection"
    elif "phishfuzzer" in name.lower():
        dataset, evaluation_task = "PhishFuzzer", task
    elif "contextual" in name.lower():
        dataset, evaluation_task = "Contextual Paired Email Injection", "injection"
    elif name == "manifest" and task in {"phishing_origin", "injection"}:
        dataset = "PhishFuzzer" if task == "phishing_origin" else "Contextual Paired Email Injection"
        evaluation_task = task
    else:
        dataset, evaluation_task = name or "Unknown dataset", task
    task_label = {
        "phishing": "Phishing detection",
        "generation_origin": "Human vs AI",
        "phishing_origin": "Phishing + Human vs AI",
        "injection": "Prompt injection detection",
    }.get(evaluation_task, evaluation_task)
    return {
        "run_name": f"{config_id} — {task_label} — {dataset} ({split})",
        "model_task": task,
        "evaluation_task": evaluation_task,
        "task_label": task_label,
        "dataset": dataset,
        "dataset_path": str(dataset_path),
        "split": split,
    }


def new_run_id(config_id: str, identity: dict) -> str:
    def slug(value: str, limit: int) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")[:limit] or "run"

    parts = [slug(config_id, 48), slug(identity["evaluation_task"], 24),
             slug(identity["dataset"], 36), slug(identity["split"], 16),
             datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), uuid.uuid4().hex[:8]]
    return "_".join(parts)


def record_identity(record: dict) -> dict:
    """Read persisted identity, deriving labels for historical runs when possible."""
    saved = record.get("reproducibility") or {}
    if saved.get("run_identity"):
        return saved["run_identity"]
    config = saved.get("config") or saved.get("protocol", {}).get("config", {})
    task = saved.get("native_task") or config.get("task") or "phishing"
    dataset = config.get("dataset_manifest") or config.get("dataset_path") or record.get("source_dataset", "")
    return run_identity(record["config_id"], task, dataset, record.get("split", "unknown"))
