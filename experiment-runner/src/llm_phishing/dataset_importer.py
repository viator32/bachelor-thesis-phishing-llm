from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .email_parser import parse_eml
from .schemas import GroundTruth, RawEmail


# Hash a file in chunks.
def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Accept JSON objects/lists or parse CSV JSON strings.
def _parse_json_field(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


# Normalize booleans and common truthy strings.
def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


# Map a dataset row to email fields, labels and extra metadata.
def row_to_email(row: dict[str, Any], row_number: int) -> RawEmail:
    instance_id = str(row.get("instance_id") or f"row-{row_number:08d}")
    ground_truth = None
    if any(row.get(key) not in (None, "", "unknown") for key in (
        "is_phishing", "has_prompt_injection", "generation_origin"
    )):
        ground_truth = GroundTruth(
            is_phishing=(
                _to_bool(row["is_phishing"])
                if row.get("is_phishing") not in (None, "", "unknown") else None
            ),
            phishing_type=row.get("phishing_type", "unknown"),
            generation_origin=row.get("generation_origin", "unknown"),
            personalization_level=(
                int(row["personalization_level"])
                if str(row.get("personalization_level", "")).isdigit()
                else "unknown"
            ),
            has_prompt_injection=(
                _to_bool(row["has_prompt_injection"])
                if row.get("has_prompt_injection") not in (None, "", "unknown") else None
            ),
            injection_goals=_parse_json_field(row.get("injection_goals"), []),
        )
    known = {
        "instance_id",
        "sender_name",
        "sender_address",
        "reply_to",
        "subject",
        "text_body",
        "html_body",
        "date",
        "language",
        "source_dataset",
        "split",
        "base_id",
        "campaign_id",
        "template_id",
        "profile_id",
        "generator_model",
        "is_phishing",
        "phishing_type",
        "generation_origin",
        "personalization_level",
        "has_prompt_injection",
        "injection_goals",
    }
    return RawEmail(
        instance_id=instance_id,
        sender_name=str(row.get("sender_name") or ""),
        sender_address=str(row.get("sender_address") or row.get("from") or ""),
        reply_to=str(row.get("reply_to") or ""),
        subject=str(row.get("subject") or ""),
        text_body=str(row.get("text_body") or row.get("body") or row.get("text") or ""),
        html_body=str(row.get("html_body") or ""),
        date=str(row.get("date") or ""),
        language=str(row.get("language") or "en"),
        source_dataset=str(row.get("source_dataset") or "unknown"),
        split=str(row.get("split") or "unspecified"),
        base_id=row.get("base_id") or None,
        campaign_id=row.get("campaign_id") or None,
        template_id=row.get("template_id") or None,
        profile_id=row.get("profile_id") or None,
        generator_model=row.get("generator_model") or None,
        ground_truth=ground_truth,
        # Any column not in `known` survives as metadata rather than being dropped, e.g. a
        # "control_group" flag that metrics.control_group_fpr later looks for.
        metadata={key: value for key, value in row.items() if key not in known},
    )


# Parse non-empty JSONL lines as email rows.
def load_jsonl(path: str | Path) -> list[RawEmail]:
    records: list[RawEmail] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if line.strip():
                records.append(row_to_email(json.loads(line), index))
    return records


# Parse a CSV file with optional UTF-8 BOM.
def load_csv(path: str | Path) -> list[RawEmail]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [row_to_email(row, i) for i, row in enumerate(csv.DictReader(handle), start=1)]


# Read EML files recursively in sorted order.
def load_eml_directory(path: str | Path) -> list[RawEmail]:
    return [parse_eml(item) for item in sorted(Path(path).glob("**/*.eml"))]


# Load CSV, JSONL or an EML directory with reproducibility hashes.
def load_dataset(path: str | Path) -> tuple[list[RawEmail], dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        records = load_eml_directory(source)
        files = list(source.glob("**/*.eml"))
        manifest = {
            "path": str(source),
            "format": "eml_directory",
            "files": len(files),
            "hashes": {str(p.relative_to(source)): file_sha256(p) for p in files},
        }
    elif source.suffix.lower() == ".jsonl":
        records = load_jsonl(source)
        manifest = {"path": str(source), "format": "jsonl", "sha256": file_sha256(source)}
    elif source.suffix.lower() == ".csv":
        records = load_csv(source)
        manifest = {"path": str(source), "format": "csv", "sha256": file_sha256(source)}
    else:
        raise ValueError("Dataset must be a CSV, JSONL, or a directory containing .eml files")
    seen: set[str] = set()
    for record in records:
        if not record.instance_id.strip() or record.instance_id in seen:
            raise ValueError(
                f"Dataset contains an empty or duplicate instance_id: {record.instance_id!r}"
            )
        seen.add(record.instance_id)
    manifest["record_count"] = len(records)
    return records, manifest


# Select a split, keeping all records when split is None.
def filter_records(records: Iterable[RawEmail], split: str | None = None) -> list[RawEmail]:
    return [record for record in records if split is None or record.split == split]
