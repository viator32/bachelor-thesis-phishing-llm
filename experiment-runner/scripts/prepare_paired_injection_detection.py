"""Derive the injection-detection view of the paired test (same 990 rows, new target).

Run from experiment-runner after prepare_phishing_injection_pairs.py:
.venv/bin/python scripts/prepare_paired_injection_detection.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = Path("data/evaluation/external")
SOURCE = EXT / "ephishllm_paired_injection.jsonl"
OUTPUT = EXT / "ephishllm_paired_injection_detection.jsonl"
MANIFEST = EXT / "ephishllm_paired_injection_detection_task_manifest.json"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    source = (ROOT / SOURCE).read_bytes()
    rows = [json.loads(line) for line in source.decode("utf-8").splitlines() if line.strip()]
    for row in rows:
        row["target"] = {"contains_prompt_injection": row["condition"] == "injected"}
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    (ROOT / OUTPUT).write_bytes(payload)
    manifest = {
        "dataset_name": "E-PhishLLM paired injection v1 — injection detection view (own derivative)",
        "evaluation_scope": "Same 990 rows as ephishllm_paired_injection.jsonl; target is "
        "contains_prompt_injection (clean=false, injected=true). Clean means no experimentally "
        "added instruction, not an independent guarantee that the carrier is free of "
        "manipulative wording.",
        "selection_policy": "Use this frozen full file without --limit",
        "reference_fields": ["contains_prompt_injection"],
        "derived_from": {"path": str(SOURCE), "sha256": sha(source)},
        "tasks": {"injection": {"splits": {"test": {
            "path": str(OUTPUT), "rows": len(rows), "sha256": sha(payload)}}}},
    }
    (ROOT / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
    print(json.dumps({"dataset": str(OUTPUT), "rows": len(rows), "sha256": sha(payload)}, indent=2))


if __name__ == "__main__":
    main()
