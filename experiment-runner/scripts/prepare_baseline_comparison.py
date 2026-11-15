"""Prepare the exact saved LLM test cases for rules/TF-IDF, without training or inference.

Run from experiment-runner:
    .venv/bin/python scripts/prepare_baseline_comparison.py
"""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT.parent / "thesis/assets/all-runs-results"
NAMES = {"PhishFuzzer": "phishfuzzer", "E-PhishLLM": "ephishllm", "Cross-Model": "crossmodel",
         "Contextual Paired Email Injection": "contextual", "LLMail-Inject": "llmail"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", default="models/tfidf_Phishing_trained_classifier.joblib",
                        help="Previously fitted project pipeline; no retraining")
    args = parser.parse_args()
    if not (ROOT / args.classifier).is_file():
        raise ValueError("Select an existing classifier with --classifier models/<file>.joblib")
    manifests = sorted(SOURCES.glob("*/task/run_manifest.json"))
    audit, annotation_rows = [], []
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    for dataset, name in NAMES.items():
        source = next((p for p in manifests if json.loads(p.read_text())["run_identity"]["dataset"] == dataset), None)
        if source is None:
            raise ValueError(f"No saved LLM test run for {dataset}")
        manifest = json.loads(source.read_text())
        records_path = source.parent.parent / "records.jsonl"
        rows = []
        for line in records_path.read_text().splitlines():
            record = json.loads(line)
            email = record["raw_email"]
            row = {k: v for k, v in email.items() if k not in {"ground_truth", "metadata", "raw_source"}}
            row.update(email.get("metadata", {}))
            row.update(record["ground_truth"])
            row["language"] = email.get("language") or "en"
            row["split"] = "test"
            rows.append(row)
        if [r["instance_id"] for r in rows] != manifest["protocol"]["selected_ids"]:
            raise ValueError(f"Selected IDs do not agree: {dataset}")
        output = processed / f"comparison_{name}_test.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        audit.append({"dataset": dataset, "path": str(output.relative_to(ROOT)), "rows": len(rows),
                      "sha256": sha(output), "source_run_id": source.parent.parent.name,
                      "source_records_sha256": sha(records_path),
                      "selected_ids_sha256": hashlib.sha256(json.dumps([r["instance_id"] for r in rows]).encode()).hexdigest()})
        if name in {"phishfuzzer", "ephishllm"}:
            annotation_rows += [{"id": r["instance_id"], "personalization_level": "", "annotation_note": "",
                                 "named_recipient": "", "role_or_organization": "", "individual_details": "",
                                 "reviewer": "", "language": r["language"], "input_text": r["text_body"]} for r in rows]
    configs = ROOT / "configs/baselines"
    def config(method, name, task="phishing"):
        return {"config_id": f"{method}-{name}-fixed-test-v1", "method": method, "task": task,
                "model": "not-applicable" if method == "rules_only" else "tfidf-logreg",
                "dataset_path": f"data/processed/comparison_{name}_test.jsonl",
                "tfidf_model_path": args.classifier, "seed": 42, "split": "test",
                "languages": [], "max_input_tokens": None, "limit": None,
                "personalization_annotations": None}
    presets = {"rules.yaml": config("rules_only", "phishfuzzer"),
               "tfidf.yaml": config("tfidf_logreg", "phishfuzzer"),
               "external_ephishllm_rules.yaml": config("rules_only", "ephishllm"),
               "external_ephishllm_tfidf.yaml": config("tfidf_logreg", "ephishllm"),
               "rules_injection.yaml": config("rules_only", "contextual", "injection")}
    for filename, value in presets.items():
        (configs / filename).write_text(yaml.safe_dump(value, sort_keys=False))
    annotations = ROOT / "data/evaluation/personalization"
    annotations.mkdir(parents=True, exist_ok=True)
    template = annotations / "comparison_review.csv"
    # Preserve any human work when preparing datasets again.
    if not template.exists():
        random.Random(42).shuffle(annotation_rows)
        with template.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(annotation_rows[0]))
            writer.writeheader()
            writer.writerows(annotation_rows)
    catalog = {"source": "saved LLM exports; unchanged inputs and reference labels", "datasets": audit,
               "classifier": args.classifier, "classifier_sha256": sha(ROOT / args.classifier),
               "annotation_template": str(template.relative_to(ROOT)),
               "note": "Test only; no training, model calls, annotation inference or changes to historical results."}
    (ROOT / "data/evaluation/baseline_comparison.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"datasets": [(x["dataset"], x["rows"]) for x in audit],
                      "configs": list(presets), "annotation_template": str(template.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
