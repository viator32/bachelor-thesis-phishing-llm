"""Convert the frozen paired injection test into a processed file for rules/TF-IDF.

Run from experiment-runner:
    .venv/bin/python scripts/prepare_paired_baseline.py

No training, model calls or changes to the paired test. Clean texts are checked against the
original E-PhishLLM selection. `has_prompt_injection` marks only the experiment-added
instruction (condition), so the existing paired metric can group clean/injected by base_id.
"""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PAIRED = ROOT / "data/evaluation/external/ephishllm_paired_injection.jsonl"
SOURCE = ROOT / "data/processed/comparison_ephishllm_test.jsonl"
OUTPUT = ROOT / "data/processed/comparison_ephishllm_paired_injection_test.jsonl"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", default="models/tfidf_Phishing_trained_classifier.joblib",
                        help="Previously fitted project pipeline; no retraining")
    args = parser.parse_args()
    if not (ROOT / args.classifier).is_file():
        raise ValueError("Select an existing classifier with --classifier models/<file>.joblib")
    source = {r["instance_id"]: r for r in map(json.loads, SOURCE.read_text().splitlines())}
    paired = [json.loads(line) for line in PAIRED.read_text().splitlines() if line.strip()]
    rows = []
    for item in paired:
        carrier = source[item["source_instance_id"]]
        if item["condition"] == "clean" and item["input_text"] != carrier["text_body"]:
            raise ValueError(f"Clean text differs from E-PhishLLM carrier: {item['id']}")
        rows.append({
            "instance_id": item["id"], "text_body": item["input_text"],
            "language": item["language"], "source_dataset": item["source_dataset"],
            "split": "test", "base_id": item["group_id"],
            "generator_model": item["generator_model"],
            "is_phishing": item["target"]["is_phishing"],
            "generation_origin": carrier["generation_origin"],
            "personalization_level": item["personalization_level"],
            "has_prompt_injection": item["condition"] == "injected",
            "group_id": item["group_id"], "condition": item["condition"],
            "injection_template_id": item["injection_template_id"],
            "experiment_added_injection": item["experiment_added_injection"],
            "annotation_note": item["annotation_note"], "input_sha256": item["input_sha256"],
        })
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    configs = ROOT / "configs/baselines"
    for method, filename in (("rules_only", "external_ephishllm_paired_rules.yaml"),
                             ("tfidf_logreg", "external_ephishllm_paired_tfidf.yaml")):
        config = {"config_id": f"{method}-ephishllm-paired-injection-fixed-test-v1",
                  "method": method, "task": "phishing",
                  "model": "not-applicable" if method == "rules_only" else "tfidf-logreg",
                  "dataset_path": str(OUTPUT.relative_to(ROOT)),
                  "tfidf_model_path": args.classifier, "seed": 42, "split": "test",
                  "languages": [], "max_input_tokens": None, "limit": None,
                  "personalization_annotations": None}
        (configs / filename).write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"path": str(OUTPUT.relative_to(ROOT)), "rows": len(rows),
                      "sha256": sha(OUTPUT), "paired_source_sha256": sha(PAIRED)}, indent=2))


if __name__ == "__main__":
    main()
