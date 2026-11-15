"""Prepare frozen external evaluation inputs from the locally stored official artifacts.

Run from experiment-runner with .venv/bin/python scripts/prepare_external_datasets.py.
No training, inference, email delivery or execution of message contents takes place.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_phishing.dataset_importer import file_sha256, load_dataset
from llm_phishing.email_parser import extract_plain_urls
from llm_phishing.task_evaluation import valid_target

SOURCE_ROOT = ROOT / "data" / "sources"
SOURCES = SOURCE_ROOT / "datasets" / "experiments"
OUT = ROOT / "data" / "processed"
AUDIT = ROOT / "data" / "evaluation" / "external"
SOURCE_PATHS: set[Path] = set()


def download_llmail() -> None:
    directory = SOURCES / "LLMail_Inject"
    manifest = json.loads((directory / "download_manifest.json").read_text())
    for entry in manifest["files"]:
        target = directory / entry["path"]
        if target.exists() and file_sha256(target) == entry["sha256"]:
            continue
        partial = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(entry["url"], timeout=60) as src, partial.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            if file_sha256(partial) != entry["sha256"]:
                raise ValueError(f"Download SHA-256 mismatch: {target.name}")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        print("Downloaded", target.name, flush=True)


def read_json(path: Path):
    SOURCE_PATHS.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


def digest(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def row(source: str, identifier: str, text: str, **fields) -> dict:
    return {
        "instance_id": f"{source}-{identifier}", "source_dataset": source,
        "split": "test", "subject": "", "text_body": text, "language": "en",
        "is_phishing": None, "phishing_type": "unknown", "generation_origin": "unknown",
        "personalization_level": "unknown", "has_prompt_injection": None,
        "injection_goals": [], **fields,
    }


def ephishllm() -> list[dict]:
    path = SOURCE_ROOT / "datasets/experiments/E_PhishLLM/ephishLLM.json"
    result = []
    for i, item in enumerate(read_json(path)):
        if item["type"] not in (0, 1):
            raise ValueError(f"Unknown E-PhishLLM type at row {i}")
        result.append(row(
            "ephishllm", str(i), item["Body"], subject=item["Subject"],
            language=item["Language"].lower(), is_phishing=bool(item["type"]),
            phishing_type="unknown" if item["type"] else "none",
            generation_origin="ai", generator_model="gpt-4o-mini",
            source_row=i, label_basis="source type; origin from E-PhishGen publication",
        ))
    return result


def crossmodel() -> list[dict]:
    path = SOURCES / "CrossModel_AI_Phishing_2026/cross-model-phishing.zip"
    SOURCE_PATHS.add(path)
    result = []
    with zipfile.ZipFile(path) as archive:
        for origin, member in (("human", "data/human_corpus_sampled.csv"),
                               ("ai", "data/llm_corpus_sampled.csv")):
            items = csv.DictReader(io.StringIO(archive.read(member).decode("utf-8-sig")))
            for i, item in enumerate(items):
                if item["label"] != "1":
                    raise ValueError(f"Unexpected Cross-Model phishing label: {item['label']}")
                result.append(row(
                    "crossmodel", f"{origin}-{i}", item["body"] or item["text"],
                    subject=item["subject"], is_phishing=True, generation_origin=origin,
                    generator_model=item.get("model") or None, source_member=member,
                    source_row=i, original_source=item["source"],
                    original_category=item.get("category"),
                    label_basis="source corpus membership and label=1; not independently relabeled",
                ))
    return result


def llmail() -> tuple[list[dict], dict]:
    directory = SOURCES / "LLMail_Inject"
    combined = {}
    source_counts = {}
    conflicts = 0
    for phase in (1, 2):
        path = directory / f"labelled_unique_submissions_phase{phase}.json"
        items = read_json(path)
        source_counts[f"phase{phase}"] = len(items)
        for text, annotation in items.items():
            value = annotation["attack_attempt"]
            label = value if type(value) is bool else {
                "True": True, "False": False, "Unclear": None,
            }.get(str(value))
            key = digest(text)
            if key not in combined:
                # Preserve the complete submitted string, including its subject/body delimiters.
                combined[key] = row(
                    "llmail", key, text, base_id=f"llmail-{key}", annotations=[],
                    has_prompt_injection=label, label_basis="published attack_attempt annotation",
                )
            combined[key]["annotations"].append({
                "phase": phase, "attack_attempt": label,
                "source_label": str(value),
                "reason": annotation.get("reason"),
                "judge_category": annotation.get("judge_category"),
            })
    for item in combined.values():
        labels = {ann["attack_attempt"] for ann in item["annotations"]}
        if True in labels and False in labels:
            item["has_prompt_injection"] = None
            conflicts += 1
        else:
            item["has_prompt_injection"] = next((v for v in (True, False) if v in labels), None)
        item["control_group"] = item["has_prompt_injection"] is False
    for i, text in enumerate(read_json(directory / "emails_for_fp_tests.json")):
        key = digest(text)
        if key in combined:
            combined[key]["annotations"].append({"source": "emails_for_fp_tests", "row": i})
            # Preserve an ambiguous/conflicting challenge annotation rather than overwrite it.
            continue
        combined[key] = row(
            "llmail", key, text, has_prompt_injection=False, control_group=True,
            label_basis="published additional false-positive controls", source_row=i,
            annotations=[{"source": "emails_for_fp_tests", "row": i}],
        )
    return list(combined.values()), {
        "source_counts": source_counts, "conflicting_binary_labels": conflicts,
        "annotation_policy": "published API/judge labels; Unclear/conflicts remain null",
        "phishing_labels": "unavailable; injection absence does not establish legitimacy",
        "original_tool_call_ASR": "not reproduced by this classifier adaptation",
    }


def exclusion_index(ephish_rows: list[dict]) -> tuple[dict[str, set[str]], dict]:
    index = defaultdict(set)
    checked = []
    training = SOURCE_ROOT / "datasets/training"
    for path in sorted(training.rglob("*.json")):
        items = read_json(path)
        count = 0
        for item in items:
            texts = []
            if "Body" in item:
                body = item["Body"] or ""
                texts = [body, (item.get("Subject") or "") + "\n" + body]
            elif "pair" in item:
                texts = [v["external_content"] for v in item["pair"].values()
                         if isinstance(v, dict) and v.get("external_content")]
            for text in texts:
                if text.strip():
                    index[digest(text)].add(str(path.relative_to(ROOT.parent)))
                    count += 1
        checked.append({"path": str(path.relative_to(ROOT.parent)), "texts_indexed": count})
    # Preserve development exclusions after removing the diagnostic datasets and builder.
    exclusions = read_json(ROOT / "data/metadata/external_exclusions.json")
    source = ROOT.parent / exclusions["source"]
    if file_sha256(source) != exclusions["source_sha256"]:
        raise ValueError("E-PhishLLM source changed; review the development exclusions")
    for i in exclusions["zero_based_rows"]:
        index[digest(ephish_rows[i]["text_body"])].add("existing diagnostic pilot selection")
    return index, {
        "checked_training_sources": checked,
        "policy": "exclude exact normalized body/full-text matches to all local training-source rows and pilot selections",
        "limits": "not a near-duplicate audit; saved model-specific train splits are absent locally; pretraining overlap unknown",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_task_inputs(name: str, rows: list[dict]) -> dict:
    task = "phishing_origin" if name in {"ephishllm", "crossmodel"} else "injection"
    selected = []
    for item in rows:
        if task == "phishing_origin":
            target = {"is_phishing": item["is_phishing"],
                      "generation_origin": {"ai": "llm", "human": "human"}.get(item["generation_origin"])}
            urls = [u.target_url for u in extract_plain_urls(item["text_body"])]
            # Match the field layout used by the joint-task training notebook.
            text = (f"Subject: {item['subject'].strip()}\nSender: \nBody:\n{item['text_body'].strip()}"
                    f"\nURLs: {json.dumps(urls, ensure_ascii=False)}\nAttachment: ")
        else:
            target = {"contains_prompt_injection": item["has_prompt_injection"]}
            text = item["text_body"]
        if not valid_target(target, task):
            continue
        selected.append({
            "id": item["instance_id"], "group_id": item.get("base_id") or item["instance_id"],
            "stratum": "|".join(str(item.get(k)) for k in (
                "language", "generation_origin", "generator_model", "has_prompt_injection"
            )),
            "input_text": text, "target": target, "language": item["language"],
            "generator_model": item.get("generator_model"),
            "source_dataset": item["source_dataset"],
        })
    path = AUDIT / f"{name}_{task}.jsonl"
    write_jsonl(path, selected)
    split_info = {"path": str(path.relative_to(ROOT)), "rows": len(selected),
                  "sha256": file_sha256(path)}
    manifest = {
        "evaluation_scope": "External source evaluation after exact local overlap exclusions; not a near-duplicate/pretraining audit",
        "selection_policy": "Use this frozen full file without --limit; groups need not be two-row pairs",
        "tasks": {task: {"splits": {"test": split_info}}},
    }
    manifest_path = AUDIT / f"{name}_task_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {**split_info, "manifest": str(manifest_path.relative_to(ROOT)),
            "omitted_missing_labels": len(rows) - len(selected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-llmail", action="store_true",
                        help="Download missing LLMail artifacts at the pinned revision (517 MB)")
    args = parser.parse_args()
    if args.download_llmail:
        download_llmail()
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    ephish = ephishllm()
    exclusions, overlap_policy = exclusion_index(ephish)
    llmail_rows, llmail_notes = llmail()
    datasets = {"ephishllm": ephish, "crossmodel": crossmodel(), "llmail": llmail_rows}
    manifest = {
        "version": 1, "prepared_on": "2026-10-21", "split": "external test only; no fitting",
        "preparation_script_sha256": file_sha256(Path(__file__)),
        "missing_label_policy": "null means unannotated; do not score as false",
        "overlap_audit": overlap_policy, "datasets": {},
    }
    excluded = []
    for name, items in datasets.items():
        kept = []
        duplicates = 0
        seen = {}
        for item in items:
            text_hash = digest(item["subject"] + "\n" + item["text_body"])
            reasons = exclusions.get(digest(item["text_body"]), set()) | exclusions.get(text_hash, set())
            if reasons:
                excluded.append({"dataset": name, "instance_id": item["instance_id"],
                                 "reason": sorted(reasons), "text_sha256": text_hash})
                continue
            if name != "llmail" and text_hash in seen:
                prior = seen[text_hash]
                prior["duplicate_source_ids"].append(item["instance_id"])
                if prior["is_phishing"] != item["is_phishing"]:
                    raise ValueError(f"Conflicting phishing labels in {name}")
                if prior["generation_origin"] != item["generation_origin"]:
                    prior["generation_origin"] = "unknown"
                    prior["generator_model"] = None
                duplicates += 1
                continue
            item["duplicate_source_ids"] = []
            seen[text_hash] = item
            kept.append(item)
        path = OUT / f"external_{name}.jsonl"
        write_jsonl(path, kept)
        # Exercise the actual backend importer and schema, not a parallel schema.
        loaded, _ = load_dataset(path)
        if len(loaded) != len(kept) or any(not r.text_body.strip() for r in loaded):
            raise ValueError(f"Invalid output for {name}")
        paths = {"full": {
            "path": str(path.relative_to(ROOT)), "rows": len(kept),
            "sha256": file_sha256(path),
        }}
        task_inputs = {"full": write_task_inputs(name, kept)}
        manifest["datasets"][name] = {
            **paths, "task_inputs": task_inputs,
            "input_rows": len(items), "duplicate_rows_collapsed": duplicates,
            "languages": dict(Counter(r["language"] for r in kept)),
            "phishing_labels": dict(Counter(str(r["is_phishing"]) for r in kept)),
            "injection_labels": dict(Counter(str(r["has_prompt_injection"]) for r in kept)),
            "origin_labels": dict(Counter(r["generation_origin"] for r in kept)),
            "generators": dict(Counter(str(r.get("generator_model")) for r in kept)),
        }
        print(name, len(kept), "full", flush=True)
    manifest["datasets"]["llmail"]["notes"] = llmail_notes
    manifest["excluded_overlaps"] = excluded
    manifest["sources"] = [
        {"path": str(p.relative_to(ROOT.parent)), "bytes": p.stat().st_size,
         "sha256": file_sha256(p)} for p in sorted(SOURCE_PATHS)
    ]
    download_manifest = SOURCES / "LLMail_Inject/download_manifest.json"
    if download_manifest.exists():
        manifest["llmail_download"] = json.loads(download_manifest.read_text())
    (AUDIT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
