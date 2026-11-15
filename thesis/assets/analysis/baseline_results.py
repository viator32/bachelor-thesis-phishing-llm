"""Read baseline exports and select one comparable run per method/task/dataset.

No predictions are generated. Raw exports are never modified or deleted.
"""

import hashlib
import gzip
import json
from collections import defaultdict


NAMES = {"rules_only": "Regeln", "tfidf_logreg": "TF-IDF"}
PAIRED = "E-PhishLLM paired injection v1"


def jsonl_path(path):
    """Resolve an original JSONL file or its losslessly compressed copy."""
    return path if path.exists() else path.with_suffix(path.suffix + ".gz")


def open_jsonl(path):
    path = jsonl_path(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_baselines(root, llm_runs):
    """llm_runs may include a reference for the paired injection set (rows, texts, truth only)."""
    references = {run["dataset"]: run for run in llm_runs}
    candidates, audit, sources = [], [], []
    for metrics_path in sorted((root / "thesis/assets/all-runs-results").glob("*/metrics.json")):
        exported = json.loads(metrics_path.read_text())
        if exported.get("native_report"):
            continue
        directory = metrics_path.parent
        context = exported["comparison"]["context"]
        method = context["model_id"]
        if method not in NAMES:
            raise ValueError(f"Unsupported result method: {directory.name}")
        rows, texts, records = [], {}, {}
        config, dataset_manifest = None, None
        clipped = 0
        with open_jsonl(directory / "records.jsonl") as stream:
            for line in stream:
                r = json.loads(line)
                if r["instance_id"] in records:
                    raise ValueError(f"Duplicate instance ID in baseline: {directory.name}")
                records[r["instance_id"]] = r
                config = config or r["reproducibility"]["config"]
                dataset_manifest = dataset_manifest or r["reproducibility"]["dataset_manifest"]
                clipped += sum(t.get("applied", False) for t in r["processed_email"]["transformations"]
                               if t["operation"] == "truncate_middle")
        if len(records) != exported["records"] or config is None:
            raise ValueError(f"Incomplete baseline export: {directory.name}")
        ids = set(records)
        dataset = context["dataset"]
        # Older TF-IDF configurations use the training-file stem as their dataset label.
        if dataset not in references:
            matches = [name for name, ref in references.items() if ids == {r["id"] for r in ref["rows"]}]
            if len(matches) != 1:
                raise ValueError(f"Cannot identify baseline dataset: {directory.name}")
            dataset = matches[0]
        ref = references[dataset]
        reference = {r["id"]: r for r in ref["rows"]}
        task = config.get("task", "phishing")
        field = "is_phishing" if task == "phishing" else "contains_prompt_injection"
        stored_field = "is_phishing" if task == "phishing" else "has_prompt_injection"
        output_field = "is_phishing" if task == "phishing" else "prompt_injection_detected"
        mismatches = 0
        for identifier, r in records.items():
            email, result, truth = r["raw_email"], r["final_result"], r["ground_truth"]
            prediction = result["prediction"].get(output_field)
            valid = result["status"] == "success" and type(prediction) is bool
            origin = {"ai": "llm", "human": "human"}.get(truth.get("generation_origin"))
            metadata = email.get("metadata") or {}
            rows.append(dict(id=identifier, timestamp=r["timestamp"], group_id=r.get("base_id") or identifier,
                             truth={field: truth[stored_field]}, prediction={field: prediction},
                             reference_origin=origin, language=email.get("language"),
                             generator_model=email.get("generator_model"),
                             source_dataset=r.get("source_dataset"), stratum=metadata.get("stratum"),
                             condition=metadata.get("condition"), injection_template_id=metadata.get("injection_template_id"),
                             personalization_level=truth.get("personalization_level") if type(truth.get("personalization_level")) is int else None,
                             schema_valid=valid, status="response" if valid else "error", error=result.get("error"),
                             latency_ms=result["measurements"].get("total_duration_ms"), usage={}))
            texts[identifier] = email["text_body"]
            if identifier in reference:
                # The native LLM files serialize the complete email into text_body.
                mismatches += (email["text_body"] != ref["texts"][identifier]
                               or any(email.get(k) for k in ["subject", "sender_address", "reply_to", "html_body"])
                               or truth.get(stored_field) != reference[identifier]["truth"].get(field)
                               or (reference[identifier]["truth"].get("generation_origin") is not None
                                   and origin != reference[identifier]["truth"]["generation_origin"]))
        missing, extra = set(reference) - ids, ids - set(reference)
        eligible = not missing and not extra and not mismatches and not clipped
        artifact = dataset_manifest.get("tfidf_model_sha256")
        if method == "tfidf_logreg" and not artifact:
            raise ValueError(f"Classifier checksum is missing: {directory.name}")
        processed = sorted((identifier, r["processed_email"]["subject"], r["processed_email"]["sender_address"],
                            r["processed_email"]["reply_to"], r["processed_email"]["text"],
                            r["ground_truth"][stored_field]) for identifier, r in records.items())
        outputs = sorted((identifier, r["final_result"]["prediction"]) for identifier, r in records.items())
        item = dict(run_id=directory.name, scope="baseline_candidate", dataset=dataset,
                    original_dataset=context["dataset"], model=NAMES[method], method=method,
                    variant="baseline", task=task, status="stored", stored_rows=len(rows),
                    selected_rows=dataset_manifest.get("selected_count"),
                    identical_input_and_truth=not (missing or extra or mismatches),
                    missing_reference_ids=len(missing), extra_ids=len(extra), input_mismatches=mismatches,
                    truncated_records=clipped, classifier_sha256=artifact,
                    processed_input_sha256=digest(processed), predictions_sha256=digest(outputs),
                    included_in_visualizations=False, selected_run_id=None)
        if not eligible:
            reasons = []
            if missing or extra:
                reasons.append("different test scope or language filter")
            if mismatches:
                reasons.append("different serialized input")
            if clipped:
                reasons.append("additional input truncation")
            item.update(scope="excluded_baseline", exclusion_reason="; ".join(reasons))
        run = dict(run_id=directory.name, scope="baseline", model=NAMES[method], model_id=method,
                   method=method, variant="baseline", task=task, dataset=dataset, rows=rows, texts=texts,
                   manifest={"protocol": {"config": config}}, report={}, exported=exported,
                   directory=directory, first_timestamp=min(r["timestamp"] for r in rows),
                   classifier_sha256=artifact, eligible=eligible, audit=item)
        candidates.append(run)
        audit.append(item)
        for path in [metrics_path, jsonl_path(directory / "records.jsonl")]:
            sources.append(dict(path=str(path.relative_to(root)), sha256=file_hash(path), bytes=path.stat().st_size))
    groups = defaultdict(list)
    for run in candidates:
        if run["eligible"]:
            groups[(run["method"], run["task"], run["dataset"])].append(run)
    selected = []
    for group in groups.values():
        group.sort(key=lambda r: (r["first_timestamp"], r["run_id"]))
        first = group[0]
        for run in group:
            if (run["classifier_sha256"], run["audit"]["processed_input_sha256"], run["audit"]["predictions_sha256"]) != (
                first["classifier_sha256"], first["audit"]["processed_input_sha256"], first["audit"]["predictions_sha256"]):
                raise ValueError(f"Distinct artifacts/inputs/predictions need explicit separate comparison: {run['run_id']}")
            run["audit"]["selected_run_id"] = first["run_id"]
            if run is first:
                run["audit"].update(scope="baseline", included_in_visualizations=True, exclusion_reason="")
            else:
                run["audit"].update(scope="duplicate_baseline", exclusion_reason="equivalent later repeat or configuration alias")
        selected.append(first)
    expected = {(m, "phishing", d) for m in NAMES for d in ["PhishFuzzer", "E-PhishLLM", "Cross-Model"]}
    expected |= {("rules_only", "injection", d) for d in ["Contextual Paired Email Injection", "LLMail-Inject"]}
    # Paired robustness test: phishing for both baselines, injection detection for the rule baseline.
    expected |= {(m, "phishing", PAIRED) for m in NAMES} | {("rules_only", "injection", PAIRED)}
    if set(groups) != expected:
        raise ValueError(f"Missing or unexpected comparable baseline combinations: {set(groups) ^ expected}")
    return selected, audit, sources
