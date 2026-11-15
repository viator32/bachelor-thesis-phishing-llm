"""Recompute saved results and create the thesis appendix; never call a model.

From the repository root, using Python with numpy and matplotlib:
    python thesis/assets/analysis/analyse.py
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

from baseline_results import PAIRED, jsonl_path, load_baselines, open_jsonl

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bachelor-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # repository root
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiment-runner/src"))
from llm_phishing.personalization import load_personalization_annotations, personalization_indicators
MODELS = ["Qwen3-4B", "Qwen3-8B", "Mistral-7B-v0.3", "Phi-4-mini", "Llama-3.1-8B"]
DATASETS = ["PhishFuzzer", "E-PhishLLM", "Cross-Model", "Contextual Paired Email Injection", "LLMail-Inject"]
SHORT = {"Contextual Paired Email Injection": "Contextual Paired"}
FIELDS = {"phishing": "is_phishing", "origin": "generation_origin", "injection": "contains_prompt_injection"}
TITLES = {"phishing": "Phishingerkennung", "origin": "Herkunftserkennung", "injection": "Injection-Erkennung"}
COLORS = {"base": "#527a96", "adapted": "#cf6e30"}
BASELINE_COLORS = {"Regeln": "#505050", "TF-IDF": "#7a5794"}
VARIANTS = {"base": "Basis", "adapted": "Angepasst"}
PARTS = [("phishing", "PhishFuzzer"), ("phishing", "E-PhishLLM"),
         ("origin", "PhishFuzzer"), ("origin", "Cross-Model"),
         ("injection", "Contextual Paired Email Injection"), ("injection", "LLMail-Inject")]
DETAIL_PARTS = [("phishing", "PhishFuzzer"), ("phishing", "E-PhishLLM"), ("phishing", "Cross-Model"),
                ("origin", "PhishFuzzer"), ("origin", "E-PhishLLM"), ("origin", "Cross-Model"),
                ("injection", "Contextual Paired Email Injection"), ("injection", "LLMail-Inject")]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with open_jsonl(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_name(identifier):
    for prefix, name in [("qwen3-4b", MODELS[0]), ("qwen3-8b", MODELS[1]),
                         ("mistral", MODELS[2]), ("phi4", MODELS[3]), ("llama31", MODELS[4])]:
        if identifier.startswith(prefix):
            return name
    return identifier


def write_csv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def ratio(a, b):
    return a / b if b else None


def value(v):
    return None if v is None else int(v == "llm") if isinstance(v, str) else int(v)


def decision(row, field):
    """No usable output means no detection; raw rows remain unchanged."""
    prediction = value(row.get("prediction", {}).get(field)) if row.get("schema_valid") else None
    return 0 if prediction is None else prediction


def evaluate(rows, field, fallback=True):
    labeled = [r for r in rows if r["truth"].get(field) is not None]
    valid = [r for r in labeled if r.get("schema_valid") and r.get("prediction", {}).get(field) is not None]
    cm = np.zeros((2, 3), dtype=int)
    for r in labeled:
        t = value(r["truth"][field])
        p = value(r["prediction"].get(field)) if r.get("schema_valid") else None
        cm[t, (0 if fallback else 2) if p is None else p] += 1
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    n, nv = len(labeled), len(valid)
    positive, negative = int(cm[1].sum()), int(cm[0].sum())
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    both = tp + fn > 0 and tn + fp > 0
    # Undefined without positive predictions; shown as a dash.
    precision = ratio(tp, tp + fp) if tp + fp else None
    f1 = ratio(2 * tp, 2 * tp + fp + fn) if both and nv else None
    f1neg = ratio(2 * tn, 2 * tn + fp + fn) if both and nv else None
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(n=n, valid=nv, invalid=n - nv, positive=positive, negative=negative,
                tp=tp, tn=tn, fp=fp, fn=fn, invalid_positive=sum(value(r["truth"][field]) == 1 for r in labeled if not r.get("schema_valid") or r.get("prediction", {}).get(field) is None), invalid_negative=sum(value(r["truth"][field]) == 0 for r in labeled if not r.get("schema_valid") or r.get("prediction", {}).get(field) is None),
                accuracy=ratio(tp + tn, n if fallback else nv), end_to_end_accuracy=ratio(tp + tn, n),
                precision=precision if both else None, recall=recall, specificity=specificity,
                f1=f1, macro_f1=(f1 + f1neg) / 2 if f1 is not None else None,
                fpr=ratio(fp, tn + fp), fnr=ratio(fn, tp + fn),
                balanced_accuracy=(recall + specificity) / 2 if both and recall is not None and specificity is not None else None,
                mcc=(tp * tn - fp * fn) / denom if denom else (0.0 if both and nv else None),
                end_to_end_positive_recall=ratio(tp, positive),
                end_to_end_negative_recall=ratio(tn, negative), confusion_matrix=cm.tolist())


def dataset_from_manifest(manifest):
    identity = manifest.get("run_identity", {})
    if identity.get("dataset"):
        return identity["dataset"]
    path = manifest["protocol"]["config"].get("dataset_manifest", "")
    for token, dataset in [("ephishllm", "E-PhishLLM"), ("crossmodel", "Cross-Model"), ("llmail", "LLMail-Inject")]:
        if token in path:
            return dataset
    return "Unbekannt"


def load_runs(annotation_path=None):
    runs, audit, input_maps, input_lengths, source_files = [], [], {}, {}, []
    normalized_inputs, trusted_tasks = {}, {}
    dataset_texts = defaultdict(dict)
    input_audit = []
    paired_reference, paired_ids = {}, set()
    for directory in sorted((ROOT / "thesis/assets/all-runs-results").iterdir()):
        if not directory.is_dir():
            continue
        if not (directory / "task/run_manifest.json").exists():
            continue
        manifest = read_json(directory / "task/run_manifest.json")
        if dataset_from_manifest(manifest) == PAIRED and manifest["protocol"]["config"]["task"] == "injection":
            # Injection-detection view of the paired test (same 990 texts); evaluated by gepaarter_injection_test.py.
            config = manifest["protocol"]["config"]
            paired_ids.add(directory.name)
            audit.append(dict(run_id=directory.name, scope="paired_injection_detection", dataset=PAIRED,
                              model=model_name(config["model_id"]), variant=config["model_variant"],
                              status=read_json(directory / "task/report.json").get("status"),
                              stored_rows=len(read_rows(directory / "task/predictions.jsonl")),
                              selected_rows=len(manifest["protocol"]["selected_ids"]), identical_input_and_truth=True,
                              included_in_visualizations=True, selected_run_id=directory.name, exclusion_reason=""))
            continue
        if dataset_from_manifest(manifest) == PAIRED:
            # Separate robustness test, evaluated by gepaarter_injection_test.py; inputs serve as baseline reference.
            config = manifest["protocol"]["config"]
            records = read_rows(directory / "records.jsonl")
            # Injection truth = experiment-added instruction, as in prepare_paired_baseline.py.
            current = {r["instance_id"]: (r["raw_email"]["text_body"], r["ground_truth"]["is_phishing"],
                                          r["raw_email"]["metadata"]["experiment_added_injection"]) for r in records}
            if paired_reference and current != paired_reference:
                raise ValueError(f"Paired input text or truth differs between models: {directory.name}")
            paired_reference = current
            paired_ids.add(directory.name)
            audit.append(dict(run_id=directory.name, scope="paired_injection", dataset=PAIRED,
                              model=model_name(config["model_id"]), variant=config["model_variant"],
                              status=read_json(directory / "task/report.json").get("status"), stored_rows=len(records),
                              selected_rows=len(manifest["protocol"]["selected_ids"]), identical_input_and_truth=True,
                              included_in_visualizations=True, selected_run_id=directory.name, exclusion_reason=""))
            continue
        report = read_json(directory / "task/report.json")
        exported = read_json(directory / "metrics.json")
        rows = read_rows(directory / "task/predictions.jsonl")
        config = manifest["protocol"]["config"]
        dataset = dataset_from_manifest(manifest)
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids) or ids != manifest["protocol"]["selected_ids"]:
            raise ValueError(f"Duplicate or missing IDs: {directory.name}")
        current = {}
        lengths = {}
        with open_jsonl(directory / "records.jsonl") as stream:
            for line in stream:
                r = json.loads(line)
                email = r["raw_email"]
                text = email["text_body"]
                if dataset not in input_maps:
                    dataset_texts[dataset][r["instance_id"]] = text
                    normalized = " ".join(text.lower().split())
                    text_hash = hashlib.sha256(normalized.encode()).hexdigest()
                    if text_hash in normalized_inputs:
                        raise ValueError("Duplicate normalized input in the five selected datasets")
                    normalized_inputs[text_hash] = r["instance_id"]
                    if dataset == DATASETS[3]:
                        task = text.split("<TRUSTED_USER_TASK>", 1)[1].split("</TRUSTED_USER_TASK>", 1)[0]
                        trusted_tasks.setdefault(str(r["base_id"]), []).append(" ".join(task.lower().split()))
                payload = json.dumps([text, r["ground_truth"]], sort_keys=True, ensure_ascii=False)
                current[r["instance_id"]] = hashlib.sha256(payload.encode()).hexdigest()
                lengths[r["instance_id"]] = len(text)
        if set(current) != set(ids):
            raise ValueError(f"Record IDs differ from native predictions: {directory.name}")
        if dataset in input_maps and current != input_maps[dataset]:
            raise ValueError(f"Input text or truth differs between models: {dataset}")
        input_maps[dataset], input_lengths[dataset] = current, lengths
        if len(rows) != exported["records"] or len(rows) != report["evaluated"]:
            raise ValueError(f"Counts differ: {directory.name}")
        run = dict(run_id=directory.name, model=model_name(config["model_id"]), model_id=config["model_id"],
                   variant=config["model_variant"], dataset=dataset, rows=rows, manifest=manifest,
                   report=report, exported=exported, directory=directory, scope="main", texts=dataset_texts[dataset])
        runs.append(run)
        audit.append(dict(run_id=directory.name, scope="main", dataset=dataset, model=run["model"],
                          variant=run["variant"], status=report.get("status"), stored_rows=len(rows),
                          selected_rows=len(ids), identical_input_and_truth=True,
                          included_in_visualizations=True, selected_run_id=directory.name, exclusion_reason=""))
        for name in ["metrics.json", "records.jsonl", "task/run_manifest.json", "task/report.json", "task/predictions.jsonl"]:
            path = jsonl_path(directory / name) if name.endswith(".jsonl") else directory / name
            source_files.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size))
    run_ids = {r["run_id"] for r in runs} | paired_ids
    pilots = []
    for path in sorted((ROOT / "experiment-runner/results/task_adapters").glob("*/run_manifest.json")):
        if path.parent.name in run_ids:
            continue
        manifest = read_json(path)
        config = manifest["protocol"]["config"]
        report_path = path.parent / "report.json"
        report = read_json(report_path) if report_path.exists() else {}
        prediction_path = path.parent / "predictions.jsonl"
        rows = read_rows(prediction_path) if prediction_path.exists() else []
        run = dict(run_id=path.parent.name, model=model_name(config["model_id"]), model_id=config["model_id"],
                   variant=config["model_variant"], dataset=dataset_from_manifest(manifest), rows=rows,
                   manifest=manifest, report=report, directory=path.parent, scope="supplementary")
        pilots.append(run)
        audit.append(dict(run_id=run["run_id"], scope="excluded_historical", dataset=run["dataset"], model=run["model"],
                          variant=run["variant"], status=report.get("status", "manifest_only"), stored_rows=len(rows),
                          selected_rows=len(manifest["protocol"].get("selected_ids", [])), identical_input_and_truth=None,
                          included_in_visualizations=False, selected_run_id=None,
                          exclusion_reason="historical/overlapping result or manifest only; outside the fixed main comparison"))
        for file in [path, report_path, prediction_path]:
            if file.exists():
                source_files.append(dict(path=str(file.relative_to(ROOT)), sha256=sha(file), bytes=file.stat().st_size))
    expected = {(m, v, d) for m in MODELS for v in VARIANTS for d in DATASETS}
    actual = [(r["model"], r["variant"], r["dataset"]) for r in runs]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("The main comparison does not contain exactly the expected 50 combinations.")
    paired = dict(dataset=PAIRED, texts={i: v[0] for i, v in paired_reference.items()},
                  rows=[dict(id=i, truth=dict(is_phishing=v[1], contains_prompt_injection=v[2]))
                        for i, v in paired_reference.items()])
    baselines, baseline_audit, baseline_files = load_baselines(ROOT, runs + [paired])
    paired_baselines = [r for r in baselines if r["dataset"] == PAIRED]
    baselines = [r for r in baselines if r["dataset"] != PAIRED]
    audit.extend(baseline_audit)
    source_files.extend(baseline_files)
    db = ROOT / "experiment-runner/results/experiments.sqlite"
    database_checked = db.exists()
    if database_checked:
        con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        db_counts = dict(con.execute("SELECT run_id, count(*) FROM experiment_records GROUP BY run_id"))
        con.close()
        if any(db_counts.get(r["run_id"]) != len(r["rows"]) for r in runs + baselines + paired_baselines):
            raise ValueError("Database and export run coverage differ.")
    else:
        print("Local SQLite database absent; recomputing from verified exports without a database cross-check.", file=sys.stderr)
    if annotation_path:
        ids = {row["id"] for run in runs for row in run["rows"]}
        annotations = load_personalization_annotations(annotation_path, ids)
        for run in runs + baselines:
            for row in run["rows"]:
                reviewed = annotations.get(row["id"], {}).get("personalization_level")
                if reviewed is not None:
                    if row.get("personalization_level") not in (None, reviewed):
                        raise ValueError(f"Conflicting personalization level: {row['id']}")
                    row["personalization_level"] = reviewed
                    row["annotation_note"] = annotations[row["id"]].get("annotation_note", "")
        annotation_dir = ROOT / "experiment-runner/data/evaluation/personalization"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        for dataset, name in [("PhishFuzzer", "phishfuzzer"), ("E-PhishLLM", "ephishllm"),
                              ("Cross-Model", "crossmodel")]:
            first = next(r for r in runs if r["dataset"] == dataset)
            with (annotation_dir / f"applied_{name}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "personalization_level", "annotation_note"])
                writer.writeheader()
                writer.writerows({key: row.get(key, "") for key in writer.fieldnames}
                                 for row in first["rows"] if row.get("personalization_level") is not None)
        source_files.append(dict(path=str(annotation_path), sha256=sha(annotation_path), bytes=annotation_path.stat().st_size))
    write_csv("laufregister.csv", audit)
    write_csv("ausgeschlossene_laeufe.csv", [row for row in audit if not row["included_in_visualizations"]])
    for dataset in DATASETS:
        first = next(r for r in runs if r["dataset"] == dataset)
        by_origin = defaultdict(list)
        for row in first["rows"]:
            by_origin[row["truth"].get("generation_origin", "unknown")].append(input_lengths[dataset][row["id"]])
        for origin, values in by_origin.items():
            input_audit.append(dict(dataset=dataset, reference_origin=origin, n=len(values), median_input_characters=float(np.median(values))))
    write_csv("datencharakteristik.csv", input_audit)
    (OUT / "eingabepruefung.json").write_text(json.dumps(dict(normalized_unique_inputs=len(normalized_inputs),
        normalization="lowercase and whitespace collapsed; complete saved input", contextual_pairs=len(trusted_tasks),
        contextual_pairs_with_different_trusted_task=sum(v[0] != v[1] for v in trusted_tasks.values()),
        note="Input equality and labels verified across models; not a new training-overlap or near-duplicate audit."), indent=2) + "\n", encoding="utf-8")
    return runs, baselines, paired_baselines, pilots, input_lengths, source_files, database_checked


def base_info(run):
    return {k: run[k] for k in ["scope", "run_id", "model", "model_id", "variant", "dataset"]}


def apply_personalization_proposals(path, runs, baselines):
    """Apply the author-adopted binary grouping; retain original rule provenance."""
    if not path:
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        proposals = list(csv.DictReader(stream))
    by_id = {row["id"]: row for row in proposals}
    canonical = {identifier: (run["dataset"], text) for run in runs
                 for identifier, text in run["texts"].items()}
    if len(by_id) != len(proposals) or set(by_id) != set(canonical):
        raise ValueError("Personalization suggestions must cover all unique selected inputs exactly once")
    for identifier, row in by_id.items():
        dataset, text = canonical[identifier]
        if row["dataset"] != dataset or row["input_sha256"] != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError(f"Suggestion refers to a different saved input: {identifier}")
        if row["annotation_method"] != "assistant_rules_v1" or row["human_verified"].lower() != "false":
            raise ValueError("This input is reserved for explicitly provisional, non-human-verified suggestions")
        if row["suggested_level"] and not row["suggested_level"].isdigit():
            raise ValueError(f"Invalid suggested personalization level: {identifier}")
    for run in runs + baselines:
        for row in run["rows"]:
            proposal = by_id[row["id"]]
            row["personalization_proposed"] = int(int(proposal["suggested_level"]) > 0) if proposal["suggested_level"] else None
    write_csv("personalisierung_binaer.csv", [dict(id=identifier, dataset=row["dataset"],
              personalization_level=int(int(row["suggested_level"]) > 0) if row["suggested_level"] else None,
              original_level=row["suggested_level"], input_sha256=row["input_sha256"],
              source_method=row["annotation_method"], adopted_by="thesis_author", adoption_date="2026-11-01",
              adoption_basis="Author requested adoption in session; no independent second review claimed")
              for identifier, row in by_id.items()])
    coverage, metrics = [], []
    for dataset in DATASETS:
        selected = [r for r in runs + baselines if r["dataset"] == dataset]
        counts = Counter(row.get("personalization_proposed") for row in selected[0]["rows"])
        coverage.append(dict(dataset=dataset, n=len(selected[0]["rows"]),
                             P0=counts[0], P1=counts[1], open=counts[None],
                             adopted_by="thesis_author", adoption_date="2026-11-01", annotation_method="assistant_rules_v1_binary_author_adopted"))
        for run in selected:
            for task, field in FIELDS.items():
                if field not in run["rows"][0]["truth"]:
                    continue
                for level in [0, 1, None]:
                    group = [r for r in run["rows"] if r.get("personalization_proposed") == level]
                    for origin in ["all", "human", "llm"]:
                        subset = group if origin == "all" else [r for r in group
                                  if r["truth"].get("generation_origin", r.get("reference_origin")) == origin]
                        if origin != "all" and not subset:
                            continue
                        metrics.append(dict(**base_info(run), task=task, group=f"P{level}" if level is not None else "open", origin=origin,
                                            annotation_kind="author_adopted_binary_grouping",
                                            **evaluate(subset, field)))
    write_csv("personalisierung_vorschlaege_abdeckung.csv", coverage)
    write_csv("personalisierung_vorschlaege_metriken.csv", metrics)
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{Personalisierungsgruppen sämtlicher ausgewerteter Eingaben. P0: ohne zugeordneten Empfängerbezug; P1: mit Empfängerbezug. Eigene Auswertung.}",
             r"\label{tab:personalisierung-vorschlaege}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Datensatz & N & P0 & P1 & Offen \\", r"\midrule"]
    for item in coverage:
        lines.append(tex(SHORT.get(item["dataset"], item["dataset"])) + " & " +
                     " & ".join(str(item[k]) for k in ["n", "P0", "P1", "open"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "personalisierung_vorschlaege.tex").write_text("\n".join(lines)+"\n")
    return coverage


def aggregate(runs, baselines):
    metrics, subgroups, errors, performance, validated = [], [], [], [], []
    for run in runs + baselines:
        rows = run["rows"]
        if not rows:
            continue
        tasks = [task for task, field in FIELDS.items() if field in rows[0]["truth"]]
        run["stats"] = {task: evaluate(rows, FIELDS[task]) for task in tasks}
        for task, stats in run["stats"].items():
            raw_stats = evaluate(rows, FIELDS[task], fallback=False)
            metrics.append({**base_info(run), "task": task, **stats})
            if run["scope"] == "main":
                native = run["report"]["per_task"][FIELDS[task]]
                if native["valid_outputs"] != stats["valid"]:
                    raise ValueError(f"Native valid count mismatch: {run['run_id']}")
                if stats["valid"] and not math.isclose(native["accuracy_on_schema_valid"], raw_stats["accuracy"], abs_tol=1e-10):
                    raise ValueError(f"Native accuracy mismatch: {run['run_id']}")
            key = {"phishing": "phishing", "origin": "generation_origin", "injection": "prompt_injection"}[task]
            saved = run["exported"]["comparison"]["classification"][key]
            cm = np.asarray(saved["confusion_matrix"])
            if not np.array_equal(cm[:2, :2], np.asarray(raw_stats["confusion_matrix"])[:, :2]):
                raise ValueError(f"Exported confusion mismatch: {run['run_id']}")
            if not math.isclose(saved["accuracy"], raw_stats["accuracy"], abs_tol=1e-10):
                raise ValueError(f"Exported accuracy mismatch: {run['run_id']}")
            validated.append(dict(run_id=run["run_id"], scope=run["scope"], task=task,
                                  native_accuracy_agrees=True if run["scope"] == "main" else None,
                                  exported_accuracy_agrees=True, exported_confusion_matrix_agrees=True))
        valid = [r for r in rows if r.get("schema_valid")]
        technical = [r for r in rows if r.get("status") != "response"]
        syntax_errors = [r for r in rows if r.get("status") == "response" and not r.get("schema_valid")]
        joint_correct = sum(all(r["prediction"].get(k) == v for k, v in r["truth"].items()) for r in valid)
        run["joint_e2e"] = joint_correct / len(rows)
        latencies = np.array([r["latency_ms"] for r in rows if r.get("status") == "response" and r.get("latency_ms") is not None])
        run["latencies"] = latencies
        tokens = [r.get("usage", {}).get("generation_tokens_per_second") for r in rows if r.get("status") == "response"]
        tokens = [x for x in tokens if isinstance(x, (int, float))]
        perf = dict(**base_info(run), n=len(rows), valid=len(valid), technical_errors=len(technical),
                    format_errors=len(syntax_errors),
                    schema_valid_rate=len(valid) / len(rows) if run["scope"] == "main" else None,
                    result_validity_rate=len(valid) / len(rows),
                    joint_end_to_end_accuracy=run["joint_e2e"],
                    responses=len(latencies) if run["scope"] == "main" else None, timed_results=len(latencies),
                    median_ms=float(np.median(latencies)) if len(latencies) else None,
                    p95_ms=float(np.percentile(latencies, 95)) if len(latencies) else None,
                    max_ms=float(np.max(latencies)) if len(latencies) else None,
                    median_generation_tokens_s=float(np.median(tokens)) if tokens else None,
                    constrained_json=run["manifest"]["protocol"]["config"].get("constrained_json") if run["scope"] == "main" else None,
                    allow_repair=run["manifest"]["protocol"]["config"].get("allow_repair") if run["scope"] == "main" else None,
                    measurement_scope="HTTP response wall time" if run["scope"] == "main" else "classification call only; excludes preprocessing and storage")
        run["performance"] = perf
        performance.append(perf)
        for r in rows:
            if r.get("schema_valid"):
                continue
            error = r.get("error") or ""
            category = "format" if r.get("status") == "response" else "HTTP_409" if "409" in error else "HTTP_400" if "400" in error else "technical_other"
            errors.append(dict(**base_info(run), instance_id=r["id"], category=category, language=r.get("language"),
                               truth=r["truth"], finish_reason=r.get("finish_reason"), error=error))
        for task in tasks:
            groups = defaultdict(list)
            for r in rows:
                if r.get("language"):
                    groups[("language", r["language"])].append(r)
                if r.get("generator_model"):
                    groups[("generator", r["generator_model"])].append(r)
                if "generation_origin" in r["truth"]:
                    groups[("true_origin", r["truth"]["generation_origin"])].append(r)
                elif r.get("reference_origin"):
                    groups[("true_origin", r["reference_origin"])].append(r)
                if "is_phishing" in r["truth"]:
                    groups[("true_phishing", str(r["truth"]["is_phishing"]))].append(r)
                if task == "injection" and run["dataset"] == DATASETS[3]:
                    groups[("scenario", r["stratum"])].append(r)
                if r.get("personalization_level") is not None:
                    groups[("reviewed_personalization", f"P{r['personalization_level']}")].append(r)
            for (dimension, group), group_rows in groups.items():
                subgroups.append(dict(**base_info(run), task=task, dimension=dimension, group=group,
                                      **evaluate(group_rows, FIELDS[task])))
    write_csv("metriken.csv", metrics)
    write_csv("teilgruppen.csv", subgroups)
    write_csv("technische_metriken.csv", performance)
    write_csv("fehler.csv", errors)
    write_csv("validierung.csv", validated)
    return metrics, subgroups, performance, errors


def personalization_analysis(runs, baselines):
    """Segment fixed predictions without inferring P0/P1 from detector outputs."""
    indicators, metrics, coverage = [], [], []
    for dataset in DATASETS[:3]:
        selected = [r for r in runs + baselines if r["dataset"] == dataset]
        first = selected[0]
        flags = {identifier: personalization_indicators(text) for identifier, text in first["texts"].items()}
        for row in first["rows"]:
            indicators.append(dict(dataset=dataset, instance_id=row["id"], language=row.get("language"),
                                   grouping_level=row.get("personalization_proposed", row.get("personalization_level")), **flags[row["id"]]))
        levels = Counter(r.get("personalization_proposed", r.get("personalization_level")) for r in first["rows"])
        coverage.append(dict(dataset=dataset, n=len(first["rows"]), P0=levels[0], P1=levels[1] + levels[2],
                             unannotated=levels[None]))
        for run in selected:
            for indicator in ["named_salutation", "recipient_context_phrase"]:
                for present in [False, True]:
                    rows = [r for r in run["rows"] if flags[r["id"]][indicator] == present]
                    metrics.append(dict(**base_info(run), criterion=indicator, group="present" if present else "absent",
                                        annotation_kind="exploratory_text_marker_not_P_level", **evaluate(rows, "is_phishing")))
    write_csv("personalisierungsindikatoren.csv", indicators)
    write_csv("personalisierungsvergleich.csv", metrics)
    write_csv("personalisierungsabdeckung.csv", coverage)
    return metrics, coverage


def paired_comparisons(runs, repetitions):
    rng = np.random.default_rng(42)
    comparisons, paired_behavior, coupled = [], [], []
    for task, dataset in PARTS:
        field = FIELDS[task]
        for model in MODELS:
            a, b = [next(r for r in runs if r["model"] == model and r["dataset"] == dataset and r["variant"] == v) for v in VARIANTS]
            ar, br = ({r["id"]: r for r in run["rows"]} for run in [a, b])
            clusters = defaultdict(list)
            counts = Counter()
            for identifier in ar:
                x, y = ar[identifier], br[identifier]
                if x["truth"] != y["truth"]:
                    raise ValueError("Truth differs between paired variants")
                ca = decision(x, field) == value(x["truth"][field])
                cb = decision(y, field) == value(y["truth"][field])
                counts[(ca, cb)] += 1
                clusters[str(x.get("group_id") or identifier)].append(int(cb) - int(ca))
            sums = np.array([sum(v) for v in clusters.values()])
            sizes = np.array([len(v) for v in clusters.values()])
            draws = rng.integers(0, len(sums), size=(repetitions, len(sums)))
            bootstrap = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
            lo, hi = np.percentile(bootstrap, [2.5, 97.5])
            comparisons.append(dict(model=model, dataset=dataset, task=task, clusters=len(clusters), n=len(ar),
                                    base_run_id=a["run_id"], adapted_run_id=b["run_id"],
                                    base_e2e=a["stats"][task]["end_to_end_accuracy"],
                                    adapted_e2e=b["stats"][task]["end_to_end_accuracy"],
                                    difference=sums.sum() / sizes.sum(), ci95_low=float(lo), ci95_high=float(hi),
                                    corrected=counts[(False, True)], newly_wrong=counts[(True, False)],
                                    both_correct=counts[(True, True)], both_wrong=counts[(False, False)],
                                    bootstrap_repetitions=repetitions))
    for run in runs:
        if run["dataset"] not in [DATASETS[0], DATASETS[3]]:
            continue
        groups = defaultdict(list)
        for r in run["rows"]:
            groups[str(r["group_id"])].append(r)
        field = "is_phishing" if run["dataset"] == DATASETS[0] else "contains_prompt_injection"
        joint_valid, both_correct, discordant, positive_positive, negative_negative, reversed_pair = 0, 0, 0, 0, 0, 0
        # Counts describe pairs of variants, not an attack on the phishing classifier.
        for pair in groups.values():
            if len(pair) != 2:
                raise ValueError("Internal holdout does not have exactly two rows per group")
            if all(r.get("schema_valid") for r in pair):
                joint_valid += 1
                pred = [r["prediction"][field] for r in pair]
                discordant += pred[0] != pred[1]
                positive_positive += all(pred)
                negative_negative += not any(pred)
                correct = [r["prediction"][field] == r["truth"][field] for r in pair]
                both_correct += all(correct)
                reversed_pair += not any(correct)
        paired_behavior.append(dict(**base_info(run), groups=len(groups), both_valid=joint_valid,
                                    both_correct=both_correct, both_correct_all_groups=both_correct / len(groups),
                                    label_disagreement=discordant, predicted_both_positive=positive_positive,
                                    predicted_both_negative=negative_negative, both_wrong=reversed_pair))
    for run in runs:
        if run["dataset"] != DATASETS[0]:
            continue
        counts = Counter()
        for r in run["rows"]:
            if not r.get("schema_valid"):
                counts["invalid"] += 1
                continue
            p = r["prediction"]["is_phishing"] == r["truth"]["is_phishing"]
            o = r["prediction"]["generation_origin"] == r["truth"]["generation_origin"]
            counts[f"phishing_{p}_origin_{o}"] += 1
        coupled.append(dict(**base_info(run), **counts))
    write_csv("adaptervergleich.csv", comparisons)
    write_csv("paarverhalten.csv", paired_behavior)
    write_csv("gemeinsame_fehler.csv", coupled)
    return comparisons


def compare_baselines(runs, baselines, repetitions):
    """Paired LLM-minus-baseline E2E differences on the identical stored test cases."""
    rng = np.random.default_rng(42)
    results = []
    for baseline in baselines:
        task, dataset = baseline["task"], baseline["dataset"]
        field = FIELDS[task]
        reference = {row["id"]: row for row in baseline["rows"]}
        for run in [r for r in runs if r["dataset"] == dataset and task in r["stats"]]:
            clusters = defaultdict(list)
            counts = Counter()
            for row in run["rows"]:
                base = reference[row["id"]]
                ca = decision(base, field) == value(base["truth"][field])
                cb = decision(row, field) == value(row["truth"][field])
                clusters[str(row.get("group_id") or row["id"])].append(int(cb) - int(ca))
                counts[(ca, cb)] += 1
            sums = np.array([sum(v) for v in clusters.values()])
            sizes = np.array([len(v) for v in clusters.values()])
            draws = rng.integers(0, len(sums), size=(repetitions, len(sums)))
            values = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
            lo, hi = np.percentile(values, [2.5, 97.5])
            results.append(dict(task=task, dataset=dataset, baseline=baseline["model"], model=run["model"],
                                variant=run["variant"], n=len(reference), clusters=len(clusters),
                                baseline_run_id=baseline["run_id"], llm_run_id=run["run_id"],
                                baseline_e2e=baseline["stats"][task]["end_to_end_accuracy"],
                                llm_e2e=run["stats"][task]["end_to_end_accuracy"],
                                difference=sums.sum() / sizes.sum(), ci95_low=float(lo), ci95_high=float(hi),
                                corrected=counts[(False, True)], newly_wrong=counts[(True, False)],
                                bootstrap_repetitions=repetitions))
    write_csv("baseline_vergleich.csv", results)
    return results


def pct(v, digits=1):
    return "–" if v is None else f"{100 * v:.{digits}f}"


def tex(text):
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def table_tex(rows, caption, label):
    lines = [r"\begin{table}[htbp]", r"\centering\scriptsize", "\\caption{" + caption + "}", "\\label{" + label + "}",
             r"\begin{tabular}{llrrrrrrrr}", r"\toprule", r"Modell & Var. & Acc. & P & R & F1 & MCC & FPR & FNR & N \\", r"\midrule"]
    for run, stats in rows:
        rates = [pct(stats[k]).replace(".", "{,}") for k in ["accuracy", "precision", "recall", "macro_f1"]]
        mcc = "--" if stats["mcc"] is None else f"{stats['mcc']:.3f}".replace(".", "{,}")
        errors = [pct(stats[k]).replace(".", "{,}") for k in ["fpr", "fnr"]]
        variant = {"base": "B", "adapted": "A", "baseline": "–"}[run["variant"]]
        lines.append(" & ".join([tex(run["model"]), variant, *rates, mcc, *errors, str(stats["n"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_tables(runs, baselines):
    lines = []
    md = ["# Vollständige Kennzahlen der ausgewählten Läufe", "",
          "B = Basis, A = angepasst. Raten in Prozent; MCC in [−1, 1]. F1 ist das ungewichtete Mittel der F1-Werte beider Klassen. P/R beziehen sich auf die positive Klasse. Ohne verwertbare Antwort gilt die negative Fallbackentscheidung. Alle Testfälle gehen ein. Ein Strich bedeutet nicht bestimmbar; keine Mittelung über Datensätze."]
    for task, dataset in DETAIL_PARTS:
        selected = [next(r for r in runs if r["model"] == m and r["variant"] == v and r["dataset"] == dataset) for m in MODELS for v in VARIANTS]
        selected += [r for r in baselines if r["dataset"] == dataset and task in r["stats"]]
        label = f"tab:result-{task}-{DATASETS.index(dataset)}"
        caption = f"{TITLES[task]} auf {SHORT.get(dataset, dataset)}: Raten in Prozent; MCC dimensionslos. F1 mittelt beide Klassen gleich. B: Basis; A: angepasst; –: Baseline. Eigene Auswertung."
        lines.append(table_tex([(r, r["stats"][task]) for r in selected], caption, label))
        md += ["", f"## {TITLES[task]} – {dataset}", "", "| Modell | Var. | N | Accuracy | Precision | Recall | F1 | MCC | FPR | FNR | TN/FP/FN/TP |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in selected:
            stats = r["stats"][task]
            values = [r["model"], VARIANTS.get(r["variant"], "Baseline"), str(stats["n"])]
            values += [pct(stats[k], 2) for k in ["accuracy", "precision", "recall", "macro_f1"]]
            values += ["–" if stats["mcc"] is None else f"{stats['mcc']:.3f}", pct(stats["fpr"], 2), pct(stats["fnr"], 2), "/".join(str(stats[k]) for k in ["tn", "fp", "fn", "tp"])]
            md.append("| " + " | ".join(values) + " |")
    (OUT / "tabellen.tex").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "TABELLEN.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def mcc_table(ax, pair, task):
    ax.axis("off")
    cells = [[VARIANTS.get(r["variant"], r["model"]),
              "–" if r["stats"][task]["mcc"] is None else f"{r['stats'][task]['mcc']:.3f}"] for r in pair]
    table = ax.table(cellText=cells, colLabels=["Variante / Methode", "MCC"], loc="center", cellLoc="center", colWidths=[.4, .2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)


def latency_panels(fig, grid, pair, task):
    """Bottom row of an LLM detail page: latency distribution and MCC/latency table."""
    ax = fig.add_subplot(grid[2, 0])
    for run in pair:
        x = np.sort(run["latencies"] / 1000)
        if len(x):
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=VARIANTS[run["variant"]], color=COLORS[run["variant"]])
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=7))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("HTTP-Latenz empfangener Antworten (s, logarithmisch)", fontsize=8)
    ax.set_ylabel("Kumulierter Anteil")
    ax.legend(fontsize=8, loc="lower right")
    ax = fig.add_subplot(grid[2, 1])
    ax.axis("off")
    cells = [[VARIANTS[r["variant"]], "–" if r["stats"][task]["mcc"] is None else f"{r['stats'][task]['mcc']:.3f}",
              f"{np.median(r['latencies'])/1000:.3f}", f"{np.percentile(r['latencies'], 95)/1000:.3f}"] for r in pair]
    table = ax.table(cellText=cells, colLabels=["Variante", "MCC", "Med. s", "p95 s"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.55)


def save_figure(fig, name, image_index, caption, detail_projections=None):
    # The figure stays open until write_appendix_pdf() has placed it behind the contents pages.
    directory = OUT / "figures"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=135, bbox_inches="tight")
    image_index.append(dict(name=name, caption=caption, pdf_page=None,
                            detail_projections=detail_projections or [], figure=fig))


def header(fig, title, subtitle):
    # Top-aligned title and subtitle occupy separate, fixed header bands on every page.
    fig.text(.055, .975, title, ha="left", va="top", fontsize=16, weight="bold")
    width = max(70, int(fig.get_figwidth() * 11))
    fig.text(.055, .922, textwrap.fill(subtitle, width), ha="left", va="top", fontsize=9,
             linespacing=1.3, color="#444444")
    fig.text(.055, .018, "Eigene Auswertung · LLM- und Baseline-Vergleich", fontsize=8, color="#555555")


def matrix(ax, stats, task, variant):
    cm = np.asarray(stats["confusion_matrix"])[:, :2]
    support = cm.sum(axis=1)
    normalized = np.divide(cm, support[:, None], out=np.zeros_like(cm, dtype=float), where=support[:, None] != 0)
    ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n{pct(normalized[i,j])} %" if support[i] else "0\nkeine Fälle",
                    ha="center", va="center", fontsize=10, color="white" if normalized[i, j] > .6 else "#172332")
    labels = ["menschlich", "LLM"] if task == "origin" else ["negativ", "positiv"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Vorhersage")
    ax.set_ylabel("Referenz")
    ax.set_title(variant, loc="left", weight="bold")


def projections(pair, task):
    return [dict(run_id=r["run_id"], task=task, model=r["model"], variant=r["variant"], dataset=r["dataset"]) for r in pair]


def phishing_cohorts(runs, baselines, dataset):
    return [next(r for r in runs if (r["model"], r["variant"], r["dataset"]) == (m, v, dataset))
            for m in MODELS for v in VARIANTS] + [r for r in baselines if r["dataset"] == dataset and r["task"] == "phishing"]


def baseline_page(pair, task, dataset, index):
    save_figure(baseline_figure(pair, task, f"{dataset} · N={len(pair[0]['rows'])}"),
                f"baseline_{task}_{DATASETS.index(dataset)}", index,
                f"{TITLES[task]}: Baselines auf {dataset}. Je Methode ein ausgewählter Lauf; keine Mehrfachzählung wiederholter Konfigurationen. MCC wird separat ausgewiesen.", projections(pair, task))


def baseline_figure(pair, task, scope):
    fig = plt.figure(figsize=(11.69, 8.27))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.7, 1.15, 1.1], top=.81, bottom=.11, hspace=.68, wspace=.35)
    header(fig, f"{TITLES[task]} · Baselines", f"{scope} · ein kanonischer Lauf je Methode; identische Referenzfälle wie im LLM-Vergleich")
    for i, run in enumerate(pair):
        matrix(fig.add_subplot(grid[0, i]), run["stats"][task], task, run["model"])
    if len(pair) == 1:
        ax = fig.add_subplot(grid[0, 1])
        ax.axis("off")
        ax.text(0, .85, "Regelbasierte Injection-Erkennung\n\nDie gespeicherte TF-IDF-Pipeline wurde\nausschließlich auf Phishing trainiert.\nFür Injection liegt kein TF-IDF-Lauf vor.\n\nProgrammatisch erzeugte Ergebnisse:\nkeine LLM-JSON-Validitätsrate.", va="top", fontsize=11, linespacing=1.6)
    ax = fig.add_subplot(grid[1, :])
    keys = ["accuracy", "precision", "recall", "macro_f1", "fpr", "fnr"]
    for i, run in enumerate(pair):
        vals = [run["stats"][task][k] for k in keys]
        bars = ax.bar(np.arange(len(keys)) + (i - (len(pair) - 1) / 2) * .36,
                      [100 * v if v is not None else np.nan for v in vals], width=.34,
                      color=BASELINE_COLORS[run["model"]], label=run["model"])
        ax.bar_label(bars, labels=[pct(v) for v in vals], fontsize=8, padding=2)
    ax.set_xticks(range(len(keys)), ["Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"])
    ax.set_ylabel("Prozent")
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 50, 100])
    mcc_table(fig.add_subplot(grid[2, :]), pair, task)
    return fig


APPENDIX_SECTIONS = ["Überblick", "Phishingerkennung", "Herkunftserkennung", "Injection-Erkennung",
                     "Zusatzbefunde", "Personalisierung"]


def appendix_section(name):
    if name in {"01_ueberblick", "injection_pairs_ueberblick", "injection_pairs_aenderung", "injection_pairs_erkennung"}:
        return APPENDIX_SECTIONS[0]
    if name.startswith("personalization_"):
        return APPENDIX_SECTIONS[5]
    for key, section in [("phishing", 1), ("origin", 2), ("injection", 3)]:
        if re.match(rf"^(\d\d_{key}_|{key}_\d_paired$|baseline_{key}_)", name):
            return APPENDIX_SECTIONS[section]
    return APPENDIX_SECTIONS[4]


def short_title(caption):
    return textwrap.shorten(caption.split(". ", 1)[0], width=115, placeholder=" …")


CONTENT_LINES = 46


def contents_lines(index):
    lines, section = [], None
    for item in index:
        if appendix_section(item["name"]) != section:
            section = appendix_section(item["name"])
            lines.append((f"{APPENDIX_SECTIONS.index(section) + 1}  {section}", None))
        lines.append((short_title(item["caption"]), item))
    return lines


def title_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "Ergebnisanhang", "Ergänzende Abbildungen zur Bachelorarbeit; die vollständigen Kennzahlentabellen stehen im Anhang der Arbeit.")
    body = """50 LLM-Läufe plus acht einmalig ausgewählte Baseline-Läufe.
49.680 Auswertungsinstanzen auf 4.296 unterschiedlichen Eingaben.
Ergänzung: zehn gepaarte Phishingläufe und zehn gepaarte Läufe der
Injection-Erkennung mit je 9.900 Bewertungen auf 495 Paaren
(clean und mit Instruktion) sowie drei Baseline-Läufe auf denselben
990 Texten.

Aufbau (Seitenzahlen im folgenden Verzeichnis)
1. Überblick: sechs Aufgaben-/Datensatzvergleiche, gepaarter Test
2. Phishing: fünf Modelle plus Regeln/TF-IDF, alle drei Quellen
3. Herkunft: fünf Modelle, jeweils alle drei Quellen
4. Injection: fünf Modelle plus Wortregel, interner und externer Test
5. Zusatzbefunde: Herkunft, Sprachen, Generatoren, Latenz, Muster
6. Personalisierung: Textmarker sowie Gruppen P0 und P1
Sortierung je Abschnitt: Modell → Datensatz; Basis und Anpassung
stehen jeweils nebeneinander.

Hinweise zur Interpretation
Konfusionsmatrizen verwenden die negative Fallbackentscheidung.
Accuracy und F1 berücksichtigen alle Testfälle; F1 mittelt beide Klassen.
Ein Strich bezeichnet nicht bestimmbare Werte.
E-PhishLLM enthält hier 495 Phishing- und 505 legitime KI-Texte.
Cross-Model enthält ausschließlich Phishing, Herkunft: 493/507.
LLMail-Inject enthält 924 positive und nur 76 negative Fälle.

Alle LLM-Läufe fordern eine JSON-Grammatik an; keine Reparatur.
JSON-Gültigkeit belegt keine inhaltliche Korrektheit.
LLM-Latenzen sind HTTP-Wandzeiten empfangener Antworten.
Baseline-Zeiten umfassen nur den Klassifikationsaufruf.
Speicher und Wiederholungsstreuung wurden nicht gemessen.
Injection-Recall ist keine Angriffserfolgsrate.

Rechenweg und Nachweise
analyse.py, metriken.csv, teilgruppen.csv, adaptervergleich.csv,
laufregister.csv, technische_metriken.csv, fehler.csv und manifest.json.
Die Dateien enthalten alle IDs und Quellenhashes für den Abgleich.
"""
    fig.text(.07, .88, body, va="top", fontsize=10, linespacing=1.55)
    fig.text(.93, .018, "1", ha="right", fontsize=8, color="#555555")
    pdf.savefig(fig)
    plt.close(fig)


def write_appendix_pdf(index):
    """Title page, contents with page numbers, then every figure in register order."""
    lines = contents_lines(index)
    chunks = [[]]
    for line in lines:
        if len(chunks[-1]) == CONTENT_LINES:
            # A section heading never ends a page.
            moved = [chunks[-1].pop()] if chunks[-1][-1][1] is None else []
            chunks.append(moved)
        chunks[-1].append(line)
    first = 2 + len(chunks)
    for number, item in enumerate(index):
        item["pdf_page"] = first + number
    # Internal links: (source page, rect in figure fractions, page size in inches, target page), pages 1-based.
    links, contents_page = [], {}
    with PdfPages(OUT / "Anhang_Ergebnisse.pdf", metadata={"Title": "Ergebnisanhang: lokale LLM-Detektion", "Author": "Serhii Radkovskyi"}) as pdf:
        title_page(pdf)
        for page, chunk in enumerate(chunks, start=2):
            fig = plt.figure(figsize=(8.27, 11.69))
            header(fig, "Verzeichnis", "Abbildungsseiten dieses Ergebnisanhangs mit Seitenzahl; Einträge sind anklickbar.")
            y = .875
            for position, (text, item) in enumerate(chunk):
                if item is None:
                    y -= .006
                    fig.text(.07, y, text, fontsize=10, weight="bold", va="top")
                    # A heading jumps to the first figure of its section.
                    target = next((i for _, i in chunk[position + 1:] if i is not None), None)
                    if target is not None:
                        links.append((page, (.07, y - .016, .93, y + .002), (8.27, 11.69), target["pdf_page"]))
                else:
                    fig.text(.09, y, textwrap.shorten(text, width=92, placeholder=" …"), fontsize=8.5, va="top")
                    fig.text(.93, y, str(item["pdf_page"]), fontsize=8.5, va="top", ha="right", color=LINK_COLOR)
                    links.append((page, (.09, y - .0145, .93, y + .002), (8.27, 11.69), item["pdf_page"]))
                    contents_page[item["name"]] = page
                y -= .0175
            fig.text(.93, .018, str(page), ha="right", fontsize=8, color="#555555")
            pdf.savefig(fig)
            plt.close(fig)
        for item in index:
            fig = item.pop("figure")
            fig.text(.955, .018, str(item["pdf_page"]), ha="right", fontsize=8, color="#555555")
            fig.text(.5, .018, "↑ zum Verzeichnis", ha="center", fontsize=8, color=LINK_COLOR)
            links.append((item["pdf_page"], (.44, .012, .56, .034), tuple(fig.get_size_inches()), contents_page[item["name"]]))
            pdf.savefig(fig)
            plt.close(fig)
        add_internal_links(pdf, links)


LINK_COLOR = "#1f5fa8"


def add_internal_links(pdf, links):
    """Add GoTo link annotations; matplotlib writes page annotations only when the file is closed."""
    from matplotlib.backends.backend_pdf import Name
    file = pdf._file
    for source, (x0, y0, x1, y1), (width, height), target in links:
        w, h = 72 * width, 72 * height
        file._annotations[source - 1][1].append({
            "Type": Name("Annot"), "Subtype": Name("Link"), "Border": [0, 0, 0],
            "Rect": [x0 * w, y0 * h, x1 * w, y1 * h],
            "Dest": [file.pageList[target - 1], Name("Fit")],
        })


def plots(runs, baselines, comparisons, personalization, personalization_coverage, proposal_coverage=None, paired_pages=None):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlepad": 10, "pdf.fonttype": 42})
    previous = read_json(OUT / "abbildungsregister.json") if (OUT / "abbildungsregister.json").exists() else []
    index = []
    # All figures stay open until the contents pages are known.
    with plt.rc_context({"figure.max_open_warning": 0}):
        def insert_paired(key):
            for item in (paired_pages or {}).get(key, []):
                save_figure(item["figure"], item["name"], index, item["caption"], item["detail_projections"])

        fig, axes = plt.subplots(3, 2, figsize=(11.69, 10.5))
        fig.subplots_adjust(top=.83, hspace=.64, wspace=.25, bottom=.10)
        header(fig, "Modellvergleich nach Aufgabe und Datensatz", "Balanced Accuracy mit negativer Fallbackentscheidung; horizontale Linien: unterstützte Baselines.")
        for ax, (task, dataset) in zip(axes.flat, PARTS):
            for k, variant in enumerate(VARIANTS):
                vals = [next(r for r in runs if r["model"] == m and r["variant"] == variant and r["dataset"] == dataset)["stats"][task]["balanced_accuracy"] for m in MODELS]
                ax.bar(np.arange(5) + (k - .5) * .36, np.asarray(vals) * 100, width=.34, color=COLORS[variant], label=VARIANTS[variant])
            for baseline in [r for r in baselines if r["dataset"] == dataset and r["task"] == task]:
                ax.axhline(100 * baseline["stats"][task]["balanced_accuracy"], color=BASELINE_COLORS[baseline["model"]],
                           linestyle=":" if baseline["model"] == "Regeln" else "-.", label=baseline["model"], linewidth=1.4)
            ax.set_xticks(range(5), MODELS, rotation=22, ha="right", fontsize=8)
            ax.set_ylim(0, 105)
            ax.set_ylabel("Balanced Accuracy (%)")
            ax.set_title(f"{TITLES[task]} · {SHORT.get(dataset, dataset)}", fontsize=10, loc="left")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), fontsize=9, ncol=4,
                   loc="upper right", bbox_to_anchor=(.94, .895), frameon=False)
        save_figure(fig, "01_ueberblick", index, "Balanced Accuracy nach Aufgabe, Modell und Datensatz; negative Fallbackentscheidung bei fehlenden Antworten.")
        insert_paired("ueberblick")
        page_number = 3
        # Explicit task → model → dataset order for all 50 runs (80 task projections).
        for task in FIELDS:
            datasets = [d for t, d in DETAIL_PARTS if t == task]
            for model in MODELS:
                for dataset in datasets:
                    pair = [next(r for r in runs if (r["model"], r["dataset"], r["variant"]) == (model, dataset, v)) for v in VARIANTS]
                    fig = plt.figure(figsize=(11.69, 8.27))
                    grid = fig.add_gridspec(3, 2, height_ratios=[1.7, 1.15, 1.1], top=.81, bottom=.11, hspace=.68, wspace=.35)
                    header(fig, f"{TITLES[task]} · {model}", f"{dataset} · N = {len(pair[0]['rows']):,} je Variante · positive Klasse: {'LLM' if task == 'origin' else 'Phishing' if task == 'phishing' else 'Injection'}")
                    for i, run in enumerate(pair):
                        matrix(fig.add_subplot(grid[0, i]), run["stats"][task], task, VARIANTS[run["variant"]])
                    ax = fig.add_subplot(grid[1, :])
                    keys = ["accuracy", "precision", "recall", "macro_f1", "fpr", "fnr"]
                    labels = ["Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"]
                    for i, run in enumerate(pair):
                        vals = [run["stats"][task][k] for k in keys]
                        bars = ax.bar(np.arange(len(keys)) + (i - .5) * .36, [100 * x if x is not None else np.nan for x in vals], width=.34, color=COLORS[run["variant"]])
                        ax.bar_label(bars, labels=[pct(x) for x in vals], padding=2, fontsize=8)
                    ax.set_ylim(0, 118)
                    ax.set_yticks([0, 50, 100])
                    ax.set_ylabel("Prozent")
                    ax.set_xticks(range(len(keys)), labels, fontsize=9)
                    latency_panels(fig, grid, pair, task)
                    name = f"{page_number:02d}_{task}_{MODELS.index(model)+1}_{DATASETS.index(dataset)+1}"
                    caption = f"{TITLES[task]}: {model}, {dataset}. Zeilen normieren auf alle Referenzfälle; fehlende Antworten zählen als negative Fallbackentscheidung. Latenzen nur für empfangene Antworten."
                    save_figure(fig, name, index, caption, projections(pair, task))
                    page_number += 1
                if task in ("phishing", "injection"):
                    insert_paired(f"{task}_{MODELS.index(model) + 1}")
            for dataset in datasets:
                pair = [r for r in baselines if r["dataset"] == dataset and r["task"] == task]
                pair.sort(key=lambda r: r["model"])
                if pair:
                    baseline_page(pair, task, dataset, index)
            insert_paired(f"{task}_baselines")
        # Origin-stratified phishing recall: paired rewrites, not personalization labels.
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 6.5))
        fig.subplots_adjust(top=.80, bottom=.22, wspace=.25)
        header(fig, "Phishingerkennung nach tatsächlicher Herkunft", "PhishFuzzer: 113 Phishing- und 99 legitime Texte je Herkunft; Fehlende Antworten zählen als fehlende Erkennung.")
        for ax, origin in zip(axes, ["human", "llm"]):
            for k, variant in enumerate(VARIANTS):
                vals = []
                for m in MODELS:
                    run = next(r for r in runs if (r["model"], r["variant"], r["dataset"]) == (m, variant, "PhishFuzzer"))
                    rows = [r for r in run["rows"] if r["truth"]["generation_origin"] == origin]
                    vals.append(evaluate(rows, "is_phishing")["end_to_end_positive_recall"] * 100)
                bars = ax.bar(np.arange(5) + (k - .5) * .36, vals, width=.34, color=COLORS[variant], label=VARIANTS[variant])
                ax.bar_label(bars, fmt="%.1f", fontsize=8)
            ax.set_xticks(range(5), MODELS, rotation=25, ha="right", fontsize=8)
            ax.set_ylim(0, 113)
            ax.set_ylabel("E2E-Phishing-Recall (%)")
            ax.set_title("Menschliche Seeds" if origin == "human" else "LLM-Varianten")
        axes[0].legend(loc="lower left")
        save_figure(fig, "33_herkunft_und_phishing", index, "Phishing-Recall nach Referenzherkunft in PhishFuzzer; keine Personalisierungsstufen.")
        fig, axes = plt.subplots(1, 3, figsize=(11.69, 6.7))
        fig.subplots_adjust(top=.80, bottom=.22, wspace=.32)
        header(fig, "Sprachbezogene Phishingerkennung auf E-PhishLLM", "Nur Gruppen mit N ≥ 100: Englisch N=690, Italienisch N=161, Deutsch N=139; zehn übrige Texte nur in CSV.")
        for ax, lang in zip(axes, ["en", "it", "de"]):
            for k, variant in enumerate(VARIANTS):
                vals = []
                for m in MODELS:
                    run = next(r for r in runs if (r["model"], r["variant"], r["dataset"]) == (m, variant, "E-PhishLLM"))
                    s = evaluate([r for r in run["rows"] if r.get("language") == lang], "is_phishing")
                    vals.append(100 * s["end_to_end_accuracy"])
                ax.bar(np.arange(5) + (k - .5) * .36, vals, width=.34, color=COLORS[variant], label=VARIANTS[variant])
            ax.set_xticks(range(5), MODELS, rotation=35, ha="right", fontsize=8)
            ax.set_ylim(0, 105)
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(lang)
        axes[0].legend(loc="lower left", fontsize=8)
        save_figure(fig, "34_sprachen", index, "Sprachgruppen der externen Phishingevaluation, deskriptiv; keine Kontrolle von Inhalt oder Profilen.")
        fig, axes = plt.subplots(1, 3, figsize=(11.69, 6.7))
        fig.subplots_adjust(top=.80, bottom=.23, wspace=.32)
        header(fig, "Herkunftserkennung nach externem Generator", "Cross-Model: je 169 LLM-Texte aus drei Generatoren; gezeigt wird LLM-Recall, keine vollständige Zwei-Klassen-Metrik.")
        for ax, generator in zip(axes, ["gpt-4.1", "deepseek3.2", "llama-3.3-70b"]):
            for k, variant in enumerate(VARIANTS):
                vals = []
                for model in MODELS:
                    run = next(r for r in runs if (r["model"], r["variant"], r["dataset"]) == (model, variant, "Cross-Model"))
                    stats = evaluate([r for r in run["rows"] if r.get("generator_model") == generator], "generation_origin")
                    vals.append(100 * stats["recall"])
                ax.bar(np.arange(5) + (k - .5) * .36, vals, width=.34, color=COLORS[variant], label=VARIANTS[variant])
            ax.set_xticks(range(5), MODELS, rotation=35, ha="right", fontsize=8)
            ax.set_ylim(0, 105)
            ax.set_ylabel("LLM-Recall (%)")
            ax.set_title(generator)
        axes[0].legend(loc="lower left", fontsize=8)
        save_figure(fig, "36_generatoren", index, "LLM-Recall nach externem Generator; menschliche Gegenklasse in den vollständigen Cross-Model-Matrizen.")
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 9.0))
        fig.subplots_adjust(top=.81, bottom=.10, hspace=.45, wspace=.25)
        header(fig, "Klassifikationsqualität und beobachtete Latenz", "Kreis: Basis; Quadrat: angepasst; Pfeile verbinden Varianten. Fehlende Antworten zählen als negative Fallbackentscheidung.")
        palette = ["#1b7285", "#94402b", "#6c5990", "#437337", "#9c790e"]
        for ax, (task, dataset) in zip(axes.flat, [PARTS[0], PARTS[1], PARTS[4], PARTS[5]]):
            for i, model in enumerate(MODELS):
                pair = [next(r for r in runs if (r["model"], r["variant"], r["dataset"]) == (model, v, dataset)) for v in VARIANTS]
                xs = [r["performance"]["median_ms"] / 1000 for r in pair]
                ys = [r["stats"][task]["end_to_end_accuracy"] * 100 for r in pair]
                ax.annotate("", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]), arrowprops={"arrowstyle": "->", "color": palette[i], "lw": 1.1})
                ax.scatter(xs[0], ys[0], marker="o", color=palette[i], s=32)
                ax.scatter(xs[1], ys[1], marker="s", color=palette[i], s=32, label=model)
            ax.set_ylim(0, 105)
            ax.set_xlabel("Median der HTTP-Latenz empfangener Antworten (s)", fontsize=8)
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"{TITLES[task]} · {SHORT.get(dataset, dataset)}", loc="left", fontsize=10)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), fontsize=9, ncol=5,
                   loc="upper right", bbox_to_anchor=(.94, .905), frameon=False)
        save_figure(fig, "37_qualitaet_latenz", index, "Qualität und Latenz innerhalb derselben Aufgabe und Testmenge; keine kausale Zuschreibung von Zeitdifferenzen an die Anpassung.")
        insert_paired("teilgruppen")
        short_models = ["Q4", "Q8", "M7", "Phi", "L8"]
        cohort_labels = [f"{m} {v}" for m in short_models for v in ["B", "A"]] + ["Regeln", "TF-IDF"]
        for indicator, title in [("named_salutation", "Namentlich wirkende Anrede"),
                                 ("recipient_context_phrase", "Empfängerbezogene Kontextformulierung")]:
            fig, axes = plt.subplots(2, 2, figsize=(11.69, 9.3))
            fig.subplots_adjust(top=.80, bottom=.12, hspace=.75, wspace=.30)
            header(fig, f"Personalisierungsindikator: {title}",
                   "Explorativer Textmarker, keine vollständige binäre Gruppenzuordnung und kein kausaler Personalisierungseffekt. Fehlende Marker bedeuten nicht fehlende Personalisierung.")
            for row_index, dataset in enumerate(DATASETS[:2]):
                for col, (key, rate) in enumerate([("end_to_end_positive_recall", "Phishing-Recall"), ("fpr", "FPR")]):
                    ax = axes[row_index, col]
                    for k, group in enumerate(["absent", "present"]):
                        cohorts = phishing_cohorts(runs, baselines, dataset)
                        entries = [next(x for x in personalization if (x["dataset"], x["model"], x["variant"], x["criterion"], x["group"]) == (dataset, r["model"], r["variant"], indicator, group)) for r in cohorts]
                        values = [100 * x[key] if x[key] is not None else np.nan for x in entries]
                        denominator = entries[0]["positive" if col == 0 else "negative"]
                        label = ("Marker fehlt" if group == "absent" else "Marker vorhanden") + f" (Referenz-n={denominator})"
                        ax.bar(np.arange(len(cohorts)) + (k - .5) * .36, values, width=.34, color=["#7893a7", "#d17b49"][k], label=label)
                    ax.set_xticks(range(len(cohorts)), cohort_labels, rotation=40, ha="right", fontsize=7)
                    ax.set_ylim(0, 135)
                    ax.set_yticks([0, 25, 50, 75, 100])
                    ax.set_ylabel("Prozent")
                    ax.set_title(f"{dataset} · {rate}", fontsize=10, loc="left")
                    ax.legend(fontsize=7, loc="upper left")
            save_figure(fig, "personalization_" + indicator, index,
                        f"Explorativer Vergleich nach Textmarker: {title}. B: Basis; A: angepasst. Referenz-n zählt alle positiven bzw. negativen Beispiele der Gruppe; die FPR verwendet alle negativen Referenzfälle. Textmarker und binäre Gruppenzuordnung sind getrennte Merkmale.")
        if proposal_coverage:
            fig = plt.figure(figsize=(11.69, 8.27))
            header(fig, "Personalisierung: P0 und P1", "Binäre Einteilung der gespeicherten Eingaben nach Empfängerbezug.")
            ax = fig.add_axes([.06, .38, .88, .38])
            ax.axis("off")
            cells = [[SHORT.get(x["dataset"], x["dataset"]), x["n"], x["P0"], x["P1"], x["open"]] for x in proposal_coverage]
            table = ax.table(cellText=cells, colLabels=["Datensatz", "N", "P0", "P1", "Offen"], colWidths=[.4, .15, .15, .15, .15], loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            fig.text(.08, .29, "P0: ohne zugeordneten Empfängerbezug. P1: mit Empfängerbezug.\n"
                     "P1 umfasst einfache Empfängerbezüge und mehrere Kontextdetails.\n"
                     "Offene Zuordnungen bleiben außerhalb des binären Gruppenvergleichs.\n"
                     "Die Gruppierung verwendet keine Detektorvorhersagen.", fontsize=10, va="top", linespacing=1.5)
            save_figure(fig, "personalization_proposed_coverage", index, "Abdeckung der Personalisierungsgruppen P0 und P1.")
            for dataset_index, dataset in enumerate(DATASETS):
                task = "injection" if dataset in DATASETS[3:] else "phishing"
                cohorts = [r for r in runs + baselines if r["dataset"] == dataset and FIELDS[task] in r["rows"][0]["truth"]]
                cohorts.sort(key=lambda r: (MODELS.index(r["model"]) if r["model"] in MODELS else 5 if r["model"] == "Regeln" else 6,
                                           list(VARIANTS).index(r["variant"]) if r["variant"] in VARIANTS else 0))
                fig, axes = plt.subplots(1, 2, figsize=(11.69, 7.3))
                fig.subplots_adjust(top=.73, bottom=.23, wspace=.28)
                header(fig, f"Personalisierung × {TITLES[task]}: {SHORT.get(dataset, dataset)}",
                       "P0: ohne zugeordneten Empfängerbezug; P1: mit Empfängerbezug. Gruppenunterschiede sind deskriptiv; leere Klassen: N/A.")
                for ax, key, title, denominator in zip(axes, ["end_to_end_positive_recall", "fpr"],
                                                      ["Recall", "Falschpositivrate"], ["positive", "negative"]):
                    for level in range(2):
                        stats = [evaluate([row for row in r["rows"] if row.get("personalization_proposed") == level], FIELDS[task]) for r in cohorts]
                        vals = [100*s[key] if s[key] is not None else np.nan for s in stats]
                        ax.bar(np.arange(len(cohorts)) + (level-.5)*.32, vals, width=.30,
                               label=f"P{level} (Referenz-n={stats[0][denominator]})")
                    labels = [f"{['Q4','Q8','M7','Phi','L8'][MODELS.index(r['model'])]} {'B' if r['variant']=='base' else 'A'}"
                              if r["model"] in MODELS else r["model"] for r in cohorts]
                    ax.set_xticks(range(len(cohorts)), labels, rotation=40, ha="right", fontsize=8)
                    ax.set_ylim(0, 135)
                    ax.set_yticks([0, 25, 50, 75, 100])
                    ax.set_ylabel("Prozent")
                    ax.set_title(title, fontsize=10, loc="left")
                    ax.legend(fontsize=8, loc="upper left")
                    if not any(r["truth"].get(FIELDS[task]) is False for r in cohorts[0]["rows"]) and key == "fpr":
                        ax.text(.5, .45, "N/A: keine negative Referenzklasse", transform=ax.transAxes, ha="center", fontsize=9)
                save_figure(fig, f"personalization_proposed_{dataset_index}", index,
                            f"{TITLES[task]} nach P0 und P1 auf {dataset}. P1 umfasst jeden erkannten Empfängerbezug.")
                if dataset == "E-PhishLLM":
                    insert_paired("personalisierung")
        if paired_pages and not proposal_coverage:
            insert_paired("personalisierung")
        write_appendix_pdf(index)
    detail = [dict(**row, figure=item["name"], pdf_page=item["pdf_page"])
              for item in index for row in item["detail_projections"]]
    keys = [(r["task"], r["model"], r["variant"], r["dataset"]) for r in detail]
    expected_projections, expected_runs = (111, 81) if paired_pages else (88, 58)
    if (len(keys) != expected_projections or len(set(keys)) != expected_projections
            or len({r["run_id"] for r in detail}) != expected_runs):
        raise ValueError("Duplicate or missing task/model/dataset detail visualization")
    write_csv("visualisierte_laeufe.csv", detail)
    (OUT / "abbildungsregister.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    current_names = {item["name"] for item in index}
    for old in previous:
        if old["name"] not in current_names:
            for suffix in [".pdf", ".png"]:
                (OUT / "figures" / (old["name"] + suffix)).unlink(missing_ok=True)
    return index


def appendix_tex(index, paired_tables):
    lines = [r"% Generated by analyse.py; paths are relative to thesis/.",
             r"\section{Vollständige Kennzahlentabellen}",
             r"Der Anhang enthält 50 LLM-Läufe und acht ausgewählte Baseline-Läufe; ergänzend enthält er zehn LLM- und drei Baseline-Läufe des gepaarten Injection-Tests in Abschnitt~\ref{app:gepaarter-injection-test}. Je Methode bzw. Modellvariante, Aufgabe und Datensatz wird genau ein Lauf ausgewertet. Liegen mehrere gleichwertige Baseline-Läufe mit identischen Eingaben, Modellartefakten und Vorhersagen vor, wird der zeitlich erste vollständige Lauf verwendet. Anderssprachige Teilmengen, zusätzliche Kürzung und andere Serialisierung werden nicht mit dem identischen LLM-Testbestand vermischt.",
             r"Die sechs Hauptvergleiche und zwei ergänzende Einklassenprojektionen berichten jede Modellvariante und unterstützte Baseline getrennt (Tabellen~\ref{tab:result-phishing-0} bis~\ref{tab:result-injection-4}). Bei Herkunft ist LLM die positive Klasse; bei Phishing und Injection jeweils das Vorliegen des Merkmals. Auf Cross-Model ist die Phishing-Accuracy lediglich der Recall; auf E-PhishLLM ist die Herkunfts-Accuracy lediglich der LLM-Recall. Ein Strich kennzeichnet nicht bestimmbare Werte. Fehlende Antworten zählen als negative Fallbackentscheidung; Rohfehler bleiben im technischen Protokoll erhalten. Die Baselines erzeugen programmatische Ergebnisse; deren Gültigkeit ist keine LLM-JSON-Validität. Alle Werte beruhen auf eigenen Versuchsdaten.",
             r"\input{assets/analysis/tabellen.tex}", paired_tables, r"\clearpage",
             r"\section{Verzeichnis des Ergebnisanhangs}\label{app:ergebnisanhang}",
             r"Die grafischen Darstellungen der ausgewerteten Läufe stehen in der separat beigefügten Datei \texttt{Anhang\_Ergebnisse.pdf}. Sie zeigen dieselben Läufe wie die vorstehenden Tabellen und enthalten keine zusätzlichen Messungen. Detailseiten enthalten je Aufgabe, Modell und Datensatz die Konfusionsmatrizen beider Varianten, die Kennzahlen sowie die Latenzverteilung empfangener Antworten. Tabelle~\ref{tab:ergebnisanhang-verzeichnis} nennt alle Abbildungsseiten mit ihrer Seitenzahl in dieser Datei.",
             r"\makeatletter\newcommand{\ergebnisseite}[2]{\phantomsection\def\@currentlabel{#1}\label{#2}#1}\makeatother",
             r"{\footnotesize",
             r"\begin{longtable}{@{}p{.86\linewidth}r@{}}",
             r"\caption{Verzeichnis der Abbildungsseiten in \texttt{Anhang\_Ergebnisse.pdf}. Eigene Darstellung.}\label{tab:ergebnisanhang-verzeichnis}\\",
             r"\toprule Abbildung & Seite \\ \midrule \endfirsthead",
             r"\toprule Abbildung & Seite \\ \midrule \endhead",
             r"\bottomrule \endlastfoot"]
    for text, item in contents_lines(index):
        if item is None:
            lines.append(r"\multicolumn{2}{@{}l}{\textbf{" + tex(text) + r"}} \\")
        else:
            label = "erg:" + item["name"].replace("_", "-")
            lines.append(r"\quad " + tex(text) + r" & \ergebnisseite{" + str(item["pdf_page"]) + "}{" + label + r"} \\")
    lines += [r"\end{longtable}", "}"]
    (OUT / "anhang_ergebnisse.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true", help="Recompute CSVs and audit without rendering figures")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--personalization-annotations", type=Path,
                        help="Separately reviewed id,personalization_level,annotation_note CSV; no inference")
    parser.add_argument("--personalization-proposals", type=Path,
                        help="Original assistant_rules_v1 suggestions; apply author-adopted binary grouping from 2026-11-01")
    args = parser.parse_args()
    runs, baselines, paired_baselines, pilots, lengths, files, database_checked = load_runs(args.personalization_annotations)
    indicator_code = ROOT / "experiment-runner/src/llm_phishing/personalization.py"
    files.append(dict(path=str(indicator_code.relative_to(ROOT)), sha256=sha(indicator_code), bytes=indicator_code.stat().st_size))
    baseline_loader = Path(__file__).with_name("baseline_results.py")
    files.append(dict(path=str(baseline_loader.relative_to(ROOT)), sha256=sha(baseline_loader), bytes=baseline_loader.stat().st_size))
    baseline_catalog = ROOT / "experiment-runner/data/evaluation/baseline_comparison.json"
    if baseline_catalog.exists():
        files.append(dict(path=str(baseline_catalog.relative_to(ROOT)), sha256=sha(baseline_catalog), bytes=baseline_catalog.stat().st_size))
    metrics, subgroups, performance, errors = aggregate(runs, baselines)
    proposal_coverage = apply_personalization_proposals(args.personalization_proposals, runs, baselines)
    if args.personalization_proposals:
        for path in [args.personalization_proposals, OUT / "personalisierung_vorschlaege.py", OUT / "personalisierung_vorschlaege_manifest.json"]:
            files.append(dict(path=str(path.resolve().relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size))
    comparisons = paired_comparisons(runs, args.bootstrap)
    baseline_comparisons = compare_baselines(runs, baselines, args.bootstrap)
    personalization, coverage = personalization_analysis(runs, baselines)
    make_tables(runs, baselines)
    write_csv("eingabelaengen.csv", [dict(dataset=d, instance_id=i, characters=n) for d, values in lengths.items() for i, n in values.items()])
    from gepaarter_injection_test import prepare
    paired_runs, paired_injection, paired_summary, detection_runs, paired_tables, paired_pages = prepare(
        baselines, paired_baselines, render_plots=not args.no_plots)
    paired_script = Path(__file__).with_name("gepaarter_injection_test.py")
    paired_sources = [paired_script]
    for run in [r for r in paired_runs if r["variant"] != "baseline"] + detection_runs:
        paired_sources += [ROOT / "thesis/assets/all-runs-results" / run["run_id"] / "task" / name
                           for name in ["run_manifest.json", "predictions.jsonl", "report.json"]]
    paired_sources += [jsonl_path(ROOT / "thesis/assets/all-runs-results" / run["run_id"] / "records.jsonl")
                       for run in paired_runs if run["variant"] != "baseline"]
    files += [dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)
              for path in paired_sources]
    if not args.no_plots:
        appendix_tex(plots(runs, baselines, comparisons, personalization, coverage, proposal_coverage, paired_pages), paired_tables)
    with (OUT / "laufregister.csv").open(encoding="utf-8") as stream:
        audit = list(csv.DictReader(stream))
    manifest = dict(analysis_date="2026-11-01", main_runs=len(runs), baseline_runs=len(baselines),
                    selected_runs=len(runs) + len(baselines), raw_export_runs=sum(row["scope"] != "excluded_historical" for row in audit),
                    inventory_directories=len(audit), supplementary_directories=len(pilots),
                    supplementary_with_predictions=sum(bool(p["rows"]) for p in pilots),
                    paired_injection_runs=len(paired_runs) - 2, paired_injection_baseline_runs=len(paired_baselines),
                    paired_injection_records=sum(2 * s["pairs"] for s in paired_summary if s["variant"] != "baseline"),
                    paired_injection_baseline_records=sum(len(r["rows"]) for r in paired_baselines),
                    paired_injection_pairs_per_run=495, paired_injection_failed=sum(s["failed"] for s in paired_summary),
                    paired_injection_detection_runs=len(detection_runs),
                    paired_injection_detection_records=sum(len(r["rows"]) for r in detection_runs),
                    total_including_paired_runs=len(runs) + len(baselines) + len(paired_runs) - 2 + len(paired_baselines) + len(detection_runs),
                    total_including_paired_records=sum(len(r["rows"]) for r in runs + baselines + paired_baselines + detection_runs)
                    + sum(2 * s["pairs"] for s in paired_summary if s["variant"] != "baseline"),
                    main_records=sum(len(r["rows"]) for r in runs), unique_main_inputs=sum(len(v) for v in lengths.values()),
                    baseline_records=sum(len(r["rows"]) for r in baselines),
                    selected_records=sum(len(r["rows"]) for r in runs + baselines),
                    duplicate_baseline_exports=sum(row["scope"] == "duplicate_baseline" for row in audit),
                    excluded_baseline_exports=sum(row["scope"] == "excluded_baseline" for row in audit),
                    selection_policy="One run per comparison cell; Qwen3-8B adapted E-PhishLLM explicitly replaced by 20261101T162628Z_12df175a; earliest equivalent complete baseline on exact LLM inputs",
                    invalid_main=sum(not r.get("schema_valid") for run in runs for r in run["rows"]),
                    input_truth_identical_across_main_runs=True, main_database_run_counts_match_exports=True if database_checked else None,
                    baseline_inputs_and_task_labels_match_llm_references=True,
                    personalization_annotations=str(args.personalization_annotations) if args.personalization_annotations else None,
                    personalization_proposals=str(args.personalization_proposals) if args.personalization_proposals else None,
                    personalization_scope="Author-adopted binary grouping P0/P1; original rule provenance retained", invalid_output_policy="negative fallback (no detection); raw exports unchanged", displayed_f1="unweighted mean of both class F1 scores",
                    validated_task_projections=len(metrics), baseline_paired_comparisons=len(baseline_comparisons), script_sha256=sha(Path(__file__)),
                    numpy_version=np.__version__, matplotlib_version=matplotlib.__version__,
                    bootstrap=dict(repetitions=args.bootstrap, seed=42, method="paired cluster percentile", metric="end_to_end_accuracy_difference", unit="stored group_id", interval="95 percent"),
                    notes=["No new model calls; no retraining; no pooling of overlapping historical runs.",
                           "Intervals quantify test-sample uncertainty conditional on stored model outputs, not repeated training/inference variation.",
                           "No multiple-comparison correction; subgroup results are exploratory.",
                           "Exported code is the export-time code; execution manifests preserve recorded run-time hashes.",
                           "Scientific source statements are in the thesis; CSVs and figures are own empirical results."],
                    sources=files)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ["sources", "notes"]}, indent=2))


if __name__ == "__main__":
    main()
