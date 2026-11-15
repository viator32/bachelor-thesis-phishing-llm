from __future__ import annotations

import json
import os
import threading
import uuid
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Literal
from zipfile import ZIP_DEFLATED, ZipFile

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .experiment_runner import ExperimentRunner, load_config
from .feature_extractor import extract_features
from .metrics import compute_metrics, first_record_error
from .model_adapter import mlx_api_key
from .preprocessor import preprocess_email
from .reporting import add_run_to_archive, records_to_dataframe, write_run_archive
from .run_identity import new_run_id, run_identity
from .schemas import ExperimentConfig, ExperimentRecord, RawEmail
from .storage import ResultStore, validate_run_id
from .task_evaluation import load_task_config, run_task_evaluation
from .tfidf_training import configure_saved_classifier, train_colab_classifier

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
STORE = ResultStore(
    os.getenv("RESULTS_DIR", str(PROJECT_ROOT / "results")), os.getenv("DATABASE_PATH") or None
)
# Job state is in-memory and therefore lost on restart; keep the most recent
# entries only so a long-lived API process does not grow without bound.
MAX_TRACKED_JOBS = 100
JOBS: OrderedDict[str, dict[str, Any]] = OrderedDict()
JOB_STOPS: dict[str, threading.Event] = {}
JOBS_LOCK = threading.Lock()
RUN_QUEUE: dict[str, Any] | None = None
RUN_QUEUE_STOP = threading.Event()
ACTIVE_STATUSES = {"queued", "running", "stopping"}

app = FastAPI(title="LLM Phishing Research Prototype", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class SingleRequest(BaseModel):
    email: RawEmail
    config: ExperimentConfig


class BatchRequest(BaseModel):
    config_path: str = "configs/baselines/rules.yaml"
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    dataset_path: str | None = None
    dataset_manifest: str | None = None
    limit: int | None = Field(default=None, ge=1)
    split: Literal["validation", "test"] = "test"


class RunQueueRequest(BaseModel):
    items: list[BatchRequest] = Field(min_length=1, max_length=200)
    stop_on_error: bool = True


class TfidfTrainingRequest(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    files: dict[str, str]


class SavedClassifierRequest(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    model_path: str
    dataset_path: str


@app.get("/classifiers/tfidf")
def saved_classifiers() -> list[dict[str, str]]:
    """List local classifier files; loading happens only when an experiment starts."""
    return [
        {"label": str(path.relative_to(PROJECT_ROOT / "models")),
         "value": str(path.relative_to(PROJECT_ROOT))}
        for path in sorted((PROJECT_ROOT / "models").rglob("*.joblib"))
    ]


@app.post("/classifiers/tfidf/experiments", status_code=201)
def configure_classifier(request: SavedClassifierRequest) -> dict[str, str]:
    try:
        return configure_saved_classifier(
            PROJECT_ROOT, request.name, Path(request.model_path), Path(request.dataset_path),
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/classifiers/tfidf/train", status_code=201)
def train_classifier(request: TfidfTrainingRequest) -> dict[str, Any]:
    """Read uploaded Colab artifacts; CPU training runs in FastAPI's worker thread."""
    required = {"run_config.json", "train.jsonl", "test.jsonl"}
    if not required.issubset(request.files) or set(request.files) - (required | {"validation.jsonl"}):
        raise HTTPException(
            status_code=400,
            detail="Select run_config.json, train.jsonl, test.jsonl and optionally validation.jsonl.",
        )
    try:
        # Uploaded names are checked above; only generated dataset/model artifacts are retained.
        with TemporaryDirectory() as directory:
            source_dir = Path(directory)
            for filename, content in request.files.items():
                (source_dir / filename).write_text(content, encoding="utf-8")
            return train_colab_classifier(PROJECT_ROOT, request.name, source_dir)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Summarize batch outcomes for dashboard polling.
def _job_outcome(
    records: list[ExperimentRecord], config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    uses_llm = config is None or config.uses_llm
    responses = sum(
        bool(record.components['llm'].details.get('model_response_received'))
        for record in records if 'llm' in record.components
    )
    if not records or (uses_llm and not responses):
        status = 'failed'
    elif any(record.final_result.status in {'error', 'timeout'} for record in records) or (
        uses_llm and responses < len(records)
    ):
        status = 'completed_with_errors'
    else:
        status = 'completed'
    error = first_record_error(records)
    if not records:
        error = "No experiment records were produced. Check dataset filters and limit."
    elif status == "failed":
        error = error or "No response was received from the configured model."
    return {
        "status": status,
        "successful": sum(r.final_result.status == "success" for r in records),
        "failed": sum(r.final_result.status == "error" for r in records),
        "timeouts": sum(r.final_result.status == "timeout" for r in records),
        "model_responses": responses,
        "error": error,
    }


# Report API status and stored run count.
@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "api", "stored_runs": len(STORE.list_runs())}


# Preprocess and classify a single email with the supplied config.
@app.post("/classify")
def classify(request: SingleRequest) -> dict[str, Any]:
    processed = preprocess_email(request.email, request.config.max_input_tokens)
    extract_features(processed)
    runner = ExperimentRunner(request.config, PROJECT_ROOT, store=STORE)
    components, final, prompt_id = runner.classify_processed(processed)
    return {
        "instance_id": request.email.instance_id,
        "prompt_id": prompt_id,
        "processed_email": processed,
        "components": components,
        "final_result": final,
    }


# Run a configured background experiment and update shared job state.
def _execute_job(
    job_id: str,
    config_path: str,
    requested_run_id: str | None,
    dataset_path: str | None = None,
    limit: int | None = None,
    split: str = "test",
    dataset_manifest: str | None = None,
) -> None:
    stop = JOB_STOPS[job_id]
    # Publish batch progress under the job-state lock.
    def update(current: int, total: int, message: str) -> None:
        with JOBS_LOCK:
            JOBS[job_id].update({"current": current, "total": total, "message": message})

    try:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "stopping" if stop.is_set() else "running"
        path = Path(config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.parent.resolve() == (PROJECT_ROOT / "configs/llm").resolve():
            if dataset_path:
                raise ValueError("For LLM runs, select a task dataset manifest instead of a processed dataset.")
            config = load_task_config(path)
            options = _task_datasets(config.task)
            selected = next((item for item in options if item["value"] == (
                dataset_manifest or config.dataset_manifest
            )), None)
            if selected is None:
                raise ValueError("Dataset manifest is not available for this model task.")
            if split not in selected["splits"]:
                raise ValueError("Selected dataset does not contain this split.")

            def task_progress(message: str) -> None:
                with JOBS_LOCK:
                    JOBS[job_id]["message"] = message

            def task_counts(current: int, total: int) -> None:
                with JOBS_LOCK:
                    JOBS[job_id].update(current=current, total=total)

            result = run_task_evaluation(
                path, PROJECT_ROOT, split=split, limit=limit,
                run_id=requested_run_id, progress=task_progress,
                manifest_path=PROJECT_ROOT / selected["value"],
                should_stop=stop.is_set, on_progress=task_counts,
            )
            records = STORE.load_run(result["run_id"])
            outcome = _job_outcome(records)
            if result["status"] == "stopped":
                outcome.update(status="stopped", error=None)
            with JOBS_LOCK:
                JOBS[job_id].update({
                    **outcome, "run_id": result["run_id"], "records": len(records),
                    "current": len(records),
                    "metrics": result["metrics"],
                })
            return
        if dataset_manifest:
            raise ValueError("Dataset manifests are for LLM task runs; use a dataset path for baselines.")
        overrides: dict[str, Any] = {"split": split}
        if dataset_path:
            overrides["dataset_path"] = dataset_path
        if limit is not None:
            overrides["limit"] = limit
        config = load_config(path, **overrides)
        runner = ExperimentRunner(config, PROJECT_ROOT, store=STORE)
        run_id, records, metrics, reports = runner.run(
            requested_run_id, progress=update, should_stop=stop.is_set,
        )
        outcome = _job_outcome(records, config)
        if metrics.get("run_status") == "stopped":
            outcome.update(status="stopped", error=None)
        with JOBS_LOCK:
            JOBS[job_id].update(
                {
                    **outcome,
                    "run_id": run_id,
                    "records": len(records),
                    "metrics": metrics,
                    "reports": reports,
                }
            )
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id].update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        with JOBS_LOCK:
            JOB_STOPS.pop(job_id, None)


def _prepare_run(request: BatchRequest) -> tuple[str, dict]:
    try:
        path = PROJECT_ROOT / request.config_path
        if path.parent.resolve() == (PROJECT_ROOT / "configs/llm").resolve():
            config = load_task_config(path)
            identity = run_identity(config.config_id, config.task,
                                    request.dataset_manifest or config.dataset_manifest, request.split)
        else:
            config = load_config(path, split=request.split)
            identity = run_identity(config.config_id, config.task,
                                    request.dataset_path or config.dataset_path, config.split)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return config.config_id, identity


def _create_job(request: BatchRequest, queue_id: str | None = None) -> dict[str, str]:
    config_id, identity = _prepare_run(request)
    run_id = request.run_id or new_run_id(config_id, identity)
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        if any(job["status"] in ACTIVE_STATUSES for job in JOBS.values()):
            raise HTTPException(status_code=409, detail="An experiment is already running")
        if RUN_QUEUE and RUN_QUEUE["status"] in ACTIVE_STATUSES:
            if queue_id != RUN_QUEUE["queue_id"] or RUN_QUEUE_STOP.is_set():
                raise HTTPException(status_code=409, detail="Stop the experiment queue before starting another run")
        for old_id in list(JOBS):
            if len(JOBS) < MAX_TRACKED_JOBS:
                break
            if old_id not in JOB_STOPS:
                JOBS.pop(old_id)
        if len(JOBS) >= MAX_TRACKED_JOBS:
            raise HTTPException(status_code=409, detail="Too many active jobs")
        JOBS[job_id] = {"status": "queued", "current": 0, "total": None,
                        "run_id": run_id, **identity}
        JOB_STOPS[job_id] = threading.Event()
        if queue_id:
            RUN_QUEUE["active_job_id"] = job_id
    return {"job_id": job_id, "run_id": run_id, "status_url": f"/jobs/{job_id}"}


def _execute_requested_job(request: BatchRequest, job: dict) -> None:
    _execute_job(job["job_id"], request.config_path, job["run_id"],
                 request.dataset_path, request.limit, request.split, request.dataset_manifest)


@app.post("/runs", status_code=202)
def start_run(request: BatchRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    job = _create_job(request)
    background_tasks.add_task(_execute_requested_job, request, job)
    return job


def _queue_snapshot() -> dict | None:
    # Caller holds JOBS_LOCK. Return a detached JSON-safe snapshot.
    if RUN_QUEUE is None:
        return None
    snapshot = json.loads(json.dumps(RUN_QUEUE))
    snapshot["active_job"] = dict(JOBS.get(snapshot.get("active_job_id"), {})) or None
    return snapshot


@app.get("/run-queue")
def get_run_queue() -> dict | None:
    with JOBS_LOCK:
        return _queue_snapshot()


def _execute_run_queue(queue_id: str, requests: list[BatchRequest]) -> None:
    had_errors, failed = False, False
    try:
        with JOBS_LOCK:
            RUN_QUEUE["status"] = "stopping" if RUN_QUEUE_STOP.is_set() else "running"
        for index, request in enumerate(requests):
            if RUN_QUEUE_STOP.is_set():
                break
            try:
                config_id, identity = _prepare_run(request)
                exists = any(run["config_id"] == config_id and
                             all(run.get(key) == identity[key] for key in ("dataset", "split", "model_task"))
                             for run in STORE.list_runs())
                if exists:
                    with JOBS_LOCK:
                        RUN_QUEUE["items"][index].update(status="skipped", message="A matching stored run already exists")
                        RUN_QUEUE["finished"] += 1
                    continue
                job = _create_job(request, queue_id)
                with JOBS_LOCK:
                    RUN_QUEUE["items"][index].update(status="running", **job)
                _execute_requested_job(request, job)
                with JOBS_LOCK:
                    result = dict(JOBS[job["job_id"]])
                status, error = result["status"], result.get("error")
            except Exception as exc:
                status = "stopped" if RUN_QUEUE_STOP.is_set() else "failed"
                error = str(exc.detail) if isinstance(exc, HTTPException) else f"{type(exc).__name__}: {exc}"
            with JOBS_LOCK:
                RUN_QUEUE["items"][index].update(status=status, message=error)
                RUN_QUEUE["finished"] += 1
                RUN_QUEUE["active_job_id"] = None
                if status in {"failed", "completed_with_errors"}:
                    had_errors = True
                    if RUN_QUEUE["stop_on_error"]:
                        failed = True
                        RUN_QUEUE["error"] = error or "An experiment finished with errors"
                        break
                if status == "stopped":
                    RUN_QUEUE_STOP.set()
                    break
    except Exception as exc:
        failed = True
        with JOBS_LOCK:
            RUN_QUEUE["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with JOBS_LOCK:
            if RUN_QUEUE_STOP.is_set():
                RUN_QUEUE["status"] = "stopped"
            else:
                RUN_QUEUE["status"] = "failed" if failed else "completed_with_errors" if had_errors else "completed"
            RUN_QUEUE["active_job_id"] = None
            for item in RUN_QUEUE["items"]:
                if item["status"] == "queued":
                    item["status"] = "cancelled"


@app.post("/run-queue", status_code=202)
def start_run_queue(request: RunQueueRequest, background_tasks: BackgroundTasks) -> dict:
    global RUN_QUEUE
    items, requests, seen = [], [], set()
    for entry in request.items:
        if entry.run_id:
            raise HTTPException(status_code=400, detail="The missing-run queue creates new run IDs; use manual resume for existing runs")
        config_id, identity = _prepare_run(entry)
        key = (config_id, identity["dataset"], identity["split"], identity["model_task"])
        if key in seen:
            continue
        seen.add(key)
        requests.append(entry)
        items.append({"config_id": config_id, **identity, "status": "queued",
                      "limit": entry.limit, "message": None})
    queue_id = uuid.uuid4().hex
    with JOBS_LOCK:
        if (RUN_QUEUE and RUN_QUEUE["status"] in ACTIVE_STATUSES) or any(
            job["status"] in ACTIVE_STATUSES for job in JOBS.values()
        ):
            raise HTTPException(status_code=409, detail="Stop the active run or queue first")
        RUN_QUEUE_STOP.clear()
        RUN_QUEUE = {"queue_id": queue_id, "status": "queued", "finished": 0,
                     "total": len(items), "items": items, "active_job_id": None,
                     "stop_on_error": request.stop_on_error, "error": None}
        snapshot = _queue_snapshot()
    background_tasks.add_task(_execute_run_queue, queue_id, requests)
    return snapshot


@app.post("/run-queue/stop")
def stop_run_queue() -> dict | None:
    with JOBS_LOCK:
        if RUN_QUEUE and RUN_QUEUE["status"] in ACTIVE_STATUSES:
            RUN_QUEUE_STOP.set()
            RUN_QUEUE["status"] = "stopping"
            stop = JOB_STOPS.get(RUN_QUEUE.get("active_job_id"))
            if stop:
                stop.set()
        return _queue_snapshot()


@app.post("/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict[str, Any]:
    """Finish the current email, persist partial results and skip remaining emails."""
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Unknown job")
        if job_id in JOB_STOPS and JOBS[job_id]["status"] in {"queued", "running", "stopping"}:
            JOB_STOPS[job_id].set()
            JOBS[job_id]["status"] = "stopping"
        return dict(JOBS[job_id])


# Read a job snapshot under the state lock.
@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Unknown job")
        return dict(JOBS[job_id])


# List stored experiment runs.
@app.get("/runs")
def list_stored_runs() -> list[dict]:
    runs = STORE.list_runs()
    with JOBS_LOCK:
        jobs = {job.get("run_id"): dict(job) for job in JOBS.values()}
    for run in runs:
        job = jobs.get(run["run_id"], {})
        if job.get("status") in {"queued", "running", "stopping", "failed", "stopped"}:
            run["status"] = job["status"]
    return runs


# Load a stored run or reject an unknown ID.
def _load_run_or_404(run_id: str) -> list[ExperimentRecord]:
    records = STORE.load_run(run_id)
    if not records:
        raise HTTPException(status_code=404, detail="Unknown run")
    return records


# Recompute metrics for a stored run.
@app.get("/runs/{run_id}/metrics")
def get_run_metrics(run_id: str) -> dict[str, Any]:
    return _metrics_for_records(_load_run_or_404(run_id))


def _metrics_for_records(records: list[ExperimentRecord]) -> dict[str, Any]:
    metrics = compute_metrics(records)
    run_id = records[0].run_id
    native_path = PROJECT_ROOT / "results" / "task_adapters" / run_id / "report.json"
    if native_path.is_file():
        # Preserve native denominators and the explicit invalid-output matrix beside shared metrics.
        metrics["native_report"] = json.loads(native_path.read_text(encoding="utf-8"))
    return metrics


@app.get("/runs/export")
def export_all_runs() -> FileResponse:
    runs = STORE.list_runs()
    if not runs:
        raise HTTPException(status_code=404, detail="No stored runs to export")
    with NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
        path = Path(temporary.name)
    try:
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            for stored_run_id in dict.fromkeys(run["run_id"] for run in runs):
                run_id = validate_run_id(stored_run_id)
                records = _load_run_or_404(run_id)
                add_run_to_archive(
                    records, _metrics_for_records(records), archive,
                    PROJECT_ROOT / "results" / "task_adapters" / run_id,
                    STORE.get_note(run_id) or "", prefix=f"{run_id}/",
                )
            archive.writestr("runs.json", json.dumps(runs, ensure_ascii=False, indent=2))
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return FileResponse(path, media_type="application/zip", filename="all-runs-results.zip",
                        background=BackgroundTask(path.unlink, missing_ok=True))


# Download all persisted evidence for the selected run.
@app.get("/runs/{run_id}/export")
def export_run(run_id: str) -> FileResponse:
    try:
        validate_run_id(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    records = _load_run_or_404(run_id)
    # Each request owns its temporary archive; deletion also runs after a download completes.
    with NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
        path = Path(temporary.name)
    try:
        write_run_archive(
            records, _metrics_for_records(records), path,
            PROJECT_ROOT / "results" / "task_adapters" / run_id, STORE.get_note(run_id) or "",
        )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return FileResponse(path, media_type="application/zip", filename=f"{run_id}-experiment.zip",
                        background=BackgroundTask(path.unlink, missing_ok=True))


@app.get("/runs/{run_id}/records")
def get_run_records(
    run_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int | None = Query(default=None, ge=1),
) -> list[dict]:
    records = _load_run_or_404(run_id)
    if offset or limit is not None:
        records = records[offset : offset + limit if limit is not None else None]
    frame = records_to_dataframe(records)
    frame = frame.astype(object).where(frame.notna(), None)
    return frame.to_dict("records")


# Load one stored email result.
@app.get("/runs/{run_id}/instances/{instance_id}")
def get_run_instance(run_id: str, instance_id: str) -> ExperimentRecord:
    record = STORE.load_instance(run_id, instance_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown instance")
    return record


# Remove one run, its note and JSONL file.
@app.delete("/runs/{run_id}")
def delete_run(run_id: str) -> dict[str, Any]:
    """Permanently remove one stored run: its records, note and JSONL file."""
    with JOBS_LOCK:
        if any(job.get("run_id") == run_id and job["status"] in {"queued", "running", "stopping"}
               for job in JOBS.values()):
            raise HTTPException(status_code=409, detail="Stop the active run before deleting it")
        deleted = STORE.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Unknown run")
    return {"run_id": run_id, "deleted_records": deleted}


# Require DELETE ALL confirmation and empty the store; reports remain.
@app.delete("/runs")
def delete_all_runs(confirm: str = Query(...)) -> dict[str, Any]:
    """Empty the result store. Requires `?confirm=DELETE ALL` so it cannot fire by accident.

    Written reports under results/reports are left alone; they are plain files and
    deleting a directory tree is not something this API should do on one request.
    """
    if confirm != "DELETE ALL":
        raise HTTPException(status_code=400, detail="Pass confirm=DELETE ALL to empty the store")
    with JOBS_LOCK:
        if (RUN_QUEUE and RUN_QUEUE["status"] in ACTIVE_STATUSES) or any(
            job["status"] in ACTIVE_STATUSES for job in JOBS.values()
        ):
            raise HTTPException(status_code=409, detail="Stop active runs and the queue before clearing the database")
        return {"deleted_records": STORE.delete_all_runs()}


class NoteRequest(BaseModel):
    note: str


# Read notes for a run.
@app.get("/runs/{run_id}/notes")
def get_run_notes(run_id: str) -> dict[str, str]:
    return {"note": STORE.get_note(run_id) or ""}


# Save the requested run note.
@app.put("/runs/{run_id}/notes")
def put_run_notes(run_id: str, request: NoteRequest) -> dict[str, str]:
    STORE.set_note(run_id, request.note)
    return {"note": request.note}


@app.get('/experiment-configs')
def experiment_configs() -> list[dict[str, Any]]:
    return _build_experiment_configs()


# Describe one config for the dashboard picker.
def _describe_config(path: Path, config: ExperimentConfig) -> dict:
    item = {
        "value": f"configs/baselines/{path.name}",
        "config_id": config.config_id,
        "name": path.stem,
        "method": config.method,
        "output_contract": config.output_contract,
        "response_schema": config.response_schema,
    }
    needs_classifier = config.baseline_component == "tfidf"
    ready = not needs_classifier or (PROJECT_ROOT / config.tfidf_model_path).is_file()
    description = "trained TF-IDF classifier" if needs_classifier else "fixed rules"
    compatible = [item for item in datasets() if config.task in item["tasks"]]
    fixed = [item for item in compatible if item["fixed_comparison"]]
    options = fixed + [item for item in compatible if item["value"] == config.dataset_path
                       and item not in fixed] if fixed else compatible
    return {
        **item,
        "label": f"{path.stem} — {description} — {'ready' if ready else 'train classifier first'}",
        "kind": "baseline",
        "task": config.task,
        "task_label": f"{description} — {config.task}",
        "dataset_path": config.dataset_path,
        "datasets": options,
        "backend": "not_applicable",
        "model_id": None,
        "classifier_path": config.tfidf_model_path if needs_classifier else None,
        "model_installed": None,
        "output_mode": None,
        "repair": None,
        "availability": "ready to run" if ready else "train classifier first",
    }

def _task_datasets(task: str) -> list[dict[str, Any]]:
    external_sources = {
        "phishing_origin": [
            ("ephishllm", "Phishing detection — E-PhishLLM (external)"),
            ("crossmodel", "Human or AI origin — Cross-Model (external)"),
            ("ephishllm_paired_injection", "Phishing robustness — own paired injection test"),
        ],
        "injection": [
            ("llmail", "Prompt injection detection — LLMail-Inject (external)"),
            ("ephishllm_paired_injection_detection", "Prompt injection detection — own paired injection test"),
        ],
    }
    sources = [
        (f"data/evaluation/external/{name}_task_manifest.json", label, True)
        for name, label in external_sources[task]
    ] + [("data/evaluation/task_adapters/manifest.json", "Internal holdout — training source", False)]
    items = []
    for value, label, external in sources:
        path = PROJECT_ROOT / value
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        splits = {
            name: info["rows"]
            for name, info in manifest["tasks"][task]["splits"].items()
            if name in {"validation", "test"}
        }
        items.append({"value": value, "label": label, "external": external, "splits": splits})
    return items


# Include native task configs even when the model server is unavailable.
def _build_experiment_configs() -> list[dict[str, Any]]:
    configs_dir = PROJECT_ROOT / "configs" / "baselines"
    items = [
        _describe_config(path, load_config(path))
        for path in sorted(configs_dir.glob("*.yaml"))
    ]
    models_by_url: dict[str, set[str] | None] = {}
    datasets_by_task = {task: _task_datasets(task) for task in ("phishing_origin", "injection")}
    key = mlx_api_key()
    model_names = {
        "Qwen/Qwen3-4B": "Qwen3 4B",
        "Qwen/Qwen3-8B": "Qwen3 8B",
        "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B Instruct",
        "microsoft/Phi-4-mini-instruct": "Phi-4 Mini Instruct",
        "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
    }
    for path in sorted((PROJECT_ROOT / "configs/llm").glob("*.yaml")):
        config = load_task_config(path)
        url = os.getenv("MLX_BASE_URL", config.base_url).rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        if url not in models_by_url:
            try:
                response = httpx.get(
                    f"{url}/models", timeout=3,
                    headers={"Authorization": f"Bearer {key}"} if key else {},
                )
                response.raise_for_status()
                models_by_url[url] = {item["id"] for item in response.json()["data"]}
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                models_by_url[url] = None
        models = models_by_url[url]
        installed = config.model_id in models if models is not None else None
        availability = (
            "availability unknown" if installed is None else
            "ready to run" if installed else "model not downloaded"
        )
        task_label = (
            "Phishing and AI origin detection" if config.task == "phishing_origin"
            else "Prompt injection detection"
        )
        variant = "Original" if config.model_variant == "base" else "Fine-tuned"
        name = f"{model_names.get(config.base_model_id, config.base_model_id)} — {variant}"
        datasets = datasets_by_task[config.task]
        if "ephishllm_paired_injection" in config.dataset_manifest:
            name += " — Paired injection"
            if config.task == "phishing_origin":
                task_label = "Phishing robustness (origin unscored)"
            datasets = [d for d in datasets if d["value"] == config.dataset_manifest]
        else:
            datasets = [d for d in datasets if "ephishllm_paired_injection" not in d["value"]]
        items.append({
            "value": f"configs/llm/{path.name}",
            "config_id": config.config_id, "name": name,
            "label": f"{name} — {task_label} — {availability}",
            "task_label": task_label,
            "description": (
                "Original model, without your task fine-tuning"
                if config.model_variant == "base" else f"Fine-tuned for {task_label.lower()}"
            ),
            "kind": "llm", "method": "llm", "backend": "mlx",
            "task": config.task, "model_variant": config.model_variant,
            "datasets": datasets,
            "model_id": config.model_id, "model_installed": installed,
            "output_contract": config.output_contract, "response_schema": config.response_schema,
            "output_mode": "JSON schema enforced" if config.constrained_json else "free JSON",
            "repair": "no repair", "availability": availability,
        })
    return items


# List processed JSONL datasets.
@lru_cache(maxsize=32)
def _dataset_summary(path: str, modified_ns: int, size: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    tasks: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                split = row.get("split", "unspecified")
                if split in {"test", "validation"}:
                    counts[split] = counts.get(split, 0) + 1
                    for task, field in (("phishing", "is_phishing"), ("injection", "has_prompt_injection")):
                        if type(row.get(field)) is bool:
                            tasks.add(task)
    return {"splits": counts, "tasks": sorted(tasks)}


@app.get("/datasets")
def datasets() -> list[dict[str, Any]]:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    if not processed_dir.is_dir():
        return []
    labels = {
        "comparison_phishfuzzer_test": "PhishFuzzer — same 424 LLM test emails (primary)",
        "comparison_ephishllm_test": "E-PhishLLM — same 1,000 LLM test emails (primary)",
        "comparison_ephishllm_paired_injection_test": "E-PhishLLM paired injection — same 990 LLM test emails (clean/injected)",
        "comparison_crossmodel_test": "Cross-Model — same 1,000 LLM test emails (phishing recall only)",
        "comparison_contextual_test": "Contextual Paired — same 872 LLM test emails (injection)",
        "comparison_llmail_test": "LLMail-Inject — same 1,000 LLM test emails (injection)",
    }
    return [{"label": labels.get(path.stem, path.name),
             "value": f"data/processed/{path.name}", "fixed_comparison": path.stem in labels,
             **_dataset_summary(str(path), path.stat().st_mtime_ns, path.stat().st_size)}
            for path in sorted(processed_dir.glob("*.jsonl"))]
