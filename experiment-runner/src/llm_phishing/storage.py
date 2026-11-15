from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .schemas import ExperimentRecord
from .run_identity import record_identity


def validate_run_id(run_id: str) -> str:
    """Keep run IDs safe to use as filenames beneath the results directory."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError(
            "run_id must start with a letter or number and contain only letters, numbers, _, . or -"
        )
    return run_id


class ResultStore:
    """Store immutable predictions and editable run notes for local evaluation."""

    # Create result directories and initialize the SQLite store.
    def __init__(
        self, results_dir: str | Path = "results", database_path: str | Path | None = None
    ) -> None:
        self.results_dir = Path(results_dir)
        self.jsonl_dir = self.results_dir / "jsonl"
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = Path(database_path or self.results_dir / "experiments.sqlite")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    # Open a transaction; commit on success, roll back on failure, always close.
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    # Create record, index and note tables if absent.
    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_records (
                    run_id TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    config_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source_dataset TEXT,
                    split TEXT,
                    status TEXT,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, instance_id)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_run ON experiment_records(run_id)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS run_notes (run_id TEXT PRIMARY KEY, note TEXT NOT NULL)"
            )

    # Look up a run/instance primary key.
    def exists(self, run_id: str, instance_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM experiment_records WHERE run_id=? AND instance_id=?",
                (run_id, instance_id),
            ).fetchone()
        return row is not None

    # Insert a new result into SQLite and JSONL, ignoring duplicate keys.
    def append(self, record: ExperimentRecord) -> bool:
        validate_run_id(record.run_id)
        serialized = record.model_dump_json()
        with self._lock:
            with self._connect() as connection:
                # INSERT OR IGNORE on the (run_id, instance_id) primary key makes append() safe to
                # call again for an already-stored record, which is what ExperimentRunner's
                # resume mode relies on instead of checking `exists` and skipping writes itself.
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO experiment_records
                    (run_id, instance_id, config_id, model_id, prompt_id, timestamp,
                     source_dataset, split, status, record_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.run_id,
                        record.instance_id,
                        record.config_id,
                        record.model_id,
                        record.prompt_id,
                        record.timestamp,
                        record.source_dataset,
                        record.split,
                        record.final_result.status,
                        serialized,
                    ),
                )
                inserted = cursor.rowcount == 1
            if inserted:  # skip the JSONL append too, so it can't drift ahead of sqlite on resume
                with (self.jsonl_dir / f"{record.run_id}.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(serialized + "\n")
        return inserted

    # Load validated results ordered by timestamp and instance ID.
    def load_run(self, run_id: str) -> list[ExperimentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT record_json FROM experiment_records WHERE run_id=? ORDER BY timestamp, instance_id",
                (run_id,),
            ).fetchall()
        return [ExperimentRecord.model_validate_json(row[0]) for row in rows]

    # Fetch one result by its run/instance primary key.
    def load_instance(self, run_id: str, instance_id: str) -> ExperimentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT record_json FROM experiment_records WHERE run_id=? AND instance_id=?",
                (run_id, instance_id),
            ).fetchone()
        return ExperimentRecord.model_validate_json(row[0]) if row else None

    # Group stored records by run, config and model.
    def list_runs(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, config_id, model_id, MIN(timestamp), COUNT(*),
                       SUM(CASE WHEN status='success' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN status='error' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN status='timeout' THEN 1 ELSE 0 END),
                       (SELECT record_json FROM experiment_records first
                        WHERE first.run_id = records.run_id
                        ORDER BY timestamp, instance_id LIMIT 1)
                FROM experiment_records records GROUP BY run_id, config_id, model_id
                ORDER BY MIN(timestamp) DESC
                """
            ).fetchall()
        summaries = []
        for row in rows:
            record = json.loads(row[8])
            saved = record.get("reproducibility") or {}
            selection = saved.get("protocol", {}).get("selection", {})
            selected_labels = selection.get("selected_labels")
            expected = (
                sum(item["count"] for item in selected_labels)
                if selected_labels is not None else
                saved.get("dataset_manifest", {}).get("selected_count")
            )
            status = "unknown" if expected is None else "incomplete"
            if expected is not None and row[4] == expected:
                status = "completed" if row[5] == row[4] else "completed_with_errors"
            summaries.append({
                "run_id": row[0],
                "config_id": row[1],
                "model_id": row[2],
                "started_at": row[3],
                "records": row[4],
                "successful": row[5],
                "failed": row[6],
                "timeouts": row[7],
                "expected_records": expected,
                "status": status,
                "model_variant": saved.get("model_variant"),
                **record_identity(record),
            })
        return summaries

    # Delete one run, its note and JSONL file.
    def delete_run(self, run_id: str) -> int:
        validate_run_id(run_id)
        with self._lock:
            with self._connect() as connection:
                deleted = connection.execute(
                    "DELETE FROM experiment_records WHERE run_id=?", (run_id,)
                ).rowcount
                connection.execute("DELETE FROM run_notes WHERE run_id=?", (run_id,))
            (self.jsonl_dir / f"{run_id}.jsonl").unlink(missing_ok=True)
        return deleted

    # Delete every stored record, note and JSONL file.
    def delete_all_runs(self) -> int:
        with self._lock:
            with self._connect() as connection:
                deleted = connection.execute("DELETE FROM experiment_records").rowcount
                connection.execute("DELETE FROM run_notes")
            for path in self.jsonl_dir.glob("*.jsonl"):
                path.unlink(missing_ok=True)
        return deleted

    # Read a run note.
    def get_note(self, run_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT note FROM run_notes WHERE run_id=?", (run_id,)
            ).fetchone()
        return row[0] if row else None

    # Insert or replace the note for a run.
    def set_note(self, run_id: str, note: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO run_notes (run_id, note) VALUES (?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET note=excluded.note",
                (run_id, note),
            )
