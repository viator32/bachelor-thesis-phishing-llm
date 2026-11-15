import { useState } from "react";

// Render comparison checkboxes and focused-run controls.
export default function RunPicker({
  runs,
  selectedRuns,
  onSelectedRunsChange,
  primaryRun,
  onPrimaryRunChange,
  scope,
}) {
  const [search, setSearch] = useState("");
  const [filters, setFilters] = useState({ status: "", model_id: "", dataset: "", model_task: "", split: "" });
  const visibleRuns = runs.filter((run) =>
    Object.entries(filters).every(([key, value]) => !value || run[key] === value)
    && [run.run_id, run.run_name, run.config_id, run.model_id, run.dataset, run.task_label, run.started_at]
      .join(" ").toLowerCase().includes(search.trim().toLowerCase())
  );
  const filterLabels = { status: "Status", model_id: "Model", dataset: "Dataset", model_task: "Task", split: "Split" };
  const filterPlaceholders = { status: "All statuses", model_id: "All models", dataset: "All datasets", model_task: "All tasks", split: "All splits" };
  // Toggle a run in the comparison selection.
  const toggle = (id) =>
    onSelectedRunsChange(
      selectedRuns.includes(id) ? selectedRuns.filter((r) => r !== id) : [...selectedRuns, id]
    );

  const allSelected = visibleRuns.length > 0 && visibleRuns.every((run) => selectedRuns.includes(run.run_id));

  return (
    <>
      <p className="panel-hint">
        {scope === "all" ? (
          <>The missing runs tracker checks all stored runs, regardless of these selections and filters.</>
        ) : scope === "compare" ? (
          <>
            This view charts the <b>{selectedRuns.length} checked</b>{" "}
            {selectedRuns.length === 1 ? "run" : "runs"}.
          </>
        ) : (
          <>
            This view charts the <b>focused</b> run only. Checkboxes affect Compare and
            Resources.
          </>
        )}
      </p>

      <input className="field" type="search" aria-label="Search stored runs"
        placeholder="Search runs, models, datasets..." value={search} onChange={(e) => setSearch(e.target.value)} />
      <div className="run-filters">
        {Object.entries(filterLabels).map(([key, label]) => <select key={key} className="field"
          aria-label={`Filter runs by ${label.toLowerCase()}`} value={filters[key]}
          onChange={(e) => setFilters((prev) => ({ ...prev, [key]: e.target.value }))}>
          <option value="">{filterPlaceholders[key]}</option>
          {[...new Set(runs.map((run) => run[key]).filter(Boolean))].sort().map((value) =>
            <option key={value} value={value}>{value.replaceAll("_", " ")}</option>)}
        </select>)}
      </div>
      <button onClick={() => { setSearch(""); setFilters({ status: "", model_id: "", dataset: "", model_task: "", split: "" }); }}>Clear filters</button>
      <p className="panel-hint">Showing {visibleRuns.length} / {runs.length} runs. Hidden selections remain checked.</p>
      <div className="table-tools">
        <button
          onClick={() => onSelectedRunsChange(allSelected
            ? selectedRuns.filter((id) => !visibleRuns.some((run) => run.run_id === id))
            : [...new Set([...selectedRuns, ...visibleRuns.map((run) => run.run_id)])])}
          disabled={!visibleRuns.length}
        >
          {allSelected ? "Uncheck shown" : "Check shown"}
        </button>
        <span>{selectedRuns.length} checked</span>
      </div>

      <div className="run-list">
        {visibleRuns.map((run) => {
          const focused = run.run_id === primaryRun;
          return (
            <div key={run.run_id} className={`run-row${focused ? " is-focused" : ""}`}>
              <input
                type="checkbox"
                checked={selectedRuns.includes(run.run_id)}
                onChange={() => toggle(run.run_id)}
                aria-label={`Include ${run.run_name || run.config_id} (${run.run_id}) in comparisons`}
              />
              <div>
                <button
                  className="run-name"
                  onClick={() => onPrimaryRunChange(run.run_id)}
                  aria-pressed={focused}
                  title={`Focus ${run.run_name || run.config_id} (${run.run_id})`}
                >
                  {run.run_name || run.config_id}
                  {focused && <span className="run-flag">FOCUSED</span>}
                </button>
                <div className="run-meta">
                  {run.model_id} · {run.records} rec
                  {run.successful != null && run.successful !== run.records
                    ? ` · ${run.successful} ok`
                    : ""}
                  {run.started_at ? ` · ${String(run.started_at).slice(0, 10)}` : ""}
                </div>
                {/* Several runs can share one config_id; the run id is what
                    actually tells two rows apart. */}
                <div className="run-meta run-id">{run.run_id}</div>
                <div className="run-meta">{run.dataset} · {run.split} · {(run.status || "unknown").replaceAll("_", " ")}
                  {run.expected_records != null ? ` · ${run.records}/${run.expected_records}` : ""}</div>
              </div>
            </div>
          );
        })}
        {!runs.length && <p className="muted">No stored runs yet.</p>}
        {runs.length > 0 && !visibleRuns.length && <p className="muted">No runs match these filters.</p>}
      </div>
    </>
  );
}
