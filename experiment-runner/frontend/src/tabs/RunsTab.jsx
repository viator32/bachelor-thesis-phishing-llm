import { useMemo, useState } from "react";
import NewRunForm from "../components/NewRunForm.jsx";
import RunQueuePanel from "../components/RunQueuePanel.jsx";
import { buildRunCoverage, runCombinationKey } from "../runCoverage.js";
import { api } from "../api.js";
import { downloadBlob, toCSV } from "../utils.js";

const activeStatuses = new Set(["queued", "running", "stopping"]);
const filterLabels = { config_id: "Configuration", model_id: "Model", dataset: "Dataset", model_task: "Task", split: "Split", status: "Status" };

export default function RunsTab({ runs, configs, configsLoading, datasets, onRefreshConfigs,
  onRefreshRuns, onFocus, onDeleteRuns, onDeleteAllRuns, onDownloadAllRuns, exportingAllRuns,
  onClassifierReady, onStartRun, onStopRun, stopping, jobMessage, jobRunning,
  selectedRuns, onSelectedRunsChange }) {
  const [view, setView] = useState("stored");
  const [search, setSearch] = useState("");
  const [filters, setFilters] = useState({});
  const [split, setSplit] = useState("test");
  const [neverRunConfig, setNeverRunConfig] = useState(false);
  const [duplicateScope, setDuplicateScope] = useState("config");
  const [marked, setMarked] = useState([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);
  const [preparedRun, setPreparedRun] = useState(null);
  const [queueRunning, setQueueRunning] = useState(false);
  const [page, setPage] = useState(0);
  const pageSize = 20;

  const { configCounts, combinationCounts } = useMemo(() => {
    const configCounts = new Map(), combinationCounts = new Map();
    for (const run of runs) {
      configCounts.set(run.config_id, (configCounts.get(run.config_id) || 0) + 1);
      const key = runCombinationKey(run);
      combinationCounts.set(key, (combinationCounts.get(key) || 0) + 1);
    }
    return { configCounts, combinationCounts };
  }, [runs]);
  const coverage = useMemo(() => buildRunCoverage(configs, runs, split), [configs, runs, split]);
  const missing = coverage.filter((row) => row.status === "Missing");
  const neverRunConfigs = configs.filter((config) => !configCounts.has(config.config_id));
  const duplicateConfigs = [...configCounts.values()].filter((count) => count > 1).length;
  const matches = (values) => values.join(" ").toLowerCase().includes(search.trim().toLowerCase());
  const filteredRuns = runs.filter((run) => Object.entries(filters).every(([key, value]) => !value || run[key] === value)
    && matches([run.run_id, run.run_name, run.config_id, run.model_id, run.dataset, run.task_label])
    && (view !== "duplicates" || (duplicateScope === "config"
      ? configCounts.get(run.config_id) > 1 : combinationCounts.get(runCombinationKey(run)) > 1)));
  if (view === "duplicates") filteredRuns.sort((a, b) => a.config_id.localeCompare(b.config_id) || String(b.started_at).localeCompare(String(a.started_at)));
  const filteredMissing = missing.filter((row) => (!neverRunConfig || !configCounts.has(row.config.config_id))
    && matches([row.config.name, row.config.config_id, row.config.task_label, row.name])
    && (!filters.config_id || row.config.config_id === filters.config_id)
    && (!filters.model_id || row.config.model_id === filters.model_id)
    && (!filters.dataset || row.name === filters.dataset)
    && (!filters.model_task || row.config.task === filters.model_task)
    && (!filters.split || row.split === filters.split));
  const rows = view === "missing" ? filteredMissing : filteredRuns;
  const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
  const currentPage = Math.min(page, pageCount - 1);
  const pageRows = rows.slice(currentPage * pageSize, (currentPage + 1) * pageSize);
  const selected = runs.filter((run) => marked.includes(run.run_id));
  const deletable = selected.filter((run) => !activeStatuses.has(run.status));
  const allShownMarked = filteredRuns.length > 0 && filteredRuns.every((run) => marked.includes(run.run_id));

  function changeView(next) {
    setView(next); setPage(0);
    if (next === "missing") setFilters((prev) => ({ ...prev, status: "", split: "" }));
  }
  function prepare(row) { setPreparedRun({ ...row }); changeView("new"); }
  async function remove(ids) {
    if (!ids.length || !window.confirm(`Delete ${ids.length} stored run(s), their records, notes and JSONL files? Saved reports and native task artifacts remain. This cannot be undone.`)) return;
    setBusy(true); setMessage(null);
    try {
      const deleted = await onDeleteRuns(ids);
      setMarked((previous) => previous.filter((id) => !deleted.includes(id)));
      setMessage({ type: "info", text: `Deleted ${deleted.length} run(s).` });
    } catch (error) { setMessage({ type: "danger", text: error.message }); }
    finally { setBusy(false); }
  }
  async function exportRun(run) {
    setMessage(null);
    try { downloadBlob(`${run.run_id}.zip`, await api.runExport(run.run_id), "application/zip"); }
    catch (error) { setMessage({ type: "danger", text: error.message }); }
  }

  return <section className="runs-overview">
    <h2>Runs overview</h2>
    <p>Missing means no matching run exists in the database. Failed, stopped, partial and successful runs all count as stored.</p>
    <div className="runs-summary" role="status">
      <span><b>{runs.length}</b> stored runs</span><span><b>{missing.length}</b> missing {split} combinations</span>
      <span><b>{neverRunConfigs.length}</b> configurations never run</span><span><b>{duplicateConfigs}</b> configurations used more than once</span>
    </div>
    <div className="table-tools runs-toolbar">
      {[["stored", "All runs"], ["missing", "Missing runs"], ["duplicates", "Duplicate runs"], ["new", "Create new run"]].map(([id, label]) =>
        <button key={id} className={view === id ? "primary" : undefined} aria-pressed={view === id} onClick={() => changeView(id)}>{label}</button>)}
      <button onClick={onRefreshRuns}>Refresh runs</button>
      <button onClick={onRefreshConfigs} disabled={configsLoading}>{configsLoading ? "Checking configurations..." : "Refresh configurations"}</button>
    </div>
    {message && <div role="status" className={`alert alert-${message.type}`}>{message.text}</div>}
    <RunQueuePanel candidates={filteredMissing} visible={view === "missing"} disabled={jobRunning || configsLoading}
      onActiveChange={setQueueRunning} onRefreshRuns={onRefreshRuns} />
    {view !== "new" && jobMessage && <div className={`alert alert-${jobMessage.type}`}>{jobMessage.text}
      {jobRunning && <button disabled={stopping} onClick={onStopRun}>{stopping ? "Stopping..." : "Stop run and save"}</button>}</div>}

    <div hidden={view !== "new"} className="run-creation">
      <h3>Create a new run</h3>
      <NewRunForm {...{ runs, configs, datasets, configsLoading, onRefreshConfigs, preparedRun,
        onClassifierReady, onStartRun, onStopRun, stopping, jobMessage, jobRunning, queueRunning }} />
    </div>
    {view !== "new" && <>
      <div className="coverage-tools">
        <label>Search<input className="field" type="search" placeholder="Run ID, configuration, model or dataset" value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(0); }} /></label>
        {Object.entries(filterLabels).filter(([key]) => view !== "missing" || !["status", "split"].includes(key)).map(([key, label]) => {
          const options = [...new Set([...runs.map((run) => run[key]), ...configs.flatMap((config) => key === "dataset"
            ? coverage.filter((row) => row.config.config_id === config.config_id).map((row) => row.name)
            : [key === "model_task" ? config.task : config[key]])].filter(Boolean))].sort();
          return <label key={key}>{label}<select className="field" value={filters[key] || ""}
            onChange={(e) => { setFilters((prev) => ({ ...prev, [key]: e.target.value })); setPage(0); }}>
            <option value="">All</option>{options.map((value) => <option key={value} value={value}>{value.replaceAll("_", " ")}</option>)}
          </select></label>;
        })}
        {view === "missing" && <label>Split<select className="field" value={split} onChange={(e) => { setSplit(e.target.value); setPage(0); }}>
          <option value="test">Test</option><option value="validation">Validation (setup)</option>
        </select></label>}
      </div>
      <button onClick={() => { setFilters({}); setSearch(""); setNeverRunConfig(false); setPage(0); }}>Clear filters</button>

      {view === "missing" ? <>
        <p>Combinations use configuration, task, dataset and split. Models, rules and classifiers are included.</p>
        <label><input type="checkbox" checked={neverRunConfig} onChange={(e) => { setNeverRunConfig(e.target.checked); setPage(0); }} /> Only configurations with no runs anywhere in the database</label>
        <div className="table-tools runs-toolbar"><button disabled={!filteredMissing.length} onClick={() => downloadBlob(`missing-runs-${split}.csv`, toCSV(filteredMissing.map((row) => ({
          config_id: row.config.config_id, config_path: row.config.value, task: row.config.task, dataset: row.name, split,
          dataset_manifest: row.config.kind === "llm" ? row.dataset.value : "", dataset_path: row.config.kind === "llm" ? "" : row.dataset.value,
        }))), "text/csv")}>Export shown missing combinations</button></div>
        <div className="table-scroll"><table className="data-table">
          <thead><tr><th>Configuration</th><th>Model / baseline</th><th>Task</th><th>Dataset</th><th>Split</th><th>Action</th></tr></thead>
          <tbody>{pageRows.map((row) => <tr key={`${row.config.config_id}:${row.dataset.value}`}>
            <td>{row.config.config_id}</td><td>{row.config.name}</td><td>{row.config.task_label}</td><td>{row.name}</td><td>{row.split}</td>
            <td><button onClick={() => prepare(row)}>Create run</button></td>
          </tr>)}</tbody>
        </table></div>
      </> : <>
        {view === "duplicates" && <><label>Show repeated runs by<select className="field" value={duplicateScope} onChange={(e) => { setDuplicateScope(e.target.value); setPage(0); }}>
          <option value="config">Same configuration (across all datasets and splits)</option>
          <option value="combination">Same configuration, task, dataset and split</option>
        </select></label><p className="muted">Repeated configurations can be intentional when datasets or splits differ. No runs are deleted automatically.</p></>}
        <div className="table-tools runs-toolbar">
          <button disabled={!filteredRuns.length} onClick={() => setMarked((prev) => allShownMarked
            ? prev.filter((id) => !filteredRuns.some((run) => run.run_id === id))
            : [...new Set([...prev, ...filteredRuns.map((run) => run.run_id)])])}>{allShownMarked ? "Unselect shown" : "Select shown"}</button>
          <span>{selected.length} selected across filters</span>
          <button disabled={!selected.length} onClick={() => onSelectedRunsChange([...new Set([...selectedRuns, ...selected.map((run) => run.run_id)])])}>Add selected to comparison</button>
          <button disabled={busy || !deletable.length} onClick={() => remove(deletable.map((run) => run.run_id))}>Delete selected ({deletable.length})</button>
          <button disabled={!filteredRuns.length} onClick={() => downloadBlob("runs-overview.csv", toCSV(filteredRuns), "text/csv")}>Export shown CSV</button>
          <button disabled={!runs.length || exportingAllRuns} onClick={onDownloadAllRuns}>{exportingAllRuns ? "Preparing ZIP..." : "Export all runs ZIP"}</button>
        </div>
        <div className="table-scroll"><table className="data-table runs-table">
          <thead><tr><th>Select</th><th>Run / configuration</th><th>Dataset / split</th><th>Status / records</th><th>Repeated runs</th><th>Actions</th></tr></thead>
          <tbody>{pageRows.map((run) => <tr key={run.run_id}>
            <td><input type="checkbox" aria-label={`Select run ${run.run_id}`} checked={marked.includes(run.run_id)} onChange={() => setMarked((prev) => prev.includes(run.run_id) ? prev.filter((id) => id !== run.run_id) : [...prev, run.run_id])} /></td>
            <td><button className="run-name" onClick={() => onFocus(run.run_id)}>{run.run_name || run.config_id}</button>
              <div className="run-meta run-id">{run.run_id}</div><div className="run-meta">{run.config_id}</div><div className="run-meta">{run.model_id} · {run.started_at?.slice(0, 19).replace("T", " ")}</div></td>
            <td>{run.dataset}<div>{run.split} · {run.task_label}</div></td>
            <td>{(run.status || "unknown").replaceAll("_", " ")}<div>{run.records} stored · {run.successful} successful · {(run.failed || 0) + (run.timeouts || 0)} failed/timed out</div></td>
            <td>{configCounts.get(run.config_id)} for config<div>{combinationCounts.get(runCombinationKey(run))} for same combination</div></td>
            <td><div className="run-actions"><button onClick={() => onFocus(run.run_id)}>Inspect</button><button onClick={() => exportRun(run)}>Export ZIP</button>
              <button disabled={busy || activeStatuses.has(run.status)} onClick={() => remove([run.run_id])}>Delete</button></div></td>
          </tr>)}</tbody>
        </table></div>
        <details className="panel"><summary>Delete all stored runs</summary>
          <p>Deletes database results, notes and shared JSONL files. Saved reports and native task artifacts remain.</p>
          <button disabled={busy || !runs.length || jobRunning || queueRunning || runs.some((run) => activeStatuses.has(run.status))} onClick={async () => {
            if (!window.confirm(`Delete ALL ${runs.length} stored runs? This cannot be undone.`)) return;
            setBusy(true); setMessage(null);
            try { await onDeleteAllRuns(); setMarked([]); } catch (error) { setMessage({ type: "danger", text: error.message }); }
            finally { setBusy(false); }
          }}>Clear database ({runs.length} runs)</button>
        </details>
      </>}
      {!rows.length && <p className="muted">{configsLoading && view === "missing" ? "Loading configurations..." : "No runs or combinations match these filters."}</p>}
      <div className="table-tools runs-toolbar"><button disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>Previous</button>
        <span>Page {currentPage + 1} / {pageCount} · {rows.length} results</span>
        <button disabled={currentPage >= pageCount - 1} onClick={() => setPage(currentPage + 1)}>Next</button></div>
    </>}
  </section>;
}
