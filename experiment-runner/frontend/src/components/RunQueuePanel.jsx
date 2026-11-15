import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

const activeStatuses = new Set(["queued", "running", "stopping"]);

export default function RunQueuePanel({ candidates, visible, disabled, onActiveChange, onRefreshRuns }) {
  const [queue, setQueue] = useState(null);
  const [limit, setLimit] = useState("1000");
  const [stopOnError, setStopOnError] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [pollError, setPollError] = useState("");
  const lastProgress = useRef("");
  const active = activeStatuses.has(queue?.status);
  const ready = candidates.filter((row) => row.config.model_installed !== false
    && row.config.availability !== "train classifier first");

  useEffect(() => { onActiveChange(active); }, [active, onActiveChange]);
  useEffect(() => {
    let disposed = false;
    let timer;
    const controller = new AbortController();
    async function poll() {
      try {
        const next = await api.runQueue(controller.signal);
        if (disposed) return;
        setQueue(next);
        setPollError("");
        const progress = `${next?.queue_id}:${next?.finished}:${next?.status}`;
        if (next && progress !== lastProgress.current) {
          lastProgress.current = progress;
          onRefreshRuns();
        }
      } catch (err) {
        if (!disposed) setPollError(err.message);
      } finally {
        if (!disposed) timer = setTimeout(poll, 2000);
      }
    }
    poll();
    return () => { disposed = true; controller.abort(); clearTimeout(timer); };
  }, [onRefreshRuns]);

  async function start() {
    setBusy(true); setError("");
    try {
      const items = ready.map((row) => ({
        config_path: row.config.value, split: row.split,
        dataset_manifest: row.config.kind === "llm" ? row.dataset.value : undefined,
        dataset_path: row.config.kind === "llm" ? undefined : row.dataset.value,
        limit: limit ? Number(limit) : undefined,
      }));
      const next = await api.startRunQueue({ items, stop_on_error: stopOnError });
      setQueue(next); onActiveChange(true);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  if (!visible && !queue && !error && !pollError) return null;
  return <section className="panel" aria-label="Sequential experiment queue">
    <h3>Automatic experiment queue</h3>
    {visible && !active && <>
      <p>Run all shown missing combinations in sequence, including rows on other pages.
        {candidates.length !== ready.length && ` ${candidates.length - ready.length} unavailable configurations are excluded.`}</p>
      <label>Max emails per experiment (empty = all)
        <input className="field" type="number" min="2" step="2" value={limit}
          onChange={(event) => setLimit(event.target.value)} /></label>
      <label><input type="checkbox" checked={stopOnError} onChange={(event) => setStopOnError(event.target.checked)} /> Stop queue if an experiment fails or finishes with errors</label>
      <p className="muted">A matching stored run is skipped, including failed or partial runs. No automatic retries.
        The queue continues when this tab closes; keep the API and oMLX running. A server restart ends the queue.</p>
      <button className="primary" disabled={busy || disabled || !ready.length || (limit !== "" && (!Number.isInteger(Number(limit)) || Number(limit) < 2 || Number(limit) % 2 !== 0))}
        onClick={start}>{busy ? "Starting queue..." : `Run shown missing sequentially (${ready.length})`}</button>
    </>}
    {queue && <>
      <p role="status"><b>{queue.status.replaceAll("_", " ")}</b> · {queue.finished} / {queue.total} experiments processed</p>
      {queue.active_job && <p>{queue.active_job.run_name}<br />
        {queue.active_job.current ?? 0} / {queue.active_job.total ?? "…"} emails · {queue.active_job.message}</p>}
      {active && <button disabled={busy || queue.status === "stopping"} onClick={async () => {
        setBusy(true); setError("");
        try { setQueue(await api.stopRunQueue()); }
        catch (err) { setError(err.message); }
        finally { setBusy(false); }
      }}>{queue.status === "stopping" ? "Stopping and saving..." : "Stop queue and save current results"}</button>}
      {queue.error && <div className="alert alert-danger">{queue.error}</div>}
      <details><summary>Queue details</summary><div className="table-scroll"><table className="data-table">
        <thead><tr><th>Configuration</th><th>Dataset</th><th>Split</th><th>Status</th><th>Message</th></tr></thead>
        <tbody>{queue.items.map((item, index) => <tr key={index}><td>{item.config_id}</td><td>{item.dataset}</td>
          <td>{item.split}</td><td>{item.status.replaceAll("_", " ")}</td><td>{item.message || "—"}</td></tr>)}</tbody>
      </table></div></details>
    </>}
    {error && <div role="alert" className="alert alert-danger">{error}</div>}
    {pollError && <div role="alert" className="alert alert-danger">Queue status unavailable: {pollError}</div>}
  </section>;
}
