const BASE_URL = import.meta.env?.VITE_API_BASE_URL || "http://localhost:8000";

// Send an API request and reject unsuccessful responses.
async function request(path, options, responseType = "json") {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  }
  return res.status === 204 ? null : responseType === "blob" ? res.blob() : res.json();
}

export const api = {
  allRunsExport: () => request("/runs/export", undefined, "blob"),
  runExport: (runId) => request(`/runs/${encodeURIComponent(runId)}/export`, undefined, "blob"),
  listRuns: () => request("/runs"),
  runMetrics: (runId) => request(`/runs/${encodeURIComponent(runId)}/metrics`),
  runRecords: (runId) => request(`/runs/${encodeURIComponent(runId)}/records`),
  runInstance: (runId, instanceId) => request(`/runs/${encodeURIComponent(runId)}/instances/${encodeURIComponent(instanceId)}`),
  getNotes: (runId) => request(`/runs/${encodeURIComponent(runId)}/notes`),
  putNotes: (runId, note) =>
    request(`/runs/${encodeURIComponent(runId)}/notes`, { method: "PUT", body: JSON.stringify({ note }) }),
  deleteRun: (runId) => request(`/runs/${encodeURIComponent(runId)}`, { method: "DELETE" }),
  deleteAllRuns: () => request("/runs?confirm=DELETE%20ALL", { method: "DELETE" }),
  configs: () => request("/experiment-configs"),
  datasets: () => request("/datasets"),
  trainClassifier: (payload) => request("/classifiers/tfidf/train", {
    method: "POST", body: JSON.stringify(payload),
  }),
  classifiers: () => request("/classifiers/tfidf"),
  configureClassifier: (payload) => request("/classifiers/tfidf/experiments", {
    method: "POST", body: JSON.stringify(payload),
  }),
  startRun: (payload) => request("/runs", { method: "POST", body: JSON.stringify(payload) }),
  runQueue: (signal) => request("/run-queue", { signal }),
  startRunQueue: (payload) => request("/run-queue", { method: "POST", body: JSON.stringify(payload) }),
  stopRunQueue: () => request("/run-queue/stop", { method: "POST" }),
  jobStatus: (jobId, signal) => request(`/jobs/${encodeURIComponent(jobId)}`, { signal }),
  stopJob: (jobId) => request(`/jobs/${encodeURIComponent(jobId)}/stop`, { method: "POST" }),
};
