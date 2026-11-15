import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import { jobStatusMessage } from "./utils.js";

// Load metrics and records, ignoring responses for old selections.
export function useRunData(runId) {
  const [state, setState] = useState({ metrics: null, records: null, loading: false, error: null });

  useEffect(() => {
    if (!runId) {
      setState({ metrics: null, records: null, loading: false, error: null });
      return;
    }
    let cancelled = false;
    setState({ metrics: null, records: null, loading: true, error: null });
    Promise.all([api.runMetrics(runId), api.runRecords(runId)])
      .then(([metrics, records]) => {
        if (!cancelled) setState({ metrics, records, loading: false, error: null });
      })
      .catch((err) => {
        if (!cancelled) setState({ metrics: null, records: null, loading: false, error: err.message });
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  return state;
}

// Load comparison metrics while retaining partial successes.
export function useRunsMetrics(runIds) {
  // Compare the selected IDs by value, even when the caller creates a new array.
  const selectionKey = JSON.stringify(runIds);
  const [metricsByRun, setMetricsByRun] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setError(null);
    const selectedRunIds = JSON.parse(selectionKey);
    if (!selectedRunIds.length) {
      setLoading(false);
      setMetricsByRun({});
      return;
    }
    let cancelled = false;
    setLoading(true);
    // Keep available runs visible when one selected run cannot be loaded.
    Promise.allSettled(selectedRunIds.map((runId) => api.runMetrics(runId))).then((results) => {
      if (cancelled) return;
      setMetricsByRun(Object.fromEntries(selectedRunIds.map((runId, index) => [runId, results[index].value ?? null])));
      const failures = results.flatMap((result, index) =>
        result.status === "rejected" ? [`${selectedRunIds[index]}: ${result.reason.message}`] : []);
      setError(failures.length ? failures.join("; ") : null);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [selectionKey]);

  return { metricsByRun, loading, error };
}

// Start and monitor the active experiment job.
export function useExperimentJob(onComplete) {
  const [jobId, setJobId] = useState(() => localStorage.getItem("activeExperimentJob"));
  const [jobMessage, setJobMessage] = useState(null);
  const [stopping, setStopping] = useState(false);

  useEffect(() => {
    if (jobId) localStorage.setItem("activeExperimentJob", jobId);
    else localStorage.removeItem("activeExperimentJob");
  }, [jobId]);

  const stopRun = useCallback(async () => {
    if (!jobId) return;
    setStopping(true);
    try {
      await api.stopJob(jobId);
      setJobMessage({ type: "info", text: "Stopping after the current email. Completed results are saved." });
    } catch (error) {
      setStopping(false);
      setJobMessage({ type: "danger", text: `Could not stop run: ${error.message}` });
    }
  }, [jobId]);

  // Submit an experiment and expose its polling status.
  const startRun = useCallback(async (payload) => {
    try {
      const job = await api.startRun(payload);
      setJobId(job.job_id);
      setStopping(false);
      setJobMessage({ type: "info", text: "Run started, waiting for progress..." });
    } catch (error) {
      setJobMessage({ type: "danger", text: `Could not start run: ${error.message}` });
    }
  }, []);

  useEffect(() => {
    if (!jobId) return;
    const controller = new AbortController();
    let timer;
    // Wait for each request to finish before scheduling another, avoiding overlap.
    async function poll() {
      try {
        const job = await api.jobStatus(jobId, controller.signal);
        if (controller.signal.aborted) return;
        setJobMessage(jobStatusMessage(job));
        setStopping(job.status === "stopping");
        if (["completed", "completed_with_errors", "failed", "stopped"].includes(job.status)) {
          setJobId(null);
          onComplete();
        } else {
          timer = setTimeout(poll, 2000);
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setJobMessage({ type: "danger", text: `Lost contact with API: ${error.message}` });
          if (error.message.startsWith("404 ")) setJobId(null);
          else timer = setTimeout(poll, 2000);
        }
      }
    }
    poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [jobId, onComplete]);

  return { startRun, stopRun, stopping, jobMessage, jobRunning: Boolean(jobId) };
}
