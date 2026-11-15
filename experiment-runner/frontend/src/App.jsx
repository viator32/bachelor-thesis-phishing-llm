import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api.js";
import { useExperimentJob, useRunData, useRunsMetrics } from "./hooks.js";
import Sidebar from "./components/Sidebar.jsx";
import Kpis from "./components/Kpis.jsx";
import { buildReportHtml, downloadBlob, toCSV } from "./utils.js";

// Load each tab on demand; chart code stays out of the initial application bundle.
const EmailsTab = lazy(() => import("./tabs/EmailsTab.jsx"));
const CompareTab = lazy(() => import("./tabs/CompareTab.jsx"));
const QualityTab = lazy(() => import("./tabs/QualityTab.jsx"));
const ResourcesTab = lazy(() => import("./tabs/ResourcesTab.jsx"));
const ThesisTab = lazy(() => import("./tabs/ThesisTab.jsx"));
const CurvesTab = lazy(() => import("./tabs/CurvesTab.jsx"));
const ExploreTab = lazy(() => import("./tabs/ExploreTab.jsx"));
const RunsTab = lazy(() => import("./tabs/RunsTab.jsx"));

// `scope` states which run selection a tab reads, which in turn drives the run
// picker's caption, the KPI strip and which sidebar sections open.
const TABS = [
  { id: "coverage", label: "Runs overview", scope: "all" },
  { id: "compare", label: "Compare", scope: "compare" },
  { id: "resources", label: "Resources", scope: "compare" },
  { id: "quality", label: "Quality", scope: "primary" },
  { id: "curves", label: "ROC / PR", scope: "primary" },
  { id: "thesis", label: "Thesis metrics", scope: "primary" },
  { id: "explore", label: "Explore", scope: "primary" },
  { id: "email", label: "Emails", scope: "primary" },
];

// Render the evaluation dashboard and manage selections.
export default function App() {
  const [runs, setRuns] = useState([]);
  const [configs, setConfigs] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [configsLoading, setConfigsLoading] = useState(false);
  const [selectedRuns, setSelectedRuns] = useState([]);
  const [primaryRun, setPrimaryRun] = useState(null);
  const [instanceId, setInstanceId] = useState(null);
  const [metricKeys, setMetricKeys] = useState(["f1", "recall", "precision"]);
  // The tab lives in the URL hash, so a view can be bookmarked, reloaded and
  // linked to without losing which tab it was.
  const [activeTab, setActiveTab] = useState(
    () => TABS.find((t) => t.id === window.location.hash.slice(1))?.id ?? "coverage"
  );
  const [notes, setNotes] = useState("");
  const [notesSaved, setNotesSaved] = useState(false);
  const [error, setError] = useState(null);
  const [exportingAllRuns, setExportingAllRuns] = useState(false);
  const tabRefs = useRef({});

  useEffect(() => {
    window.location.hash = activeTab;
  }, [activeTab]);

  useEffect(() => {
    // Select the tab identified by the current URL hash.
    const onHashChange = () => {
      const id = TABS.find((t) => t.id === window.location.hash.slice(1))?.id;
      if (id) setActiveTab(id);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  // Reload run summaries and initialize empty selections.
  const refreshRuns = useCallback(() => {
    setError(null);
    return api
      .listRuns()
      .then((list) => {
        setRuns(list);
        setSelectedRuns((prev) => (prev.length ? prev.filter((id) => list.some((run) => run.run_id === id)) : list.slice(0, 5).map((r) => r.run_id)));
        setPrimaryRun((prev) => list.some((run) => run.run_id === prev) ? prev : list[0]?.run_id || null);
      })
      .catch((error) => setError(error.message));
  }, []);

  // Recheck model availability on the server, bypassing cached statuses.
  const refreshConfigs = useCallback(() => {
    setConfigsLoading(true);
    return api.configs()
      .then(setConfigs)
      .catch((error) => setError(error.message))
      .finally(() => setConfigsLoading(false));
  }, []);

  useEffect(() => {
    refreshRuns();
    refreshConfigs();
    api.datasets().then(setDatasets).catch((error) => setError(error.message));
  }, [refreshRuns, refreshConfigs]);

  useEffect(() => {
    setNotes("");
    setNotesSaved(false);
    if (!primaryRun) return;
    let cancelled = false;
    api
      .getNotes(primaryRun)
      .then((result) => {
        if (!cancelled) setNotes(result.note);
      })
      .catch((error) => {
        if (!cancelled) setError(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, [primaryRun]);

  const { metrics, records, loading, error: runError } = useRunData(primaryRun);
  const { startRun, stopRun, stopping, jobMessage, jobRunning } = useExperimentJob(refreshRuns);

  function refreshAfterClassifierSetup() {
    refreshConfigs();
    api.datasets().then(setDatasets).catch((error) => setError(error.message));
  }

  const comparedRuns = useMemo(
    () => (selectedRuns.length ? selectedRuns : primaryRun ? [primaryRun] : []),
    [selectedRuns, primaryRun]
  );
  const { metricsByRun, loading: comparedLoading, error: comparisonError } = useRunsMetrics(comparedRuns);

  useEffect(() => {
    setInstanceId(records?.[0]?.instance_id ?? null);
  }, [records]);

  // Save notes for the focused run.
  const saveNotes = () => {
    if (!primaryRun) return;
    api
      .putNotes(primaryRun, notes)
      .then(() => setNotesSaved(true))
      .catch((error) => setError(error.message));
  };

  // Delete requested runs, preserve failures and refresh the database view once.
  const deleteRuns = async (ids) => {
    const deleted = [], failures = [];
    for (const id of ids) {
      try { await api.deleteRun(id); deleted.push(id); }
      catch (error) { failures.push(`${id}: ${error.message}`); }
    }
    setSelectedRuns((prev) => prev.filter((id) => !deleted.includes(id)));
    setPrimaryRun((prev) => deleted.includes(prev) ? null : prev);
    await refreshRuns();
    if (failures.length) throw new Error(`Deleted ${deleted.length} run(s). Failed to delete: ${failures.join("; ")}`);
    return deleted;
  };

  const deleteAllRuns = async () => {
    await api.deleteAllRuns();
    setSelectedRuns([]);
    setPrimaryRun(null);
    await refreshRuns();
  };

  // Download focused-run predictions.
  const downloadCSV = () =>
    records && downloadBlob(`${primaryRun}-predictions.csv`, toCSV(records), "text/csv");
  // Download focused-run metrics.
  const downloadMetrics = () =>
    metrics &&
    downloadBlob(
      `${primaryRun}-metrics.json`,
      JSON.stringify(metrics, null, 2),
      "application/json"
    );
  // Download the focused-run HTML report.
  const downloadReport = () =>
    metrics &&
    downloadBlob(`${primaryRun}-report.html`, buildReportHtml(primaryRun, metrics), "text/html");

  const downloadArchive = async () => {
    if (!primaryRun) return;
    try {
      const archive = await api.runExport(primaryRun);
      downloadBlob(`${primaryRun}-experiment.zip`, archive, "application/zip");
    } catch (error) {
      setError(error.message);
    }
  };

  const downloadAllRuns = async () => {
    if (!runs.length || exportingAllRuns) return;
    setExportingAllRuns(true);
    setError(null);
    try {
      const archive = await api.allRunsExport();
      downloadBlob("all-runs-results.zip", archive, "application/zip");
    } catch (error) {
      setError(error.message);
    } finally {
      setExportingAllRuns(false);
    }
  };

  const tabScope = TABS.find((t) => t.id === activeTab)?.scope ?? "primary";
  const focusedRun = runs.find((r) => r.run_id === primaryRun);

  // Move keyboard focus between tabs with the arrow keys.
  const onTabKeyDown = (event) => {
    const delta = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (!delta) return;
    event.preventDefault();
    const index = TABS.findIndex((t) => t.id === activeTab);
    const next = TABS[(index + delta + TABS.length) % TABS.length];
    setActiveTab(next.id);
    tabRefs.current[next.id]?.focus();
  };

  return (
    <div className={`layout${tabScope === "all" ? " layout-runs" : ""}`}>
      {tabScope !== "all" && <Sidebar
        runs={runs}
        selectedRuns={selectedRuns}
        onSelectedRunsChange={setSelectedRuns}
        primaryRun={primaryRun}
        onPrimaryRunChange={setPrimaryRun}
        metricKeys={metricKeys}
        onMetricKeysChange={setMetricKeys}
        notes={notes}
        onNotesChange={(value) => {
          setNotes(value);
          setNotesSaved(false);
        }}
        onSaveNotes={saveNotes}
        notesSaved={notesSaved}
        onDownloadArchive={downloadArchive}
        onDownloadCSV={downloadCSV}
        onDownloadMetrics={downloadMetrics}
        onDownloadReport={downloadReport}
        activeTab={activeTab}
        tabScope={tabScope}
      />}

      <div className="main">
        <div className="topbar">
          <h1>LLM Phishing Evaluation</h1>
          {jobRunning && <button onClick={() => setActiveTab("coverage")}>Run in progress — manage run</button>}
          <button onClick={refreshRuns}>Refresh runs</button>
        </div>

        {/* A screenshot of any tab carries the provenance of what it shows. */}
        <div className="context-strip">
          {tabScope === "all" ? <span className="ctx-item">Manage all stored runs and experiment configurations</span> : tabScope === "compare" ? (
            <span className="ctx-item">
              Comparing <b>{comparedRuns.length}</b> {comparedRuns.length === 1 ? "run" : "runs"}
            </span>
          ) : (
            <>
              <span className="ctx-item">
                Run <b>{focusedRun?.run_id ?? "—"}</b>
              </span>
              <span className="ctx-item">
                Config <b>{focusedRun?.config_id ?? "—"}</b>
              </span>
              <span className="ctx-item">
                Task <b>{focusedRun?.task_label ?? "—"}</b>
              </span>
              <span className="ctx-item">
                Dataset <b>{focusedRun?.dataset ?? "—"}</b> ({focusedRun?.split ?? "—"})
              </span>
              <span className="ctx-item">
                Model <b>{focusedRun?.model_id ?? "—"}</b>
              </span>
              <span className="ctx-item">
                Records <b>{focusedRun?.records ?? "—"}</b>
                {focusedRun && focusedRun.successful !== focusedRun.records
                  ? ` (${focusedRun.successful} ok)`
                  : ""}
              </span>
              <span className="ctx-item">
                Started <b>{focusedRun?.started_at?.slice(0, 19).replace("T", " ") ?? "—"}</b>
              </span>
            </>
          )}
          <span className="ctx-item" style={{ marginLeft: "auto" }}>
            Email content is rendered as inert text only
          </span>
        </div>

        {(error || runError || comparisonError) && (
          <div role="alert" className="alert alert-danger">
            {error || runError || comparisonError}
          </div>
        )}

        <div className="tabs" role="tablist" aria-label="Dashboard views" onKeyDown={onTabKeyDown}>
          {TABS.filter((t) => t.scope === "all").map((tab) => (
            <TabButton key={tab.id} tab={tab} active={activeTab} onSelect={setActiveTab} refs={tabRefs} />
          ))}
          <span className="tab-divider" aria-hidden="true" />
          <span className="tab-group-label">All checked runs</span>
          {TABS.filter((t) => t.scope === "compare").map((tab) => (
            <TabButton key={tab.id} tab={tab} active={activeTab} onSelect={setActiveTab} refs={tabRefs} />
          ))}
          <span className="tab-divider" aria-hidden="true" />
          <span className="tab-group-label">Focused run</span>
          {TABS.filter((t) => t.scope === "primary").map((tab) => (
            <TabButton key={tab.id} tab={tab} active={activeTab} onSelect={setActiveTab} refs={tabRefs} />
          ))}
        </div>

        {tabScope === "primary" && activeTab !== "email" && (
          <Kpis
            phishing={metrics?.phishing}
            caption={`Phishing classification for the focused run${
              focusedRun ? ` · ${focusedRun.run_name || focusedRun.config_id}` : ""
            }. Hover f(x) for each definition.`}
          />
        )}

        {!runs.length && activeTab !== "coverage" ? (
          <div className="alert alert-info">
            No experiment records yet. Open “Runs overview → Create new run”, or use <code>phishlab run --config ...</code>.
          </div>
        ) : (
          <div
            id={`panel-${activeTab}`}
            role="tabpanel"
            aria-labelledby={`tab-${activeTab}`}
            /* Hold the previous render rather than flashing an empty view. */
            className={tabScope !== "all" && (loading || comparedLoading) ? "is-stale" : undefined}
          >
            <Suspense
              fallback={
                <p className="muted" role="status">
                  Loading view...
                </p>
              }
            >
              {activeTab === "coverage" && <RunsTab
                runs={runs} configs={configs} configsLoading={configsLoading} datasets={datasets}
                onRefreshConfigs={refreshConfigs} onRefreshRuns={refreshRuns}
                onFocus={(id) => { setPrimaryRun(id); setActiveTab("quality"); }}
                onDeleteRuns={deleteRuns} onDeleteAllRuns={deleteAllRuns}
                onDownloadAllRuns={downloadAllRuns} exportingAllRuns={exportingAllRuns}
                onClassifierReady={refreshAfterClassifierSetup} onStartRun={startRun}
                onStopRun={stopRun} stopping={stopping} jobMessage={jobMessage} jobRunning={jobRunning}
                selectedRuns={selectedRuns} onSelectedRunsChange={setSelectedRuns} />}
              {activeTab === "email" && (
                <EmailsTab
                  runId={primaryRun}
                  records={records}
                  instanceId={instanceId}
                  onInstanceChange={setInstanceId}
                />
              )}
              {activeTab === "compare" && (
                <CompareTab
                  runs={runs}
                  selectedRuns={comparedRuns}
                  metricsByRun={metricsByRun}
                  metricKeys={metricKeys}
                />
              )}
              {activeTab === "quality" && <QualityTab metrics={metrics} />}
              {activeTab === "resources" && (
                <ResourcesTab
                  metrics={metrics}
                  records={records}
                  selectedRuns={comparedRuns}
                  metricsByRun={metricsByRun}
                  runs={runs}
                />
              )}
              {activeTab === "thesis" && <ThesisTab metrics={metrics} />}
              {activeTab === "curves" && <CurvesTab records={records} />}
              {activeTab === "explore" && <ExploreTab records={records} />}
            </Suspense>
          </div>
        )}
      </div>
    </div>
  );
}

// Render an accessible tab selector.
function TabButton({ tab, active, onSelect, refs }) {
  const selected = active === tab.id;
  return (
    <button
      id={`tab-${tab.id}`}
      role="tab"
      aria-selected={selected}
      aria-controls={`panel-${tab.id}`}
      tabIndex={selected ? 0 : -1}
      ref={(el) => {
        refs.current[tab.id] = el;
      }}
      onClick={() => onSelect(tab.id)}
    >
      {tab.label}
    </button>
  );
}
