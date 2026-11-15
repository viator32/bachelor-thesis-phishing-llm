import RunPicker from "./RunPicker.jsx";

const METRIC_OPTIONS = [
  "accuracy",
  "precision",
  "recall",
  "f1",
  "specificity",
  "balanced_accuracy",
  "mcc",
  "fpr",
  "fnr",
];

// Each section says which tabs read it, and opens only when one of them is
// active, so the column is not six always-expanded groups of unrelated controls.
// Render a sidebar section opened for relevant tabs.
function Panel({ title, usedBy, activeTab, children, forceOpen }) {
  const relevant = forceOpen || usedBy.includes(activeTab);
  return (
    <details className="panel" open={relevant}>
      <summary>
        {title}
        {relevant && !forceOpen && <span className="panel-scope">this view</span>}
      </summary>
      {children}
    </details>
  );
}

// Render comparison selections, metrics, notes and focused-run exports.
export default function Sidebar({
  runs,
  selectedRuns,
  onSelectedRunsChange,
  primaryRun,
  onPrimaryRunChange,
  metricKeys,
  onMetricKeysChange,
  notes,
  onNotesChange,
  onSaveNotes,
  notesSaved,
  onDownloadArchive,
  onDownloadCSV,
  onDownloadMetrics,
  onDownloadReport,
  activeTab,
  tabScope,
}) {
  // Toggle a metric in the comparison selection.
  const toggleMetric = (key) => {
    onMetricKeysChange(
      metricKeys.includes(key) ? metricKeys.filter((m) => m !== key) : [...metricKeys, key]
    );
  };

  return (
    <div className="sidebar">
      <details className="panel" open>
        <summary>Runs</summary>
        <RunPicker
          runs={runs}
          selectedRuns={selectedRuns}
          onSelectedRunsChange={onSelectedRunsChange}
          primaryRun={primaryRun}
          onPrimaryRunChange={onPrimaryRunChange}
          scope={tabScope}
        />
      </details>

      <Panel title="Compared metrics" usedBy={["compare"]} activeTab={activeTab}>
        <p className="panel-hint">Series drawn in the Compare bar chart.</p>
        <div className="checklist">
          {METRIC_OPTIONS.map((key) => (
            <label key={key}>
              <input
                type="checkbox"
                checked={metricKeys.includes(key)}
                onChange={() => toggleMetric(key)}
              />{" "}
              {key}
            </label>
          ))}
        </div>
      </Panel>

      <Panel title="Notes" usedBy={[]} activeTab={activeTab}>
        <p className="panel-hint">Saved against the focused run.</p>
        <textarea
          className="field"
          placeholder="Experiment intent, observations..."
          aria-label="Notes for the focused run"
          value={notes}
          onChange={(e) => onNotesChange(e.target.value)}
          disabled={!primaryRun}
        />
        <button className="w-full" disabled={!primaryRun} onClick={onSaveNotes}>
          {notesSaved ? "Saved" : "Save note"}
        </button>
      </Panel>

      <Panel title="Export focused run" usedBy={[]} activeTab={activeTab}>
        <button className="w-full" disabled={!primaryRun} onClick={onDownloadArchive}>
          Complete experiment ZIP
        </button>
        <button className="w-full" disabled={!primaryRun} onClick={onDownloadCSV}>
          Predictions CSV
        </button>
        <button className="w-full" disabled={!primaryRun} onClick={onDownloadMetrics}>
          metrics.json
        </button>
        <button className="w-full" disabled={!primaryRun} onClick={onDownloadReport}>
          HTML report
        </button>
      </Panel>
    </div>
  );
}
