import { useEffect, useState } from "react";
import ConfigPicker from "./ConfigPicker.jsx";
import ClassifierTraining from "./ClassifierTraining.jsx";
import { buildRunCoverage } from "../runCoverage.js";

export default function NewRunForm({ runs, configs, datasets, configsLoading, onRefreshConfigs, preparedRun, onClassifierReady, onStartRun, onStopRun, stopping, jobMessage, jobRunning, queueRunning }) {
  const [configPath, setConfigPath] = useState("");
  const [datasetPath, setDatasetPath] = useState("");
  const [limits, setLimits] = useState(() => {
    try { return JSON.parse(localStorage.getItem("datasetEmailLimits") || "{}"); }
    catch { return {}; }
  });
  useEffect(() => {
    localStorage.setItem("datasetEmailLimits", JSON.stringify(limits));
  }, [limits]);
  const [runId, setRunId] = useState("");
  const [split, setSplit] = useState("test");
  const [missingOnly, setMissingOnly] = useState(true);
  const [datasetManifest, setDatasetManifest] = useState("");
  const missingRuns = buildRunCoverage(configs, runs, split).filter((row) => row.status === "Missing");
  const visibleConfigs = missingOnly ? configs.map((config) => ({
    ...config,
    datasets: missingRuns.filter((row) => row.config.value === config.value).map((row) => row.dataset),
  })).filter((config) => config.datasets.length > 0) : configs;
  const selectedConfig = visibleConfigs.find((config) => config.value === configPath);
  const isTaskRun = selectedConfig?.kind === "llm";
  const taskDatasets = selectedConfig?.datasets || [];
  const taskDataset = taskDatasets.find((dataset) => dataset.value === datasetManifest) || taskDatasets[0];
  const baselineDatasets = (selectedConfig?.datasets || [])
    .filter((dataset) => dataset.splits?.[split] > 0);
  const baselineDataset = baselineDatasets.find((dataset) => dataset.value === datasetPath)
    || baselineDatasets.find((dataset) => dataset.value === selectedConfig?.dataset_path) || baselineDatasets[0];
  const effectiveDatasetPath = baselineDataset?.value || "";
  const isExternal = isTaskRun && taskDataset?.external;
  const effectiveSplit = isTaskRun && taskDataset?.splits[split] === undefined ? "test" : split;
  const limitKey = isTaskRun ? `${taskDataset?.value}:${effectiveSplit}` : effectiveDatasetPath || configPath;
  const limit = limits[limitKey] ?? "";
  const setLimit = (value) => setLimits((previous) => ({ ...previous, [limitKey]: value }));

  useEffect(() => {
    if (!preparedRun) return;
    setConfigPath(preparedRun.config.value);
    setDatasetManifest(preparedRun.dataset.value);
    setDatasetPath(preparedRun.dataset.value);
    setSplit(preparedRun.split);
    setMissingOnly(preparedRun.status === "Missing");
    setRunId("");
  }, [preparedRun]);

  return <>
      <details className="panel"><summary>Train or register a TF-IDF classifier</summary>
        <ClassifierTraining datasets={datasets} onExperimentReady={(result) => {
          setMissingOnly(false);
          setConfigPath(result.config_path);
          setDatasetPath(result.dataset_path);
          setLimits((previous) => ({ ...previous, [result.dataset_path]: "" }));
          setRunId("");
          onClassifierReady();
        }} />
      </details>

      <section className="new-run-form">
        <button className="w-full" onClick={onRefreshConfigs} disabled={configsLoading}>
          {configsLoading ? "Checking models..." : "Refresh models"}
        </button>
        <label className="checklist">
          <span><input type="checkbox" checked={missingOnly} onChange={(e) => setMissingOnly(e.target.checked)} /> Missing runs only</span>
        </label>
        {missingOnly && <>
          <select className="field" aria-label="Missing runs split" value={split} onChange={(e) => setSplit(e.target.value)}>
            <option value="test">Test</option>
            <option value="validation">Validation (setup)</option>
          </select>
          <p className="panel-hint">{missingRuns.length} experiment–dataset combinations have no stored {split} run.
            Includes models, rules and classifiers. Any stored run counts, including failed or stopped runs. Turn this filter off to repeat runs.</p>
        </>}
        {missingOnly && !visibleConfigs.length
          ? <p className="panel-hint">{configsLoading ? "Checking model configurations..." : configs.length ? "No missing combinations for this split." : "No model configurations loaded."}</p>
          : <ConfigPicker configs={visibleConfigs} value={selectedConfig?.value || ""} onChange={setConfigPath} />}
        {isTaskRun ? <>
          <select className="field" aria-label="Evaluation dataset" value={taskDataset?.value || ""}
            onChange={(e) => setDatasetManifest(e.target.value)}>
            {taskDatasets.map((dataset) => <option key={dataset.value} value={dataset.value}>
              {dataset.label}
            </option>)}
          </select>
          {!missingOnly && <select className="field" aria-label="Evaluation split" value={effectiveSplit}
            onChange={(e) => setSplit(e.target.value)}>
            {Object.entries(taskDataset?.splits || {}).map(([name, rows]) => (
              <option key={name} value={name}>{name === "validation" ? "Validation (setup)" : "Test"} — {rows.toLocaleString()} emails</option>
            ))}
          </select>}
          <p className="panel-hint">{isExternal
            ? "Random sample across all available label combinations, preserving their proportions. The same maximum and seed select the same emails across models."
            : "Different examples from the training source. Partial limits must be even and preserve pairs and strata."}</p>
        </> : selectedConfig && <><select
          className="field"
          aria-label="Baseline evaluation dataset"
          value={effectiveDatasetPath}
          onChange={(e) => setDatasetPath(e.target.value)}
        >
          {baselineDatasets.map((d) => (
            <option key={d.value} value={d.value}>
              {d.label}
            </option>
          ))}
        </select>
          {!missingOnly && <select className="field" aria-label="Baseline evaluation split" value={split} onChange={(e) => setSplit(e.target.value)}>
            <option value="test">Test</option><option value="validation">Validation (setup)</option>
          </select>}
        </>}
        {!isTaskRun && selectedConfig && <p className="panel-hint">Limited runs randomly sample across available labels instead of taking the first emails.
          {baselineDataset?.fixed_comparison && " Leave Max emails empty to evaluate exactly the saved LLM test cases, including all languages."}
          {selectedConfig.availability === "train classifier first" && " Train this classifier in the TF-IDF classifier panel first."}</p>}
        <div className="field-row">
          <input
            className="field"
            type="number"
            min={isTaskRun && !isExternal ? "2" : "1"}
            step={isTaskRun && !isExternal ? "2" : "1"}
            placeholder="Max emails (all if empty)"
            aria-label="Maximum emails for selected dataset"
            value={limit}
            onChange={(e) => setLimit(e.target.value)}
          />
          <input
            className="field"
            type="text"
            placeholder="Run ID"
            aria-label="Existing run ID to resume"
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
          />
        </div>
        <p className="panel-hint">
          Leave Run ID empty to name a new run automatically with its configuration, task,
          dataset and split; enter an existing ID to resume it without
          overwriting stored records.
        </p>
        <button
          className="primary w-full"
          disabled={!selectedConfig || jobRunning || queueRunning || (isTaskRun ? !taskDataset : !baselineDataset)
            || selectedConfig?.availability === "train classifier first"}
          onClick={() =>
            onStartRun({
              config_path: configPath,
              dataset_path: isTaskRun ? undefined : effectiveDatasetPath || undefined,
              dataset_manifest: isTaskRun ? taskDataset?.value : undefined,
              split: effectiveSplit,
              limit: limit ? Number(limit) : undefined,
              run_id: runId || undefined,
            })
          }
        >
          {jobRunning ? "Run in progress..." : "Start run"}
        </button>
        {jobRunning && <button className="w-full" onClick={onStopRun} disabled={stopping}>
          {stopping ? "Stopping and saving..." : "Stop run and save results"}
        </button>}
        {jobMessage && <div className={`alert alert-${jobMessage.type}`}>{jobMessage.text}</div>}
      </section>

  </>;
}
