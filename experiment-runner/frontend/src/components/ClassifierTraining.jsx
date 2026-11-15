import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";

export default function ClassifierTraining({ datasets, onExperimentReady }) {
  const [mode, setMode] = useState("train");
  const [name, setName] = useState("");
  const [files, setFiles] = useState([]);
  const [classifiers, setClassifiers] = useState([]);
  const [modelPath, setModelPath] = useState("");
  const [datasetPath, setDatasetPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);

  const refreshClassifiers = useCallback(() => {
    api.classifiers().then(setClassifiers)
      .catch((error) => setMessage({ type: "danger", text: error.message }));
  }, []);
  useEffect(refreshClassifiers, [refreshClassifiers]);

  const filenames = files.map((file) => file.name);
  const validName = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name);
  const hasSavedSplits = ["run_config.json", "train.jsonl", "test.jsonl"]
    .every((filename) => filenames.includes(filename));
  const ready = validName && (mode === "train" ? hasSavedSplits : modelPath && datasetPath);

  async function prepareExperiment() {
    setBusy(true);
    setMessage(null);
    try {
      let result;
      if (mode === "train") {
        const contents = Object.fromEntries(await Promise.all(
          files.map(async (file) => [file.name, await file.text()])
        ));
        result = await api.trainClassifier({ name, files: contents });
        setMessage({ type: "success", text:
          `Trained on ${result.trained_records} emails with seed ${result.seed}. ` +
          `${result.counts.test} test emails are ready. The new experiment is selected below.` });
        refreshClassifiers();
      } else {
        result = await api.configureClassifier({ name, model_path: modelPath, dataset_path: datasetPath });
        setMessage({ type: "success", text:
          "Experiment created using the saved classifier without retraining. It is selected below." });
      }
      onExperimentReady(result);
    } catch (error) {
      setMessage({ type: "danger", text: error.message });
    } finally {
      setBusy(false);
    }
  }

  return <>
    <select className="field" aria-label="Classifier setup" value={mode} disabled={busy}
      onChange={(event) => { setMode(event.target.value); setMessage(null); }}>
      <option value="train">Train on saved Colab splits</option>
      <option value="saved">Use saved classifier (no retraining)</option>
    </select>
    <input className="field" aria-label="Classifier experiment name" placeholder="Unique experiment name"
      value={name} onChange={(event) => setName(event.target.value)} disabled={busy} maxLength={64} />
    {mode === "train" ? <>
      <p className="panel-hint">
        Select train.jsonl, test.jsonl, run_config.json and optionally validation.jsonl
        from one phishing_origin Colab run. Seed 42; existing splits are kept.
      </p>
      <input className="field" type="file" accept=".json,.jsonl" multiple
        aria-label="Saved Colab run files" disabled={busy}
        onChange={(event) => { setFiles(Array.from(event.target.files)); setMessage(null); }} />
      <p className="panel-hint">
        Learns phishing versus legitimate from words and word pairs. Only train.jsonl is used
        for fitting; test emails are reserved for experiments.
      </p>
    </> : <>
      <p className="panel-hint">
        Place your fitted TF-IDF pipeline (.joblib, saved by this project) in the backend models/
        folder and refresh files. This uses its existing weights without training again.
      </p>
      <button className="w-full" onClick={refreshClassifiers} disabled={busy}>Refresh classifier files</button>
      <select className="field" aria-label="Saved classifier" value={modelPath} disabled={busy}
        onChange={(event) => setModelPath(event.target.value)}>
        <option value="">Select saved classifier</option>
        {classifiers.map((model) => <option key={model.value} value={model.value}>{model.label}</option>)}
      </select>
      <select className="field" aria-label="Classifier test dataset" value={datasetPath} disabled={busy}
        onChange={(event) => setDatasetPath(event.target.value)}>
        <option value="">Select evaluation dataset</option>
        {datasets.filter((dataset) => dataset.tasks?.includes("phishing")).map((dataset) => <option key={dataset.value} value={dataset.value}>{dataset.label}</option>)}
      </select>
      <p className="panel-hint">Use the fixed PhishFuzzer and E-PhishLLM test sets for the main LLM comparison. Cross-Model measures phishing recall only. This classifier does not predict origin or injection.</p>
    </>}
    <button className="primary w-full" disabled={!ready || busy} onClick={prepareExperiment}>
      {busy ? "Preparing classifier experiment..." : mode === "train" ? "Train TF-IDF classifier" : "Create experiment with saved classifier"}
    </button>
    {message && <div role="status" className={`alert alert-${message.type}`}>{message.text}</div>}
  </>;
}
