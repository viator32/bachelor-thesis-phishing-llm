export const THESIS_METRICS = [
  ["phishing.accuracy", "Accuracy"],
  ["phishing.precision", "Precision"],
  ["phishing.recall", "Recall (Sensitivity / TPR)"],
  ["phishing.f1", "F1-Wert"],
  ["phishing.specificity", "Specificity (TNR)"],
  ["phishing.fpr", "False Positive Rate"],
  ["phishing.fnr", "False Negative Rate"],
  ["phishing.balanced_accuracy", "Balanced Accuracy"],
  ["phishing.mcc", "Matthews Correlation Coefficient (MCC)"],
  ["generation_origin.macro_f1", "Macro-F1 (Human / AI)"],
  ["generation_origin.balanced_accuracy", "Balanced Accuracy (Human / AI)"],
  ["json_initial_validity_rate", "JSON-Validitätsrate (initial)"],
  ["json_final_validity_rate", "Finale Validitätsrate"],
  ["manipulation_detection_rate", "Manipulationserkennungsrate"],
  [
    "paired_robustness.attack_success_rate_warning_suppression",
    "Gepaarte Warnunterdrückung (Label true → false)",
  ],
  ["control_group_fpr.fpr", "False-Positive-Rate (Kontrollgruppe)"],
  ["latency_ms.median", "Latenz (Median, ms)"],
  ["model_load_ms.median", "Modellladezeit (Median, ms)"],
  ["memory_delta_mb.median", "RSS-Änderung (Median, MiB)"],
  ["throughput_records_per_min.rate", "Durchsatz (Records/min)"],
];

// Formulas describe the values returned by metrics.py; units are explicit, never inferred from names.
export const METRIC_DETAILS = {
  "phishing.accuracy": ["ratio", "(TP + TN) / N; N = TP + TN + FP + FN"],
  "phishing.precision": ["ratio", "TP / (TP + FP)"],
  "phishing.recall": ["ratio", "TP / (TP + FN)"],
  "phishing.f1": ["ratio", "2TP / (2TP + FP + FN)"],
  "phishing.specificity": ["ratio", "TN / (TN + FP)"],
  "phishing.fpr": ["percent", "FP / (FP + TN)"],
  "phishing.fnr": ["percent", "FN / (FN + TP)"],
  "phishing.balanced_accuracy": ["ratio", "(Recall + Specificity) / 2"],
  "phishing.mcc": ["−1 to 1", "(TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))"],
  "generation_origin.macro_f1": ["ratio", "Mean of class F1 for human and ai; unknown predictions count as abstentions"],
  "generation_origin.balanced_accuracy": ["ratio", "Mean recall over human and ai; unavailable if a reference class is absent"],
  "json_initial_validity_rate": ["percent", "Fully valid initial reports / available initial reports; calls without a report excluded"],
  "json_final_validity_rate": ["percent", "Fully valid final reports / records with initial report; repair report takes precedence"],
  "manipulation_detection_rate": ["percent", "Injection TP / (Injection TP + Injection FN); supported tasks with predictions only"],
  "paired_robustness.attack_success_rate_warning_suppression": ["percent", "Injected phishing false / annotated phishing pairs correctly detected in the clean variant"],
  "control_group_fpr.fpr": ["percent", "FP / (FP + TN) within records marked control_group"],
  "latency_ms.median": ["ms", "Median measured method duration; general detector includes configured repair calls"],
  "model_load_ms.median": ["ms", "Median available load_duration_ms; external servers may not report it"],
  "memory_delta_mb.median": ["MiB", "Median (RSS after − RSS before); not peak unified memory"],
  "throughput_records_per_min.rate": ["records/min", "60 × record count / (last − first result timestamp in seconds); excludes first inference span"],
};

// Format a metric using its declared unit.
export function formatMetric(key, value) {
  return fmt(value, METRIC_DETAILS[key]?.[0] === "percent");
}

// Look up a dotted object path.
export function getNested(obj, dottedKey) {
  return dottedKey.split(".").reduce((node, part) => (node == null ? null : node[part]), obj);
}

// Format values with optional ratio-to-percent conversion.
export function fmt(value, asPercent = false) {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "boolean") return String(value);
  if (typeof value === "number") return asPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
  return String(value);
}

// Two configs can truncate to the same string, which would silently merge two
// runs into one category. Number the collisions instead.
// Number collisions between shortened run labels.
export function runAxisLabels(runs) {
  const seen = new Map();
  return runs.map((run) => {
    const base = run.task_label && run.dataset
      ? `${truncateMiddle(run.config_id, 24)} · ${run.task_label} · ${run.dataset} (${run.split})`
      : truncateMiddle(run.config_id, 28);
    const count = (seen.get(base) ?? 0) + 1;
    seen.set(base, count);
    return count === 1 ? base : `${base} (${count})`;
  });
}

// Shorten a string while retaining both ends.
export function truncateMiddle(value, max) {
  const text = String(value ?? "");
  if (text.length <= max) return text;
  const head = Math.ceil((max - 1) / 2);
  return `${text.slice(0, head)}…${text.slice(text.length - (max - 1 - head))}`;
}

// Recharts hands raw floats to tooltips; 0.8666666666666667 is not a readable value.
// Format finite chart values with fixed precision.
export function chartNumber(value, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "N/A";
  return Number(value).toFixed(digits);
}

// Serialize rows with CSV quoting.
export function toCSV(rows) {
  if (!rows.length) return "";
  const columns = Object.keys(rows[0]);
  // Quote a CSV value containing commas, quotes or newlines.
  const escape = (value) => {
    const str = value === null || value === undefined ? "" : String(value);
    return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
  };
  const lines = [columns.join(",")];
  for (const row of rows) lines.push(columns.map((c) => escape(row[c])).join(","));
  return lines.join("\n");
}

// Escape text before inserting it into exported HTML.
export function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

// Keep every backend metric available, including notes and unavailable values.
// Arrays remain one JSON value so matrix labels and counts stay together.
export function metricEntries(value, prefix = "") {
  if (value !== null && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length) {
    return Object.entries(value).flatMap(([key, child]) =>
      metricEntries(child, prefix ? `${prefix}.${key}` : key));
  }
  return [[prefix, value]];
}

export function metricText(value) {
  if (value == null) return "N/A";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

// Build an export with metric formulas and confusion matrix.
export function buildReportHtml(runId, metrics) {
  runId = escapeHtml(runId);
  const phishing = metrics.phishing || {};
  const rows = THESIS_METRICS.map(([key, label]) => {
    const [unit, formula] = METRIC_DETAILS[key];
    return `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(formatMetric(key, getNested(metrics, key)))}</td><td>${escapeHtml(unit)}</td><td>${escapeHtml(formula)}</td></tr>`;
  }).join("");
  const [tn, fp] = phishing.confusion_matrix?.[0] || ["N/A", "N/A"];
  const [fn, tp] = phishing.confusion_matrix?.[1] || ["N/A", "N/A"];
  const completeRows = metricEntries(metrics).map(([path, value]) =>
    `<tr><td>${escapeHtml(path)}</td><td>${escapeHtml(metricText(value))}</td></tr>`).join("");
  return `<!doctype html>
<html><head><meta charset="utf-8"><title>${runId} report</title>
<style>
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 2rem; }
table { border-collapse: collapse; width: 100%; max-width: 640px; }
td, th { border: 1px solid #e1e0d9; padding: 6px 10px; text-align: left; }
</style></head>
<body>
<h1>LLM Phishing Evaluation Report</h1>
<h2>Run: ${runId}</h2>
<h3>Thesis metrics (§5.2)</h3>
<table><tr><th>Metrik</th><th>Wert</th><th>Unit</th><th>Formula / denominator</th></tr>${rows}</table>
<h3>Confusion matrix</h3>
<table>
<tr><th></th><th>Pred: legitimate</th><th>Pred: phishing</th></tr>
<tr><th>True: legitimate</th><td>${tn}</td><td>${fp}</td></tr>
<tr><th>True: phishing</th><td>${fn}</td><td>${tp}</td></tr>
</table>
<h3>Complete metrics (including all tasks and saved native report)</h3>
<table><tr><th>Metric path</th><th>Value</th></tr>${completeRows}</table>
</body></html>`;
}

// Trigger a browser download and release its temporary URL.
export function downloadBlob(filename, content, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

// Average finite numeric values.
export function mean(values) {
  const measuredValues = values.filter(Number.isFinite);
  if (!measuredValues.length) return null;
  return measuredValues.reduce((sum, value) => sum + value, 0) / measuredValues.length;
}

// Match NumPy linear percentile interpolation.
export function percentile(values, p) {
  const clean = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!clean.length) return null;
  // Match NumPy's default linear interpolation, including even-sized medians.
  const index = ((clean.length - 1) * Math.max(0, Math.min(100, p))) / 100;
  const lower = Math.floor(index);
  const weight = index - lower;
  return clean[lower] + weight * (clean[Math.ceil(index)] - clean[lower]);
}

// Score rows whose truth and prediction are booleans.
export function binaryMetrics(rows, trueKey, predKey) {
  const scored = rows.filter((r) => typeof r[trueKey] === "boolean" && typeof r[predKey] === "boolean");
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (const r of scored) {
    if (r[trueKey] && r[predKey]) tp += 1;
    else if (!r[trueKey] && r[predKey]) fp += 1;
    else if (!r[trueKey] && !r[predKey]) tn += 1;
    else fn += 1;
  }
  return metricsFromCounts(tp, fp, tn, fn);
}

// Group rows by a field, using unknown for missing values.
export function groupBy(rows, key) {
  const groups = new Map();
  for (const row of rows) {
    const groupValue = row[key] ?? "unknown";
    if (!groups.has(groupValue)) groups.set(groupValue, []);
    groups.get(groupValue).push(row);
  }
  return groups;
}

// Keep API job states and their user-facing messages in one place.
// Translate API job state into a dashboard notification.
export function jobStatusMessage(job) {
  if (job.status === "stopped") {
    return { type: "warning", text: `Run ${job.run_id} stopped. ${job.records} completed email results saved. Use the same Run ID and settings to resume.` };
  }
  if (job.status === "stopping") {
    return { type: "info", text: `Stopping run ${job.run_id} after the current email. ${job.current || 0} results saved.` };
  }
  if (job.status === "failed") {
    return { type: "danger", text: `Run failed: ${job.error || "no model response"}` };
  }
  if (job.status === "completed") {
    return {
      type: "success",
      text: `Run ${job.run_id} completed: ${job.successful}/${job.records} successful, ${job.model_responses} model responses.`,
    };
  }
  if (job.status === "completed_with_errors") {
    return {
      type: "warning",
      text: `Run ${job.run_id} completed with errors: ${job.successful}/${job.records} successful, ${job.failed} failed, ${job.timeouts} timed out, ${job.model_responses} model responses.`,
    };
  }
  const detail = job.total ? `${job.current}/${job.total}` : String(job.current || 0);
  return { type: "info", text: `Run ${job.run_id}: ${detail} emails. ${job.message || ""}` };
}

// Match backend zero-division conventions when records exist; no records remains N/A.
// Calculate binary metrics using backend zero-division rules.
export function metricsFromCounts(tp, fp, tn, fn) {
  const n = tp + fp + tn + fn;
  // Divide counts while preserving the no-records state.
  const divide = (numerator, denominator) => {
    if (!n) return null;
    return denominator ? numerator / denominator : 0;
  };
  return {
    n, tp, fp, tn, fn,
    precision: divide(tp, tp + fp),
    recall: divide(tp, tp + fn),
    f1: divide(2 * tp, 2 * tp + fp + fn),
    accuracy: divide(tp + tn, n),
  };
}
