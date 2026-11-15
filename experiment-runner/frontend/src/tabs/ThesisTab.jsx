import { useState } from "react";
import Heatmap from "../components/Heatmap.jsx";
import Unavailable from "../components/Unavailable.jsx";
import { THESIS_METRICS, METRIC_DETAILS, formatMetric, getNested, metricEntries, metricText } from "../utils.js";

// Sections mirror the thesis structure so the table is readable as a figure
// rather than as one undifferentiated 20-row block.
const SECTIONS = [
  ["Phishing classification", (key) => key.startsWith("phishing.")],
  ["Generation origin", (key) => key.startsWith("generation_origin.")],
  [
    "Output validity and robustness",
    (key) =>
      key.startsWith("json_") ||
      key.startsWith("manipulation_") ||
      key.startsWith("paired_robustness.") ||
      key.startsWith("control_group_fpr."),
  ],
  ["Resources", () => true],
];

// Render thesis metrics with their definitions.
export default function ThesisTab({ metrics }) {
  const [search, setSearch] = useState("");
  if (!metrics) return <div className="alert alert-info">Focus a run in the sidebar.</div>;

  const remaining = [...THESIS_METRICS];
  const sections = SECTIONS.map(([title, match]) => {
    const taken = remaining.filter(([key]) => match(key));
    for (const entry of taken) remaining.splice(remaining.indexOf(entry), 1);
    return [title, taken];
  });

  return (
    <div>
      <Unavailable unavailable={metrics.unavailable} />
      {Object.entries(metrics.comparison?.classification || {}).map(([task, result]) =>
        result.confusion_matrix && <Heatmap key={task} title={`${task}: valid supported predictions`}
          matrix={result.confusion_matrix} labels={result.labels.map(String)}
          note={`Rows = truth; columns = prediction. ${result.evaluated_records} scored / ${result.eligible_records} eligible; ${result.invalid_outputs} invalid or missing. Invalid outputs are excluded from this matrix.`} />)}
      {Object.entries(metrics.native_report?.per_task || {}).map(([task, result]) => {
        const matrix = result.confusion_matrix_with_invalid;
        if (!matrix) return null;
        const rows = Object.keys(matrix);
        const columns = Object.keys(matrix[rows[0]]);
        return <Heatmap key={`native-${task}`} title={`${task}: saved native matrix including invalid outputs`}
          labels={rows} columnLabels={columns} matrix={rows.map((row) => columns.map((col) => matrix[row][col]))} />;
      })}
      <div className="card">
      <h4>Thesis metrics (§5.2)</h4>
      <p className="metric-note">
        Computed metrics with their units, formulas and denominators. TP/FP count phishing
        predictions; TN/FN count legitimate predictions. N/A means unavailable; backend
        zero-denominator ratios use 0 where records exist.
      </p>
      <div className="table-scroll" style={{ maxHeight: "none" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Metrik</th>
              <th className="num">Wert</th>
              <th>Unit</th>
              <th style={{ whiteSpace: "normal" }}>Formula / denominator</th>
            </tr>
          </thead>
          {sections.map(([title, entries]) =>
            entries.length ? (
              <tbody key={title}>
                <tr>
                  <th colSpan={4} style={{ position: "static" }}>
                    {title}
                  </th>
                </tr>
                {entries.map(([key, label]) => {
                  const [unit, formula] = METRIC_DETAILS[key];
                  return (
                    <tr key={key}>
                      <td style={{ whiteSpace: "normal" }}>{label}</td>
                      <td className="num">{formatMetric(key, getNested(metrics, key))}</td>
                      <td>{unit}</td>
                      <td style={{ whiteSpace: "normal", maxWidth: "44ch" }}>{formula}</td>
                    </tr>
                  );
                })}
              </tbody>
            ) : null
          )}
        </table>
      </div>
      </div>
      <div className="card">
        <h4>Complete metric inventory</h4>
        <p className="metric-note">All returned metrics, denominators, per-class values and saved native report. N/A preserves missing values. Numeric paths are selectable in Compare.</p>
        <input aria-label="Filter metric paths" placeholder="Filter by task, metric or subgroup" value={search} onChange={(event) => setSearch(event.target.value)} />
        <div className="table-scroll"><table className="data-table">
          <thead><tr><th>Metric path</th><th>Value</th></tr></thead>
          <tbody>{metricEntries(metrics).filter(([path]) => path.toLowerCase().includes(search.toLowerCase())).map(([path, value]) =>
            <tr key={path}><td>{path}</td><td style={{ whiteSpace: "normal" }}>{metricText(value)}</td></tr>)}</tbody>
        </table></div>
      </div>
    </div>
  );
}
