import { useMemo, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Legend,
  ScatterChart,
  Scatter,
  LineChart,
  Line,
  LabelList,
} from "recharts";
import ChartCard, { CartesianGrid, ChartTooltip, axisProps, gridProps } from "../components/ChartCard.jsx";
import { colorAt } from "../palette.js";
import { chartNumber, fmt, runAxisLabels, metricEntries, getNested, metricText, toCSV, downloadBlob } from "../utils.js";
import ComparisonMetrics from "../components/ComparisonMetrics.jsx";

const TREND_METRICS = ["f1", "recall", "precision", "accuracy", "balanced_accuracy", "mcc"];

// Render selected-run metrics and comparisons.
export default function CompareTab({ runs, selectedRuns, metricsByRun, metricKeys }) {
  const [selectedMetric, setSelectedMetric] = useState("comparison.classification.phishing.f1");
  const [trendMetric, setTrendMetric] = useState("f1");

  const rows = useMemo(() => {
    const resolved = selectedRuns.map((id) => runs.find((r) => r.run_id === id)).filter(Boolean);
    const labels = runAxisLabels(resolved);
    return resolved.map((run, index) => {
          const metrics = metricsByRun[run.run_id] || {};
          const phishing = metrics.phishing || {};
          return {
            run,
            label: labels[index],
            phishing,
            phishing_f1: phishing.f1 ?? null,
            phishing_recall: phishing.recall ?? null,
            injection_recall: metrics.prompt_injection?.recall ?? null,
            json_validity: metrics.json_initial_validity_rate ?? null,
            median_latency_ms: metrics.latency_ms?.median ?? null,
          };
    });
  }, [selectedRuns, runs, metricsByRun]);

  // Each backend numeric leaf can be compared, including per-class and subgroup metrics.
  const numericPaths = [...new Set(rows.flatMap((row) =>
    metricEntries(metricsByRun[row.run.run_id] || {})
      .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
      .map(([path]) => path)))].sort();
  const activeMetric = numericPaths.includes(selectedMetric) ? selectedMetric : numericPaths[0];
  const numericRows = rows.map((row) => ({ label: row.label,
    value: getNested(metricsByRun[row.run.run_id], activeMetric || "") ?? null }));

  if (!rows.length) {
    return <div className="alert alert-info">Check at least one run in the sidebar to compare.</div>;
  }

  const barData = rows.map((r) => ({ label: r.label, ...r.phishing }));
  const scatterData = rows
    .filter((r) => r.phishing_f1 != null && r.median_latency_ms != null)
    .map((r) => ({ x: r.median_latency_ms, y: r.phishing_f1, name: r.run.run_name || r.run.config_id }));
  const trendData = [...rows]
    .sort((a, b) => new Date(a.run.started_at) - new Date(b.run.started_at))
    .map((r) => ({ label: r.label, value: r.phishing?.[trendMetric] ?? null }));

  const signed = metricKeys.includes("mcc");
  // Horizontal bars: config identifiers are far too long for readable x-axis
  // ticks, and grow one row per run rather than squeezing the plot.
  const barHeight = Math.max(220, rows.length * Math.max(28, metricKeys.length * 22) + 60);

  return (
    <div>
      <button onClick={() => downloadBlob("comparison-metrics.csv", toCSV(rows.flatMap(({ run }) =>
        metricEntries(metricsByRun[run.run_id] || {}).map(([metric, value]) => ({
          run_id: run.run_id, config_id: run.config_id, model_id: run.model_id,
          run_name: run.run_name, task: run.evaluation_task, dataset: run.dataset,
          dataset_path: run.dataset_path, split: run.split,
          metric, value: value == null ? "" : metricText(value),
        })))), "text/csv")}>
        Export all selected-run metrics CSV
      </button>
      {activeMetric && <ChartCard title="Any recorded numeric metric"
        note="Raw metric units; missing values have no bar. Compare matching tasks, inputs, language filters and denominators. Matrix cells are shown in the focused-run heatmaps."
        control={<select aria-label="Metric path to compare" value={activeMetric} onChange={(event) => setSelectedMetric(event.target.value)}>
          {numericPaths.map((path) => <option key={path} value={path}>{path}</option>)}
        </select>}>
        <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 45)}>
          <BarChart data={numericRows} layout="vertical" margin={{ right: 50 }}>
            <CartesianGrid {...gridProps} /><XAxis type="number" {...axisProps} />
            <YAxis type="category" dataKey="label" width={200} {...axisProps} /><ChartTooltip />
            <Bar isAnimationActive={false} dataKey="value" fill={colorAt(0)}>
              <LabelList dataKey="value" position="right" formatter={(value) => chartNumber(value)} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <table className="data-table"><thead><tr><th>Run</th><th>{activeMetric}</th></tr></thead>
          <tbody>{numericRows.map((row, index) => <tr key={rows[index].run.run_id}><td>{rows[index].run.run_id}</td><td>{fmt(row.value)}</td></tr>)}</tbody>
        </table>
      </ChartCard>}
      <ComparisonMetrics runs={rows.map((row) => row.run)} metricsByRun={metricsByRun} />
      <ChartCard
        title="Phishing-classification metrics by run"
        note="Metrics use stored boolean labels and available ground truth. Compare runs on the same dataset and split; see Thesis metrics for formulas. No confidence intervals are shown."
      >
        <ResponsiveContainer width="100%" height={barHeight}>
          <BarChart data={barData} layout="vertical" margin={{ left: 8, right: 48 }} barCategoryGap="22%">
            <CartesianGrid {...gridProps} vertical horizontal={false} />
            <XAxis type="number" domain={signed ? [-1, 1] : [0, 1]} {...axisProps} />
            <YAxis type="category" dataKey="label" width={200} {...axisProps} />
            <ChartTooltip />
            <Legend />
            {metricKeys.map((key, i) => (
              <Bar isAnimationActive={false} key={key} dataKey={key} name={key} fill={colorAt(i)} radius={[0, 4, 4, 0]}>
                {/* The palette warns on contrast against the surface, so every bar
                    carries its value rather than relying on hue alone. */}
                {metricKeys.length <= 3 && (
                  <LabelList
                    dataKey={key}
                    position="right"
                    formatter={(v) => chartNumber(v, 2)}
                    style={{ fontSize: 10, fill: "#52514e" }}
                  />
                )}
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="card">
        <h4>Runs in these figures</h4>
        <p className="metric-note">
          The table is the WCAG-clean twin of the charts above: every plotted value is readable here
          as text, and the axis label maps to its full run identifier.
        </p>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Axis label</th>
                <th>run_id</th>
                <th>model_id</th>
                <th className="num">records</th>
                <th className="num">phishing F1</th>
                <th className="num">phishing recall</th>
                <th className="num">injection recall</th>
                <th className="num">JSON validity</th>
                <th className="num">median latency (ms)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.run.run_id}>
                  <td>{r.label}</td>
                  <td>{r.run.run_id}</td>
                  <td>{r.run.model_id}</td>
                  <td className="num">{r.run.records}</td>
                  <td className="num">{fmt(r.phishing_f1)}</td>
                  <td className="num">{fmt(r.phishing_recall)}</td>
                  <td className="num">{fmt(r.injection_recall)}</td>
                  <td className="num">{fmt(r.json_validity, true)}</td>
                  <td className="num">{chartNumber(r.median_latency_ms, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <ChartCard
        title="Metric across runs, ordered by start time"
        note="Run snapshots ordered by start time, not a learning curve. Lines connect observed values without smoothing."
        control={
          <label className="muted">
            Metric{" "}
            <select value={trendMetric} onChange={(e) => setTrendMetric(e.target.value)}>
              {TREND_METRICS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
        }
      >
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trendData} margin={{ left: 72, right: 72, bottom: 64 }}>
            <CartesianGrid {...gridProps} />
            <XAxis
              dataKey="label"
              interval={0}
              angle={-25}
              textAnchor="end"
              height={84}
              {...axisProps}
            />
            <YAxis domain={trendMetric === "mcc" ? [-1, 1] : [0, 1]} {...axisProps} />
            <ChartTooltip />
            <Line isAnimationActive={false}
              type="linear"
              dataKey="value"
              name={trendMetric}
              stroke={colorAt(0)}
              strokeWidth={2}
              dot={{ r: 4, strokeWidth: 2, stroke: "#fcfcfb" }}
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {scatterData.length > 1 && (
        <ChartCard
          title="Quality against latency"
          note="Each point is one run: x = median initial-call latency in ms, y = phishing F1. Runs missing either measurement are excluded."
        >
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ left: 8, right: 24, bottom: 24 }}>
              <CartesianGrid {...gridProps} vertical />
              <XAxis
                type="number"
                dataKey="x"
                name="median latency (ms)"
                label={{ value: "median latency (ms)", position: "insideBottom", offset: -12, fontSize: 11 }}
                {...axisProps}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="phishing F1"
                domain={[0, 1]}
                label={{ value: "phishing F1", angle: -90, position: "insideLeft", fontSize: 11 }}
                {...axisProps}
              />
              <ChartTooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter isAnimationActive={false} data={scatterData} fill={colorAt(0)} r={6}>
                <LabelList
                  dataKey="name"
                  position="top"
                  style={{ fontSize: 10, fill: "#52514e" }}
                />
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>
      )}
    </div>
  );
}
