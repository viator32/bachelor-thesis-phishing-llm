import { useMemo, useState } from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Legend } from "recharts";
import SortableTable from "../components/SortableTable.jsx";
import Unavailable from "../components/Unavailable.jsx";
import ChartCard, { CartesianGrid, ChartTooltip, axisProps, gridProps } from "../components/ChartCard.jsx";
import { colorAt, STATUS_GOOD, STATUS_CRITICAL } from "../palette.js";
import { fmt, groupBy, mean, percentile, runAxisLabels } from "../utils.js";

// 24 columns at once is unreadable and unscreenshotable; these are the questions
// the table actually gets asked, one at a time.
const COLUMN_SETS = {
  "Timing": [
    "instance_id",
    "status",
    "latency_ms",
    "inference_duration_ms",
    "time_to_first_token_ms",
    "model_load_ms",
    "tokens_per_second",
  ],
  "Tokens": ["instance_id", "status", "prompt_tokens", "output_tokens", "total_tokens", "cached_prompt_tokens"],
  "Output validity": [
    "instance_id",
    "json_syntax_initial_valid",
    "schema_initial_valid",
    "json_initial_valid",
    "repair_attempted",
    "json_final_valid",
    "error",
  ],
  "Request provenance": [
    "instance_id",
    "backend",
    "backend_model",
    "model_response_received",
    "prompt_id",
    "prompt_instance_hash",
    "request_temperature",
    "request_seed",
    "request_max_tokens",
  ],
};

// A rate is a status, so it ships with a word as well as a colour — the colour
// alone is not an accessible encoding.
// Render a validation rate with a textual status.
function ValidityTile({ rate, title, note }) {
  const known = rate != null;
  const good = known && rate >= 0.9;
  return (
    <div className="kpi-card">
      <div className="kpi-title">
        <span>{title}</span>
        {known && (
          <span className={`pill ${good ? "pill-good" : "pill-bad"}`}>
            {good ? "at target" : "below target"}
          </span>
        )}
      </div>
      <div
        className="kpi-value"
        style={{ color: known ? (good ? STATUS_GOOD : STATUS_CRITICAL) : "inherit" }}
      >
        {fmt(rate, true)}
      </div>
      <div className="kpi-sub">{note}</div>
    </div>
  );
}

// Check for at least one reported summary statistic.
function hasAnyValue(rows) {
  return rows.some((row) => ["mean", "median", "p95"].some((k) => row[k] != null));
}

// Summarize one measurement by record status.
function distributionByStatus(records, column) {
  const groups = groupBy(records, "status");
  return [...groups.entries()].map(([status, rows]) => {
    const values = rows.map((r) => r[column]);
    return {
      status: `${status} (n=${rows.length})`,
      mean: mean(values),
      median: percentile(values, 50),
      p95: percentile(values, 95),
    };
  });
}

// Render timing, token, validity and resource comparisons.
export default function ResourcesTab({ metrics, records, selectedRuns, metricsByRun, runs }) {
  const [columnSet, setColumnSet] = useState("Timing");

  const latencyDist = useMemo(() => distributionByStatus(records || [], "latency_ms"), [records]);
  const tokensDist = useMemo(
    () => distributionByStatus(records || [], "tokens_per_second"),
    [records]
  );

  if (!metrics || !records) return <div className="alert alert-info">Focus a run in the sidebar.</div>;

  const comparedRunRows = selectedRuns
    .map((id) => runs.find((r) => r.run_id === id))
    .filter(Boolean);
  const comparedLabels = runAxisLabels(comparedRunRows);
  const resourcesData = comparedRunRows.map((run, index) => ({
    label: comparedLabels[index],
    "Median model load (ms)": metricsByRun[run.run_id]?.model_load_ms?.median ?? null,
    "Median latency (ms)": metricsByRun[run.run_id]?.latency_ms?.median ?? null,
  }));

  return (
    <div>
      <Unavailable
        unavailable={metrics.unavailable}
        only={[
          "model_responses_received",
          "time_to_first_token_ms",
          "prompt_duration_ms",
          "generation_duration_ms",
          "prompt_tokens_per_second",
          "model_load_ms",
          "memory_delta_mb",
        ]}
      />
      <div className="kpi-row">
        <ValidityTile
          rate={metrics.json_initial_validity_rate}
          title="Initial JSON validity"
          note="Fully valid initial reports / available reports"
        />
        <ValidityTile
          rate={metrics.json_final_validity_rate}
          title="Final (repaired) validity"
          note="Repair report takes precedence where present"
        />
        <ValidityTile
          rate={metrics.manipulation_detection_rate}
          title="Manipulation detection"
          note="Injection recall on supported tasks"
        />
      </div>

      <div className="grid-2">
        <ChartCard
          title="Latency by record status (ms)"
          note="Mean = Σx/n. Median and p95 use linear interpolation at (n−1)p, excluding missing values. Initial-call latency excludes repair time."
        >
          {!hasAnyValue(latencyDist) ? (
            <p className="muted">
              No latency measurements in this run — deterministic baselines record no model timing.
            </p>
          ) : (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={latencyDist} margin={{ left: 8, right: 16, bottom: 8 }} barCategoryGap="24%">
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="status" {...axisProps} />
              <YAxis {...axisProps} />
              <ChartTooltip digits={0} suffix=" ms" />
              <Legend />
              <Bar isAnimationActive={false} dataKey="mean" fill={colorAt(0)} radius={[4, 4, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="median" fill={colorAt(1)} radius={[4, 4, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="p95" fill={colorAt(2)} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          )}
        </ChartCard>

        <ChartCard
          title="Generation rate by record status (tokens/s)"
          note="Distribution of per-response token rates, not total run throughput. Rates use server timing where available, otherwise HTTP wall time."
        >
          {!hasAnyValue(tokensDist) ? (
            <p className="muted">
              No generation-rate measurements in this run — deterministic baselines call no model.
            </p>
          ) : (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={tokensDist} margin={{ left: 8, right: 16, bottom: 8 }} barCategoryGap="24%">
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="status" {...axisProps} />
              <YAxis {...axisProps} />
              <ChartTooltip digits={1} suffix=" tok/s" />
              <Legend />
              <Bar isAnimationActive={false} dataKey="mean" fill={colorAt(0)} radius={[4, 4, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="median" fill={colorAt(1)} radius={[4, 4, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="p95" fill={colorAt(2)} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          )}
        </ChartCard>
      </div>

      <ChartCard
        title="Model load time and latency across checked runs"
        note="Separate medians of available measurements. A missing model-load measurement stays absent rather than being plotted as zero — external model servers need not report it."
      >
        <ResponsiveContainer width="100%" height={Math.max(220, resourcesData.length * 52 + 60)}>
          <BarChart data={resourcesData} layout="vertical" margin={{ left: 8, right: 32 }} barCategoryGap="24%">
            <CartesianGrid {...gridProps} vertical horizontal={false} />
            <XAxis type="number" {...axisProps} />
            <YAxis type="category" dataKey="label" width={200} {...axisProps} />
            <ChartTooltip digits={0} suffix=" ms" />
            <Legend />
            <Bar isAnimationActive={false} dataKey="Median model load (ms)" fill={colorAt(0)} radius={[0, 4, 4, 0]} />
            <Bar isAnimationActive={false} dataKey="Median latency (ms)" fill={colorAt(1)} radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="card">
        <div className="card-head">
          <h4>Per-record measurements</h4>
          <label className="muted">
            Columns{" "}
            <select value={columnSet} onChange={(e) => setColumnSet(e.target.value)}>
              {Object.keys(COLUMN_SETS).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="metric-note">
          One row per record of the focused run. Export the full column set with “Predictions CSV”
          in the sidebar.
        </p>
        <SortableTable columns={COLUMN_SETS[columnSet]} rows={records} />
      </div>
    </div>
  );
}
