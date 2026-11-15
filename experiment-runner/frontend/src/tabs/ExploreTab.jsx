import { useMemo, useState } from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, LabelList } from "recharts";
import ChartCard, { CartesianGrid, ChartTooltip, axisProps, gridProps } from "../components/ChartCard.jsx";
import { binaryMetrics, chartNumber, fmt, groupBy, mean, truncateMiddle } from "../utils.js";
import { colorAt } from "../palette.js";

const FILTER_FIELDS = [
  ["source_dataset", "Dataset"],
  ["status", "Status"],
  ["true_personalization_level", "Personalization"],
  ["true_is_phishing", "True label"],
];

const METRIC_FIELDS = [
  "latency_ms",
  "tokens_per_second",
  "phishing_risk_score",
  "prompt_injection_risk_score",
  "prompt_tokens",
  "output_tokens",
];

const GROUP_FIELDS = [
  "source_dataset",
  "split",
  "status",
  "true_personalization_level",
  "true_is_phishing",
  "true_phishing_type",
];

// Collect distinct non-missing field values in their original order.
function uniqueValues(records, key) {
  return [...new Set(records.map((r) => r[key]).filter((v) => v !== null && v !== undefined))];
}

// Render filters and grouped prediction analysis.
export default function ExploreTab({ records }) {
  const [filters, setFilters] = useState({});
  const [metric, setMetric] = useState(METRIC_FIELDS[0]);
  const [groupField, setGroupField] = useState(GROUP_FIELDS[0]);

  const filtered = useMemo(() => {
    if (!records) return [];
    return records.filter((r) =>
      Object.entries(filters).every(([key, value]) => value === "" || String(r[key]) === value)
    );
  }, [records, filters]);

  const chartData = useMemo(() => {
    const groups = groupBy(filtered, groupField);
    return [...groups.entries()]
      .map(([key, rows]) => ({
        group: truncateMiddle(String(key), 24),
        value: mean(rows.map((r) => r[metric])),
        n: rows.length,
      }))
      .sort((a, b) => (a.group > b.group ? 1 : -1));
  }, [filtered, groupField, metric]);

  const segmentMetrics = useMemo(
    () => binaryMetrics(filtered, "true_is_phishing", "pred_is_phishing"),
    [filtered]
  );

  if (!records) return <div className="alert alert-info">Focus a run in the sidebar.</div>;

  const active = Object.entries(filters).filter(([, v]) => v !== "");

  return (
    <div>
      {/* One filter row above everything it scopes, rather than a control strip
          inside each chart card. */}
      <div className="filter-bar">
        {FILTER_FIELDS.map(([key, label]) => (
          <label key={key}>
            {label}
            <select
              value={filters[key] ?? ""}
              onChange={(e) => setFilters((f) => ({ ...f, [key]: e.target.value }))}
            >
              <option value="">all</option>
              {uniqueValues(records, key).map((v) => (
                <option key={String(v)} value={String(v)}>
                  {String(v)}
                </option>
              ))}
            </select>
          </label>
        ))}
        {active.length > 0 && <button onClick={() => setFilters({})}>Clear filters</button>}
        <span className="spacer muted">
          {filtered.length} / {records.length} records in this segment
        </span>
      </div>

      <div className="card">
        <h4>Segment classification</h4>
        <p className="metric-note">
          Recomputed in the browser from the filtered records using stored boolean labels. Denominators
          change with the filter, so these are exploratory summaries, not thesis figures.
        </p>
        <div className="kpi-row" style={{ marginBottom: 0 }}>
          {[
            ["Accuracy", segmentMetrics.accuracy],
            ["Precision", segmentMetrics.precision],
            ["Recall", segmentMetrics.recall],
            ["F1", segmentMetrics.f1],
            ["Records scored", segmentMetrics.n],
          ].map(([label, value]) => (
            <div className="kpi-card" key={label}>
              <div className="kpi-title">{label}</div>
              <div className="kpi-value">
                {label === "Records scored" ? value : fmt(value)}
              </div>
            </div>
          ))}
        </div>
      </div>

      <ChartCard
        title={`Mean ${metric} by ${groupField}`}
        note="Each bar is the arithmetic mean Σx/n of finite values in that group after filtering. Group sizes (n) appear in the tooltip; grouping and filtering both change the denominator."
        control={
          <span className="muted">
            <label>
              Metric{" "}
              <select value={metric} onChange={(e) => setMetric(e.target.value)}>
                {METRIC_FIELDS.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>{" "}
            <label>
              Group by{" "}
              <select value={groupField} onChange={(e) => setGroupField(e.target.value)}>
                {GROUP_FIELDS.map((g) => (
                  <option key={g} value={g}>
                    {g}
                  </option>
                ))}
              </select>
            </label>
          </span>
        }
      >
        {!chartData.some((d) => d.value != null) ? (
          <p className="muted">
            {chartData.length
              ? `No finite ${metric} values in this segment — the focused run records no such measurement.`
              : "No records match this segment."}
          </p>
        ) : (
          <ResponsiveContainer width="100%" height={Math.max(220, chartData.length * 40 + 50)}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 56 }}>
              <CartesianGrid {...gridProps} vertical horizontal={false} />
              <XAxis type="number" {...axisProps} />
              <YAxis type="category" dataKey="group" width={170} {...axisProps} />
              <ChartTooltip
                digits={2}
                labelFormatter={(label) =>
                  `${label} · n=${chartData.find((d) => d.group === label)?.n ?? 0}`
                }
              />
              <Bar isAnimationActive={false} dataKey="value" name={`mean(${metric})`} fill={colorAt(3)} radius={[0, 4, 4, 0]}>
                <LabelList
                  dataKey="value"
                  position="right"
                  formatter={(v) => chartNumber(v, 1)}
                  style={{ fontSize: 10, fill: "#52514e" }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </ChartCard>
    </div>
  );
}
