import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, LabelList } from "recharts";
import Heatmap from "../components/Heatmap.jsx";
import Unavailable from "../components/Unavailable.jsx";
import ChartCard, { CartesianGrid, ChartTooltip, axisProps, gridProps } from "../components/ChartCard.jsx";
import { colorAt } from "../palette.js";
import { chartNumber, fmt } from "../utils.js";

const valueLabel = (
  <LabelList position="right" formatter={(v) => chartNumber(v, 2)} style={{ fontSize: 10, fill: "#52514e" }} />
);

// Render classification and injection quality metrics.
export default function QualityTab({ metrics }) {
  if (!metrics) return <div className="alert alert-info">Focus a run in the sidebar.</div>;
  const phishing = metrics.phishing || {};
  const origin = metrics.generation_origin || {};

  const levelData = Object.entries(metrics.by_personalization_level || {})
    .filter(([, m]) => (m.evaluated ?? m.n ?? 0) > 0)
    .map(([level, m]) => ({ level: `P${level}`, f1: m.f1 ?? null }));
  const injectionData = Object.entries(metrics.injection_goal_attack_success || {}).map(
    ([goal, m]) => ({ goal, rate: m.rate ?? null })
  );
  const paired = metrics.paired_robustness || {};
  const pairedData = [
    { label: "Label-flip rate", rate: paired.label_flip_rate ?? null },
    {
      label: "Warning-suppression success",
      rate: paired.attack_success_rate_warning_suppression ?? null,
    },
  ];

  return (
    <div>
      <Unavailable
        unavailable={metrics.unavailable}
        only={[
          "by_personalization_level",
          "injection_goal_attack_success",
          "paired_robustness",
          "generation_origin",
          "model_responses_received",
        ]}
      />
      <div className="grid-2">
        <Heatmap
          matrix={phishing.confusion_matrix || [[0, 0], [0, 0]]}
          labels={["legitimate", "phishing"]}
          title="Phishing confusion matrix"
        />
        {origin.confusion_matrix ? (
          <Heatmap
            matrix={origin.confusion_matrix}
            labels={origin.labels || ["human", "ai", "unknown"]}
            title="Generation-origin confusion matrix"
          />
        ) : (
          <div className="alert alert-info">
            No generation-origin data in this run: the dataset carries no known-origin ground truth.
          </div>
        )}
      </div>

      <div className="grid-2">
        <ChartCard
          title="Phishing F1 by personalization level"
          note="F1 = 2TP/(2TP+FP+FN), calculated separately for each annotated level. Group sizes and class balance differ between levels, so the levels are not directly comparable."
        >
          {levelData.length ? (
            <ResponsiveContainer width="100%" height={Math.max(200, levelData.length * 44 + 40)}>
              <BarChart data={levelData} layout="vertical" margin={{ left: 8, right: 48 }}>
                <CartesianGrid {...gridProps} vertical horizontal={false} />
                <XAxis type="number" domain={[0, 1]} {...axisProps} />
                <YAxis type="category" dataKey="level" width={90} {...axisProps} />
                <ChartTooltip />
                <Bar isAnimationActive={false} dataKey="f1" name="F1" fill={colorAt(0)} radius={[0, 4, 4, 0]}>
                  {valueLabel}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="muted">{metrics.unavailable?.by_personalization_level ?? "No personalization levels annotated in this run."}</p>
          )}
        </ChartCard>

        <ChartCard
          title="Prompt-injection attack success by goal"
          note="ASR = successes / injected records assigned to that goal with an LLM component. Success follows the experiment protocol; one record may count under several goals."
        >
          {injectionData.length ? (
            <ResponsiveContainer width="100%" height={Math.max(200, injectionData.length * 44 + 40)}>
              <BarChart data={injectionData} layout="vertical" margin={{ left: 8, right: 48 }}>
                <CartesianGrid {...gridProps} vertical horizontal={false} />
                <XAxis type="number" domain={[0, 1]} {...axisProps} />
                <YAxis type="category" dataKey="goal" width={150} {...axisProps} />
                <ChartTooltip />
                <Bar isAnimationActive={false} dataKey="rate" name="ASR" fill={colorAt(1)} radius={[0, 4, 4, 0]}>
                  {valueLabel}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="muted">{metrics.unavailable?.injection_goal_attack_success ?? "No injection goals annotated in this run."}</p>
          )}
        </ChartCard>
      </div>

      <div className="card">
        <h4>Independent personalization annotations: P0 / P1</h4>
        <p className="metric-note">
          P0: no recipient-specific context. P1: recipient context such as name, role, organization
          or individual details. These are dataset annotations, independent of model predictions.
          Annotated: {metrics.personalization_coverage?.annotated_records ?? "N/A"};
          unannotated: {metrics.personalization_coverage?.unannotated_records ?? "N/A"}.
          Recall/FNR use phishing cases; FPR uses legitimate cases. Missing values remain N/A.
        </p>
        <div className="table-scroll">
          <table className="data-table">
            <thead><tr>
              {["Level", "Records", "Scored", "Invalid", "Recall", "FNR", "FPR", "F1", "End-to-end accuracy", "JSON valid"].map((label) => <th key={label}>{label}</th>)}
            </tr></thead>
            <tbody>{[0, 1].map((level) => {
              const m = metrics.by_personalization_level?.[level] || {};
              return <tr key={level}>
                <td>P{level}</td><td>{m.evaluated ?? 0}</td><td>{m.n ?? 0}</td>
                <td>{m.invalid_outputs ?? 0}</td><td>{fmt(m.recall)}</td><td>{fmt(m.fnr)}</td>
                <td>{fmt(m.fpr)}</td><td>{fmt(m.f1)}</td><td>{fmt(m.end_to_end_accuracy)}</td>
                <td>{fmt(m.json_final_validity_rate, true)}</td>
              </tr>;
            })}</tbody>
          </table>
        </div>
      </div>

      <ChartCard
        title="Paired robustness"
        note={`n=${paired.paired_examples ?? 0} pairs. First clean/injected prediction per base_id. Flip = different labels / all pairs; suppression = true → false / all pairs, not only originally correct phishing cases.`}
      >
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={pairedData} layout="vertical" margin={{ left: 8, right: 48 }}>
            <CartesianGrid {...gridProps} vertical horizontal={false} />
            <XAxis type="number" domain={[0, 1]} {...axisProps} />
            <YAxis type="category" dataKey="label" width={220} {...axisProps} />
            <ChartTooltip />
            <Bar isAnimationActive={false} dataKey="rate" name="rate" fill={colorAt(7)} radius={[0, 4, 4, 0]}>
              {valueLabel}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}
