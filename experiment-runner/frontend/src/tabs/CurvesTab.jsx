import { useMemo, useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, ReferenceLine, ReferenceDot } from "recharts";
import Heatmap from "../components/Heatmap.jsx";
import ChartCard, { CartesianGrid, ChartTooltip, axisProps, gridProps } from "../components/ChartCard.jsx";
import { computeCurves, confusionAtThreshold } from "../curves.js";
import { colorAt, AXIS } from "../palette.js";
import { fmt } from "../utils.js";

// Render ROC/PR plots and threshold analysis.
export default function CurvesTab({ records }) {
  const [threshold, setThreshold] = useState(50);
  const curves = useMemo(() => computeCurves(records || []), [records]);
  const stats = useMemo(() => confusionAtThreshold(records || [], threshold), [records, threshold]);

  // The operating point the threshold slider currently selects, marked on the ROC.
  const operatingPoint = useMemo(() => {
    if (!stats.n) return null;
    const negatives = stats.fp + stats.tn;
    const positives = stats.tp + stats.fn;
    if (!negatives || !positives) return null;
    return { fpr: stats.fp / negatives, tpr: stats.tp / positives };
  }, [stats]);

  if (!records) return <div className="alert alert-info">Focus a run in the sidebar.</div>;
  if (!curves.roc.length) {
    return (
      <div className="alert alert-warning">
        Not enough scored records with both a ground-truth label and a phishing_risk_score to plot
        ROC/PR curves — both classes must be present.
      </div>
    );
  }

  return (
    <div>
      <div className="filter-bar">
        <label>
          Decision threshold
          <input
            aria-label="Phishing risk decision threshold"
            type="range"
            min={0}
            max={100}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            style={{ width: 220 }}
          />
        </label>
        <strong style={{ fontVariantNumeric: "tabular-nums" }}>{threshold}</strong>
        <span className="spacer muted">
          Classify as phishing when phishing_risk_score ≥ threshold. Recomputed live from stored
          per-record scores; this can differ from the model’s stored label. Choose final thresholds
          on development data only.
        </span>
      </div>

      <div className="grid-2">
        <ChartCard
          title={`ROC curve${curves.rocAuc != null ? ` — AUC ${curves.rocAuc.toFixed(3)}` : ""}`}
          note={`TPR = TP/(TP+FN); FPR = FP/(FP+TN). Area by trapezoids; the dashed diagonal marks chance. The marked point is the current threshold. n=${curves.n} scored records.`}
        >
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={curves.roc} margin={{ left: 8, right: 16, bottom: 20 }}>
              <CartesianGrid {...gridProps} vertical />
              <XAxis
                type="number"
                dataKey="fpr"
                domain={[0, 1]}
                label={{ value: "FPR", position: "insideBottom", offset: -10, fontSize: 11 }}
                {...axisProps}
              />
              <YAxis
                type="number"
                domain={[0, 1]}
                label={{ value: "TPR", angle: -90, position: "insideLeft", fontSize: 11 }}
                {...axisProps}
              />
              <ChartTooltip />
              <ReferenceLine
                segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                stroke={AXIS}
                strokeDasharray="4 4"
              />
              <Line isAnimationActive={false} type="linear" dataKey="tpr" name="TPR" stroke={colorAt(0)} dot={false} strokeWidth={2} />
              {operatingPoint && (
                <ReferenceDot
                  x={operatingPoint.fpr}
                  y={operatingPoint.tpr}
                  r={5}
                  fill={colorAt(1)}
                  stroke="#fcfcfb"
                  strokeWidth={2}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title={`Precision-recall curve${
            curves.prAuc != null ? ` — trapezoidal AUC ${curves.prAuc.toFixed(3)}` : ""
          }`}
          note="Precision = TP/(TP+FP); Recall = TP/(TP+FN). Includes the conventional (0,1) start. The trapezoidal area is explicitly not average precision."
        >
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={curves.pr} margin={{ left: 8, right: 16, bottom: 20 }}>
              <CartesianGrid {...gridProps} vertical />
              <XAxis
                type="number"
                dataKey="recall"
                domain={[0, 1]}
                label={{ value: "Recall", position: "insideBottom", offset: -10, fontSize: 11 }}
                {...axisProps}
              />
              <YAxis
                type="number"
                domain={[0, 1]}
                label={{ value: "Precision", angle: -90, position: "insideLeft", fontSize: 11 }}
                {...axisProps}
              />
              <ChartTooltip />
              <Line isAnimationActive={false}
                type="linear"
                dataKey="precision"
                name="Precision"
                stroke={colorAt(1)}
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <div className="grid-2">
        <Heatmap
          matrix={[
            [stats.tn, stats.fp],
            [stats.fn, stats.tp],
          ]}
          labels={["legitimate", "phishing"]}
          title={`Confusion matrix at threshold ${threshold} (n=${stats.n})`}
          note="Recomputed from stored risk scores at the threshold set above. Rows = ground truth; columns = prediction; the percentage is the share of that ground-truth row."
        />
        <div className="card">
          <h4>Classification at threshold {threshold}</h4>
          <p className="metric-note">
            Derived from the same counts as the matrix beside it, using the backend's zero-division
            convention.
          </p>
          <div className="kpi-row" style={{ marginBottom: 0, gridTemplateColumns: "repeat(2, 1fr)" }}>
            {[
              ["Precision", stats.precision],
              ["Recall", stats.recall],
              ["F1", stats.f1],
              ["Accuracy", stats.accuracy],
            ].map(([label, value]) => (
              <div className="kpi-card" key={label}>
                <div className="kpi-title">{label}</div>
                <div className="kpi-value">{fmt(value)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
