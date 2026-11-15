import { fmt, METRIC_DETAILS } from "../utils.js";

const CARDS = [
  ["accuracy", "Accuracy"],
  ["precision", "Precision"],
  ["recall", "Recall"],
  ["f1", "F1"],
  ["balanced_accuracy", "Bal. accuracy"],
  ["mcc", "MCC"],
];

// The formula stays available but no longer occupies three wrapped lines under
// every tile: the short form is visible, the full denominator is on hover/focus.
// Render focused-run classification KPIs.
export default function Kpis({ phishing, caption }) {
  return (
    <>
      {caption && <p className="muted" style={{ margin: "0 0 var(--s2)" }}>{caption}</p>}
      <div className="kpi-row">
        {CARDS.map(([key, title]) => {
          const [, formula] = METRIC_DETAILS[`phishing.${key}`];
          return (
            <div className="kpi-card" key={key}>
              <div className="kpi-title">
                <span>{title}</span>
                <abbr className="formula" title={formula} tabIndex={0}>
                  f(x)
                </abbr>
              </div>
              <div className="kpi-value">{phishing ? fmt(phishing[key]) : "N/A"}</div>
            </div>
          );
        })}
      </div>
    </>
  );
}
