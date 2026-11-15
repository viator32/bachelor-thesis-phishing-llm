import { Fragment } from "react";
import { SEQUENTIAL_BLUE } from "../palette.js";

// Use light ink on the darkest background.
const INK_ON_STEP = ["#0b0b0b", "#0b0b0b", "#0b0b0b", "#0b0b0b", "#ffffff"];

// Render an accessible confusion matrix with row shares.
export default function Heatmap({ matrix, labels, columnLabels = labels, title, note }) {
  const max = Math.max(1, ...matrix.flat());
  // Map a cell count to the heatmap color ramp.
  const stepFor = (value) =>
    Math.min(SEQUENTIAL_BLUE.length - 1, Math.round((value / max) * (SEQUENTIAL_BLUE.length - 1)));

  return (
    <div className="card">
      <h4>{title}</h4>
      <p className="metric-note">
        {note ||
          "Rows = ground truth; columns = prediction. Cells are counts, with the share of that ground-truth row beneath. Colour = count / largest count in this matrix."}
      </p>
      <div className="heatmap" style={{ gridTemplateColumns: `auto repeat(${columnLabels.length}, 1fr)` }}>
        <div />
        {columnLabels.map((label) => (
          <div className="heatmap-label" key={`col-${label}`}>
            Pred: {label}
          </div>
        ))}
        {matrix.map((row, r) => {
          const rowTotal = row.reduce((a, b) => a + b, 0);
          return (
            <Fragment key={`row-${r}`}>
              <div className="heatmap-label">True: {labels[r]}</div>
              {row.map((value, c) => {
                const step = stepFor(value);
                return (
                  <div
                    key={`cell-${r}-${c}`}
                    className="heatmap-cell"
                    style={{ background: SEQUENTIAL_BLUE[step], color: INK_ON_STEP[step] }}
                    title={`True ${labels[r]} → predicted ${columnLabels[c]}: ${value}`}
                  >
                    {value}
                    <small>{rowTotal ? `${((value / rowTotal) * 100).toFixed(0)}%` : "—"}</small>
                  </div>
                );
              })}
            </Fragment>
          );
        })}
      </div>
    </div>
  );
}
