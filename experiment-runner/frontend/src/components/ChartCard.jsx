import { useRef, useState } from "react";
import { CartesianGrid, Tooltip } from "recharts";
import { AXIS, GRIDLINE, INK_SECONDARY, SURFACE } from "../palette.js";
import { chartNumber, downloadBlob } from "../utils.js";

// Wrap a chart with its title, note and optional control.
export default function ChartCard({ title, note, control, children }) {
  const chartRef = useRef(null);
  const [exportError, setExportError] = useState(null);
  function exportSvg() {
    const source = chartRef.current?.querySelector("svg.recharts-surface");
    if (!source) {
      setExportError("Chart is not rendered yet.");
      return;
    }
    const copy = source.cloneNode(true);
    copy.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    copy.setAttribute("width", source.getBoundingClientRect().width);
    copy.setAttribute("height", source.getBoundingClientRect().height);
    // Copy computed styles so the downloaded SVG does not depend on dashboard CSS.
    const originals = [source, ...source.querySelectorAll("*")];
    const copies = [copy, ...copy.querySelectorAll("*")];
    originals.forEach((element, index) => {
      const style = getComputedStyle(element);
      for (const property of ["fill", "stroke", "stroke-width", "font-family", "font-size", "font-weight", "opacity"]) {
        copies[index].style.setProperty(property, style.getPropertyValue(property));
      }
    });
    downloadBlob(`${title.replace(/[^a-z0-9]+/gi, "-")}.svg`, new XMLSerializer().serializeToString(copy), "image/svg+xml");
    setExportError(null);
  }
  return (
    <div className="card">
      <div className="card-head">
        <h4>{title}</h4>
        {control}
        <button onClick={exportSvg}>Export SVG</button>
      </div>
      {note && <p className="metric-note">{note}</p>}
      <div className="chart-wrap" ref={chartRef}>{children}</div>
      {exportError && <p role="alert">{exportError}</p>}
    </div>
  );
}

// Hairline, solid, one shade off the surface; horizontal rules only.
export const gridProps = { stroke: GRIDLINE, strokeWidth: 1, vertical: false };

export const axisProps = {
  stroke: AXIS,
  tickLine: false,
  tick: { fontSize: 11, fill: INK_SECONDARY },
};

// Render a chart tooltip with shared styles and number formatting.
export function ChartTooltip({ digits = 3, suffix = "", ...rest }) {
  return (
    <Tooltip
      cursor={{ fill: "rgba(42, 120, 214, 0.06)" }}
      contentStyle={{
        background: SURFACE,
        border: `1px solid ${GRIDLINE}`,
        borderRadius: 6,
        fontSize: 12,
        boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
      }}
      labelStyle={{ color: INK_SECONDARY, fontSize: 11, marginBottom: 2 }}
      formatter={(value, name) => [
        value == null ? "N/A" : `${chartNumber(value, digits)}${suffix}`,
        name,
      ]}
      {...rest}
    />
  );
}

export { CartesianGrid };
