// Three different causes all print as "N/A": the dataset never annotated the
// field, the model server never reported the measurement, or nothing survived to
// be scored. The backend distinguishes them, so show which one applies.
// Explain missing metrics, optionally restricted to named keys.
export default function Unavailable({ unavailable, only }) {
  const entries = Object.entries(unavailable || {}).filter(
    ([key]) => !only || only.includes(key)
  );
  if (!entries.length) return null;

  return (
    <details className="card">
      <summary>
        <b>Why {entries.length} {entries.length === 1 ? "metric is" : "metrics are"} unavailable</b>
      </summary>
      <dl className="why-list" style={{ marginTop: "var(--s3)" }}>
        {entries.map(([key, reason]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{reason}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}
