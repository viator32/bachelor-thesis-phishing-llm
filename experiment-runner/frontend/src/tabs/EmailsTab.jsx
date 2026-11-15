import { useEffect, useMemo, useState } from "react";
import { api } from "../api.js";
import { fmt } from "../utils.js";

const FILTERS = [
  ["all", "All"],
  ["mismatch", "Misclassified"],
  ["phishing", "True phishing"],
  ["legitimate", "True legitimate"],
  ["failed", "No model reply"],
];

// Apply the selected email outcome filter.
function matchesFilter(row, filter) {
  if (filter === "mismatch") {
    return (
      typeof row.true_is_phishing === "boolean" &&
      typeof row.pred_is_phishing === "boolean" &&
      row.true_is_phishing !== row.pred_is_phishing
    );
  }
  if (filter === "phishing") return row.true_is_phishing === true;
  if (filter === "legitimate") return row.true_is_phishing === false;
  if (filter === "failed") return row.status !== "success";
  return true;
}

// A coloured dot is not enough on its own; the letter carries the meaning too.
// Render an email outcome as text and color.
function Outcome({ row }) {
  if (row.status !== "success") return <span className="pill pill-bad">error</span>;
  if (typeof row.true_is_phishing !== "boolean" || typeof row.pred_is_phishing !== "boolean") {
    return <span className="pill pill-neutral">n/a</span>;
  }
  const correct = row.true_is_phishing === row.pred_is_phishing;
  return (
    <span className={`pill ${correct ? "pill-good" : "pill-bad"}`}>
      {correct ? "ok" : row.pred_is_phishing ? "FP" : "FN"}
    </span>
  );
}

// Render a detail label and value with a missing-value fallback.
function Field({ label, children }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{children ?? <span className="muted">—</span>}</dd>
    </>
  );
}

// Render a prediction beside available ground truth.
function Verdict({ label, predicted, truth }) {
  if (predicted == null) return <Field label={label}>{null}</Field>;
  const known = typeof truth === "boolean" || (truth != null && truth !== "");
  const correct = known ? String(predicted) === String(truth) : null;
  return (
    <Field label={label}>
      <span
        className={`pill ${correct === null ? "pill-neutral" : correct ? "pill-good" : "pill-bad"}`}
      >
        {String(predicted)}
      </span>{" "}
      {known && (
        <span className="muted">
          truth: {String(truth)}
          {correct ? "" : " · mismatch"}
        </span>
      )}
    </Field>
  );
}

// Render filtered email records and load the selected detail.
export default function EmailsTab({ runId, records, instanceId, onInstanceChange }) {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [record, setRecord] = useState(null);
  const [error, setError] = useState(null);

  const rows = useMemo(() => {
    const term = search.trim().toLowerCase();
    return (records || []).filter(
      (r) =>
        matchesFilter(r, filter) &&
        (!term || String(r.instance_id).toLowerCase().includes(term))
    );
  }, [records, filter, search]);

  // Keep the selection inside the visible list, so filtering never strands it.
  useEffect(() => {
    if (!rows.length) return;
    if (!rows.some((r) => r.instance_id === instanceId)) onInstanceChange(rows[0].instance_id);
  }, [rows, instanceId, onInstanceChange]);

  useEffect(() => {
    setRecord(null);
    setError(null);
    if (!runId || !instanceId) return;
    let cancelled = false;
    api
      .runInstance(runId, instanceId)
      .then((r) => !cancelled && setRecord(r))
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [runId, instanceId]);

  if (!runId) return <div className="alert alert-info">Focus a run in the sidebar.</div>;
  if (!records) {
    return (
      <p className="muted" role="status">
        Loading records...
      </p>
    );
  }

  const position = rows.findIndex((r) => r.instance_id === instanceId);
  // Select the adjacent visible email when available.
  const step = (delta) => {
    const next = rows[position + delta];
    if (next) onInstanceChange(next.instance_id);
  };

  return (
    <div>
      <div className="filter-bar">
        <label>
          Show
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            {FILTERS.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Find
          <input
            type="search"
            placeholder="instance id"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </label>
        <span className="spacer muted">
          {rows.length} of {records.length} emails
        </span>
      </div>

      <div className="email-layout">
        <div className="card email-list-card">
          <h4>Emails in this run</h4>
          <p className="metric-note">
            Every record the run stored. FP = predicted phishing but legitimate; FN = predicted
            legitimate but phishing.
          </p>
          <div className="email-list">
            {rows.map((row) => (
              <button
                key={row.instance_id}
                className={`email-row${row.instance_id === instanceId ? " is-selected" : ""}`}
                onClick={() => onInstanceChange(row.instance_id)}
                aria-current={row.instance_id === instanceId}
              >
                <Outcome row={row} />
                <span className="email-id">{row.instance_id}</span>
                <span className="email-score">
                  {row.phishing_risk_score == null ? "—" : row.phishing_risk_score}
                </span>
              </button>
            ))}
            {!rows.length && <p className="muted">No email matches this filter.</p>}
          </div>
          <div className="table-tools">
            <button disabled={position <= 0} onClick={() => step(-1)}>
              Previous
            </button>
            <span>
              {position < 0 ? "—" : position + 1} / {rows.length}
            </span>
            <button disabled={position < 0 || position >= rows.length - 1} onClick={() => step(1)}>
              Next
            </button>
          </div>
        </div>

        <div className="email-detail">
          {error && <div className="alert alert-danger">{error}</div>}
          {!error && !record && (
            <p className="muted" role="status">
              Loading email...
            </p>
          )}
          {record && <EmailDetail record={record} />}
        </div>
      </div>
    </div>
  );
}

// Render stored email content and predictions as inert text.
function EmailDetail({ record }) {
  const {
    processed_email: email,
    ground_truth: truth,
    final_result: finalResult,
    components,
  } = record;
  const prediction = finalResult.prediction || {};
  const rawResponse = finalResult.raw_response || components.llm?.raw_response || "N/A";

  return (
    <>
      <div className="card">
        <h4>Prediction against ground truth — {record.instance_id}</h4>
        <p className="metric-note">
          Green means the stored prediction matches the stored label for that field; grey means no
          ground truth is available to compare against.
        </p>
        <dl className="kv">
          <Verdict label="Is phishing" predicted={prediction.is_phishing} truth={truth?.is_phishing} />
          <Verdict label="Phishing type" predicted={prediction.phishing_type} truth={truth?.phishing_type} />
          <Verdict
            label="Generation origin"
            predicted={prediction.generation_origin}
            truth={truth?.generation_origin}
          />
          <Verdict
            label="Personalization"
            predicted={prediction.personalization_level}
            truth={truth?.personalization_level}
          />
          <Verdict
            label="Prompt injection"
            predicted={prediction.prompt_injection_detected}
            truth={truth?.has_prompt_injection}
          />
          <Field label="Phishing risk">{fmt(prediction.phishing_risk_score)}</Field>
          <Field label="Injection risk">{fmt(prediction.prompt_injection_risk_score)}</Field>
          <Field label="Status">{finalResult.status}</Field>
          <Field label="Method">{finalResult.method}</Field>
          {finalResult.error && <Field label="Error">{finalResult.error}</Field>}
        </dl>
        {prediction.reasoning && (
          <>
            <h4 style={{ marginTop: "var(--s4)" }}>Model reasoning</h4>
            <pre className="detail">{prediction.reasoning}</pre>
          </>
        )}
      </div>

      <div className="card">
        <h4>Inert email view</h4>
        <p className="metric-note">
          Rendered as plain text. No HTML is executed, no URL is opened and no remote content is
          loaded.
        </p>
        <dl className="kv">
          <Field label="From">
            {email.sender_name} &lt;{email.sender_address}&gt;
          </Field>
          <Field label="Reply-To">{email.reply_to}</Field>
          <Field label="Subject">{email.subject}</Field>
        </dl>
        <pre className="detail" style={{ marginTop: "var(--s2)" }}>
          {email.text}
        </pre>
      </div>

      <div className="card">
        <h4>URLs ({email.urls.length}) — never opened</h4>
        {email.urls.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  {Object.keys(email.urls[0]).map((k) => (
                    <th key={k}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {email.urls.map((u, i) => (
                  <tr key={i}>
                    {Object.values(u).map((v, j) => (
                      <td key={j}>{String(v)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">No URLs in this message.</p>
        )}
      </div>

      {/* Raw payloads stay available but collapsed: they are for debugging one
          record, not for reading on every selection. */}
      {[
        ["Raw model response", rawResponse],
        ["Pipeline components", JSON.stringify(components, null, 2)],
        ["Full ground truth", JSON.stringify(truth, null, 2)],
        ["Preprocessing transformations", JSON.stringify(email.transformations, null, 2)],
      ].map(([title, body]) => (
        <details className="card" key={title}>
          <summary>
            <b>{title}</b>
          </summary>
          <pre className="detail" style={{ marginTop: "var(--s2)" }}>
            {body}
          </pre>
        </details>
      ))}
    </>
  );
}
