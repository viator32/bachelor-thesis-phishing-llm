import { fmt } from "../utils.js";

const classificationFields = [
  "accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc",
  "false_positive_rate", "false_negative_rate", "tp", "tn", "fp", "fn",
  "end_to_end_accuracy",
];

export default function ComparisonMetrics({ runs, metricsByRun }) {
  const available = runs.filter((run) => metricsByRun[run.run_id]?.comparison);
  if (!available.length) return null;
  return (
    <div className="card">
      <h4>All tasks and required comparison metrics</h4>
      <p className="metric-note">
        N/A means unsupported, unannotated or unmeasured. Origin precision, recall and F1
        are macro averages over human/AI. Phishing and injection use the positive class.
        Model JSON validity is separate from deterministic result serialization.
      </p>
      <div className="table-scroll">
        <table className="data-table">
          <thead><tr>
            <th>Run</th><th>Task</th><th>Evaluated / eligible</th><th>Invalid</th>
            {classificationFields.map((key) => <th key={key}>{key}</th>)}
            <th>Confusion matrix</th><th>Availability</th>
          </tr></thead>
          <tbody>{available.flatMap((run) => Object.entries(
            metricsByRun[run.run_id].comparison.classification,
          ).map(([task, result]) => (
            <tr key={`${run.run_id}-${task}`}>
              <td>{run.run_name || run.config_id}</td><td>{task}</td>
              <td>{result.evaluated_records} / {result.eligible_records}</td>
              <td>{result.invalid_outputs}</td>
              {classificationFields.map((key) => <td key={key}>{fmt(result[key])}</td>)}
              <td>{result.confusion_matrix ? JSON.stringify(result.confusion_matrix) : "N/A"}
                <small>{JSON.stringify(result.labels)}</small></td>
              <td>{result.reason || result.status}</td>
            </tr>
          )))}</tbody>
        </table>
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <thead><tr><th>Run</th><th>Latency mean / median / p95 (ms)</th>
            <th>Tokens/s</th><th>Model RSS mean (MB)</th><th>Server peak (GiB)</th>
            <th>Model JSON validity</th><th>Schema validity</th><th>Result validity</th>
            <th>Attack success rate</th><th>Eligible attack pairs</th>
            <th>Benign suspicious-wording FPR</th></tr></thead>
          <tbody>{available.map((run) => {
            const { performance: p, validity: v, robustness: r } = metricsByRun[run.run_id].comparison;
            return <tr key={run.run_id}>
              <td>{run.run_name || run.config_id}</td>
              <td>{[p.latency_ms.mean, p.latency_ms.median, p.latency_ms.p95].map((n) => fmt(n)).join(" / ")}</td>
              <td>{fmt(p.tokens_per_second.mean)}</td><td>{fmt(p.model_process_rss_mb.mean)}</td>
              <td title={p.memory_note}>{fmt(p.server_peak_memory_gib)}</td>
              <td title={v.note}>{fmt(v.json_validity_rate, true)}</td>
              <td>{fmt(v.schema_validity_rate, true)}</td><td>{fmt(v.result_validity_rate, true)}</td>
              <td title={r.note}>{fmt(r.prompt_injection_attack_success_rate, true)}</td>
              <td>{r.eligible_attack_pairs}</td><td>{fmt(r.benign_suspicious_wording_fpr, true)}</td>
            </tr>;
          })}</tbody>
        </table>
      </div>
    </div>
  );
}
