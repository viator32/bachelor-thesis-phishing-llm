import { binaryMetrics } from "./utils.js";

// Threshold curves use stored risk scores, not the model's separate boolean decision.
// Curves require known truth and a score on the same 0–100 scale as the threshold.
function scoredRecords(records) {
  return records.filter(
    (record) => typeof record.true_is_phishing === "boolean" && Number.isFinite(record.phishing_risk_score) &&
      record.phishing_risk_score >= 0 && record.phishing_risk_score <= 100
  );
}

// Integrate ordered curve points with trapezoids.
function trapezoidArea(points, xKey, yKey) {
  let area = 0;
  for (let index = 1; index < points.length; index++) {
    const intervalWidth = points[index][xKey] - points[index - 1][xKey];
    area += (intervalWidth * (points[index][yKey] + points[index - 1][yKey])) / 2;
  }
  return area;
}

// Build ROC and PR curves from risk scores, grouping tied scores.
export function computeCurves(records) {
  const evaluatedRecords = scoredRecords(records).sort(
    (first, second) => second.phishing_risk_score - first.phishing_risk_score
  );
  const positiveCount = evaluatedRecords.filter((record) => record.true_is_phishing).length;
  const negativeCount = evaluatedRecords.length - positiveCount;
  if (!evaluatedRecords.length || !positiveCount || !negativeCount) {
    return { roc: [], pr: [], rocAuc: null, prAuc: null, n: evaluatedRecords.length };
  }

  const roc = [{ fpr: 0, tpr: 0, threshold: null }];
  const pr = [{ recall: 0, precision: 1, threshold: null }];
  let tp = 0;
  let fp = 0;
  // Include an entire tied-score group before recording its threshold (score >= threshold).
  for (let index = 0; index < evaluatedRecords.length; index++) {
    const record = evaluatedRecords[index];
    if (record.true_is_phishing) tp += 1;
    else fp += 1;
    if (evaluatedRecords[index + 1]?.phishing_risk_score === record.phishing_risk_score) continue;
    const threshold = record.phishing_risk_score;
    roc.push({ fpr: fp / negativeCount, tpr: tp / positiveCount, threshold });
    pr.push({ recall: tp / positiveCount, precision: tp / (tp + fp), threshold });
  }

  return {
    roc,
    pr,
    rocAuc: trapezoidArea(roc, "fpr", "tpr"),
    // Integrate the displayed PR curve; this is trapezoidal AUC, not average precision.
    prAuc: trapezoidArea(pr, "recall", "precision"),
    n: evaluatedRecords.length,
  };
}

// Classify stored risk scores at a selected threshold.
export function confusionAtThreshold(records, threshold) {
  const rows = scoredRecords(records).map((record) => ({
    true_is_phishing: record.true_is_phishing,
    pred_is_phishing: record.phishing_risk_score >= threshold,
  }));
  return binaryMetrics(rows, "true_is_phishing", "pred_is_phishing");
}
