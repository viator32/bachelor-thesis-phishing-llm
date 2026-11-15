// Config IDs distinguish original/adapted models and task-specific adapters.
// Dataset labels normalize container (/app) and host paths through run_identity.
export function coverageDataset(path, task) {
  const name = path.split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
  if (name.toLowerCase().includes("ephishllm_paired_injection")) return "E-PhishLLM paired injection v1";
  if (name.toLowerCase().includes("ephishllm")) return "E-PhishLLM";
  if (name.toLowerCase().includes("crossmodel")) return "Cross-Model";
  if (name.toLowerCase().includes("llmail")) return "LLMail-Inject";
  if (name.toLowerCase().includes("phishfuzzer")) return "PhishFuzzer";
  if (name.toLowerCase().includes("contextual")) return "Contextual Paired Email Injection";
  if (name === "manifest") return task === "injection" ? "Contextual Paired Email Injection" : "PhishFuzzer";
  return name;
}

export function runCombinationKey(run) {
  return JSON.stringify([run.config_id, run.dataset, run.split, run.model_task]);
}

export function buildRunCoverage(configs, runs, split = "test") {
  return configs.flatMap((config) =>
    (config.datasets || []).filter((dataset) => dataset.splits[split] > 0).map((dataset) => {
      const name = coverageDataset(dataset.value, config.task);
      const attempts = runs.filter((run) => run.config_id === config.config_id
        && run.dataset === name && run.split === split && run.model_task === config.task);
      return {
        config, dataset, name, split, attempts,
        status: attempts.length > 1 ? "Duplicate" : attempts.length === 1 ? "Stored" : "Missing",
      };
    })
  );
}
