const PILL_TITLES = {
  schema: "Output is constrained to the JSON schema",
  free: "Generation is unconstrained; the response must still be a strict JSON object",
  "no repair": "Replies are never modified; schema-invalid replies may be re-requested (schema_retries)",
  ready: "Model is listed by the configured oMLX server; it may still need to load into memory",
  "not available": "Model ID is not listed by oMLX; check the model name and server settings",
  unknown: "Could not check availability; check oMLX connectivity and authentication",
  "no LLM": "Runs without an LLM server; TF-IDF still requires a trained classifier",
  "train first": "Train the TF-IDF classifier before starting this experiment",
};

const AVAILABILITY_PILL = {
  "ready to run": ["ready", "pill-good"],
  "model not downloaded": ["not available", "pill-bad"],
  "availability unknown": ["unknown", "pill-neutral"],
};

// Render a configuration status badge with its explanation.
function Pill({ text, tone = "pill-neutral" }) {
  return (
    <span className={`pill pill-xs ${tone}`} title={PILL_TITLES[text] || text}>
      {text}
    </span>
  );
}

// Describe model output, repair and availability settings.
function pillsFor(config) {
  if (config.kind !== "llm") return config.availability === "train classifier first"
    ? [["no LLM", "pill-neutral"], ["train first", "pill-bad"]]
    : [["no LLM", "pill-neutral"]];
  const [availability, tone] = AVAILABILITY_PILL[config.availability] ?? [
    config.availability,
    "pill-neutral",
  ];
  return [
    [config.output_mode === "JSON schema enforced" ? "schema" : "free", "pill-neutral"],
    ["no repair", "pill-neutral"],
    [availability, tone],
  ];
}

// Render grouped experiment choices.
export default function ConfigPicker({ configs, value, onChange }) {
  const groups = [
    ["Phishing and AI origin detection", configs.filter((c) => c.kind === "llm" && c.task === "phishing_origin")],
    ["Prompt injection detection", configs.filter((c) => c.kind === "llm" && c.task === "injection")],
    ["Rules and trained classifiers", configs.filter((c) => c.kind !== "llm")],
  ];

  if (!configs.length) return <p className="panel-hint">No experiment configs found.</p>;

  return (
    <div className="picker" role="radiogroup" aria-label="Experiment config">
      {groups.map(([title, items]) =>
        items.length ? (
          <div key={title}>
            <div className="picker-group">{title}</div>
            {items.map((config) => {
              const selected = config.value === value;
              return (
                <button
                  key={config.value}
                  role="radio"
                  aria-checked={selected}
                  title={config.label}
                  className={`picker-row${selected ? " is-selected" : ""}`}
                  onClick={() => onChange(selected ? "" : config.value)}
                >
                  <span className="picker-name">{config.name}</span>
                  <span className="picker-sub">{config.description || config.classifier_path || config.model_id || config.method}</span>
                  <span className="picker-pills">
                    {pillsFor(config).map(([text, tone]) => (
                      <Pill key={text} text={text} tone={tone} />
                    ))}
                  </span>
                </button>
              );
            })}
          </div>
        ) : null
      )}
    </div>
  );
}
