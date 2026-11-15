# Experiment runner

Research prototype that evaluates phishing, origin and prompt-injection detection with local
LLMs (oMLX on Apple Silicon), a keyword rule baseline and a TF-IDF baseline. It never opens
URLs, renders HTML or sends mail.

```text
src/llm_phishing/   Python core for the CLI (`phishlab`) and the FastAPI server
frontend/           React dashboard
configs/llm/        LLM runs: one file per model, variant (base/adapted) and task
configs/baselines/  rule and TF-IDF runs
configs/baseline_training/  TF-IDF training settings
scripts/            one-off preparation of test sets and baseline inputs
data/               test sets and dataset sources (see data/README.md)
models/             trained TF-IDF classifier (not in Git)
results/            run outputs and dashboard database (not in Git)
```

## Setup

Needs Python 3.11 and [oMLX](https://omlx.ai) running on the host (`localhost:8080`).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # set MLX_API_KEY (oMLX Settings > API Authentication)
phishlab check-mlx --model Qwen3-4B-4bit
```

Docker (API on 127.0.0.1:8001, dashboard on 127.0.0.1:5173; oMLX stays on the host):

```bash
cp .env.example .env    # MLX_BASE_URL=http://host.docker.internal:8080
docker compose up -d --build
```

## Running

```bash
# LLM (base and adapted get identical inputs, prompt, schema and token budget)
phishlab evaluate-adapter --config configs/llm/qwen3-4b-phishing-origin-4bit.yaml \
  [--dataset-manifest data/evaluation/external/ephishllm_task_manifest.json] [--limit 1000]

# Baselines
phishlab run --config configs/baselines/rules.yaml
phishlab run --config configs/baselines/tfidf.yaml
phishlab train-tfidf-colab --name phishfuzzer-all-42   # trains on data/training/phishing_origin/
```

Resume an interrupted run with the same `--run-id`. A response that does not match the JSON
schema is requested again up to `schema_retries` times; nothing is repaired. Thesis runs used
`schema_retries: 0`, except adapted Phi-4-mini (3).

## Outputs

```text
results/task_adapters/<run_id>/   LLM runs: run_manifest.json, predictions.jsonl, report.json
results/jsonl/, results/reports/  baseline records and metrics
results/experiments.sqlite        dashboard store
```

The dashboard export (`GET /runs/export`) is what lives in `thesis/assets/all-runs-results/`.

## Rule baseline

Fixed points per matched category; `score >= 40` means phishing. Weights are hand-picked, not
calibrated (patterns in `src/llm_phishing/feature_extractor.py`).

| Signal | Points |
|---|---:|
| Urgency | 15 |
| Credentials | 30 |
| Payment | 25 |
| Instruction override | 35 |
| Link text domain ≠ target host | 15 each, max. 25 |
| Sender ≠ Reply-To domain | 15 |

## Limitations

Token counts for truncation are approximated offline; memory use is not measured; models
return labels, not calibrated risk scores.
