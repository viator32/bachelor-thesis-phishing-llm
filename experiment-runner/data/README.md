# Data

Email texts are not in Git (redistribution rights unverified); manifests, hashes and source
lists are.

```text
sources/datasets/      all raw datasets: training/ (PhishFuzzer, Contextual Paired),
                       experiments/ (E-PhishLLM, Cross-Model, LLMail-Inject)
evaluation/task_adapters/  internal validation/test splits for LLM runs
evaluation/external/   external test sets and the paired injection test
evaluation/personalization/  P0/P1 annotations (read by the thesis analysis)
processed/             the same test emails in the baseline format (rules, TF-IDF)
training/              training split from Colab, used to train TF-IDF
metadata/              IDs excluded from external tests (leakage audit)
```

## Rules

- `is_phishing`, `generation_origin` and `contains_prompt_injection` are independent labels.
  A missing label stays `null`; metrics only score available labels.
- Variants of the same email or pair always stay in one split.
- Test data is never used for training, prompt design or threshold tuning.
- Labels and source metadata never reach the model prompt.
