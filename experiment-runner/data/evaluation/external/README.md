# Externe Testbestände

Nie für Training verwendet. Hashes, Herkunft, Verteilungen und Ausschlüsse stehen in
[`manifest.json`](manifest.json).

| Datensatz | Aufgabe | Vorbereitete Texte |
|---|---|---|
| E-PhishLLM | Phishing (alle KI-generiert) | 16.606 (8.213 Phishing / 8.393 legitim) |
| Cross-Model | Herkunft (alle Phishing) | 9.840 (4.854 menschlich / 4.986 KI) |
| LLMail-Inject | Injection | 153.986 mit eindeutigem Label |

In der Arbeit wird je Bestand eine Stichprobe von 1.000 Texten (`--limit 1000`, Seed 42) verwendet.

## Dateien

| Aufgabe | Eingabe | Manifest |
|---|---|---|
| Phishing/Herkunft | `ephishllm_phishing_origin.jsonl` | `ephishllm_task_manifest.json` |
| Phishing/Herkunft | `crossmodel_phishing_origin.jsonl` | `crossmodel_task_manifest.json` |
| Injection | `llmail_injection.jsonl` | `llmail_task_manifest.json` |
| Gepaarter Test (Phishing-Entscheidung) | `ephishllm_paired_injection.jsonl` | `ephishllm_paired_injection_task_manifest.json` |
| Gepaarter Test (Injection-Erkennung) | `ephishllm_paired_injection_detection.jsonl` | `ephishllm_paired_injection_detection_task_manifest.json` |

Die Detektionsvariante enthält dieselben 990 Zeilen; Ziel ist `contains_prompt_injection`
(`clean` = false, `injected` = true). Configs: `configs/llm/*-injection*-paired-injection-detection.yaml`.

Für Regeln und TF-IDF liegen dieselben Texte unter `data/processed/`.

## Neu erzeugen

```bash
.venv/bin/python scripts/prepare_external_datasets.py [--download-llmail]
.venv/bin/python scripts/prepare_phishing_injection_pairs.py
.venv/bin/python scripts/prepare_paired_injection_detection.py
```

## Ausführen

```bash
.venv/bin/phishlab evaluate-adapter \
  --config configs/llm/qwen3-4b-phishing-origin-4bit.yaml \
  --dataset-manifest data/evaluation/external/ephishllm_task_manifest.json --limit 1000
```

Für LLMail eine Injection-Konfiguration, für die Basis die `*-base-4bit-*`-Konfiguration wählen.
`--limit` zieht eine reproduzierbare, nach Labels stratifizierte Stichprobe; die gewählten IDs
stehen im Run-Manifest.

## Grenzen

- **Ausschlüsse:** 10 E-PhishLLM- und 146 Cross-Model-Texte, die textgleich in den
  Trainingsquellen vorkommen. Ähnliche Texte sind damit nicht ausgeschlossen.
- **E-PhishLLM:** keine menschlichen Texte, daher keine vollständige Herkunftsauswertung.
- **Cross-Model:** nur Phishing, daher keine Falsch-Positiv-Rate; Labels nicht neu annotiert.
- **LLMail:** Labels aus API-Auslösung bzw. Modellbewertung, keine manuelle Goldannotation;
  `Unclear` bleibt ausgeschlossen. Ein Injection-Label ist kein Phishing-Label.

Quellen: E-PhishLLM `pajola2025ephishgen`, Cross-Model `gutierrez2026crossmodel`,
LLMail `abdelnabi2025llmail`.
