# Rohdatensätze

| Datensatz | Ordner | Verwendung |
|---|---|---|
| PhishFuzzer | `training/PhishFuzzer` | Training Phishing + Herkunft (Phishing/Valid) |
| Contextual Paired Email Injection | `training/Contextual_Paired_Email_Injection` | Training Injection |
| E-PhishLLM | `experiments/E_PhishLLM` | externer Phishing-Test |
| Cross-Model-Phishing | `experiments/CrossModel_AI_Phishing_2026` | externer Herkunftstest |
| LLMail-Inject | `experiments/LLMail_Inject` | externer Injection-Test |

`SOURCES.csv` enthält Download- und Paper-URLs, `CHECKSUMS.sha256` die Prüfsummen der
Trainingsdaten und von E-PhishLLM. Jeder Datenordner hat ein eigenes README.
Aufbereitung der Testbestände: [externe Testbestände](../../evaluation/external/README.md).

## Regeln

- Labels sind unabhängig; fehlende Labels werden nicht ergänzt.
- PhishFuzzer wird nach `Original_ID` gruppiert gesplittet, Injection-Paare nach `id`.
- Testdaten (`experiments/`) nie für Training, Promptwahl oder Schwellenwerte verwenden.
- LLMail-Inject-Dateien (517 MB) liegen nicht in Git:
  `scripts/prepare_external_datasets.py --download-llmail`.
