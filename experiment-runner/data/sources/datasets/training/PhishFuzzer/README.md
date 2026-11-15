# PhishFuzzer

Trainingsquelle für den gemeinsamen Phishing-/Herkunftsadapter.

- `PhishFuzzer_emails_original_seed_v1.json`: 3.300 Seed-E-Mails (1.126 Phishing, 1.074 Spam,
  1.100 Valid), überwiegend menschlich.
- `PhishFuzzer_emails_entity_rephrased_v1.json`: 19.800 LLM-Varianten, sechs je Seed
  (gleiche `Original_ID`).

**Verwendung in der Arbeit:** nur Phishing/Valid; je Seed mit `Created by = Human` eine
zufällige LLM-Variante, sodass Mensch und LLM gleich häufig sind. Split gruppiert nach
`Original_ID`.

**Felder im Prompt:** `Subject`, `Sender`, `Body`, `URL`, `File`. Metadaten wie `Created by`,
`Source`, `Year`, `Entity_Type` gelangen nicht in den Prompt.

**Hinweise:** 75 exakte Duplikate unter den Seeds (werden entfernt), zwei leere Seeds, mögliche
Überschneidung mit Cross-Model-Quellen, keine Lizenzdatei.

Paper: https://arxiv.org/abs/2511.21448
