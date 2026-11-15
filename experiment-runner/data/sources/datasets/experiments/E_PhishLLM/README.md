# E-PhishLLM

Externer Phishing-Testbestand. Alle E-Mails sind laut Publikation mit GPT-4o-mini erzeugt.

`ephishLLM.json`: 16.616 Einträge, 8.218 Phishing (`type = 1`) / 8.398 legitim (`type = 0`);
11.502 Englisch, 2.701 Italienisch, 2.349 Deutsch, 64 weitere.

- **Felder:** `Subject`, `Body`, `type`, `Language`. Kein Absender, keine URLs, keine
  Personalisierungs- oder Injection-Labels.
- **Labels:** `is_phishing` aus `type`; `generation_origin = llm` für alle Einträge.
- **Grenzen:** keine menschlichen Gegenbeispiele; keine Persona-ID für gruppierte Splits;
  keine Lizenzdatei im offiziellen Repository.

Paper: https://arxiv.org/abs/2509.01791
