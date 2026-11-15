# Contextual Paired Email Injection

Trainingsquelle für den Injection-Adapter. Kein Phishing-Datensatz.

`paired_emails_dataset.json`: 4.200 Paare (8.400 E-Mails), je eine Angriffs- und eine legitime
Variante.

- **Felder:** `id`, `scenario`, `category`, `pair.attack` / `pair.benign` mit `user_prompt`
  (vertrauenswürdige Aufgabe) und `external_content` (E-Mail).
- **Labels:** Angriff `contains_prompt_injection = true`, legitim `false`.
- **Eingabe:** Aufgabe und E-Mail getrennt markiert; `expected_action`, `injection_technique`
  und `category` gelangen nicht in den Prompt.
- **Split:** nach `id`, beide Varianten im selben Split.
- **Hinweise:** 21 Kategorienamen, teils Schreibvarianten derselben Kategorie; keine Lizenzdatei.

Paper: https://arxiv.org/abs/2605.17634
