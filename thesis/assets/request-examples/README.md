# Beispielanfragen an das Modell

JSON-Anfragekörper für `POST http://localhost:8080/v1/chat/completions`, offline erzeugt
(keine Inferenz, kein API-Schlüssel, keine Holdout-Daten).

| Datei | Inhalt |
|---|---|
| `llama31_phishing_origin_request.json` | Aufgabe `phishing_origin` mit Llama-Parametern |
| `qwen3_4b_injection_request.json` | Aufgabe `injection` mit Qwen3-4B-Parametern |

Aufbau: eine `system`-Nachricht und eine `user`-Nachricht mit dem gespeicherten `input_text` und
der Formatvorgabe. Das Modell erhält keine Labels, Gruppen-IDs oder `expected_action`.
