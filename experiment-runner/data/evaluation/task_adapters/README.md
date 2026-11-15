# Interne Validierungs- und Testdaten

Byte-identische Kopien der Splits aus den Colab-Trainingsläufen. Nicht in Git; nach einem neuen
Checkout aus den Run-Ordnern kopieren und mit [`manifest.json`](manifest.json) abgleichen.

| Aufgabe | Validation | Test |
|---|---:|---:|
| Phishing/Herkunft | 416 | 424 |
| Injection | 824 | 872 |

Das Modell erhält nur `input_text`; `target` ist die Referenz.
