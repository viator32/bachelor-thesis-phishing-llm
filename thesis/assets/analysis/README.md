# Ergebnisanalyse

Berechnet alle Kennzahlen, Tabellen und Abbildungen der Arbeit aus den exportierten Läufen in
`thesis/assets/all-runs-results/`. Es werden keine Modelle aufgerufen.

## Umfang

| Teil | Läufe | Bewertungen |
|---|---:|---:|
| Hauptvergleich (50 LLM + 8 Baselines) | 58 | 49.680 |
| Gepaarter Injection-Test (10 LLM + 3 Baselines) | 13 | 12.870 |
| Injection-Erkennung auf denselben Paaren (10 LLM) | 10 | 9.900 |

Je Kombination aus Verfahren, Aufgabe und Testbestand geht genau ein vollständiger Lauf ein.

## Ausführen

Aus dem Repository-Stamm mit Python 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r thesis/assets/analysis/requirements.txt
.venv/bin/python thesis/assets/analysis/analyse.py \
  --personalization-proposals thesis/assets/analysis/personalisierung_vorschlaege.csv
```

Alle benötigten Ergebnisdaten liegen in `thesis/assets/all-runs-results/`.
`records.jsonl.gz` wird direkt gelesen; manuelles Entpacken ist nicht nötig.
Die SQLite-Datenbank des Prototyps ist optional und dient nur einem zusätzlichen Abgleich.
Die Ablage enthält die Eingaben für Berechnung, Abbildungen und Audit, keine vollständigen
Export-ZIPs mit Codekopien und doppelten CSV-Darstellungen.

## Ausgaben

| Datei | Inhalt |
|---|---|
| `metriken.csv`, `teilgruppen.csv` | Kennzahlen je Lauf bzw. Teilgruppe |
| `adaptervergleich.csv`, `baseline_vergleich.csv` | gepaarte Bootstrap-Differenzen |
| `gepaarte_injection.csv` | Ergebnisse des gepaarten Injection-Tests (ASR) |
| `gepaarte_injection_erkennung*.csv` | Injection-Erkennung auf denselben 990 Texten (Erkennung, Fehlalarm, abgefangene Angriffe) |
| `personalisierung_*.csv` | P0/P1-Zuordnung und Auswertung |
| `technische_metriken.csv`, `fehler.csv` | Latenz und die fünf fehlenden Antworten |
| `laufregister.csv`, `validierung.csv`, `manifest.json` | Laufauswahl, Prüfungen, Hashes |
| `tabellen.tex`, `anhang_ergebnisse.tex`, `gepaarter_injection_test.tex` | Tabellen für die Arbeit |
| `Anhang_Ergebnisse.pdf`, `figures/` | Abbildungen des Ergebnisanhangs |

Alle Dateien werden erzeugt; nicht von Hand ändern.

## Regeln der Berechnung

- Eine fehlende oder ungültige Antwort zählt als negative Entscheidung (kein Phishing, keine
  Injection, menschliche Herkunft).
- Angezeigter F1 ist das Mittel beider Klassen-F1 (Makro-F1); MCC steht separat.
- Personalisierung: binär P0/P1, 4.155 von 4.296 Eingaben zugeordnet, regelgestützt, ohne
  Zweitannotation.
