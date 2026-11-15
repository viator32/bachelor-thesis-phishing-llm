# Eingaben der Ergebnisanalyse

Diese Ablage enthält die gespeicherten Daten, die `../analysis/analyse.py` für die
Kennzahlen, Abbildungen und die Prüfung der Laufauswahl verwendet. Sie ist eine
bereinigte Zusammenstellung der Laufexporte, kein vollständiges Exportarchiv.

## Umfang

- 50 LLM-Hauptläufe auf den fünf Testbeständen.
- Zehn gepaarte Phishing-/Herkunftsläufe und zehn gepaarte Injection-Erkennungsläufe.
- 35 Baseline-Exporte. Die Analyse wählt daraus acht Haupt- und drei gepaarte Läufe;
  die übrigen Exporte dokumentieren abweichende Eingaben oder gleichwertige Wiederholungen.

## Dateien je Lauf

| Datei | Verwendung |
|---|---|
| `records.jsonl.gz` | Unveränderte vollständige Ergebniszeilen mit Eingabe, Referenz, Vorhersage und Messwerten; verlustfrei komprimiert |
| `metrics.json` | Gespeicherte Kennzahlen zum Abgleich und Baseline-Metadaten |
| `task/run_manifest.json` | Aufgaben- und Modellkonfiguration, ausgewählte IDs und protokollierte Prüfsummen |
| `task/predictions.jsonl` | Native Einzelvorhersagen, Rohantworten, Fehler und Laufzeiten der LLMs |
| `task/report.json` | Gespeicherter nativer Bericht zur Prüfung von Umfang und Kennzahlen |

Je nach Auswertungspfad sind nur die benötigten Dateien vorhanden. Insbesondere
benötigen die gepaarten Injection-Erkennungsläufe nur die drei Dateien unter `task/`.

Codekopien, doppelte CSV-Darstellungen, normalisierte `comparison`-Kopien,
Export-Inventare, leere Notizen und die globale Laufübersicht wurden entfernt.
Die Analyse verwendet den gemeinsamen Code unter `../analysis/` und im Prototyp.
Die Laufmanifeste behalten die ursprünglich protokollierten Code- und Prompt-Prüfsummen.
Die Inhalte der beibehaltenen Ergebnis- und Konfigurationsdateien wurden nicht verändert.

## Neuberechnung

Vom Repository-Stamm mit den Abhängigkeiten aus `../analysis/requirements.txt`:

```bash
python thesis/assets/analysis/analyse.py \
  --personalization-proposals thesis/assets/analysis/personalisierung_vorschlaege.csv
```

Gzip-Dateien werden direkt gelesen. Die SQLite-Datenbank und erneute Modellaufrufe
sind für die Neuberechnung nicht erforderlich. Die Ergebniszeilen enthalten
Nachrichtentexte aus den dokumentierten Forschungsquellen.
