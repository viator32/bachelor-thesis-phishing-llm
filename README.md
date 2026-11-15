# Bachelorarbeit

**LLM-basierte Verfahren zur Erkennung KI-generierter Spear-Phishing-Mails unter
Berücksichtigung von Personalisierung und Prompt Injection**

```text
bachelor/
├── model-training/     QLoRA-Training, Merge und MLX-Konvertierung (Colab)
├── experiment-runner/  Prototyp: CLI, API und Dashboard für die Läufe (oMLX)
└── thesis/             Arbeit (thesis.tex), Literatur, Folien
    └── assets/         Auswertung, Läufe, Diagramme, Quellen-PDFs
```

## Ablauf

1. **Trainieren** in Colab: [model-training](model-training/README.md).
2. **Läufe ausführen** mit dem [experiment-runner](experiment-runner/README.md) und als ZIP nach
   `thesis/assets/all-runs-results/` exportieren.
3. **Auswerten** (aus dem Repository-Stamm):

   ```bash
   experiment-runner/.venv/bin/python thesis/assets/analysis/analyse.py \
     --personalization-proposals thesis/assets/analysis/personalisierung_vorschlaege.csv
   ```

   Details: [Ergebnisanalyse](thesis/assets/analysis/README.md).
4. **Arbeit bauen:** `bash thesis/build.sh` (pdfLaTeX, `biber`, `latexmk`; Hilfsdateien in `thesis/build/`).

## Hinweise

- Dateien in `thesis/assets/analysis/` (Tabellen, Abbildungen, `Anhang_Ergebnisse.pdf`) erzeugt
  `analyse.py`; nicht von Hand ändern.
- Die für die Analyse benötigten Eingaben, Referenzlabels und Vorhersagen liegen in
  `thesis/assets/all-runs-results/`; vollständige Datensätze sind verlustfrei als
  `records.jsonl.gz` komprimiert. Die Auswertung liest diese Dateien direkt.
- `experiment-runner/results/` und die SQLite-Datenbank bleiben lokal. Die Neuberechnung
  der Kennzahlen und Abbildungen benötigt weder diese Datenbank noch neue Modellaufrufe.
- Modelle: Qwen3-4B, Qwen3-8B, Mistral-7B-Instruct-v0.3, Phi-4-mini-instruct,
  Llama-3.1-8B-Instruct (Hugging-Face-Freigabe nötig), jeweils Basis und QLoRA.
- Läufe mit `schema_retries: 0` wiederholen, das angepasste Phi-4-mini mit `3`.
