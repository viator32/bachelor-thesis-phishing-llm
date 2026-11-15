# Training und MLX-Konvertierung (Colab)

Alle Schritte laufen in Google Colab; auf dem Mac werden nur die fertigen MLX-Ordner in oMLX
eingebunden. Zellen jeweils von oben nach unten ausführen.

1. **Adapter trainieren** (GPU), `TASK = 'phishing_origin'` oder `'injection'`:
   [Qwen3-4B](colab_qlora_qwen3_4b_configurable.ipynb) ·
   [Qwen3-8B](colab_qlora_qwen3_8b_configurable.ipynb) ·
   [Mistral-7B](colab_qlora_mistral_7b_instruct_configurable.ipynb) ·
   [Phi-4-mini](colab_qlora_phi4_mini_instruct_configurable.ipynb) ·
   [Llama-3.1-8B](colab_qlora_llama31_8b_instruct_configurable.ipynb)
2. **Adapter mit dem Basismodell zusammenführen** (GPU):
   [Merge](colab_merge_adapter_for_omlx.ipynb) → `merged_hf/` im Run-Ordner.
3. **Nach MLX/4 Bit konvertieren:**
   [angepasste Modelle](colab_convert_merged_for_omlx.ipynb) ·
   [Basismodelle ohne Adapter](colab_download_base_models_for_omlx.ipynb).
4. Ordner aus `omlx_ready/` in Drive nach `~/.omlx/models/` kopieren.

Läufe und Ergebnisse liegen in Drive unter `llm_phishing_finetune_workflow_v3/runs_v4/`.
Verlauf aller Trainingsläufe: [Trainingsjournal](model_training_progress.ipynb).
