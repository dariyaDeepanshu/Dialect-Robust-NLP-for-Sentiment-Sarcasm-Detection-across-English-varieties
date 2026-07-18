# BESSTIE-CW-26: Sarcasm Detection across English Varieties

NLP coursework (COMM061, University of Surrey) for sarcasm detection across en-UK, en-AU, and en-IN using classical baselines, fine-tuned DistilBERT, and Qwen2.5-1.5B + LoRA adapters.

## Notebooks

| # | Notebook | Content |
|---|---|---|
| Q1 | `01_dataset_analysis.ipynb` | Distribution visualisation, vocabulary overlap (Jaccard/TF-IDF), linguistic distance |
| Q2.1 | `02_baseline_vs_transformer.ipynb` | TF-IDF + LR/SVM baselines vs fine-tuned DistilBERT (67M) |
| Q2.2 | `03_cross_variety_evaluation.ipynb` | 3x3 cross-variety Sarcasm matrix |
| Q2.3 | `04_lora_adapters.ipynb` | LoRA adapters (r=16) per variety on Qwen2.5-1.5B-Instruct |
| Q3 | `05_evaluation.ipynb` | Comprehensive metrics, heatmaps, confusion matrices |
| Q4 | `06_error_analysis.ipynb` | Error analysis with 4-shot prompting |

## Deployment (Q5)

```bash
uv run python deployment/app.py      # Gradio web service
uv run python deployment/benchmark.py # Inference timing
```

LoRA adapters are hosted on HuggingFace Hub (`Dush91/besstie-sarcasm-lora-*`) and loaded via `from_pretrained()`.

## Setup

```bash
uv sync
```
