# QA Abstention Pipeline

Stanford CS 525 project studying when QA models should abstain from answering. Fine-tunes Qwen2.5-1.5B-Instruct with LoRA on different dataset compositions to learn when to output `<NO-ANSWER>` vs. provide an answer.

## Datasets

| Dataset | Size | Composition | Purpose |
|---------|------|-------------|---------|
| dataset1 | 10k | 100% answerable | Training (no abstention signal) |
| dataset2 | 10k | 50% answerable, 50% unanswerable | Training (balanced) |
| dataset3 | 10k | 50/50, 4 unanswerable types (1250 each) | Training (diverse types) |
| dataset3c | 10k | 70/30 answerable/unanswerable | Training (ablation) |
| dataset3e | 10k | 90/10 answerable/unanswerable | Training (ablation) |
| dataset4 | 2k | 50/50 (1k answerable + 1k unanswerable) | Eval only |

Unanswerable types: False Premise, Underspecified Context, Answer Unknown, Subjective.

## Setup

Upload the repo to Google Drive at `My Drive/abstention-data/`, then run notebooks on Colab with a GPU runtime (A100 recommended).

```
pip install -r requirements.txt
```

## Notebooks

Run in order on Colab (all training notebooks need GPU, notebook 06 does not):

| Notebook | Description | GPU |
|----------|-------------|-----|
| `01_setup_and_baseline.ipynb` | Baseline eval (no finetuning) on datasets 1/2/4 | Yes |
| `02_lora_dataset1_eval4.ipynb` | Train on D1, eval on D4 | Yes |
| `03_lora_dataset2_eval4.ipynb` | Train on D2, eval on D4 | Yes |
| `04_compare_results.ipynb` | EM metrics comparison (baseline, D1, D2) | No |
| `05_lora_dataset3_eval4.ipynb` | Train on D3 (50/50), eval on D4 | Yes |
| `06_eval_llm_judge.ipynb` | LLM-as-judge eval, comparison, plots (auto-discovers all runs) | No |
| `07_lora_dataset3c_eval4.ipynb` | Train on D3c (70/30), eval on D4 | Yes |
| `08_lora_dataset3e_eval4.ipynb` | Train on D3e (90/10), eval on D4 | Yes |

## Training Configuration

All training runs use identical hyperparameters for controlled comparison:

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Quantization | 4-bit |
| LoRA r / alpha | 16 / 32 |
| Learning rate | 1e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Epochs | 1 |
| Batch size | 4 |
| Gradient accumulation | 8 |
| Effective batch size | 32 |
| Train/val split | 9000/1000 |

## Evaluation

- **Exact Match (EM)**: Normalized text comparison after lowercasing, removing articles/punctuation
- **LLM-as-Judge**: GPT-5.2 judges semantic equivalence for answerable predictions (e.g. "slaughtered them" ~ "executed")
- **Abstention metrics**: Precision, recall, F1 for `<NO-ANSWER>` predictions
- **Per-type breakdown**: Abstention recall per unanswerable type

Notebook 06 auto-discovers all training runs from `outputs/notebooks/lora_*/eval/` — no manual editing needed when adding new runs.

## Project Structure

```
abstention-data/
  data/               # JSONL datasets
  src/
    abstention_pipeline/
      config.py       # Model name, dataset paths, system prompt
      data.py         # JSONL loading, field validation
      evaluation.py   # Benchmark runner, save/load predictions
      llm_judge.py    # LLM-as-judge (OpenAI API)
      metrics.py      # EM + judge metrics computation
      modeling.py      # Model/tokenizer loading, adapter loading
      prompting.py    # Chat template, prediction normalization
      training.py     # LoRA training with eval monitoring
      visualization.py # Training curve plotting
  notebooks/          # Colab notebooks
  requirements.txt
```
