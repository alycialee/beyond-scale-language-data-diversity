# Experiment 03: End-to-End Diversity Pipeline

Complete, dispatchable pipeline: **compute diversity coefficient** → **train model** → **evaluate on benchmarks** → **correlate diversity with performance**.

Inspired by the "Break it down in parts. And start dispatching them" philosophy
from the Gemini analysis document.

## The Pipeline

```
Dataset (UDACA/PileSubsets)
    │
    ▼
[Stage 1] Compute Task2Vec Diversity Coefficient
    │       (src/diversity/ — FIM diagonal extraction via probe network)
    │       Output: div_coeff scalar per dataset mixture
    │
    ▼
[Stage 2] Train Model from Scratch
    │       (src/training/train.py — GPT-2 or LLaMA-2 via HF Trainer)
    │       Output: trained checkpoint on HuggingFace
    │
    ▼
[Stage 3] Evaluate on Downstream Benchmarks
    │       (lm-evaluation-harness — MMLU with --log_samples)
    │       Output: per-sample log-likelihoods in JSONL
    │
    ▼
[Stage 4] Extract Soft Metrics & Correlate
            (experiments/02_div_vs_llm_bench/analyze.py)
            Output: log_p_correct, log_p_contrast, correlation plots
```

## Quick Start

```bash
# Give the runbook to Claude Code:
cat experiments/03_end_to_end_div_pipeline/run_pipeline.md

# Or run individual stages:
python experiments/03_end_to_end_div_pipeline/pipeline.py --stage diversity --dataset uspto
python experiments/03_end_to_end_div_pipeline/pipeline.py --stage train --dataset uspto --model gpt2
python experiments/03_end_to_end_div_pipeline/pipeline.py --stage eval --model_path UDACA/GPT2_117M_2.2B_USPTO
python experiments/03_end_to_end_div_pipeline/pipeline.py --stage analyze
```

## Files

- `pipeline.py` — Orchestration script: runs any/all stages
- `run_pipeline.md` — Claude Code runbook for fully automated execution
- `configs.py` — Dataset configs, training hyperparameters, stage definitions
