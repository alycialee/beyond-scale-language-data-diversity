# Experiment 02: Diversity Coefficient vs. LLM Benchmark Soft Metrics

Does training on more diverse data improve downstream benchmark performance?

## Key Insight

Accuracy is a harsh, emergent metric that can hide smooth trends (Schaeffer &
Miranda 2023, "Are Emergent Abilities a Mirage?"). Instead we track **soft
metrics** — log-likelihoods — that reveal continuous improvement even when
accuracy appears flat:

1. **log P(correct)** — how much probability mass the model puts on the right answer
2. **log P(correct) − mean log P(incorrect)** — how much more the model prefers the right answer over wrong ones

## Setup

26 models trained at 3 diversity levels (Task2Vec diversity coefficient):

| Training Data | Diversity Coeff |
|---|---|
| USPTO | 0.158 |
| PubMed | 0.168 |
| USPTO + PubMed | 0.195 |

Models: GPT-2 (51M, 117M, 204M, 345M, 810M, 1.5B) + LLaMA-2 7B, all on
HuggingFace under `UDACA/`.

## Quick Start

```bash
# 1. Run evaluations (or use existing results)
bash run_evals.sh --output_dir ./results --gpu 0

# 2. Extract metrics + plot
python analyze.py all --results_dir ./results
```

Or give `run_pipeline.md` to Claude Code for fully automated execution.

## Files

- `models.py` — model registry (diversity coefficients, families, colors)
- `analyze.py` — metric extraction, correlation analysis, plotting
- `run_evals.sh` — batch lm_eval runner
- `run_pipeline.md` — Claude Code prompt for end-to-end execution
