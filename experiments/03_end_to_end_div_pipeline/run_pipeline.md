# Claude Code Prompt: End-to-End Diversity → Training → Evaluation → Correlation Pipeline

**Give this file to Claude Code to execute the full causal experiment from the "Beyond Scale" paper.**

You are running the core causal experiment that proves data diversity drives downstream performance.
The pipeline has 4 stages, each dispatchable independently or chained end-to-end.

## Key References

- **Paper**: "Beyond Scale: The Diversity Coefficient" (Miranda, Lee et al.)
- **Core Insight**: Task2Vec diversity coefficient (Fisher Information Matrix diagonal) of training data causally predicts model performance
- **Soft Metrics**: Per Schaeffer & Miranda ("Are Emergent Abilities a Mirage?"), track log-likelihoods not accuracy — accuracy is a harsh, jumpy metric that hides smooth trends

## Overview of the 4 Stages

```
Stage 1: DIVERSITY  → Compute Task2Vec div_coeff for each dataset mixture
Stage 2: TRAIN      → Train model(s) from scratch on each dataset
Stage 3: EVAL       → Run MMLU with --log_samples for per-choice log-likelihoods
Stage 4: ANALYZE    → Extract soft metrics, correlate with div_coeff, generate plots
```

---

## Prerequisites

```bash
# Verify GPU access
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# Install dependencies
pip install -e ~/beyond-scale-language-data-diversity
pip install lm-eval wandb

# Verify imports
python -c "from diversity.task2vec import Task2Vec; print('Task2Vec OK')"
python -c "import lm_eval; print('lm_eval OK')"
```

---

## Stage 1: Compute Diversity Coefficient

Measures how diverse each training dataset is using the Task2Vec framework:
1. Sample random batches from the dataset
2. Pass each batch through a GPT-2 probe network
3. Extract the diagonal of the Fisher Information Matrix (= Task2Vec embedding)
4. Compute expected pairwise cosine distance between all embeddings
5. Result = diversity coefficient (scalar)

```bash
cd ~/beyond-scale-language-data-diversity

# Compute for each dataset (3 diversity levels)
python experiments/03_end_to_end_div_pipeline/pipeline.py \
    --stage diversity --dataset uspto --output_dir ./pipeline_output

python experiments/03_end_to_end_div_pipeline/pipeline.py \
    --stage diversity --dataset pubmed --output_dir ./pipeline_output

python experiments/03_end_to_end_div_pipeline/pipeline.py \
    --stage diversity --dataset uspto_pubmed --output_dir ./pipeline_output
```

**Expected results:**

| Dataset | Expected div_coeff |
|---------|-------------------|
| USPTO | ~0.158 |
| PubMed | ~0.168 |
| USPTO + PubMed | ~0.195 |

**Alternative:** If you only want to verify the diversity coefficients (they've already been computed), skip to Stage 2. The known values are hardcoded in `configs.py`.

---

## Stage 2: Train Models

Train a model from scratch on each dataset to create the causal intervention.
The key insight: same architecture, same compute budget, only the training data diversity changes.

```bash
# Quick test run (GPT-2, 5K steps, ~30 min)
python experiments/03_end_to_end_div_pipeline/pipeline.py \
    --stage train --dataset uspto --model gpt2_small --output_dir ./pipeline_output

python experiments/03_end_to_end_div_pipeline/pipeline.py \
    --stage train --dataset pubmed --model gpt2_small --output_dir ./pipeline_output

python experiments/03_end_to_end_div_pipeline/pipeline.py \
    --stage train --dataset uspto_pubmed --model gpt2_small --output_dir ./pipeline_output
```

For full reproduction (30K steps, ~10 hours each):
```bash
python experiments/03_end_to_end_div_pipeline/pipeline.py \
    --stage train --dataset uspto --model gpt2 --output_dir ./pipeline_output
# ... repeat for pubmed and uspto_pubmed
```

**Alternative:** If using the pre-trained models already on HuggingFace (UDACA org), skip directly to Stage 3.

---

## Stage 3: Evaluate on MMLU

Run lm-evaluation-harness with `--log_samples` to get per-choice log-likelihoods.
This is the critical step that enables soft metric analysis.

```bash
# Using pre-trained UDACA models (recommended — already trained at scale):
python experiments/03_end_to_end_div_pipeline/pipeline.py \
    --stage eval \
    --model_path UDACA/GPT2_117M_2.2B_USPTO \
    --output_dir ./pipeline_output

python experiments/03_end_to_end_div_pipeline/pipeline.py \
    --stage eval \
    --model_path UDACA/GPT2_117M_2.2B_PubMedAbs \
    --output_dir ./pipeline_output

python experiments/03_end_to_end_div_pipeline/pipeline.py \
    --stage eval \
    --model_path UDACA/GPT2_117M_2.2B_USPTOAndPubMedAbs \
    --output_dir ./pipeline_output
```

For the full model suite (all 26 models), use the batch runner from experiment 02:
```bash
bash experiments/02_div_vs_llm_bench/run_evals.sh \
    --output_dir ./pipeline_output/eval_results --gpu 0
```

**Output format (JSONL, per sample):**
```json
{
  "doc_id": 0,
  "target": "1",
  "resps": [
    [["-3.23", "False"]],
    [["-3.69", "False"]],
    [["-4.38", "False"]],
    [["-4.45", "False"]]
  ],
  "acc": 0.0
}
```
Where `resps[i][0][0]` = log P(choice i). This is what makes soft metric analysis possible.

---

## Stage 4: Analyze — Extract Soft Metrics & Correlate

```bash
python experiments/03_end_to_end_div_pipeline/pipeline.py \
    --stage analyze \
    --results_dir ./pipeline_output/eval_results \
    --output_dir ./pipeline_output
```

This runs `experiments/02_div_vs_llm_bench/analyze.py` under the hood, producing:

### Metrics extracted (in priority order):
1. **log_p_correct** = mean log P(correct choice) — the model's confidence in the right answer
2. **log_p_contrast** = mean [log P(correct) − mean log P(incorrect)] — how much more the model prefers the right answer over wrong ones
3. **accuracy** = argmax match (tracked for reference, NOT the primary metric)

### Outputs:
- `results.csv` — per-model metrics
- `div_coeff_vs_mmlu_log_p_correct.png/pdf` — diversity vs. log P(correct) with R², Pearson, Spearman, Kendall
- `div_coeff_vs_mmlu_log_p_contrast.png/pdf` — diversity vs. contrast score
- `correlation_summary.txt` — statistical summary

### What to verify:
- **Positive correlation**: higher div_coeff → higher log_p_correct and log_p_contrast
- **Smooth trend**: even if accuracy looks flat, log-likelihoods should show smooth improvement
- **Per-family R²**: within each model family (same architecture, different data), the correlation should be strong

---

## Run Everything at Once

```bash
# Full pipeline for one dataset/model combo:
python experiments/03_end_to_end_div_pipeline/pipeline.py \
    --stage all \
    --dataset uspto_pubmed \
    --model gpt2_small \
    --output_dir ./pipeline_output
```

This chains: diversity → train → eval → analyze in sequence.

---

## The Causal Argument

The key scientific claim this pipeline proves:

```
Same architecture + Same compute + Different data diversity = Different performance
                                                             ↑
                                               Diversity is the causal variable
```

By training models on USPTO (div=0.158), PubMed (div=0.168), and USPTO+PubMed
(div=0.195) and showing that downstream log-likelihoods increase monotonically
with diversity, we establish that **diversity is not merely correlated with but
causally drives model capability** — even when total training tokens are held constant.

This introduces a new axis to neural scaling laws: performance scales with
data diversity, not just data volume.

---

## File Inventory

| File | Purpose |
|------|---------|
| `pipeline.py` | Stage orchestrator (diversity/train/eval/analyze) |
| `configs.py` | Dataset configs, model configs, hyperparameters |
| `run_pipeline.md` | This file — Claude Code runbook |
| `README.md` | Experiment overview |
