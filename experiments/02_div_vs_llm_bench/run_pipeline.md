# Claude Code Prompt: Diversity Coefficient vs. Benchmark Soft Metrics Pipeline

**Give this file to Claude Code to run the full experiment end-to-end.**

You are running an experiment to test whether higher Task2Vec diversity coefficient
of training data correlates with better downstream benchmark performance. The key
insight (from Schaeffer & Miranda, "Are Emergent Abilities a Mirage?") is that
**accuracy is a harsh, jumpy metric** — we must track the continuous log-likelihood
of the correct answer to see smooth scaling trends.

## Metrics to track (in priority order)

1. **log P(correct choice)** — the log-likelihood the model assigns to the correct MMLU answer
2. **log P(correct) − mean log P(incorrect)** — contrast score, how much more likely the correct answer is vs. wrong ones
3. **Accuracy** — tracked for reference only, NOT the primary metric

## Pre-trained models

All models are on HuggingFace under the `UDACA` org. They were trained on Pile
subsets at 3 diversity levels:

| Training Data | Task2Vec Diversity Coefficient |
|---|---|
| USPTO only | 0.158 |
| PubMed only | 0.168 |
| USPTO + PubMed | 0.195 |

See `models.py` in this directory for the full model list (26 models total:
GPT-2 variants from 51M to 1.5B, plus LLaMA-2 7B).

---

## Step 0 — Environment setup

```bash
# Check GPU availability
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# Create/activate conda environment with lm-evaluation-harness
conda create -n lm_eval_bench python=3.11 -y
conda activate lm_eval_bench
pip install lm-eval transformers torch accelerate

# Verify lm_eval is available
lm_eval --help | head -5
```

---

## Step 1 — Run MMLU evaluations on all models

Run the batch evaluation script. It automatically skips models that already have results.

```bash
cd ~/beyond-scale-language-data-diversity/experiments/02_div_vs_llm_bench

# Dry run first — see what will be evaluated
bash run_evals.sh --dry-run --output_dir ./results --gpu 0

# Actually run (this will take a while — ~10-20 min per GPT-2 model, longer for LLaMA-2)
bash run_evals.sh --output_dir ./results --gpu 0
```

**Important flags used by `run_evals.sh`:**
- `--log_samples` — captures per-question log-likelihoods in JSONL format (critical!)
- `--batch_size 8` for GPT-2 models, `--batch_size 2` for LLaMA-2
- Results are saved to `./results/<model_name>/`

If running on SNAP cluster with DFS, use:
```bash
bash run_evals.sh --output_dir /dfs/scratch0/brando9/data/beyond_scale/eval_results --gpu 0
```

### Parallelizing across GPUs

If multiple GPUs are free, run models in parallel:
```bash
# Terminal 1 (GPU 0): GPT-2 small models
GPU=0 OUTPUT_DIR=./results bash -c '
for m in GPT2_51M_1.31B_USPTO GPT2_51M_1.31B_PubMedAbs GPT2_51M_1.31B_USPTOAndPubMedAbs; do
    CUDA_VISIBLE_DEVICES=$GPU lm_eval --model hf --model_args pretrained=UDACA/$m \
        --tasks mmlu --device cuda --batch_size 8 --output_path $OUTPUT_DIR/$m --log_samples
done
'

# Terminal 2 (GPU 1): LLaMA-2 models
GPU=1 OUTPUT_DIR=./results bash -c '
for m in LLama2_Uspto_Ckpt_1 LLama2_Pubmed_Ckpt_2 LLama2_Pubmed_Ckpt_7; do
    CUDA_VISIBLE_DEVICES=$GPU lm_eval --model hf --model_args pretrained=UDACA/$m \
        --tasks mmlu --device cuda --batch_size 2 --output_path $OUTPUT_DIR/$m --log_samples
done
'
```

---

## Step 2 — Extract soft metrics from JSONL results

```bash
cd ~/beyond-scale-language-data-diversity/experiments/02_div_vs_llm_bench

python analyze.py extract --results_dir ./results --csv results.csv
```

This reads every `samples_*.jsonl` file produced by lm_eval and computes:
- `log_p_correct`: mean log P(correct choice) over all MMLU questions
- `log_p_contrast`: mean [log P(correct) − mean log P(incorrect)]
- `acc`: accuracy (argmax match, for reference)

Output: `results.csv` with columns: `model, family, div_coeff, log_p_correct, log_p_contrast, acc, n_samples`

---

## Step 3 — Generate correlation plots and summary

```bash
python analyze.py plot --csv results.csv --output_dir .
```

This produces:
- `div_coeff_vs_mmlu_log_p_correct.png/pdf` — diversity vs. log P(correct) with linear fits
- `div_coeff_vs_mmlu_log_p_contrast.png/pdf` — diversity vs. contrast score
- `div_coeff_vs_mmlu_acc.png/pdf` — diversity vs. accuracy (for comparison)
- `correlation_summary.txt` — text report with R², Pearson, Spearman, Kendall stats

Each plot has two panels:
1. **Left**: all models with global linear fit + correlation stats
2. **Right**: per-family fits (GPT2-51M, GPT2-117M, ..., LLaMA2-7B) with R² per family

Or run everything in one command:
```bash
python analyze.py all --results_dir ./results --csv results.csv --output_dir .
```

---

## Step 4 — Verify results

Check that:

1. **Positive correlation**: higher div_coeff → higher log_p_correct and log_p_contrast
2. **R² is meaningful**: especially for per-family fits where model size is controlled
3. **Smooth trend in log-likelihoods**: even if accuracy looks flat across diversity levels, log P(correct) should show a smooth increasing trend
4. **Sample counts**: all models should have ~14,042 samples (57 MMLU subjects × ~246 questions each)

```bash
cat correlation_summary.txt
```

Key expected outcome: log-likelihood metrics show a clearer positive correlation with
diversity than accuracy does, confirming that accuracy is too harsh a metric to
capture the smooth effect of data diversity on model quality.

---

## Step 5 — (Optional) Extend to more benchmarks

To add ARC, HellaSwag, or WinoGrande:

```bash
# Run with multiple tasks
bash run_evals.sh --tasks "mmlu,arc_challenge,hellaswag" --output_dir ./results --gpu 0

# The analyze.py script will need minor extension to parse non-MMLU JSONL
# (the format is the same — resps + target — but file paths differ)
```

---

## File inventory

| File | Purpose |
|------|---------|
| `models.py` | Model registry: name → div_coeff, family, colors |
| `analyze.py` | Extract metrics from JSONL, compute correlations, generate plots |
| `run_evals.sh` | Batch runner for lm_eval across all models |
| `run_pipeline.md` | This file — the Claude Code prompt |
| `README.md` | Experiment overview |
| `results.csv` | (generated) Per-model metrics |
| `correlation_summary.txt` | (generated) Statistical summary |
| `div_coeff_vs_mmlu_*.png/pdf` | (generated) Correlation plots |

---

## References

- Task2Vec Diversity Coefficient: https://arxiv.org/abs/2306.13840
- "Are Emergent Abilities of Large Language Models a Mirage?": Schaeffer, Miranda, Koyejo (2023)
- EleutherAI lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- Polo et al. (2024) log-likelihood scoring: https://arxiv.org/abs/2406.04391
