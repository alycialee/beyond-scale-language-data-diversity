# Div Coeff vs MMLU Log-Likelihood Benchmark Scores

**TL;DR:** Run MMLU evals (log-likelihood scoring) on all trained GPT-2 and LLaMA-2 models from HuggingFace (UDACA org), aggregate per-model MMLU accuracy, map to Task2Vec diversity coefficient of training data, and produce a scatter plot with linear fit + R². Confirms whether higher training data diversity → higher MMLU benchmark accuracy.

---

## Context

We trained models at 3 diversity levels using Pile subsets:

| Training Data | Task2Vec Diversity Coefficient |
|---|---|
| USPTO only | 0.158 |
| PubMed only | 0.168 |
| USPTO + PubMed | 0.195 |

Models already trained and pushed to HuggingFace under the `UDACA` org:

```python
diversity_coefficients = {
    # GPT-2 variants
    "UDACA/GPT2_51M_1.31B_USPTO":               0.158,
    "UDACA/GPT2_51M_1.31B_PubMedAbs":           0.168,
    "UDACA/GPT2_51M_1.31B_USPTOAndPubMedAbs":   0.195,
    "UDACA/GPT2_51M_557M_USPTO":                0.158,
    "UDACA/GPT2_51M_557M_PubMedAbs":            0.168,
    "UDACA/GPT2_51M_557M_USPTOAndPubMedAbs":    0.195,
    "UDACA/GPT2_117M_2.2B_USPTO":               0.158,
    "UDACA/GPT2_117M_2.2B_PubMedAbs":           0.168,
    "UDACA/GPT2_117M_2.2B_USPTOAndPubMedAbs":   0.195,
    "UDACA/GPT2_204M_USPTO":                    0.158,
    "UDACA/GPT2_204M_PubMedAbs":                0.168,
    "UDACA/GPT2_204M_USPTOAndPubMedAbs":        0.195,
    "UDACA/GPT2_345M_2.2B_USPTO":               0.158,
    "UDACA/GPT2_345M_2.2B_PubMedAbs":           0.168,
    "UDACA/GPT2_345M_2.2B_USPTOAndPubMedAbs":   0.195,
    "UDACA/GPT2_810M_PubMedAbs":                0.168,
    "UDACA/GPT2_810M_2.2B_USPTOAndPubMedAbs":   0.195,
    "UDACA/GPT2_1.5B_180M_USPTO":               0.158,
    "UDACA/GPT2_1.5B_180M_PubMedAbs":           0.168,
    "UDACA/GPT2_1.5B_180M_USPTOAndPubMedAbs":   0.195,
    # LLaMA-2 7B variants
    "UDACA/LLama2_Uspto_Ckpt_1":                0.158,
    "UDACA/LLama2_Pubmed_Ckpt_2":               0.168,
    "UDACA/LLama2_Uspto_Pubmed_Ckpt_3":         0.195,
    "UDACA/LLama2_Uspto_Pubmed_Ckpt_4":         0.195,
    "UDACA/LLama2_Uspto_Pubmed_Ckpt_5":         0.195,
    "UDACA/LLama2_Uspto_Pubmed_Ckpt_6":         0.195,
    "UDACA/LLama2_Pubmed_Ckpt_7":               0.168,
}
```

Partial MMLU eval results (log-likelihood JSONL) may already exist at:
```
/lfs/skampere1/0/brando9/data/beyond_scale/eval_results_back_up/   # original location (skampere1-local)
/dfs/scratch0/brando9/data/beyond_scale/eval_results/               # canonical DFS location (all servers)
```

Existing CE loss + PPL plots (already showing positive correlation) are in:
```
~/beyond-scale-language-data-diversity/src/data_analysis/
```

The metric used is `-log P^{Vocab}(CorrectChoice)` — the log-likelihood the model assigns to the correct MMLU answer choice. Accuracy = argmax over choices matches target.

**Canonical results location going forward: `/dfs/scratch0/brando9/data/beyond_scale/eval_results/`**
This DFS path is accessible from all SNAP servers (skampere1, skampere2, ampere8, mercury1, etc.).

---

## Step 0 — Discover hardware and set up environment

```bash
hostname
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
# pick a free GPU (memory.used < 1000 MiB)

conda activate eleuther_lm_eval_harness_20240927
# if env doesn't exist on this server:
conda create -n eleuther_lm_eval_harness_20240927 python=3.11 -y
conda activate eleuther_lm_eval_harness_20240927
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness /lfs/$(hostname -s)/0/brando9/lm-evaluation-harness
cd /lfs/$(hostname -s)/0/brando9/lm-evaluation-harness && pip install -e . && cd ~
```

---

## Step 0.5 — Copy existing results from skampere1 to DFS (run once, then results are everywhere)

```bash
# Create DFS destination
mkdir -p /dfs/scratch0/brando9/data/beyond_scale/eval_results

# Copy from skampere1 (run this from any server that can reach skampere1 via SSH)
rsync -av --progress \
    brando9@skampere1.stanford.edu:/lfs/skampere1/0/brando9/data/beyond_scale/eval_results_back_up/ \
    /dfs/scratch0/brando9/data/beyond_scale/eval_results/

# Verify
ls /dfs/scratch0/brando9/data/beyond_scale/eval_results/
```

If you are already on skampere1, copy directly:
```bash
mkdir -p /dfs/scratch0/brando9/data/beyond_scale/eval_results
rsync -av /lfs/skampere1/0/brando9/data/beyond_scale/eval_results_back_up/ \
          /dfs/scratch0/brando9/data/beyond_scale/eval_results/
```

After this step, all servers can read from `/dfs/scratch0/brando9/data/beyond_scale/eval_results/`.

---

## Step 1 — Check what eval results already exist

```bash
# Check DFS (canonical, accessible from all servers):
ls /dfs/scratch0/brando9/data/beyond_scale/eval_results/ 2>/dev/null || echo "no DFS results yet — run Step 0.5"
# Also check skampere1 original backup (if on skampere1):
ls /lfs/skampere1/0/brando9/data/beyond_scale/eval_results_back_up/ 2>/dev/null || echo "not on skampere1 or path missing"
```

For each model that already has results, verify the JSONL contains `acc` and `resps` fields (log-likelihoods per choice). Skip re-running those.

---

## Step 2 — Run MMLU evals for missing models

Output dir: `/dfs/scratch0/brando9/data/beyond_scale/eval_results/` (DFS — shared across all servers)

```bash
OUTPUT_DIR=/dfs/scratch0/brando9/data/beyond_scale/eval_results
mkdir -p $OUTPUT_DIR

# Example for one model (adapt GPU index as needed):
CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model hf \
    --model_args pretrained=UDACA/GPT2_117M_2.2B_USPTO \
    --tasks mmlu \
    --device cuda \
    --batch_size 8 \
    --output_path $OUTPUT_DIR/GPT2_117M_2.2B_USPTO \
    # NOTE: OUTPUT_DIR=/dfs/scratch0/brando9/data/beyond_scale/eval_results — written to DFS so all servers see it
    --log_samples
```

Run this for every model in the `diversity_coefficients` dict above that is missing results.
For LLaMA-2 models, use `--batch_size 2` and a single A100 GPU.

**Important:** use `--log_samples` so we get per-sample log-likelihoods, not just aggregate accuracy.

---

## Step 3 — Aggregate MMLU accuracy per model

Write a script `src/data_analysis/aggregate_mmlu_results.py` that:

1. Reads each model's result directory (from `lm_eval` output JSON or JSONL)
2. Extracts mean MMLU accuracy across all subjects
3. Maps model name → div_coeff using the dict above
4. Outputs a CSV: `model_name, div_coeff, mmlu_acc`

Expected output format:
```
model_name,div_coeff,mmlu_acc
GPT2_117M_2.2B_USPTO,0.158,0.243
GPT2_117M_2.2B_PubMedAbs,0.168,0.251
GPT2_117M_2.2B_USPTOAndPubMedAbs,0.195,0.267
...
```

---

## Step 4 — Plot div_coeff vs MMLU accuracy with R²

Write `src/data_analysis/plot_div_coeff_vs_mmlu_acc.py` that:

1. Loads the CSV from Step 3
2. Groups by model family (GPT2-51M, GPT2-117M, ..., LLaMA2-7B)
3. For each family: scatter plot (div_coeff on x, mmlu_acc on y), linear fit, R² annotation
4. Overall scatter + fit across all models
5. Saves to `experiments/00_div_vs_benchmark_scores/div_coeff_vs_mmlu_acc.png` (and .pdf)

Style should match the existing `src/data_analysis/div_vs_ce_with_r2.py` plots (same color scheme, axis labels, grid).

X-axis label: `Task2Vec Diversity Coefficient of Training Dataset`
Y-axis label: `MMLU Accuracy`
Title: `Diversity Coefficient vs. MMLU Benchmark Accuracy`

---

## Step 5 — Verify and report

- Confirm the plot shows positive correlation: div_coeff↑ → MMLU acc↑
- Report R² overall and per model family
- Save final outputs:
  - `experiments/00_div_vs_benchmark_scores/div_coeff_vs_mmlu_acc.png`
  - `experiments/00_div_vs_benchmark_scores/div_coeff_vs_mmlu_acc.pdf`
  - `experiments/00_div_vs_benchmark_scores/mmlu_results.csv`

---

## Reference files

```
experiments/00_div_vs_benchmark_scores/summary.md                    # experiment overview
experiments/2024/01_10_2024_compiling_list_dict_name_2_div.md        # name→div_coeff dict
experiments/2024/27_28_09_2024_ckpts_for_mmlu_evals.md               # ckpt paths + wandb links
experiments/2024/01_10_2024_plot_log_p_v_correct_data.md             # MMLU JSONL format example
src/data_analysis/div_vs_ce_with_r2.py                               # reference plot style (CE loss)
src/data_analysis/div_vs_ppl.py                                      # reference plot style (PPL)
```
