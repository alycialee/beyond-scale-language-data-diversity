# Experiment 00: Diversity Coefficient vs. Benchmark Scores

**TL;DR:** Investigate whether higher Task2Vec diversity coefficient of training data correlates with higher benchmark scores (MMLU accuracy, lower CE loss, lower perplexity). Models (GPT-2 variants + LLaMA-2 7B) trained on USPTO (div=0.158), PubMed (div=0.168), and USPTO+PubMed (div=0.195) are already uploaded to HuggingFace. Eval results (log-likelihoods on MMLU) were partially collected. Goal: plot div_coeff vs. benchmark score with R² and confirm positive correlation.

---

## What we already have

### Trained models — uploaded to HuggingFace (UDACA org)

Training data subsets from `UDACA/PileSubsets`. Three diversity levels (x-axis):

| Dataset | Diversity Coefficient |
|---|---|
| USPTO | 0.158 |
| PubMed | 0.168 |
| USPTO + PubMed | 0.195 |

Models trained (each at all 3 diversity levels unless noted):

```
GPT2_51M_1.31B_{USPTO,PubMedAbs,USPTOAndPubMedAbs}
GPT2_51M_557M_{USPTO,PubMedAbs,USPTOAndPubMedAbs}
GPT2_117M_2.2B_{USPTO,PubMedAbs,USPTOAndPubMedAbs}
GPT2_204M_{USPTO,PubMedAbs,USPTOAndPubMedAbs}
GPT2_345M_2.2B_{USPTO,PubMedAbs,USPTOAndPubMedAbs}
GPT2_810M_PubMedAbs, GPT2_810M_2.2B_USPTOAndPubMedAbs
GPT2_1.5B_180M_{USPTO,PubMedAbs,USPTOAndPubMedAbs}
LLama2_Uspto_Ckpt_1, LLama2_Pubmed_Ckpt_2, LLama2_Uspto_Pubmed_Ckpt_{3,4,5,6,7}
```

See `experiments/2024/01_10_2024_compiling_list_dict_name_2_div.md` for the full name→div_coeff dict.

### Existing plots (div_coeff vs CE loss / PPL)

Located in `src/data_analysis/`:

| Script | What it plots |
|---|---|
| `ally_div_vs_ce_main_paper.py` | div_coeff vs CE loss on OWT2 + C4 (9 GPT-2 + LLaMA-2 models) |
| `div_vs_ce_with_r2.py` | Same but with R² annotations per model |
| `div_vs_ppl.py` | div_coeff vs perplexity on Pile-all + OWT2 (3 points: USPTO/PubMed/USPTO+PubMed) |
| `ally_div_vs_pply_cs197.py` | Single model (GPT2-117M) — student exercise version |

These already show a **positive correlation**: as div_coeff increases (0.158→0.168→0.195), CE loss decreases and perplexity decreases, with R²≈1 for most models.

### MMLU eval results (partial)

- Eval pipeline: EleutherAI `lm-evaluation-harness`, metric = `-log P^{Vocab}(CorrectChoice)` per answer choice
- Eval results backed up at: `/lfs/skampere1/0/brando9/data/beyond_scale/eval_results_back_up`
- Format: JSONL with per-doc `acc`, `resps` (log-likelihoods for each choice), `target`
- W&B project: `brando/beyond-scale` — table at https://wandb.ai/brando/beyond-scale/table

---

## What still needs to be done

1. **Collect/confirm all MMLU eval results** for all models across all 3 diversity levels.
   - Script to run: `lm_eval --model hf --model_args pretrained=UDACA/<model_name> --tasks mmlu --device cuda`
2. **Aggregate** per-model MMLU accuracy (mean over subjects) and map to div_coeff.
3. **Plot** div_coeff (x) vs MMLU accuracy (y), per model family, with linear fit + R².
4. **Compare** against the existing CE/PPL plots — do benchmark scores follow the same trend?

---

## Key hypothesis

> Models trained on more formally diverse data (higher Task2Vec div_coeff) achieve higher MMLU accuracy and lower CE loss, confirming that diversity is a causal driver of generalization.

---

## Relevant files

```
src/data_analysis/ally_div_vs_ce_main_paper.py   # main paper CE plot
src/data_analysis/div_vs_ce_with_r2.py           # CE plot with R² annotations
src/data_analysis/div_vs_ppl.py                  # PPL plot with R²
src/training/train.py                            # training code
src/training/eval.py                             # eval code
experiments/2024/27_28_09_2024_ckpts_for_mmlu_evals.md   # ckpt paths + wandb links
experiments/2024/01_10_2024_compiling_list_dict_name_2_div.md  # name→div_coeff mapping
experiments/2024/01_10_2024_plot_log_p_v_correct_data.md       # MMLU jsonl format example
```

---

## Claude Code agent prompt (for `claude_code.md`)

See `claude_code.md` in this directory for the runbook to dispatch an agent to run the full pipeline.
