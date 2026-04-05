# CLAUDE.md

@/dfs/scratch0/brando9/CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Official implementation of the **Task2Vec Diversity Coefficient** — a metric for measuring natural language data diversity. The paper shows LLMs are pre-trained on formally diverse data.

- **Paper:** https://arxiv.org/abs/2306.13840
- **Core idea:** Compute Task2Vec embeddings (diagonal of Fisher Information Matrix) for dataset batches, then measure pairwise cosine distances to get a diversity coefficient.
- **Probe network:** GPT-2 (by default)
- **Supported datasets:** C4, WikiText-103, The Pile (and its sub-datasets), GINC (synthetic)

---

## Installation

```bash
# Conda (recommended)
conda create -n beyond_scale_div_coeff python=3.11 -y
conda activate beyond_scale_div_coeff
pip install -e ~/beyond-scale-language-data-diversity

# Or venv
python3.11 -m venv ~/.virtualenvs/beyond_scale_div_coeff
source ~/.virtualenvs/beyond_scale_div_coeff/bin/activate
pip install -e ~/beyond-scale-language-data-diversity
```

The `install.sh` script installs via conda and also sets up dependencies (`ultimate-utils`, `ultimate-anatome`).

---

## Running Experiments

**Compute diversity coefficient (main workflow):**
```bash
python src/diversity/main.py \
  --task_name c4        # or wikitext, the_pile
  --num_tasks 200 \
  --batch_size 512 \
  --buffer_size 500000 \
  --finetune --pretrained \
  --output_dir ./output_dir \
  --cache_dir ./cache_dir
```

**Batch runners (used in paper):**
```bash
# C4, WikiText-103, The Pile — 200 tasks each
bash src/diversity/scripts/runner.sh

# Individual Pile sub-datasets
bash src/diversity/scripts/runner_thepile_subdataset.sh

# GINC diversity
bash src/diversity/scripts/runner_ginc.sh
```

**GINC synthetic data:**
```bash
# Generate datasets (HMMs with varying symbols)
bash src/ginc/scripts/runner_generate.sh

# Train GPT-2 on GINC
bash src/ginc/scripts/runner_train.sh

# Or directly:
python src/diversity/main_ginc.py \
  --batch_size 512 --finetune --pretrained \
  --cache_dir ./cache_dir --n_hmms=10 --n_symbols=50
```

**Model training (LLaMA-2, GPT-2, Mistral via HF Trainer + TRL/PEFT):**
```bash
python src/training/train.py
python src/training/eval.py
```

---

## Architecture

### `src/diversity/` — Core module

| File | Role |
|------|------|
| `main.py` | CLI entry point; loads dataset, runs Task2Vec embedding loop |
| `div_coeff.py` | `get_diversity_coefficient()`, `cross_diversity_coefficient()` — main API |
| `task2vec.py` | Task2Vec class; computes diagonal FIM via montecarlo/variational/autoregressive methods |
| `task_similarity.py` | `pdist()` (pairwise cosine), `stats_of_distance_matrix()`, `plot_distance_matrix()` |
| `data_mixtures.py` | Mixture definitions: Uniform, DoReMi, LLaMA v1 for C4+WikiText; 5-subset Pile |
| `utils.py` | `AverageMeter`, `get_error()` (autoregressive loss), `seed_everything()` |
| `main_ginc.py` | GINC-specific entry point |
| `scripts/` | Shell runners for paper experiments |
| `notebooks/` | `plot.ipynb` reproduces paper figures; `plot-ginc.ipynb` for GINC plots |

`get_diversity_coefficient()` returns a dict with: `div_coeff`, `div_coeff_ci`, `embeddings`, `distance_matrix`, `losses`, `num_batches`.

Comments marked `## LLM DIV` indicate modifications from the original CV Task2Vec for language model use.

### `src/alignment/` — Alignment/relevance coefficients

- `align_t2v_coeff.py`: `relevance_coeff_task2vec_via_full_embed_dataset()`, `alignment_with_diversity_coefficient()`
- `_align.py`: Alignment framework

### `src/training/` — LM fine-tuning

HuggingFace Trainer + TRL/PEFT (LoRA/QLoRA). Supports GPT-2, LLaMA-2, Mistral, C4, UDACA PileSubsets.

### `src/ginc/` — Synthetic in-context learning data

Generates datasets using HMMs with varying number of symbols/HMMs. Has its own conda env (`conda-env.yml`) and runner scripts.

### `src/data_analysis/` — Paper figures

Scripts correlating diversity coefficient vs. cross-entropy loss and perplexity (R² analysis).

---

## Key API Usage

```python
from diversity.div_coeff import get_diversity_coefficient

results = get_diversity_coefficient(
    dataset,
    probe_network,  # typically GPT-2
    num_tasks=200,
    batch_size=512,
)
print(results['div_coeff'], results['div_coeff_ci'])
```

---

## SNAP Cluster Notes

- Repo lives on DFS: `/dfs/scratch0/brando9/beyond-scale-language-data-diversity`
- LFS symlink: `~/beyond-scale-language-data-diversity`
- GPU selection: `main_krbtmux.sh` handles free-GPU discovery and tmux session setup
- Outputs (embeddings, distance matrices, loss files) go to `./output_dir` by default — use LFS paths for speed
- wandb tracking is used in GINC training scripts
