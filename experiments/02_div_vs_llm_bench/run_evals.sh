#!/usr/bin/env bash
# run_evals.sh — Run lm-evaluation-harness (MMLU) on all UDACA models.
#
# Usage:
#   bash run_evals.sh                           # run all missing evals
#   bash run_evals.sh --dry-run                 # print commands without executing
#   bash run_evals.sh --output_dir /my/path     # custom output directory
#   bash run_evals.sh --gpu 0                   # specify GPU index
#
# Prerequisites:
#   conda activate eleuther_lm_eval_harness_20240927
#   pip install lm-eval  (or install from source)
#
# The script:
#   1. Loops over all models in the UDACA org on HuggingFace
#   2. Skips models whose output directory already contains JSONL samples
#   3. Runs lm_eval with --log_samples to capture per-choice log-likelihoods
#   4. Uses batch_size=8 for GPT-2 models, batch_size=2 for LLaMA-2

set -euo pipefail

# ---- Defaults ----
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
GPU="${GPU:-0}"
DRY_RUN=false
TASKS="mmlu"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=true; shift ;;
        --output_dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --gpu)           GPU="$2"; shift 2 ;;
        --tasks)         TASKS="$2"; shift 2 ;;
        *)               echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# ---- Model list ----
# GPT-2 models (batch_size=8)
GPT2_MODELS=(
    "GPT2_51M_1.31B_USPTO"
    "GPT2_51M_1.31B_PubMedAbs"
    "GPT2_51M_1.31B_USPTOAndPubMedAbs"
    "GPT2_51M_557M_USPTO"
    "GPT2_51M_557M_PubMedAbs"
    "GPT2_51M_557M_USPTOAndPubMedAbs"
    "GPT2_117M_2.2B_USPTO"
    "GPT2_117M_2.2B_PubMedAbs"
    "GPT2_117M_2.2B_USPTOAndPubMedAbs"
    "GPT2_204M_USPTO"
    "GPT2_204M_PubMedAbs"
    "GPT2_204M_USPTOAndPubMedAbs"
    "GPT2_345M_2.2B_USPTO"
    "GPT2_345M_2.2B_PubMedAbs"
    "GPT2_345M_2.2B_USPTOAndPubMedAbs"
    "GPT2_810M_PubMedAbs"
    "GPT2_810M_2.2B_USPTOAndPubMedAbs"
    "GPT2_1.5B_180M_USPTO"
    "GPT2_1.5B_180M_PubMedAbs"
    "GPT2_1.5B_180M_USPTOAndPubMedAbs"
)

# LLaMA-2 models (batch_size=2, need more VRAM)
LLAMA_MODELS=(
    "LLama2_Uspto_Ckpt_1"
    "LLama2_Pubmed_Ckpt_2"
    "LLama2_Pubmed_Ckpt_7"
    "LLama2_Uspto_Pubmed_Ckpt_3"
    "LLama2_Uspto_Pubmed_Ckpt_4"
    "LLama2_Uspto_Pubmed_Ckpt_5"
    "LLama2_Uspto_Pubmed_Ckpt_6"
)

# ---- Helper ----
run_eval() {
    local model_name="$1"
    local batch_size="$2"
    local model_out="$OUTPUT_DIR/$model_name"

    # Skip if results already exist (check for samples JSONL)
    if find "$model_out" -name "samples_*.jsonl" -print -quit 2>/dev/null | grep -q .; then
        echo "[SKIP] $model_name — results already exist at $model_out"
        return 0
    fi

    local cmd="CUDA_VISIBLE_DEVICES=$GPU lm_eval \
    --model hf \
    --model_args pretrained=UDACA/$model_name \
    --tasks $TASKS \
    --device cuda \
    --batch_size $batch_size \
    --output_path $model_out \
    --log_samples"

    if $DRY_RUN; then
        echo "[DRY-RUN] $cmd"
    else
        echo "[RUNNING] $model_name (batch_size=$batch_size) ..."
        eval "$cmd"
        echo "[DONE] $model_name"
    fi
}

# ---- Run all ----
echo "=== Diversity vs. Benchmark Evals ==="
echo "Output dir: $OUTPUT_DIR"
echo "GPU: $GPU"
echo "Tasks: $TASKS"
echo "Dry run: $DRY_RUN"
echo ""

for m in "${GPT2_MODELS[@]}"; do
    run_eval "$m" 8
done

for m in "${LLAMA_MODELS[@]}"; do
    run_eval "$m" 2
done

echo ""
echo "=== All evaluations complete ==="
echo "Run analysis:  python analyze.py all --results_dir $OUTPUT_DIR"
