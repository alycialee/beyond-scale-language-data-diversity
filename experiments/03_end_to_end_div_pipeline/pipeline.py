#!/usr/bin/env python3
"""
End-to-end diversity pipeline orchestrator.

Chains four stages:
  1. diversity  — Compute Task2Vec diversity coefficient for a dataset
  2. train      — Train a model on that dataset
  3. eval       — Evaluate trained model on MMLU via lm-evaluation-harness
  4. analyze    — Extract soft metrics and correlate with diversity

Each stage can be run independently or all at once.

Usage:
  # Run a single stage
  python pipeline.py --stage diversity --dataset uspto
  python pipeline.py --stage train --dataset uspto --model gpt2_small
  python pipeline.py --stage eval --model_path UDACA/GPT2_117M_2.2B_USPTO
  python pipeline.py --stage analyze --results_dir ./eval_results

  # Run everything
  python pipeline.py --stage all --dataset uspto --model gpt2_small
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(SCRIPT_DIR))
from configs import (
    DATASET_CONFIGS,
    MODEL_CONFIGS,
    STAGES,
    DiversityConfig,
    EvalConfig,
)


# ---------------------------------------------------------------------------
# Stage 1: Compute Task2Vec Diversity Coefficient
# ---------------------------------------------------------------------------

def run_diversity(dataset_name: str, output_dir: Path, div_cfg: DiversityConfig | None = None):
    """Compute diversity coefficient using src/diversity/ infrastructure.

    This stage:
      1. Loads the dataset from UDACA/PileSubsets
      2. Initializes a GPT-2 probe network
      3. Computes Task2Vec embeddings for num_batches random batches
      4. Returns the expected pairwise cosine distance (= diversity coefficient)
    """
    if div_cfg is None:
        div_cfg = DiversityConfig()
    ds_cfg = DATASET_CONFIGS[dataset_name]

    print(f"\n{'='*60}")
    print(f"STAGE 1: DIVERSITY COEFFICIENT — {ds_cfg.name}")
    print(f"{'='*60}")
    print(f"  Dataset: {ds_cfg.hf_paths} / {ds_cfg.hf_names}")
    print(f"  Expected div_coeff: {ds_cfg.expected_div_coeff}")
    print(f"  Probe network: {div_cfg.probe_network}")
    print(f"  Num batches: {div_cfg.num_batches}, Batch size: {div_cfg.batch_size}")

    # Use the existing CLI entry point (src/diversity/main.py)
    # This respects the full Task2Vec pipeline already implemented in the repo
    cmd = [
        sys.executable, str(REPO_ROOT / "src" / "diversity" / "main.py"),
        "--task_name", ds_cfg.hf_names[0],
        "--num_tasks", str(div_cfg.num_batches),
        "--batch_size", str(div_cfg.batch_size),
        "--buffer_size", str(div_cfg.buffer_size),
        "--finetune", "--pretrained",
        "--output_dir", str(output_dir / "diversity" / ds_cfg.name),
        "--cache_dir", str(output_dir / "cache"),
    ]
    print(f"\n  CMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  ERROR: diversity computation failed (exit code {result.returncode})")
        return False
    print(f"\n  Diversity results saved to: {output_dir / 'diversity' / ds_cfg.name}")
    return True


# ---------------------------------------------------------------------------
# Stage 2: Train Model
# ---------------------------------------------------------------------------

def run_train(dataset_name: str, model_name: str, output_dir: Path):
    """Train a model on the specified dataset.

    This stage uses the existing src/training/train.py infrastructure.
    For full control, modify configs.py or pass custom training arguments.
    """
    ds_cfg = DATASET_CONFIGS[dataset_name]
    model_cfg = MODEL_CONFIGS[model_name]

    print(f"\n{'='*60}")
    print(f"STAGE 2: TRAIN — {model_cfg.name} on {ds_cfg.name}")
    print(f"{'='*60}")
    print(f"  Model: {model_cfg.pretrained_model_name_or_path}")
    print(f"  Dataset: {ds_cfg.hf_paths} / {ds_cfg.hf_names}")
    print(f"  Max steps: {model_cfg.max_steps}")
    print(f"  Batch size: {model_cfg.batch_size} x {model_cfg.gradient_accumulation_steps} (grad accum)")
    print(f"  Max length: {model_cfg.max_length}")

    # Build the training command
    # The existing train.py uses hardcoded configs, so we generate a small
    # wrapper script that sets the right variables before calling train()
    train_script = output_dir / "train" / f"run_train_{ds_cfg.name}_{model_cfg.name}.py"
    train_script.parent.mkdir(parents=True, exist_ok=True)

    # Generate paths for interleaved datasets
    paths_str = str(ds_cfg.hf_paths)
    names_str = str(ds_cfg.hf_names)
    ckpt_dir = output_dir / "checkpoints" / f"{model_cfg.name}_{ds_cfg.name}"

    script_content = f'''#!/usr/bin/env python3
"""Auto-generated training script for {model_cfg.name} on {ds_cfg.name}."""
import sys
sys.path.insert(0, "{REPO_ROOT / "src"}")

import datetime
import os
import torch
import math
import wandb
from pathlib import Path
from datasets import load_dataset, interleave_datasets
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, Trainer, TrainingArguments
from training.utils import eval_hf_with_subsample, get_column_names, group_texts, get_data_from_hf_dataset

torch.cuda.empty_cache()

# -- Config
hf_paths = {ds_cfg.hf_paths}
hf_names = {ds_cfg.hf_names}
pretrained = "{model_cfg.pretrained_model_name_or_path}"
max_steps = {model_cfg.max_steps}
batch_size = {model_cfg.batch_size}
grad_accum = {model_cfg.gradient_accumulation_steps}
max_length = {model_cfg.max_length}
lr = {model_cfg.learning_rate}
output_dir = Path("{ckpt_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

# -- Wandb
today = datetime.datetime.now().strftime("%Y-m%m-d%d-t%Hh_%Mm_%Ss")
run_name = f"e2e_pipeline: {{pretrained}} on {ds_cfg.name} (div={ds_cfg.expected_div_coeff}) {{today}}"
run = wandb.init(mode="online", project="beyond-scale", name=run_name, save_code=True)
wandb.config.update(dict(
    hf_paths=hf_paths, hf_names=hf_names, pretrained=pretrained,
    max_steps=max_steps, batch_size=batch_size, max_length=max_length,
    div_coeff={ds_cfg.expected_div_coeff}, dataset="{ds_cfg.name}",
))

# -- Load model + tokenizer
if "gpt2" in pretrained:
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(pretrained)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    block_size = tokenizer.model_max_length
else:
    from transformers import AutoModelForCausalLM
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token
    block_size = max_length

print(f"Parameters: {{sum(p.numel() for p in model.parameters()):,}}")
print(f"Expected random loss: {{math.log(len(tokenizer)):.4f}}")

# -- Load data
train_datasets = [
    load_dataset(p, n, streaming=True, split="train").with_format("torch")
    for p, n in zip(hf_paths, hf_names)
]
probs = [1.0 / len(train_datasets)] * len(train_datasets)
raw_train = interleave_datasets(train_datasets, probs)
remove_cols = get_column_names(raw_train)
tokenize_fn = lambda ex: tokenizer(ex["text"])
tokenized = raw_train.map(tokenize_fn, batched=True, remove_columns=remove_cols)
_group = lambda ex: group_texts(ex, block_size)
train_dataset = tokenized.map(_group, batched=True)

# -- Train
training_args = TrainingArguments(
    output_dir=str(output_dir),
    max_steps=max_steps,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    learning_rate=lr,
    warmup_ratio=0.01,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=max_steps // 3,
    logging_first_step=True,
    report_to="wandb",
    fp16=False,
    bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
trainer.save_model(output_dir=str(output_dir / "final_ckpt"))

# -- Eval on held-out sets
print("\\n---- Evaluating trained model ----")
eval_hf_with_subsample("UDACA/pile_openwebtext2", None, "validation", model, tokenizer, block_size, output_dir, max_eval_samples=8, print_str="> Eval OWT2")
eval_hf_with_subsample("c4", "en", "validation", model, tokenizer, block_size, output_dir, max_eval_samples=8, print_str="> Eval C4")

print(f"\\nCheckpoint saved to: {{output_dir / 'final_ckpt'}}")
print(f"{{wandb.config=}}")
print("Done!")
'''
    train_script.write_text(script_content)
    print(f"\n  Generated training script: {train_script}")

    cmd = [sys.executable, str(train_script)]
    print(f"  CMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  ERROR: training failed (exit code {result.returncode})")
        return False
    print(f"\n  Checkpoint saved to: {ckpt_dir / 'final_ckpt'}")
    return True


# ---------------------------------------------------------------------------
# Stage 3: Evaluate on Benchmarks
# ---------------------------------------------------------------------------

def run_eval(model_path: str, output_dir: Path, eval_cfg: EvalConfig | None = None):
    """Run lm-evaluation-harness on a trained model.

    Uses --log_samples to capture per-choice log-likelihoods (critical
    for soft metrics per the Schaeffer/Miranda "Elusive Abilities" insight).
    """
    if eval_cfg is None:
        eval_cfg = EvalConfig()

    # Determine batch size based on model name
    bs = eval_cfg.batch_size_llama if "llama" in model_path.lower() else eval_cfg.batch_size_gpt2
    model_short = model_path.replace("/", "_").replace("UDACA_", "")

    print(f"\n{'='*60}")
    print(f"STAGE 3: EVALUATE — {model_path}")
    print(f"{'='*60}")
    print(f"  Tasks: {eval_cfg.tasks}")
    print(f"  Batch size: {bs}")
    print(f"  Log samples: {eval_cfg.log_samples}")

    eval_output = output_dir / "eval_results" / model_short
    eval_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", eval_cfg.tasks,
        "--device", "cuda",
        "--batch_size", str(bs),
        "--output_path", str(eval_output),
    ]
    if eval_cfg.log_samples:
        cmd.append("--log_samples")

    print(f"\n  CMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: evaluation failed (exit code {result.returncode})")
        return False
    print(f"\n  Eval results saved to: {eval_output}")
    return True


# ---------------------------------------------------------------------------
# Stage 4: Analyze — Extract soft metrics and correlate
# ---------------------------------------------------------------------------

def run_analyze(results_dir: Path, output_dir: Path):
    """Run the analysis pipeline from experiment 02.

    Extracts log_p_correct, log_p_contrast, acc from lm_eval JSONL output
    and generates correlation plots against diversity coefficients.
    """
    print(f"\n{'='*60}")
    print(f"STAGE 4: ANALYZE")
    print(f"{'='*60}")

    analyze_script = REPO_ROOT / "experiments" / "02_div_vs_llm_bench" / "analyze.py"
    csv_out = output_dir / "results.csv"

    cmd = [
        sys.executable, str(analyze_script),
        "all",
        "--results_dir", str(results_dir),
        "--csv", str(csv_out),
        "--output_dir", str(output_dir),
    ]
    print(f"\n  CMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: analysis failed (exit code {result.returncode})")
        return False
    print(f"\n  Analysis outputs in: {output_dir}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end diversity pipeline: diversity → train → eval → analyze",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--stage", required=True, choices=STAGES,
                        help="Pipeline stage to run (or 'all' for everything)")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset config name (for diversity/train stages)")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()),
                        default="gpt2_small",
                        help="Model config name (for train stage)")
    parser.add_argument("--model_path", type=str,
                        help="HuggingFace model path (for eval stage, e.g. UDACA/GPT2_117M_2.2B_USPTO)")
    parser.add_argument("--results_dir", type=Path,
                        help="Eval results directory (for analyze stage)")
    parser.add_argument("--output_dir", type=Path,
                        default=SCRIPT_DIR / "output",
                        help="Root output directory for all stages")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    if args.stage in ("diversity", "all"):
        if not args.dataset:
            parser.error("--dataset required for diversity stage")
        run_diversity(args.dataset, args.output_dir)

    if args.stage in ("train", "all"):
        if not args.dataset:
            parser.error("--dataset required for train stage")
        run_train(args.dataset, args.model, args.output_dir)

    if args.stage in ("eval", "all"):
        if args.stage == "eval" and not args.model_path:
            parser.error("--model_path required for eval stage")
        model_path = args.model_path or str(
            args.output_dir / "checkpoints" / f"{args.model}_{args.dataset}" / "final_ckpt"
        )
        run_eval(model_path, args.output_dir)

    if args.stage in ("analyze", "all"):
        results_dir = args.results_dir or (args.output_dir / "eval_results")
        run_analyze(results_dir, args.output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline complete. Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
