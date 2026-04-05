#!/usr/bin/env python3
"""
Diversity Coefficient vs. Benchmark Soft Metrics — end-to-end analysis.

Extracts *soft* metrics from lm-evaluation-harness JSONL output (--log_samples)
and correlates them with the Task2Vec diversity coefficient of training data.

Motivation (Schaeffer & Miranda, "Are Emergent Abilities a Mirage?"):
  Accuracy is a harsh, jumpy metric that can hide smooth underlying trends.
  Log-likelihoods reveal the model's continuous confidence in the correct
  answer, even when argmax accuracy appears flat.

Metrics per question:
  1. log_p_correct   = log P(correct choice)
  2. log_p_contrast  = log P(correct) - mean log P(incorrect choices)
  3. acc             = 1 if argmax == correct else 0   (tracked for reference)

Usage:
  # Extract metrics from lm_eval results → CSV
  python analyze.py extract --results_dir /path/to/eval_results

  # Generate correlation plots from CSV
  python analyze.py plot --csv results.csv

  # Both in one shot
  python analyze.py all --results_dir /path/to/eval_results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

# Import model registry (single source of truth)
sys.path.insert(0, str(Path(__file__).parent))
from models import (
    DIVERSITY_COEFFICIENTS,
    DIV_LABELS,
    FAMILY_COLORS,
    MODEL_FAMILY,
)

# Default paths (relative to this script)
SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_CSV = SCRIPT_DIR / "results.csv"


# ---------------------------------------------------------------------------
# 1. Extract soft metrics from lm_eval JSONL output
# ---------------------------------------------------------------------------

def extract_metrics_from_jsonl(model_dir: Path) -> dict | None:
    """Parse lm_eval --log_samples JSONL and return per-model aggregate metrics.

    Returns dict with keys: log_p_correct, log_p_contrast, acc, n_samples
    or None if no JSONL files found.

    JSONL format (from lm_eval --log_samples):
      {"doc_id": 0, "target": "1", "resps": [
          [["-3.23", "False"]], [["-3.69", "False"]],
          [["-4.38", "False"]], [["-4.45", "False"]]
      ], "acc": 0.0, ...}

    resps[i][0][0] = log-likelihood for choice i
    target = index of correct choice
    """
    sample_jsonls = list(model_dir.glob("*/*/samples_*.jsonl"))
    if not sample_jsonls:
        # Also try one level up (directory structure can vary)
        sample_jsonls = list(model_dir.glob("*/samples_*.jsonl"))
    if not sample_jsonls:
        return None

    log_liks_correct: list[float] = []
    log_liks_contrast: list[float] = []
    accs: list[float] = []

    for jsonl_path in sample_jsonls:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                target_idx = int(sample["target"])
                resps = sample["resps"]
                n_choices = len(resps)

                # Log-likelihood of correct choice
                lp_correct = float(resps[target_idx][0][0])

                # Contrast: log P(correct) - mean log P(incorrect)
                lp_incorrect = np.mean([
                    float(resps[i][0][0])
                    for i in range(n_choices) if i != target_idx
                ])

                log_liks_correct.append(lp_correct)
                log_liks_contrast.append(lp_correct - lp_incorrect)

                # Accuracy: was the highest log-likelihood the correct one?
                all_lps = [float(resps[i][0][0]) for i in range(n_choices)]
                accs.append(1.0 if np.argmax(all_lps) == target_idx else 0.0)

    if not log_liks_correct:
        return None

    return {
        "log_p_correct": float(np.mean(log_liks_correct)),
        "log_p_contrast": float(np.mean(log_liks_contrast)),
        "acc": float(np.mean(accs)),
        "n_samples": len(log_liks_correct),
    }


def extract_all(results_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Walk results_dir, extract metrics for each model, save CSV."""
    rows = []
    for model_name, div_coeff in DIVERSITY_COEFFICIENTS.items():
        model_dir = results_dir / model_name
        if not model_dir.exists():
            print(f"  MISSING dir: {model_name}")
            continue

        metrics = extract_metrics_from_jsonl(model_dir)
        if metrics is None:
            print(f"  NO JSONL:    {model_name}")
            continue

        row = {
            "model": model_name,
            "family": MODEL_FAMILY[model_name],
            "div_coeff": div_coeff,
            **metrics,
        }
        rows.append(row)
        print(f"  {model_name:45s}  div={div_coeff:.3f}  "
              f"log_p_correct={metrics['log_p_correct']:.4f}  "
              f"log_p_contrast={metrics['log_p_contrast']:.4f}  "
              f"acc={metrics['acc']:.4f}  "
              f"n={metrics['n_samples']}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nWARNING: No results found. Check --results_dir path.")
        return df

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved CSV → {output_csv}  ({len(df)} models)")
    return df


# ---------------------------------------------------------------------------
# 2. Correlation analysis
# ---------------------------------------------------------------------------

def correlations(x: np.ndarray, y: np.ndarray) -> dict:
    """Compute linear fit + correlation statistics."""
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = r2_score(y, y_pred)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    kt, kp = stats.kendalltau(x, y)
    return dict(
        slope=slope, intercept=intercept, r2=r2,
        pearson_r=pr, pearson_p=pp,
        spearman_r=sr, spearman_p=sp,
        kendall_t=kt, kendall_p=kp,
    )


def print_correlations(label: str, c: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  fit: y = {c['slope']:.4f}x + {c['intercept']:.4f}")
    print(f"  R²          = {c['r2']:.4f}")
    print(f"  Pearson  r  = {c['pearson_r']:.4f}  (p={c['pearson_p']:.4e})")
    print(f"  Spearman ρ  = {c['spearman_r']:.4f}  (p={c['spearman_p']:.4e})")
    print(f"  Kendall  τ  = {c['kendall_t']:.4f}  (p={c['kendall_p']:.4e})")


def write_summary_report(df: pd.DataFrame, output_dir: Path) -> None:
    """Write a plain-text summary of correlation results."""
    report_path = output_dir / "correlation_summary.txt"
    lines = ["Diversity Coefficient vs. Benchmark Soft Metrics — Summary\n",
             "=" * 60, "\n"]

    x_all = df["div_coeff"].values
    for metric, label in [
        ("log_p_correct", "log P(correct)"),
        ("log_p_contrast", "log P(correct) - mean log P(incorrect)"),
        ("acc", "Accuracy"),
    ]:
        if metric not in df.columns or df[metric].isna().all():
            lines.append(f"\n--- {label}: NO DATA ---")
            continue
        mask = ~df[metric].isna()
        y_all = df.loc[mask, metric].values
        x_sub = df.loc[mask, "div_coeff"].values
        c = correlations(x_sub, y_all)
        lines.append(f"\n--- {label} (all {len(df)} models) ---")
        lines.append(f"  y = {c['slope']:.4f}x + {c['intercept']:.4f}")
        lines.append(f"  R² = {c['r2']:.4f}")
        lines.append(f"  Pearson  r = {c['pearson_r']:.4f}  (p={c['pearson_p']:.2e})")
        lines.append(f"  Spearman ρ = {c['spearman_r']:.4f}  (p={c['spearman_p']:.2e})")
        lines.append(f"  Kendall  τ = {c['kendall_t']:.4f}  (p={c['kendall_p']:.2e})")

        lines.append(f"\n  Per-family breakdown:")
        df_valid = df[mask]
        for family, gdf in df_valid.groupby("family"):
            if len(gdf) < 2:
                lines.append(f"    {family}: only {len(gdf)} point(s), skipping")
                continue
            fc = correlations(gdf["div_coeff"].values, gdf[metric].values)
            r2_str = f"R²={fc['r2']:.3f}" if len(gdf) >= 3 else "R²=n/a(<3pts)"
            lines.append(
                f"    {family:12s}  n={len(gdf)}  {r2_str}  "
                f"Pearson={fc['pearson_r']:.3f}  Spearman={fc['spearman_r']:.3f}"
            )

    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved summary → {report_path}")


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------

def plot_all(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate correlation plots: one figure per metric, two subplots each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    x_all = df["div_coeff"].values

    metric_info = [
        ("log_p_correct",
         "Mean log P(correct choice) on MMLU",
         "div_coeff_vs_mmlu_log_p_correct"),
        ("log_p_contrast",
         "Mean [log P(correct) \u2212 mean log P(incorrect)] on MMLU",
         "div_coeff_vs_mmlu_log_p_contrast"),
        ("acc",
         "MMLU Accuracy (for reference)",
         "div_coeff_vs_mmlu_acc"),
    ]

    def _fit_label(c: dict) -> str:
        sign = "+" if c["intercept"] >= 0 else "\u2212"
        return (f"y = {c['slope']:.3f}x {sign} {abs(c['intercept']):.3f}  "
                f"R\u00b2={c['r2']:.3f}")

    def _add_vlines(ax: plt.Axes) -> None:
        for dv, dl in DIV_LABELS:
            ax.axvline(dv, color="black", linestyle="dotted", linewidth=0.8)
            ax.annotate(dl, xy=(dv, ax.get_ylim()[0]),
                        xytext=(3, 4), textcoords="offset points",
                        fontsize=7, rotation=90)

    for metric, ylabel, fname in metric_info:
        if metric not in df.columns or df[metric].isna().all():
            print(f"Skipping {metric}: no data")
            continue

        df_plot = df[~df[metric].isna()]
        x_all = df_plot["div_coeff"].values
        y_all = df_plot[metric].values
        c_all = correlations(x_all, y_all)

        # Print correlations to stdout too
        print_correlations(f"div_coeff vs {ylabel}", c_all)

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # --- Left: all models, overall fit ---
        ax = axes[0]
        for family, gdf in df_plot.groupby("family"):
            color = FAMILY_COLORS.get(family, "gray")
            ax.scatter(gdf["div_coeff"], gdf[metric],
                       label=family, color=color, s=80, zorder=3)

        x_line = np.linspace(x_all.min() - 0.003, x_all.max() + 0.003, 100)
        ax.plot(x_line, c_all["slope"] * x_line + c_all["intercept"],
                "k--", linewidth=1.5, label=_fit_label(c_all))

        ax.annotate(
            f"{_fit_label(c_all)}\n"
            f"Pearson r={c_all['pearson_r']:.3f} (p={c_all['pearson_p']:.3f})\n"
            f"Spearman \u03c1={c_all['spearman_r']:.3f} (p={c_all['spearman_p']:.3f})\n"
            f"Kendall \u03c4={c_all['kendall_t']:.3f} (p={c_all['kendall_p']:.3f})",
            xy=(0.03, 0.97), xycoords="axes fraction",
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )
        _add_vlines(ax)
        ax.set_xlabel("Task2Vec Diversity Coefficient of Training Dataset")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Div Coeff vs. {ylabel} (all models)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

        # --- Right: per-family fits ---
        ax2 = axes[1]
        for family, gdf in df_plot.groupby("family"):
            color = FAMILY_COLORS.get(family, "gray")
            fx = gdf["div_coeff"].values
            fy = gdf[metric].values
            ax2.scatter(fx, fy, color=color, s=80, zorder=3)
            if len(gdf) >= 2:
                fc = correlations(fx, fy)
                eq = (f"y={fc['slope']:.3f}x"
                      f"{'+' if fc['intercept'] >= 0 else ''}"
                      f"{fc['intercept']:.3f}")
                r2_str = f" R\u00b2={fc['r2']:.2f}" if len(gdf) >= 3 else ""
                ax2.plot(np.sort(fx),
                         np.poly1d(np.polyfit(fx, fy, 1))(np.sort(fx)),
                         color=color, linewidth=1.5,
                         label=f"{family} ({eq}{r2_str})")

        _add_vlines(ax2)
        ax2.set_xlabel("Task2Vec Diversity Coefficient of Training Dataset")
        ax2.set_ylabel(ylabel)
        ax2.set_title(f"Div Coeff vs. {ylabel} (per family)")
        ax2.legend(fontsize=7, loc="lower right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        for ext in ("png", "pdf"):
            out_path = output_dir / f"{fname}.{ext}"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {out_path}")
        plt.close()


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diversity coefficient vs. benchmark soft metrics analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- extract --
    p_extract = sub.add_parser(
        "extract",
        help="Parse lm_eval JSONL results and produce a CSV.",
    )
    p_extract.add_argument(
        "--results_dir", type=Path, required=True,
        help="Root directory containing per-model lm_eval output folders.",
    )
    p_extract.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"Output CSV path (default: {DEFAULT_CSV}).",
    )

    # -- plot --
    p_plot = sub.add_parser(
        "plot",
        help="Generate correlation plots from CSV.",
    )
    p_plot.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"Input CSV path (default: {DEFAULT_CSV}).",
    )
    p_plot.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for plots and summary (default: {DEFAULT_OUTPUT_DIR}).",
    )

    # -- all --
    p_all = sub.add_parser(
        "all",
        help="Extract metrics then generate plots (end-to-end).",
    )
    p_all.add_argument(
        "--results_dir", type=Path, required=True,
        help="Root directory containing per-model lm_eval output folders.",
    )
    p_all.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"Output CSV path (default: {DEFAULT_CSV}).",
    )
    p_all.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for plots and summary (default: {DEFAULT_OUTPUT_DIR}).",
    )

    args = parser.parse_args()

    if args.command == "extract":
        print(f"Extracting metrics from: {args.results_dir}")
        extract_all(args.results_dir, args.csv)

    elif args.command == "plot":
        if not args.csv.exists():
            print(f"ERROR: CSV not found at {args.csv}. Run 'extract' first.")
            sys.exit(1)
        df = pd.read_csv(args.csv)
        print(f"Loaded {len(df)} models from {args.csv}")
        plot_all(df, args.output_dir)
        write_summary_report(df, args.output_dir)

    elif args.command == "all":
        print(f"Extracting metrics from: {args.results_dir}")
        df = extract_all(args.results_dir, args.csv)
        if not df.empty:
            plot_all(df, args.output_dir)
            write_summary_report(df, args.output_dir)


if __name__ == "__main__":
    main()
