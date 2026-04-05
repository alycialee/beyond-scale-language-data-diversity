"""
Plot diversity coefficient vs MMLU log-likelihood scores with full correlation analysis.

Two metrics from Polo et al. (2024) https://arxiv.org/abs/2406.04391:
  1. log P(correct choice)
  2. log P(correct choice) - mean log P(incorrect choices)

Results dir: /dfs/scratch0/brando9/data/beyond_scale/eval_results/
Output:      experiments/00_div_vs_benchmark_scores/
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# 1. Model → diversity coefficient / family mapping
# ---------------------------------------------------------------------------
DIVERSITY_COEFFICIENTS: dict[str, float] = {
    "GPT2_51M_1.31B_USPTO":              0.158,
    "GPT2_51M_1.31B_PubMedAbs":          0.168,
    "GPT2_51M_1.31B_USPTOAndPubMedAbs":  0.195,
    "GPT2_51M_557M_USPTO":               0.158,
    "GPT2_51M_557M_PubMedAbs":           0.168,
    "GPT2_51M_557M_USPTOAndPubMedAbs":   0.195,
    "GPT2_117M_2.2B_USPTO":              0.158,
    "GPT2_117M_2.2B_PubMedAbs":          0.168,
    "GPT2_117M_2.2B_USPTOAndPubMedAbs":  0.195,
    "GPT2_204M_USPTO":                   0.158,
    "GPT2_204M_PubMedAbs":               0.168,
    "GPT2_204M_USPTOAndPubMedAbs":       0.195,
    "GPT2_345M_2.2B_USPTO":              0.158,
    "GPT2_345M_2.2B_PubMedAbs":          0.168,
    "GPT2_345M_2.2B_USPTOAndPubMedAbs":  0.195,
    "GPT2_810M_PubMedAbs":               0.168,
    "GPT2_810M_2.2B_USPTOAndPubMedAbs":  0.195,
    "GPT2_1.5B_180M_USPTO":              0.158,
    "GPT2_1.5B_180M_PubMedAbs":          0.168,
    "GPT2_1.5B_180M_USPTOAndPubMedAbs":  0.195,
    "LLama2_Uspto_Ckpt_1":               0.158,
    "LLama2_Pubmed_Ckpt_2":              0.168,
    "LLama2_Pubmed_Ckpt_7":              0.168,
    "LLama2_Uspto_Pubmed_Ckpt_3":        0.195,
    "LLama2_Uspto_Pubmed_Ckpt_4":        0.195,
    "LLama2_Uspto_Pubmed_Ckpt_5":        0.195,
    "LLama2_Uspto_Pubmed_Ckpt_6":        0.195,
}

MODEL_FAMILY: dict[str, str] = {
    "GPT2_51M_1.31B_USPTO":              "GPT2-51M",
    "GPT2_51M_1.31B_PubMedAbs":          "GPT2-51M",
    "GPT2_51M_1.31B_USPTOAndPubMedAbs":  "GPT2-51M",
    "GPT2_51M_557M_USPTO":               "GPT2-51M",
    "GPT2_51M_557M_PubMedAbs":           "GPT2-51M",
    "GPT2_51M_557M_USPTOAndPubMedAbs":   "GPT2-51M",
    "GPT2_117M_2.2B_USPTO":              "GPT2-117M",
    "GPT2_117M_2.2B_PubMedAbs":          "GPT2-117M",
    "GPT2_117M_2.2B_USPTOAndPubMedAbs":  "GPT2-117M",
    "GPT2_204M_USPTO":                   "GPT2-204M",
    "GPT2_204M_PubMedAbs":               "GPT2-204M",
    "GPT2_204M_USPTOAndPubMedAbs":       "GPT2-204M",
    "GPT2_345M_2.2B_USPTO":              "GPT2-345M",
    "GPT2_345M_2.2B_PubMedAbs":          "GPT2-345M",
    "GPT2_345M_2.2B_USPTOAndPubMedAbs":  "GPT2-345M",
    "GPT2_810M_PubMedAbs":               "GPT2-810M",
    "GPT2_810M_2.2B_USPTOAndPubMedAbs":  "GPT2-810M",
    "GPT2_1.5B_180M_USPTO":              "GPT2-1.5B",
    "GPT2_1.5B_180M_PubMedAbs":          "GPT2-1.5B",
    "GPT2_1.5B_180M_USPTOAndPubMedAbs":  "GPT2-1.5B",
    "LLama2_Uspto_Ckpt_1":               "LLaMA2-7B",
    "LLama2_Pubmed_Ckpt_2":              "LLaMA2-7B",
    "LLama2_Pubmed_Ckpt_7":              "LLaMA2-7B",
    "LLama2_Uspto_Pubmed_Ckpt_3":        "LLaMA2-7B",
    "LLama2_Uspto_Pubmed_Ckpt_4":        "LLaMA2-7B",
    "LLama2_Uspto_Pubmed_Ckpt_5":        "LLaMA2-7B",
    "LLama2_Uspto_Pubmed_Ckpt_6":        "LLaMA2-7B",
}

FAMILY_COLORS = {
    "GPT2-51M":  "royalblue",
    "GPT2-117M": "deepskyblue",
    "GPT2-204M": "darkturquoise",
    "GPT2-345M": "mediumslateblue",
    "GPT2-810M": "rebeccapurple",
    "GPT2-1.5B": "darkviolet",
    "LLaMA2-7B": "crimson",
}

RESULTS_DIR = Path("/dfs/scratch0/brando9/data/beyond_scale/eval_results")
OUT_DIR = Path(__file__).parent.parent.parent / "experiments/00_div_vs_benchmark_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Aggregate both metrics from JSONL samples
# ---------------------------------------------------------------------------
def get_mmlu_metrics(model_dir: Path) -> tuple[float, float] | tuple[None, None]:
    """Return (mean log P(correct), mean [log P(correct) - mean log P(incorrect)]).

    From Polo et al. 2024 (https://arxiv.org/abs/2406.04391):
      metric1 = log P(correct choice)
      metric2 = log P(correct choice) - mean_j≠correct log P(choice_j)

    resps[i] = [[loglik_i, is_greedy]]  for choice i in {A,B,C,D}
    target    = index of correct choice
    """
    sample_jsonls = list(model_dir.glob("*/*/samples_*.jsonl"))
    if not sample_jsonls:
        return None, None

    log_liks_correct = []
    log_liks_contrast = []

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

                lp_correct = float(resps[target_idx][0][0])
                lp_incorrect = np.mean([
                    float(resps[i][0][0])
                    for i in range(n_choices) if i != target_idx
                ])

                log_liks_correct.append(lp_correct)
                log_liks_contrast.append(lp_correct - lp_incorrect)

    if not log_liks_correct:
        return None, None
    return float(np.mean(log_liks_correct)), float(np.mean(log_liks_contrast))


rows = []
for model_name, div_coeff in DIVERSITY_COEFFICIENTS.items():
    model_dir = RESULTS_DIR / model_name
    if not model_dir.exists():
        print(f"MISSING: {model_name}")
        continue
    lp, lp_contrast = get_mmlu_metrics(model_dir)
    if lp is None:
        print(f"NO RESULTS: {model_name}")
        continue
    rows.append({
        "model":        model_name,
        "family":       MODEL_FAMILY[model_name],
        "div_coeff":    div_coeff,
        "log_p_correct":   lp,
        "log_p_contrast":  lp_contrast,
    })
    print(f"{model_name:45s}  div={div_coeff:.3f}  "
          f"log_p_correct={lp:.4f}  log_p_contrast={lp_contrast:.4f}")

df = pd.DataFrame(rows)
csv_path = OUT_DIR / "mmlu_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved CSV → {csv_path}")
print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# 3. Correlation analysis helper
# ---------------------------------------------------------------------------
def correlations(x: np.ndarray, y: np.ndarray) -> dict:
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = r2_score(y, y_pred)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    kt, kp = stats.kendalltau(x, y)
    return dict(slope=slope, intercept=intercept, r2=r2,
                pearson_r=pr, pearson_p=pp,
                spearman_r=sr, spearman_p=sp,
                kendall_t=kt, kendall_p=kp)


def print_correlations(label: str, c: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  fit: y = {c['slope']:.4f}x + {c['intercept']:.4f}")
    print(f"  R²          = {c['r2']:.4f}")
    print(f"  Pearson  r  = {c['pearson_r']:.4f}  (p={c['pearson_p']:.4e})")
    print(f"  Spearman ρ  = {c['spearman_r']:.4f}  (p={c['spearman_p']:.4e})")
    print(f"  Kendall  τ  = {c['kendall_t']:.4f}  (p={c['kendall_p']:.4e})")


x_all = df["div_coeff"].values

for metric, ylabel_short in [
    ("log_p_correct",  "log P(correct)"),
    ("log_p_contrast", "log P(correct) − mean log P(incorrect)"),
]:
    y_all = df[metric].values
    c_all = correlations(x_all, y_all)
    print_correlations(f"div_coeff vs MMLU {ylabel_short} — all models", c_all)

    print(f"\n  Per-family ({ylabel_short}):")
    for family, gdf in df.groupby("family"):
        if len(gdf) < 3:
            print(f"    {family}: only {len(gdf)} points, skipping")
            continue
        fc = correlations(gdf["div_coeff"].values, gdf[metric].values)
        print(f"    {family:12s}  n={len(gdf)}  R²={fc['r2']:.3f}  "
              f"y={fc['slope']:.3f}x+{fc['intercept']:.3f}  "
              f"Pearson={fc['pearson_r']:.3f}(p={fc['pearson_p']:.3e})  "
              f"Spearman={fc['spearman_r']:.3f}  Kendall={fc['kendall_t']:.3f}")


# ---------------------------------------------------------------------------
# 4. Plot — one figure per metric, each with two subplots (all / per-family)
# ---------------------------------------------------------------------------
METRIC_INFO = [
    ("log_p_correct",
     "Mean log P(correct choice) on MMLU",
     "div_coeff_vs_mmlu_log_p_correct"),
    ("log_p_contrast",
     "Mean [log P(correct) − mean log P(incorrect)] on MMLU",
     "div_coeff_vs_mmlu_log_p_contrast"),
]

DIV_LABELS = [(0.158, "USPTO"), (0.168, "PubMed"), (0.195, "USPTO+PubMed")]


def add_vlines(ax: plt.Axes) -> None:
    for dv, dl in DIV_LABELS:
        ax.axvline(dv, color="black", linestyle="dotted", linewidth=0.8)
        ax.annotate(dl, xy=(dv, ax.get_ylim()[0]),
                    xytext=(3, 4), textcoords="offset points", fontsize=7, rotation=90)


def fit_label(c: dict) -> str:
    sign = "+" if c["intercept"] >= 0 else "-"
    return (f"y = {c['slope']:.3f}x {sign} {abs(c['intercept']):.3f}  "
            f"R²={c['r2']:.3f}")


for metric, ylabel, fname in METRIC_INFO:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # --- left: all models, overall fit ---
    ax = axes[0]
    y_all = df[metric].values
    c_all = correlations(x_all, y_all)

    for family, gdf in df.groupby("family"):
        color = FAMILY_COLORS.get(family, "gray")
        ax.scatter(gdf["div_coeff"], gdf[metric],
                   label=family, color=color, s=80, zorder=3)

    x_line = np.linspace(x_all.min() - 0.003, x_all.max() + 0.003, 100)
    ax.plot(x_line, c_all["slope"] * x_line + c_all["intercept"],
            "k--", linewidth=1.5, label=fit_label(c_all))

    ax.annotate(
        f"{fit_label(c_all)}\n"
        f"Pearson r={c_all['pearson_r']:.3f} (p={c_all['pearson_p']:.3f})\n"
        f"Spearman ρ={c_all['spearman_r']:.3f} (p={c_all['spearman_p']:.3f})\n"
        f"Kendall τ={c_all['kendall_t']:.3f} (p={c_all['kendall_p']:.3f})",
        xy=(0.03, 0.97), xycoords="axes fraction",
        va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
    )
    add_vlines(ax)
    ax.set_xlabel("Task2Vec Diversity Coefficient of Training Dataset")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Div Coeff vs. {ylabel} (all models)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # --- right: per-family fits ---
    ax2 = axes[1]
    for family, gdf in df.groupby("family"):
        color = FAMILY_COLORS.get(family, "gray")
        fx = gdf["div_coeff"].values
        fy = gdf[metric].values
        ax2.scatter(fx, fy, color=color, s=80, zorder=3)
        if len(gdf) >= 2:
            fc = correlations(fx, fy)
            eq = (f"y={fc['slope']:.3f}x"
                  f"{'+' if fc['intercept']>=0 else ''}{fc['intercept']:.3f}")
            r2_str = f" R²={fc['r2']:.2f}" if len(gdf) >= 3 else ""
            ax2.plot(np.sort(fx),
                     np.poly1d(np.polyfit(fx, fy, 1))(np.sort(fx)),
                     color=color, linewidth=1.5,
                     label=f"{family} ({eq}{r2_str})")

    add_vlines(ax2)
    ax2.set_xlabel("Task2Vec Diversity Coefficient of Training Dataset")
    ax2.set_ylabel(ylabel)
    ax2.set_title(f"Div Coeff vs. {ylabel} (per family)")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"{fname}.{ext}"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    plt.close()
