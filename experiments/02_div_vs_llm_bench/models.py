"""
Model registry: diversity coefficients, model families, and plot colors.

Single source of truth for the mapping between model names (as they appear
in HuggingFace / lm_eval output directories) and their Task2Vec diversity
coefficients.

Training data subsets from UDACA/PileSubsets:
  USPTO           → div_coeff = 0.158
  PubMed          → div_coeff = 0.168
  USPTO + PubMed  → div_coeff = 0.195
"""

# Model short name (matches lm_eval --output_path leaf dir) → diversity coeff
DIVERSITY_COEFFICIENTS: dict[str, float] = {
    # GPT-2 51M (two token budgets)
    "GPT2_51M_1.31B_USPTO":              0.158,
    "GPT2_51M_1.31B_PubMedAbs":          0.168,
    "GPT2_51M_1.31B_USPTOAndPubMedAbs":  0.195,
    "GPT2_51M_557M_USPTO":               0.158,
    "GPT2_51M_557M_PubMedAbs":           0.168,
    "GPT2_51M_557M_USPTOAndPubMedAbs":   0.195,
    # GPT-2 117M
    "GPT2_117M_2.2B_USPTO":              0.158,
    "GPT2_117M_2.2B_PubMedAbs":          0.168,
    "GPT2_117M_2.2B_USPTOAndPubMedAbs":  0.195,
    # GPT-2 204M
    "GPT2_204M_USPTO":                   0.158,
    "GPT2_204M_PubMedAbs":               0.168,
    "GPT2_204M_USPTOAndPubMedAbs":       0.195,
    # GPT-2 345M
    "GPT2_345M_2.2B_USPTO":              0.158,
    "GPT2_345M_2.2B_PubMedAbs":          0.168,
    "GPT2_345M_2.2B_USPTOAndPubMedAbs":  0.195,
    # GPT-2 810M (partial — missing USPTO-only)
    "GPT2_810M_PubMedAbs":               0.168,
    "GPT2_810M_2.2B_USPTOAndPubMedAbs":  0.195,
    # GPT-2 1.5B
    "GPT2_1.5B_180M_USPTO":              0.158,
    "GPT2_1.5B_180M_PubMedAbs":          0.168,
    "GPT2_1.5B_180M_USPTOAndPubMedAbs":  0.195,
    # LLaMA-2 7B
    "LLama2_Uspto_Ckpt_1":               0.158,
    "LLama2_Pubmed_Ckpt_2":              0.168,
    "LLama2_Pubmed_Ckpt_7":              0.168,
    "LLama2_Uspto_Pubmed_Ckpt_3":        0.195,
    "LLama2_Uspto_Pubmed_Ckpt_4":        0.195,
    "LLama2_Uspto_Pubmed_Ckpt_5":        0.195,
    "LLama2_Uspto_Pubmed_Ckpt_6":        0.195,
}

# HuggingFace model ID prefix (prepend "UDACA/" to get the full HF path)
HF_PREFIX = "UDACA"

# Model short name → family label (for grouping in plots)
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

# Consistent colors for each model family in plots
FAMILY_COLORS: dict[str, str] = {
    "GPT2-51M":  "royalblue",
    "GPT2-117M": "deepskyblue",
    "GPT2-204M": "darkturquoise",
    "GPT2-345M": "mediumslateblue",
    "GPT2-810M": "rebeccapurple",
    "GPT2-1.5B": "darkviolet",
    "LLaMA2-7B": "crimson",
}

# Diversity level labels for vertical reference lines in plots
DIV_LABELS: list[tuple[float, str]] = [
    (0.158, "USPTO"),
    (0.168, "PubMed"),
    (0.195, "USPTO+PubMed"),
]

# Models that require smaller batch size during lm_eval (e.g. 7B params)
LARGE_MODELS: set[str] = {
    name for name, family in MODEL_FAMILY.items() if "LLaMA" in family
}
