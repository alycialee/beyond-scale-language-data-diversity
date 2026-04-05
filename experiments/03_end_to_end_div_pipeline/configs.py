"""
Pipeline configuration: dataset definitions, training hyperparameters, stage metadata.

Three diversity levels from UDACA/PileSubsets:
  USPTO           → div_coeff = 0.158
  PubMed          → div_coeff = 0.168
  USPTO + PubMed  → div_coeff = 0.195
"""
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """A dataset mixture to train on and measure diversity for."""
    name: str                           # short identifier
    hf_paths: list[str]                 # HuggingFace dataset paths
    hf_names: list[str]                 # HuggingFace dataset config names
    expected_div_coeff: float           # known diversity coefficient (if available)
    description: str = ""


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "uspto": DatasetConfig(
        name="uspto",
        hf_paths=["UDACA/PileSubsets"],
        hf_names=["uspto"],
        expected_div_coeff=0.158,
        description="USPTO patent text (low diversity)",
    ),
    "pubmed": DatasetConfig(
        name="pubmed",
        hf_paths=["UDACA/PileSubsets"],
        hf_names=["pubmed"],
        expected_div_coeff=0.168,
        description="PubMed abstracts (medium diversity)",
    ),
    "uspto_pubmed": DatasetConfig(
        name="uspto_pubmed",
        hf_paths=["UDACA/PileSubsets", "UDACA/PileSubsets"],
        hf_names=["uspto", "pubmed"],
        expected_div_coeff=0.195,
        description="USPTO + PubMed interleaved (high diversity)",
    ),
}


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """A model architecture to train."""
    name: str
    pretrained_model_name_or_path: str
    max_length: int = 1024
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    max_steps: int = 30_000
    description: str = ""


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "gpt2": ModelConfig(
        name="gpt2",
        pretrained_model_name_or_path="gpt2",
        max_length=1024,
        batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        max_steps=30_000,
        description="GPT-2 124M from scratch (reinit weights)",
    ),
    "gpt2_small": ModelConfig(
        name="gpt2_small",
        pretrained_model_name_or_path="gpt2",
        max_length=1024,
        batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        max_steps=5_000,
        description="GPT-2 124M quick training run for testing",
    ),
    "llama2_7b": ModelConfig(
        name="llama2_7b",
        pretrained_model_name_or_path="meta-llama/Llama-2-7b-hf",
        max_length=4096,
        batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        max_steps=30_000,
        description="LLaMA-2 7B with reinitialized weights",
    ),
}


# ---------------------------------------------------------------------------
# Diversity computation defaults
# ---------------------------------------------------------------------------

@dataclass
class DiversityConfig:
    """Settings for Task2Vec diversity coefficient computation."""
    probe_network: str = "gpt2"         # probe network for FIM extraction
    num_batches: int = 200              # number of Task2Vec embeddings to compute
    batch_size: int = 512               # samples per batch
    buffer_size: int = 500_000          # shuffle buffer for streaming datasets
    distance: str = "cosine"            # distance metric for pairwise comparison
    seed: int = 42


# ---------------------------------------------------------------------------
# Evaluation defaults
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Settings for lm-evaluation-harness runs."""
    tasks: str = "mmlu"
    batch_size_gpt2: int = 8
    batch_size_llama: int = 2
    log_samples: bool = True            # CRITICAL: enables per-sample log-likelihoods


# ---------------------------------------------------------------------------
# Pipeline stage names (for CLI)
# ---------------------------------------------------------------------------

STAGES = ["diversity", "train", "eval", "analyze", "all"]
