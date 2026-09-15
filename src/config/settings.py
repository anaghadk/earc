"""
Hierarchical configuration system for the EARC pipeline.

Loads from YAML files, supports environment variable overrides,
and allows CLI argument overrides. All configuration is validated
using Pydantic models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-configuration models
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    """Data loading and sampling configuration."""
    seed: int = Field(default=42, description="Random seed for reproducibility")
    n_examples_per_dataset: int = Field(default=5000, ge=1)
    datasets: list[str] = Field(default=["nq", "hotpotqa", "triviaqa"])
    nq_split: str = Field(default="train", description="NQ split to sample from")
    hotpotqa_split: str = Field(default="train", description="HotpotQA split")
    triviaqa_split: str = Field(default="train", description="TriviaQA split")
    cache_dir: str = Field(default="data/cache")
    processed_dir: str = Field(default="data/processed")
    raw_dir: str = Field(default="data/raw")


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    backend: str = Field(default="dense", description="bm25, dense, hybrid, or rag_project")
    top_k: int = Field(default=5, ge=1)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    index_dir: str = Field(default="data/indexes")
    hybrid_bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    hybrid_dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rag_project_dir: Optional[str] = Field(default=None, description="Path to RAG_Project dir; enables RAGProjectRetriever when set")
    retrieval_method: str = Field(default="hybrid", description="Retrieval mode: dense | bm25 | hybrid")


class CompressionConfig(BaseModel):
    """Compression algorithm configuration (paper §Methodology)."""
    alpha: float = Field(default=0.7, ge=0.0, description="Semantic similarity weight")
    beta: float = Field(default=0.3, ge=0.0, description="Evidence score weight")
    redundancy_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Cross-document redundancy cosine threshold (tau)"
    )
    token_budget: int = Field(default=200, ge=0, description="Token budget T")
    spacy_model: str = Field(default="en_core_web_sm")
    stopwords: list[str] = Field(default_factory=list, description="Extra stopwords")
    use_regex_fallback: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_weights(self) -> "CompressionConfig":
        total = self.alpha + self.beta
        if abs(total - 1.0) > 1e-6:
            import warnings
            warnings.warn(
                f"alpha ({self.alpha}) + beta ({self.beta}) = {total} != 1.0. "
                f"This is allowed but diverges from the paper default."
            )
        return self


class OllamaConfig(BaseModel):
    """Ollama local inference configuration."""
    base_url: str = Field(default="http://localhost:11434")
    model: str = Field(default="llama3.2")
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=256, ge=1)
    timeout: int = Field(default=120, ge=1)
    max_retries: int = Field(default=3, ge=0)


class MistralConfig(BaseModel):
    """Mistral API configuration."""
    model: str = Field(default="ministral-8b-latest")
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=256, ge=1)
    timeout: int = Field(default=120, ge=1)
    max_retries: int = Field(default=3, ge=0)
    api_key: Optional[str] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def load_api_key_from_env(self) -> "MistralConfig":
        if self.api_key is None:
            self.api_key = os.environ.get("MISTRAL_API_KEY")
        return self


class GenerationConfig(BaseModel):
    """Generation configuration."""
    providers: list[str] = Field(default=["ollama", "mistral"])
    default_provider: str = Field(default="ollama")


class HardwareConfig(BaseModel):
    """Hardware configuration."""
    device: str = Field(default="cuda", description="cuda or cpu")
    embedding_batch_size: int = Field(default=16, ge=1)
    fp16: bool = Field(default=True)
    max_gpu_memory_fraction: float = Field(default=0.9, ge=0.1, le=1.0)


class EvaluationConfig(BaseModel):
    """Evaluation and statistical testing configuration."""
    bootstrap_resamples: int = Field(default=2000, ge=100)
    significance_resamples: int = Field(default=5000, ge=100)
    significance_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    bootstrap_seed: int = Field(default=42)
    token_budgets_for_curve: list[int] = Field(
        default=[50, 100, 150, 200, 250, 300, 400, 500]
    )


class OutputConfig(BaseModel):
    """Output directory configuration."""
    base_dir: str = Field(default="outputs")
    log_level: str = Field(default="INFO")
    save_predictions: bool = Field(default=True)
    save_figures: bool = Field(default=True)
    figure_formats: list[str] = Field(default=["png", "pdf"])


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------

class EARCConfig(BaseModel):
    """Root configuration for the EARC pipeline."""
    data: DataConfig = Field(default_factory=DataConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    mistral: MistralConfig = Field(default_factory=MistralConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> EARCConfig:
    """
    Load configuration from a YAML file with optional overrides.

    Priority (highest to lowest):
    1. Explicit overrides dict
    2. Environment variables (EARC_ prefix)
    3. YAML config file
    4. Pydantic defaults

    Args:
        config_path: Path to YAML config file.
        overrides: Dict of overrides (dot-separated keys supported).

    Returns:
        Validated EARCConfig instance.
    """
    config_data: dict[str, Any] = {}

    # Load from YAML
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                yaml_data = yaml.safe_load(f) or {}
            config_data = _deep_merge(config_data, yaml_data)

    # Apply environment variable overrides
    env_overrides = _collect_env_overrides()
    if env_overrides:
        config_data = _deep_merge(config_data, env_overrides)

    # Apply explicit overrides
    if overrides:
        expanded = _expand_dot_keys(overrides)
        config_data = _deep_merge(config_data, expanded)

    return EARCConfig(**config_data)


def _collect_env_overrides() -> dict[str, Any]:
    """Collect EARC_* environment variables as config overrides."""
    result: dict[str, Any] = {}
    prefix = "EARC_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            parts = key[len(prefix):].lower().split("__")
            current = result
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            # Try to parse as int/float/bool
            current[parts[-1]] = _parse_env_value(value)
    return result


def _parse_env_value(value: str) -> Any:
    """Parse a string environment variable to appropriate type."""
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _expand_dot_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Expand dot-separated keys into nested dicts."""
    result: dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result
