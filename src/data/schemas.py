"""
Core data schemas for the EARC pipeline.

All pipeline entities are represented as strongly-typed dataclasses.
These schemas are the single source of truth for data structures
flowing through retrieval, compression, generation, and evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

import numpy as np


class DatasetName(str, Enum):
    """Supported dataset identifiers."""
    NQ = "nq"
    HOTPOTQA = "hotpotqa"
    TRIVIAQA = "triviaqa"


class SentenceStatus(str, Enum):
    """Status of a candidate sentence after the compression pipeline."""
    SELECTED = "selected"
    REJECTED_REDUNDANCY = "rejected_redundancy"
    REJECTED_BUDGET = "rejected_budget"
    NEVER_REACHED = "never_reached"


class LLMProviderName(str, Enum):
    """Supported LLM provider identifiers."""
    OLLAMA = "ollama"
    MISTRAL = "mistral"
    MOCK = "mock"


class RetrieverName(str, Enum):
    """Supported retriever identifiers."""
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Document & QA schemas
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single document or passage associated with a QA example."""
    doc_id: str
    title: str
    text: str
    is_supporting: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class QAExample:
    """Normalized representation of a single QA example from any dataset."""
    id: str
    dataset: str
    question: str
    answers: list[str]
    documents: list[Document]
    supporting_documents: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @property
    def num_documents(self) -> int:
        return len(self.documents)

    @property
    def total_text_length(self) -> int:
        return sum(len(doc.text) for doc in self.documents)


# ---------------------------------------------------------------------------
# Retrieval schemas
# ---------------------------------------------------------------------------

@dataclass
class RetrievedDocument:
    """A document returned by the retriever."""
    doc_id: str
    title: str
    text: str
    score: float
    rank: int
    retriever: str
    dataset: str = ""
    query: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "retriever": self.retriever,
        }


# ---------------------------------------------------------------------------
# Compression schemas
# ---------------------------------------------------------------------------

@dataclass
class CandidateSentence:
    """
    A candidate sentence produced by the compression pipeline.

    Each field corresponds to a specific stage of the EARC algorithm.
    """
    sentence_id: str
    document_id: str
    title: str
    text: str
    source_rank: int
    # Stage 3: Semantic relevance
    semantic_similarity: float = 0.0
    # Stage 4: Evidence scoring (raw components)
    entity_count: int = 0
    number_count: int = 0
    keyword_hits: int = 0
    # Stage 4: Evidence scoring (aggregate)
    evidence_score: float = 0.0
    # Stage 4: Normalized evidence
    normalized_evidence: float = 0.0
    # Stage 5: Hybrid score
    hybrid_score: float = 0.0
    # Embedding vector (not serialized to JSON by default)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    # Final pipeline status
    status: SentenceStatus = SentenceStatus.NEVER_REACHED
    # Token count for budget tracking
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, excluding the embedding array."""
        return {
            "sentence_id": self.sentence_id,
            "document_id": self.document_id,
            "title": self.title,
            "text": self.text,
            "source_rank": self.source_rank,
            "semantic_similarity": self.semantic_similarity,
            "entity_count": self.entity_count,
            "number_count": self.number_count,
            "keyword_hits": self.keyword_hits,
            "evidence_score": self.evidence_score,
            "normalized_evidence": self.normalized_evidence,
            "hybrid_score": self.hybrid_score,
            "status": self.status.value,
            "token_count": self.token_count,
        }

    @staticmethod
    def generate_id(doc_id: str, sentence_index: int) -> str:
        """Generate a stable, deterministic sentence ID."""
        raw = f"{doc_id}::sent_{sentence_index}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


@dataclass
class CompressedContext:
    """Result of the compression pipeline for a single query."""
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    reduction_percentage: float
    selected_sentences: list[CandidateSentence] = field(default_factory=list)
    all_candidates: list[CandidateSentence] = field(default_factory=list)
    # Alias: some callers use 'candidates' instead of 'all_candidates'
    candidates: list[CandidateSentence] = field(default_factory=list)
    original_text: str = ""

    def __post_init__(self):
        # Merge candidates alias into all_candidates
        if self.candidates and not self.all_candidates:
            self.all_candidates = self.candidates
        elif self.all_candidates and not self.candidates:
            self.candidates = self.all_candidates

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_count": len(self.selected_sentences),
            "candidate_count": len(self.all_candidates),
            "compressed_text": self.compressed_text,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "reduction_percentage": self.reduction_percentage,
            "selected_sentences": [s.to_dict() for s in self.selected_sentences],
        }


# ---------------------------------------------------------------------------
# Generation schemas
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Result from an LLM generation call."""
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency": self.latency,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Evaluation schemas
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """A single metric computation result."""
    name: str
    value: float
    per_example: Optional[list[float]] = None
    std: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


@dataclass
class SignificanceResult:
    """Result of a paired-bootstrap significance test."""
    comparison: str
    dataset: str
    metric: str
    observed_delta: float
    p_value: float
    ci_lower: float
    ci_upper: float
    is_significant: bool
    n_resamples: int
    alpha: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Experiment schemas
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Complete result for a single QA example in an experiment run."""
    example_id: str
    dataset: str
    question: str
    gold_answers: list[str]
    retrieved_documents: list[dict[str, Any]]
    candidate_count: int
    selected_sentences: list[dict[str, Any]]
    original_tokens: int
    compressed_tokens: int
    compression_percentage: float
    alpha: float
    beta: float
    redundancy_threshold: float
    token_budget: int
    llm_provider: str
    llm_model: str
    prediction: str
    exact_match: float
    f1: float
    retrieval_latency: float
    compression_latency: float
    generation_latency: float
    end_to_end_latency: float
    peak_gpu_memory_mb: float
    method: str = "proposed"
    retriever_type: str = "dense"
    seed: int = 42
    # Extracted answer used for EM/F1 (may differ from raw prediction)
    evaluated_answer: str = ""
    # Compressed context text that was sent to the LLM
    compressed_context_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class ExperimentConfig:
    """Snapshot of configuration for an experiment run."""
    run_id: str
    method: str
    dataset: str
    llm_provider: str
    llm_model: str
    retriever_type: str
    top_k: int
    token_budget: int
    alpha: float
    beta: float
    redundancy_threshold: float
    seed: int
    n_examples: int
    embedding_model: str
    hardware: dict[str, Any] = field(default_factory=dict)
    software: dict[str, Any] = field(default_factory=dict)
    git_commit: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
