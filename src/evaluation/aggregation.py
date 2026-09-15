"""
Result aggregation across datasets and LLM backends.

Computes per-dataset, per-backend, and cross-dataset summary statistics.
Results must never be reported as only one pooled score.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from src.data.schemas import ExperimentResult
from src.evaluation.exact_match import normalize_answer

logger = logging.getLogger(__name__)


def compute_retrieval_hit(
    retrieved_docs: list[dict[str, Any]],
    gold_answers: list[str],
    k: Optional[int] = None,
) -> float:
    """
    Check if any normalized gold answer appears as a substring in the top-k retrieved docs.

    Returns:
        1.0 if at least one gold answer is found, else 0.0.
    """
    docs = retrieved_docs[:k] if k is not None else retrieved_docs
    norm_golds = [normalize_answer(a) for a in gold_answers if normalize_answer(a)]
    if not norm_golds or not docs:
        return 0.0
    for doc in docs:
        doc_text = normalize_answer(doc.get("text", "") + " " + doc.get("title", ""))
        for gold in norm_golds:
            if gold in doc_text:
                return 1.0
    return 0.0


def aggregate_by_dataset(
    results: list[ExperimentResult],
) -> dict[str, dict[str, Any]]:
    """
    Aggregate experiment results by dataset.

    For each dataset, reports:
    - n_examples, em_mean, em_std, f1_mean, f1_std
    - hit_at_1, hit_at_5, hit_at_k (retrieval recall metrics)
    - avg_prompt_tokens, avg_compressed_tokens, compression_pct
    - latency stats, gpu_memory_peak

    Args:
        results: List of ExperimentResult.

    Returns:
        Dict of dataset_name -> aggregated metrics.
    """
    by_dataset: dict[str, list[ExperimentResult]] = defaultdict(list)
    for r in results:
        by_dataset[r.dataset].append(r)

    aggregated = {}
    for dataset, examples in by_dataset.items():
        ems = [e.exact_match for e in examples]
        f1s = [e.f1 for e in examples]
        orig_tokens = [e.original_tokens for e in examples]
        comp_tokens = [e.compressed_tokens for e in examples]
        comp_pcts = [e.compression_percentage for e in examples]
        e2e_lats = [e.end_to_end_latency for e in examples]
        gpu_mems = [e.peak_gpu_memory_mb for e in examples]

        hit_1 = [compute_retrieval_hit(e.retrieved_documents, e.gold_answers, k=1) for e in examples]
        hit_5 = [compute_retrieval_hit(e.retrieved_documents, e.gold_answers, k=5) for e in examples]
        hit_k = [compute_retrieval_hit(e.retrieved_documents, e.gold_answers, k=None) for e in examples]

        aggregated[dataset] = {
            "n_examples": len(examples),
            "em_mean": float(np.mean(ems)),
            "em_std": float(np.std(ems)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "hit_at_1": float(np.mean(hit_1)),
            "hit_at_5": float(np.mean(hit_5)),
            "hit_at_k": float(np.mean(hit_k)),
            "avg_original_tokens": float(np.mean(orig_tokens)),
            "avg_compressed_tokens": float(np.mean(comp_tokens)),
            "avg_compression_pct": float(np.mean(comp_pcts)),
            "avg_latency": float(np.mean(e2e_lats)),
            "median_latency": float(np.median(e2e_lats)),
            "p95_latency": float(np.percentile(e2e_lats, 95)),
            "peak_gpu_memory_mb": float(np.max(gpu_mems)) if gpu_mems else 0.0,
        }

    return aggregated


def aggregate_by_provider(
    results: list[ExperimentResult],
) -> dict[str, dict[str, Any]]:
    """
    Aggregate experiment results by LLM provider.

    Never pools predictions from different providers into a single score.
    """
    by_provider: dict[str, list[ExperimentResult]] = defaultdict(list)
    for r in results:
        by_provider[r.llm_provider].append(r)

    aggregated = {}
    for provider, examples in by_provider.items():
        ems = [e.exact_match for e in examples]
        f1s = [e.f1 for e in examples]

        aggregated[provider] = {
            "n_examples": len(examples),
            "em_mean": float(np.mean(ems)),
            "f1_mean": float(np.mean(f1s)),
            "avg_compressed_tokens": float(np.mean([e.compressed_tokens for e in examples])),
            "avg_generation_latency": float(np.mean([e.generation_latency for e in examples])),
        }

    return aggregated


def aggregate_by_dataset_and_provider(
    results: list[ExperimentResult],
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Produce dataset × method × LLM backend results.

    Returns nested dict: dataset -> provider -> metrics.
    """
    nested: dict[str, dict[str, list[ExperimentResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        nested[r.dataset][r.llm_provider].append(r)

    aggregated = {}
    for dataset, providers in nested.items():
        aggregated[dataset] = {}
        for provider, examples in providers.items():
            ems = [e.exact_match for e in examples]
            f1s = [e.f1 for e in examples]
            comp_tokens = [e.compressed_tokens for e in examples]
            lats = [e.end_to_end_latency for e in examples]

            aggregated[dataset][provider] = {
                "n_examples": len(examples),
                "em_mean": float(np.mean(ems)),
                "em_std": float(np.std(ems)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
                "avg_compressed_tokens": float(np.mean(comp_tokens)),
                "avg_latency": float(np.mean(lats)),
            }

    return aggregated


def compute_overall_summary(
    results: list[ExperimentResult],
) -> dict[str, Any]:
    """
    Compute overall summary across all 15,000 examples.
    This is SUPPLEMENTARY to per-dataset reporting, not a replacement.
    """
    if not results:
        return {"n_examples": 0}

    ems = [e.exact_match for e in results]
    f1s = [e.f1 for e in results]
    orig_tokens = [e.original_tokens for e in results]
    comp_tokens = [e.compressed_tokens for e in results]
    comp_pcts = [e.compression_percentage for e in results]
    e2e_lats = [e.end_to_end_latency for e in results]

    datasets = set(r.dataset for r in results)
    providers = set(r.llm_provider for r in results)

    return {
        "n_examples": len(results),
        "datasets": sorted(datasets),
        "providers": sorted(providers),
        "em_mean": float(np.mean(ems)),
        "em_std": float(np.std(ems)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "avg_original_tokens": float(np.mean(orig_tokens)),
        "avg_compressed_tokens": float(np.mean(comp_tokens)),
        "avg_compression_pct": float(np.mean(comp_pcts)),
        "avg_latency": float(np.mean(e2e_lats)),
        "total_runtime_s": float(np.sum(e2e_lats)),
    }
