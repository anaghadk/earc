"""
Paired-bootstrap significance testing.

Paper specification: 5,000 paired bootstrap resamples for
two-sided significance testing. Uses paired comparisons on
matched examples (same example IDs across methods).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.schemas import SignificanceResult


def paired_bootstrap_test(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    comparison_name: str,
    dataset: str,
    metric: str = "f1",
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> SignificanceResult:
    """
    Perform a two-sided paired bootstrap significance test.

    Tests whether scores_a (proposed method) is significantly different
    from scores_b (baseline) using paired bootstrap resampling.

    IMPORTANT: scores_a and scores_b must be paired — i.e., the i-th
    score in each corresponds to the same example. Never independently
    resample each method.

    Args:
        scores_a: Per-example scores for system A (proposed method).
        scores_b: Per-example scores for system B (baseline).
        comparison_name: Description (e.g., "Proposed vs Standard RAG").
        dataset: Dataset name.
        metric: Metric name (e.g., "f1").
        n_resamples: Number of paired bootstrap resamples (paper: 5,000).
        alpha: Significance level (paper: 0.05).
        seed: Random seed for reproducibility.

    Returns:
        SignificanceResult with observed delta, p-value, CI, significance.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)

    assert len(a) == len(b), (
        f"Paired test requires equal-length score arrays. "
        f"Got {len(a)} vs {len(b)}"
    )

    n = len(a)

    if n == 0:
        return SignificanceResult(
            comparison=comparison_name,
            dataset=dataset,
            metric=metric,
            observed_delta=0.0,
            p_value=1.0,
            ci_lower=0.0,
            ci_upper=0.0,
            is_significant=False,
            n_resamples=n_resamples,
            alpha=alpha,
        )

    # Observed difference (A - B)
    observed_delta = float(a.mean() - b.mean())

    rng = np.random.RandomState(seed)

    # Paired bootstrap: resample the SAME indices for both systems
    bootstrap_deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        resample_a = a[idx]
        resample_b = b[idx]
        bootstrap_deltas[i] = resample_a.mean() - resample_b.mean()

    # Two-sided p-value: proportion of bootstrap deltas that are
    # at least as extreme as observed delta (in either direction)
    # We use the centered approach: shift under H0 (delta = 0)
    centered_deltas = bootstrap_deltas - bootstrap_deltas.mean()
    p_value = float(
        np.mean(np.abs(centered_deltas) >= np.abs(observed_delta))
    )

    # 95% CI for the delta
    ci_lower = float(np.percentile(bootstrap_deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2)))

    return SignificanceResult(
        comparison=comparison_name,
        dataset=dataset,
        metric=metric,
        observed_delta=observed_delta,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_significant=p_value < alpha,
        n_resamples=n_resamples,
        alpha=alpha,
    )


def run_all_significance_tests(
    proposed_scores: dict[str, list[float]],
    baseline_scores: dict[str, dict[str, list[float]]],
    dataset: str,
    metric: str = "f1",
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[SignificanceResult]:
    """
    Run paired bootstrap tests for proposed vs all baselines.

    Args:
        proposed_scores: Dict with metric name -> per-example scores for proposed.
        baseline_scores: Dict of baseline_name -> {metric_name -> scores}.
        dataset: Dataset name.
        metric: Which metric to test.
        n_resamples: Number of bootstrap resamples.
        alpha: Significance level.
        seed: Base random seed.

    Returns:
        List of SignificanceResult for each comparison.
    """
    results = []

    proposed = proposed_scores.get(metric, [])

    for i, (baseline_name, scores_dict) in enumerate(baseline_scores.items()):
        baseline = scores_dict.get(metric, [])

        if len(proposed) != len(baseline):
            import logging
            logging.getLogger(__name__).warning(
                f"Skipping comparison Proposed vs {baseline_name}: "
                f"unequal lengths ({len(proposed)} vs {len(baseline)})"
            )
            continue

        result = paired_bootstrap_test(
            scores_a=proposed,
            scores_b=baseline,
            comparison_name=f"Proposed vs {baseline_name}",
            dataset=dataset,
            metric=metric,
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed + i,
        )
        results.append(result)

    return results
