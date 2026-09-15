"""
Bootstrap confidence intervals for evaluation metrics.

Paper specification: 2,000 bootstrap resamples for 95% confidence intervals.
Uses deterministic seeding for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BootstrapCI:
    """Result of a bootstrap confidence interval computation."""
    metric_name: str
    observed: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_resamples: int
    std_error: float

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "observed": self.observed,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "n_resamples": self.n_resamples,
            "std_error": self.std_error,
        }


def bootstrap_confidence_interval(
    scores: list[float] | np.ndarray,
    metric_name: str = "metric",
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
    statistic: str = "mean",
) -> BootstrapCI:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        scores: Per-example metric scores.
        metric_name: Name of the metric for reporting.
        n_resamples: Number of bootstrap resamples (paper default: 2,000).
        confidence_level: Confidence level (paper default: 0.95).
        seed: Random seed for reproducibility.
        statistic: Statistic to compute ('mean' or 'median').

    Returns:
        BootstrapCI with observed value, CI bounds, and metadata.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    n = len(scores_arr)

    if n == 0:
        return BootstrapCI(
            metric_name=metric_name,
            observed=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            std_error=0.0,
        )

    rng = np.random.RandomState(seed)

    # Observed statistic
    stat_fn = np.mean if statistic == "mean" else np.median
    observed = float(stat_fn(scores_arr))

    # Bootstrap resamples
    bootstrap_stats = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        resample_idx = rng.randint(0, n, size=n)
        resample = scores_arr[resample_idx]
        bootstrap_stats[i] = stat_fn(resample)

    # Percentile method for CI
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    std_error = float(np.std(bootstrap_stats))

    return BootstrapCI(
        metric_name=metric_name,
        observed=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        std_error=std_error,
    )


def compute_bootstrap_cis(
    metrics: dict[str, list[float]],
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, BootstrapCI]:
    """
    Compute bootstrap CIs for multiple metrics.

    Args:
        metrics: Dict of metric_name -> per-example scores.
        n_resamples: Number of resamples.
        confidence_level: Confidence level.
        seed: Base seed (each metric gets seed + offset for independence).

    Returns:
        Dict of metric_name -> BootstrapCI.
    """
    results = {}
    for i, (name, scores) in enumerate(metrics.items()):
        results[name] = bootstrap_confidence_interval(
            scores=scores,
            metric_name=name,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=seed + i,
        )
    return results
