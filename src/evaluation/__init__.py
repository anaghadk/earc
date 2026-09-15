"""Evaluation module for EARC pipeline."""

from src.evaluation.exact_match import exact_match_score, compute_exact_match
from src.evaluation.f1 import token_f1_score, max_token_f1_score, compute_f1
from src.evaluation.token_metrics import TokenMetrics
from src.evaluation.latency import LatencyTracker
from src.evaluation.bootstrap import bootstrap_confidence_interval, compute_bootstrap_cis
from src.evaluation.significance import paired_bootstrap_test, run_all_significance_tests

__all__ = [
    "exact_match_score",
    "compute_exact_match",
    "token_f1_score",
    "max_token_f1_score",
    "compute_f1",
    "TokenMetrics",
    "LatencyTracker",
    "bootstrap_confidence_interval",
    "compute_bootstrap_cis",
    "paired_bootstrap_test",
    "run_all_significance_tests",
]
