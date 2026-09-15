import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import numpy as np
from src.evaluation.bootstrap import bootstrap_confidence_interval

def test_deterministic_seed():
    scores = [0.8, 0.9, 0.7, 0.6, 0.85]
    mean1, ci1 = bootstrap_confidence_interval(scores, seed=42)
    mean2, ci2 = bootstrap_confidence_interval(scores, seed=42)
    assert mean1 == mean2
    assert ci1 == ci2

def test_ci_contains_observed():
    scores = np.random.rand(100).tolist()
    mean, (lower, upper) = bootstrap_confidence_interval(scores, seed=42)
    assert lower <= mean <= upper

def test_n_resamples():
    scores = [1.0] * 10
    mean, (lower, upper) = bootstrap_confidence_interval(scores, n_resamples=100)
    assert mean == 1.0
    assert lower == 1.0
    assert upper == 1.0
