import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.evaluation.significance import paired_bootstrap_test

def test_identical_scores():
    sysA = [1.0, 1.0, 0.0, 0.0, 1.0]
    sysB = [1.0, 1.0, 0.0, 0.0, 1.0]
    p_val = paired_bootstrap_test(sysA, sysB, seed=42)
    assert p_val > 0.05 # Not significant

def test_clearly_different():
    sysA = [1.0] * 50
    sysB = [0.0] * 50
    p_val = paired_bootstrap_test(sysA, sysB, seed=42)
    assert p_val < 0.05 # Significant

def test_paired_same_length():
    with pytest.raises(AssertionError):
        paired_bootstrap_test([1.0], [1.0, 0.0])
