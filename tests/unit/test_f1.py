import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.evaluation.f1 import max_token_f1_score

def test_perfect_f1():
    assert max_token_f1_score("Paris", ["Paris"]) == 1.0

def test_partial_overlap():
    f1 = max_token_f1_score("Paris France", ["Paris"])
    # Prediction: "Paris France" -> 2 tokens
    # Gold: "Paris" -> 1 token
    # Overlap: "Paris" -> 1 token
    # Precision = 1/2 = 0.5, Recall = 1/1 = 1.0
    # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 2/3 ≈ 0.666
    assert pytest.approx(f1, 0.01) == 0.666

def test_no_overlap():
    assert max_token_f1_score("London", ["Paris"]) == 0.0

def test_empty_prediction():
    assert max_token_f1_score("", ["Paris"]) == 0.0

def test_multiple_gold():
    assert max_token_f1_score("NYC", ["New York City", "NYC"]) == 1.0
