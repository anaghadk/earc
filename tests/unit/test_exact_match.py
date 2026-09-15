import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.evaluation.exact_match import exact_match_score

def test_exact_match():
    assert exact_match_score("Paris", ["Paris"]) == 1.0

def test_case_insensitive():
    assert exact_match_score("paris", ["Paris"]) == 1.0

def test_article_removal():
    assert exact_match_score("the Paris", ["Paris"]) == 1.0
    assert exact_match_score("a cat", ["cat"]) == 1.0
    assert exact_match_score("an apple", ["apple"]) == 1.0

def test_no_match():
    assert exact_match_score("London", ["Paris"]) == 0.0

def test_multiple_aliases():
    assert exact_match_score("NYC", ["New York City", "NYC", "New York"]) == 1.0

def test_empty_prediction():
    assert exact_match_score("", ["Paris"]) == 0.0
