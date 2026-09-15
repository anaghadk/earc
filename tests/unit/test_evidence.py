import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.compression.evidence import EvidenceScorer
from src.data.schemas import CandidateSentence

@pytest.fixture
def scorer():
    return EvidenceScorer()

def test_entity_counting(scorer):
    sentence = CandidateSentence(sentence_id="1", doc_id="1", text="Apple Inc. is in Cupertino.", score=0.0)
    score = scorer.score([sentence], "Where is Apple?")[0]
    assert score > 0

def test_number_detection(scorer):
    sentence = CandidateSentence(sentence_id="1", doc_id="1", text="42 items at $3.14 in 2024", score=0.0)
    # The evidence scorer counts numbers
    score = scorer.score([sentence], "How many items?")[0]
    assert score > 0

def test_keyword_overlap(scorer):
    sentence = CandidateSentence(sentence_id="1", doc_id="1", text="Paris is the capital of France", score=0.0)
    score = scorer.score([sentence], "capital of France")[0]
    assert score > 0

def test_min_max_normalization():
    scorer = EvidenceScorer()
    scores = [1.0, 2.0, 3.0]
    normalized = scorer._normalize(scores)
    assert normalized == [0.0, 0.5, 1.0]

def test_constant_score_edge_case():
    scorer = EvidenceScorer()
    scores = [2.0, 2.0, 2.0]
    normalized = scorer._normalize(scores)
    assert normalized == [0.0, 0.0, 0.0]

def test_zero_evidence(scorer):
    sentence = CandidateSentence(sentence_id="1", doc_id="1", text="a b c d e", score=0.0)
    score = scorer.score([sentence], "x y z")[0]
    assert score == 0.0
