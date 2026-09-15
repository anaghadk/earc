import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import warnings
from src.compression.ranking import HybridRanker
from src.data.schemas import CandidateSentence

def test_hybrid_score_computation():
    ranker = HybridRanker(alpha=0.7, beta=0.3)
    score = ranker._compute_hybrid_score(0.8, 0.6)
    assert pytest.approx(score, 0.01) == 0.74

def test_descending_sort():
    ranker = HybridRanker(alpha=0.5, beta=0.5)
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a", score=0.0, relevance_score=0.2, evidence_score=0.2)
    s2 = CandidateSentence(sentence_id="2", doc_id="1", text="b", score=0.0, relevance_score=0.8, evidence_score=0.8)
    
    ranked = ranker.rank([s1, s2])
    assert ranked[0].sentence_id == "2"
    assert ranked[1].sentence_id == "1"

def test_deterministic_tiebreak():
    ranker = HybridRanker(alpha=0.5, beta=0.5)
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a", score=0.0, relevance_score=0.5, evidence_score=0.5)
    s2 = CandidateSentence(sentence_id="2", doc_id="1", text="b", score=0.0, relevance_score=0.5, evidence_score=0.5)
    
    ranked1 = ranker.rank([s1, s2])
    ranked2 = ranker.rank([s2, s1])
    assert [s.sentence_id for s in ranked1] == [s.sentence_id for s in ranked2]

def test_alpha_beta_validation():
    with pytest.warns(UserWarning):
        HybridRanker(alpha=0.8, beta=0.3)
