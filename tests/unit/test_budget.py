import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.compression.budget import TokenBudgetSelector, SimpleTokenCounter
from src.data.schemas import CandidateSentence, SentenceStatus

@pytest.fixture
def counter():
    return SimpleTokenCounter()

@pytest.fixture
def selector(counter):
    return TokenBudgetSelector(counter, max_tokens=10)

def test_exact_budget(selector):
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a b c", score=0.0, status=SentenceStatus.SELECTED)
    s2 = CandidateSentence(sentence_id="2", doc_id="1", text="d e f", score=0.0, status=SentenceStatus.SELECTED)
    s3 = CandidateSentence(sentence_id="3", doc_id="1", text="g h i", score=0.0, status=SentenceStatus.SELECTED)
    
    selected = selector.select([s1, s2, s3])
    # Total tokens = 3 + 3 + 3 = 9 < 10
    assert len(selected) == 3
    assert all(s.status == SentenceStatus.SELECTED for s in selected)

def test_zero_budget(counter):
    selector = TokenBudgetSelector(counter, max_tokens=0)
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a", score=0.0, status=SentenceStatus.SELECTED)
    selected = selector.select([s1])
    assert len(selected) == 0
    assert s1.status == SentenceStatus.REJECTED_BUDGET

def test_sentence_too_long(selector):
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a b c d e f g h i j k l", score=0.0, status=SentenceStatus.SELECTED)
    selected = selector.select([s1])
    assert len(selected) == 0
    assert s1.status == SentenceStatus.REJECTED_BUDGET

def test_exact_fit(selector):
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a b c d e f g h i j", score=0.0, status=SentenceStatus.SELECTED)
    selected = selector.select([s1])
    assert len(selected) == 1
    assert s1.status == SentenceStatus.SELECTED

def test_one_token_over(selector):
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a b c d e f g h i j k", score=0.0, status=SentenceStatus.SELECTED)
    selected = selector.select([s1])
    assert len(selected) == 0
    assert s1.status == SentenceStatus.REJECTED_BUDGET

def test_status_marking(selector):
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a b c d e", score=0.0, status=SentenceStatus.SELECTED)
    s2 = CandidateSentence(sentence_id="2", doc_id="1", text="f g h i j k l", score=0.0, status=SentenceStatus.SELECTED)
    s3 = CandidateSentence(sentence_id="3", doc_id="1", text="m n", score=0.0, status=SentenceStatus.PENDING) # Suppose it's pending/selected
    
    # Actually redundancy might mark as REJECTED_REDUNDANCY, budget selector skips them
    s4 = CandidateSentence(sentence_id="4", doc_id="1", text="o", score=0.0, status=SentenceStatus.REJECTED_REDUNDANCY)
    
    selected = selector.select([s1, s2, s3, s4])
    
    assert len(selected) == 2 # s1 (5), s3 (2)
    assert selected[0].sentence_id == "1"
    assert selected[1].sentence_id == "3"
    
    assert s1.status == SentenceStatus.SELECTED
    assert s2.status == SentenceStatus.REJECTED_BUDGET
    assert s3.status == SentenceStatus.SELECTED
    assert s4.status == SentenceStatus.REJECTED_REDUNDANCY
