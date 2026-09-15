import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import numpy as np
from src.compression.redundancy import RedundancyFilter
from src.data.schemas import CandidateSentence, SentenceStatus

@pytest.fixture
def filter_obj():
    class DummyEmbeddingEngine:
        def embed_sentences(self, sentences):
            # A, B are similar (0.9), C is different (0.0)
            return np.array([
                [1.0, 0.0],
                [0.9, np.sqrt(1 - 0.9**2)],
                [0.0, 1.0]
            ])
    return RedundancyFilter(DummyEmbeddingEngine(), tau=0.85)

def test_above_threshold(filter_obj):
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a", score=0.0)
    s2 = CandidateSentence(sentence_id="2", doc_id="1", text="b", score=0.0)
    
    filtered = filter_obj.filter([s1, s2])
    assert len(filtered) == 2
    assert filtered[0].status == SentenceStatus.SELECTED
    assert filtered[1].status == SentenceStatus.REJECTED_REDUNDANCY

def test_below_threshold(filter_obj):
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a", score=0.0)
    s3 = CandidateSentence(sentence_id="3", doc_id="1", text="c", score=0.0)
    
    filtered = filter_obj.filter([s1, s3])
    # The dummy embedder will get only 2 items, which changes indices
    # We should override the embed_sentences behavior
    
    class FilterDummy:
        def embed_sentences(self, sentences):
            return np.array([
                [1.0, 0.0],
                [0.0, 1.0]
            ])
    
    f2 = RedundancyFilter(FilterDummy(), tau=0.85)
    filtered = f2.filter([s1, s3])
    
    assert len(filtered) == 2
    assert filtered[0].status == SentenceStatus.SELECTED
    assert filtered[1].status == SentenceStatus.SELECTED

def test_threshold_boundary():
    class FilterDummy:
        def embed_sentences(self, sentences):
            return np.array([
                [1.0, 0.0],
                [0.85, np.sqrt(1 - 0.85**2)]
            ])
    f = RedundancyFilter(FilterDummy(), tau=0.85)
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a", score=0.0)
    s2 = CandidateSentence(sentence_id="2", doc_id="1", text="b", score=0.0)
    
    filtered = f.filter([s1, s2])
    assert filtered[1].status == SentenceStatus.SELECTED # > not >=

def test_cross_document():
    class FilterDummy:
        def embed_sentences(self, sentences):
            return np.array([
                [1.0, 0.0],
                [0.9, np.sqrt(1 - 0.9**2)]
            ])
    f = RedundancyFilter(FilterDummy(), tau=0.85)
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a", score=0.0)
    s2 = CandidateSentence(sentence_id="2", doc_id="2", text="a", score=0.0)
    
    filtered = f.filter([s1, s2])
    assert filtered[1].status == SentenceStatus.REJECTED_REDUNDANCY

def test_first_always_selected(filter_obj):
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a", score=0.0)
    filtered = filter_obj.filter([s1])
    assert filtered[0].status == SentenceStatus.SELECTED
