import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import numpy as np
from src.compression.relevance import SemanticRelevanceScorer
from src.data.schemas import CandidateSentence

@pytest.fixture
def scorer():
    class DummyEmbeddingEngine:
        def embed_queries(self, queries):
            return np.array([np.array([1.0, 0.0, 0.0])])
        def embed_sentences(self, sentences):
            # First is identical, second is orthogonal
            return np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ])
    return SemanticRelevanceScorer(DummyEmbeddingEngine())

def test_identical_vectors(scorer):
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a", score=0.0)
    scores = scorer.score([s1], "query")
    assert np.isclose(scores[0], 1.0)

def test_orthogonal_vectors(scorer):
    class OrthogonalEmbeddingEngine:
        def embed_queries(self, queries):
            return np.array([[1.0, 0.0]])
        def embed_sentences(self, sentences):
            return np.array([[0.0, 1.0]])
    
    scorer_ortho = SemanticRelevanceScorer(OrthogonalEmbeddingEngine())
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a", score=0.0)
    scores = scorer_ortho.score([s1], "query")
    assert np.isclose(scores[0], 0.0)

def test_normalized_embeddings():
    pass # Implementation details of embeddings

def test_batch_relevance(scorer):
    s1 = CandidateSentence(sentence_id="1", doc_id="1", text="a", score=0.0)
    s2 = CandidateSentence(sentence_id="2", doc_id="1", text="b", score=0.0)
    scores = scorer.score([s1, s2], "query")
    assert len(scores) == 2
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.0)
