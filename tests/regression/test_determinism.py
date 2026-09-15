import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.utils.seed import set_seed
from src.config.settings import EARCConfig
from src.data.schemas import RetrievedDocument
from src.compression.compressor import EvidenceAwareCompressor
from src.baselines.standard_rag import StandardRAGBaseline

def test_compression_determinism():
    config = EARCConfig()
    docs = [
        RetrievedDocument(doc_id="1", text="The quick brown fox jumps over the lazy dog.", score=1.0),
        RetrievedDocument(doc_id="2", text="A fast, dark-colored fox leaps over a resting hound.", score=0.9)
    ]
    query = "What did the fox do?"
    
    set_seed(42)
    comp1 = EvidenceAwareCompressor(config)
    res1 = comp1.compress(query, docs, config.compression.budget)
    
    set_seed(42)
    comp2 = EvidenceAwareCompressor(config)
    res2 = comp2.compress(query, docs, config.compression.budget)
    
    assert [s.sentence_id for s in res1.selected_sentences] == [s.sentence_id for s in res2.selected_sentences]

def test_no_answer_leakage():
    config = EARCConfig()
    docs = [
        RetrievedDocument(doc_id="1", text="The answer is 42.", score=1.0)
    ]
    query = "What is the answer?"
    
    comp = EvidenceAwareCompressor(config)
    res = comp.compress(query, docs, config.compression.budget)
    
    # We shouldn't see any explicit gold answer structures in the compression inputs
    assert "gold_answers" not in res.original_query
    for s in res.selected_sentences:
        assert "gold_answers" not in s.text

def test_baseline_independence():
    config = EARCConfig()
    docs = [RetrievedDocument(doc_id="1", text="text", score=1.0)]
    baseline = StandardRAGBaseline(config)
    ctx = baseline.prepare_context("query", docs)
    assert hasattr(ctx, "original_query")
    assert ctx.selected_sentences[0].text == "text"
    assert ctx.metrics['budget_used'] > 0
