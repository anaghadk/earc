import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.compression.segmentation import SentenceSegmenter
from src.data.schemas import RetrievedDocument

@pytest.fixture
def segmenter():
    return SentenceSegmenter()

def test_normal_text(segmenter):
    doc = RetrievedDocument(doc_id="1", text="The cat sat. The dog ran.", score=1.0)
    sentences = segmenter.segment([doc])
    assert len(sentences) == 2
    assert sentences[0].text.strip() == "The cat sat."
    assert sentences[1].text.strip() == "The dog ran."

def test_abbreviations(segmenter):
    doc = RetrievedDocument(doc_id="1", text="Dr. Smith went to Washington. He arrived at noon.", score=1.0)
    sentences = segmenter.segment([doc])
    assert len(sentences) == 2
    assert "Dr. Smith" in sentences[0].text

def test_decimals(segmenter):
    doc = RetrievedDocument(doc_id="1", text="The value is 3.14. It is important.", score=1.0)
    sentences = segmenter.segment([doc])
    assert len(sentences) == 2
    assert "3.14" in sentences[0].text

def test_empty_text(segmenter):
    doc = RetrievedDocument(doc_id="1", text="", score=1.0)
    sentences = segmenter.segment([doc])
    assert len(sentences) == 0

def test_regex_fallback():
    segmenter = SentenceSegmenter(use_spacy=False)
    doc = RetrievedDocument(doc_id="1", text="The cat sat. The dog ran.", score=1.0)
    sentences = segmenter.segment([doc])
    assert len(sentences) >= 1

def test_duplicate_removal(segmenter):
    doc1 = RetrievedDocument(doc_id="1", text="The cat sat.", score=1.0)
    doc2 = RetrievedDocument(doc_id="2", text="The cat sat.", score=1.0)
    sentences = segmenter.segment([doc1, doc2])
    assert len(sentences) == 1

def test_sentence_ids_stable(segmenter):
    doc = RetrievedDocument(doc_id="1", text="The cat sat.", score=1.0)
    sentences1 = segmenter.segment([doc])
    sentences2 = segmenter.segment([doc])
    assert sentences1[0].sentence_id == sentences2[0].sentence_id
