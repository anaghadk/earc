import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.prompting.builder import PromptBuilder
from src.data.schemas import CompressedContext, CandidateSentence

@pytest.fixture
def builder():
    return PromptBuilder()

def test_question_included(builder):
    ctx = CompressedContext(
        original_query="What is the capital of France?",
        selected_sentences=[CandidateSentence(sentence_id="1", doc_id="1", text="Paris is the capital of France.", score=1.0)],
        metrics={}
    )
    prompt = builder.build_prompt(ctx)
    assert "What is the capital of France?" in prompt

def test_context_included(builder):
    ctx = CompressedContext(
        original_query="q",
        selected_sentences=[CandidateSentence(sentence_id="1", doc_id="1", text="Paris is the capital of France.", score=1.0)],
        metrics={}
    )
    prompt = builder.build_prompt(ctx)
    assert "Paris is the capital of France." in prompt

def test_no_hidden_labels(builder):
    ctx = CompressedContext(
        original_query="q",
        selected_sentences=[CandidateSentence(sentence_id="1", doc_id="1", text="Paris is the capital of France.", score=1.0)],
        metrics={}
    )
    prompt = builder.build_prompt(ctx)
    assert "gold_answer" not in prompt
    assert "supporting" not in prompt

def test_stable_formatting(builder):
    ctx = CompressedContext(
        original_query="q",
        selected_sentences=[CandidateSentence(sentence_id="1", doc_id="1", text="Paris is the capital of France.", score=1.0)],
        metrics={}
    )
    prompt1 = builder.build_prompt(ctx)
    prompt2 = builder.build_prompt(ctx)
    assert prompt1 == prompt2
