import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.data.schemas import QAExample, Document, RetrievedDocument
from src.config.settings import EARCConfig
from src.retrieval.bm25 import BM25Retriever
from src.compression.compressor import EvidenceAwareCompressor
from src.llm.factory import create_provider
from src.prompting.builder import PromptBuilder
from src.evaluation.exact_match import compute_exact_match

def test_pipeline_completes():
    config = EARCConfig()
    
    # 1. Dataset
    examples = [
        QAExample(
            query="What is the capital of France?",
            gold_answers=["Paris"],
            documents=[Document(doc_id="1", title="France", text="Paris is the capital and most populous city of France.")]
        )
    ]
    
    # 2. Retriever
    retriever = BM25Retriever()
    retriever.index(examples[0].documents)
    retrieved = retriever.retrieve("What is the capital of France?", top_k=1)
    
    # 3. Compressor
    compressor = EvidenceAwareCompressor(config)
    compressed_ctx = compressor.compress(
        query=examples[0].query,
        documents=retrieved,
        budget=config.compression.budget
    )
    
    assert len(compressed_ctx.selected_sentences) > 0
    assert compressed_ctx.metrics['budget_used'] <= config.compression.budget
    
    # 4. Prompt Builder
    builder = PromptBuilder()
    prompt = builder.build_prompt(compressed_ctx)
    assert "Paris" in prompt
    
    # 5. LLM
    llm = create_provider("mock", None)
    result = llm.generate(prompt)
    
    # 6. Evaluation
    em = compute_exact_match([result.text], [examples[0].gold_answers])
    assert em >= 0.0 # Just verify it runs
