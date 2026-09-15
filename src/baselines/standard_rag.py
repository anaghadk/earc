import logging
from typing import List, Dict, Any
from src.data.schemas import RetrievedDocument
from src.prompting.builder import PromptBuilder
from src.llm.base import TokenCounter

logger = logging.getLogger(__name__)

class StandardRAGBaseline:
    """Standard RAG Baseline without any compression."""
    
    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder
        logger.info("Initialized StandardRAGBaseline")
        
    def run(self, query: str, retrieved_docs: List[RetrievedDocument], token_counter: TokenCounter) -> Dict[str, Any]:
        """Runs standard RAG with no compression."""
        
        # Concatenate ALL retrieved document texts
        doc_texts = []
        for doc in retrieved_docs:
            title_text = doc.title if doc.title else "Untitled"
            doc_texts.append(f"Title: {title_text}\n{doc.text}")
            
        concatenated_text = "\n\n".join(doc_texts)
        
        # Call prompt builder (not strictly necessary for the return dict, but for side-effects or interface adherence)
        _ = self.prompt_builder.build(query, concatenated_text)
        
        # Count tokens
        original_tokens = token_counter.count(concatenated_text)
        compressed_tokens = original_tokens  # No compression
        
        result = {
            "compressed_text": concatenated_text,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": 1.0,
            "reduction_percentage": 0.0
        }
        
        return result
