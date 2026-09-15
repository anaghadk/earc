import logging
from typing import List, Dict, Any, Optional
from src.data.schemas import RetrievedDocument
from src.prompting.builder import PromptBuilder
from src.llm.base import TokenCounter

logger = logging.getLogger(__name__)

class TopKBaseline:
    """Top-K Documents Baseline."""
    
    def __init__(self, prompt_builder: PromptBuilder, k: int = 5):
        self.prompt_builder = prompt_builder
        self.k = k
        logger.info(f"Initialized TopKBaseline with k={k}")
        
    def run(self, query: str, retrieved_docs: List[RetrievedDocument], token_counter: TokenCounter, budget: Optional[int] = None) -> Dict[str, Any]:
        """Runs Top-K document selection."""
        
        # Take the top-k documents (assume already sorted by retrieval score)
        top_docs = retrieved_docs[:self.k]
        
        doc_texts = []
        current_tokens = 0
        
        for doc in top_docs:
            title_text = doc.title if doc.title else "Untitled"
            doc_text = f"Title: {title_text}\n{doc.text}"
            
            if budget is not None:
                doc_tokens = token_counter.count(doc_text)
                if current_tokens + doc_tokens > budget and current_tokens > 0:
                    break
                current_tokens += doc_tokens
                
            doc_texts.append(doc_text)
            
        concatenated_text = "\n\n".join(doc_texts)
        
        # Build prompt
        _ = self.prompt_builder.build(query, concatenated_text)
        
        # Calculate tokens
        all_doc_texts = [f"Title: {doc.title if doc.title else 'Untitled'}\n{doc.text}" for doc in retrieved_docs]
        full_text = "\n\n".join(all_doc_texts)
        
        original_tokens = token_counter.count(full_text)
        compressed_tokens = token_counter.count(concatenated_text)
        
        compression_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 0.0
        reduction_percentage = max(0.0, 1.0 - (compressed_tokens / original_tokens)) if original_tokens > 0 else 0.0
        
        result = {
            "compressed_text": concatenated_text,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compression_ratio,
            "reduction_percentage": reduction_percentage
        }
        
        return result
