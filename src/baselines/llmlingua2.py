import logging
import time
import re
from typing import List, Dict, Any, Optional
from src.data.schemas import RetrievedDocument
from src.prompting.builder import PromptBuilder
from src.llm.base import TokenCounter

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None

logger = logging.getLogger(__name__)

class LLMLingua2Baseline:
    """Prompt Compression Baseline using LLMLingua-2."""
    
    def __init__(self, prompt_builder: PromptBuilder, device: str = 'cuda', target_token: int = 200):
        if PromptCompressor is None:
            raise ImportError("llmlingua library is not installed. Please install it with 'pip install llmlingua'.")
            
        self.prompt_builder = prompt_builder
        self.device = device
        self.target_token = target_token
        
        logger.info("Initializing LLMLingua2 PromptCompressor...")
        self.compressor = PromptCompressor(
            model_name='microsoft/llmlingua-2-xlm-roberta-large-meetingbank',
            use_llmlingua2=True,
            device_map=self.device
        )
        logger.info("LLMLingua2 PromptCompressor initialized.")
        
    def _extract_keywords(self, query: str) -> List[str]:
        words = re.findall(r'\b\w{4,}\b', query.lower())
        return list(set(words))
        
    def run(self, query: str, retrieved_docs: List[RetrievedDocument], token_counter: TokenCounter, budget: int = 200) -> Dict[str, Any]:
        """Runs prompt compression using LLMLingua-2."""
        
        doc_texts = []
        for doc in retrieved_docs:
            title_text = doc.title if doc.title else "Untitled"
            doc_texts.append(f"Title: {title_text}\n{doc.text}")
            
        concatenated_text = "\n\n".join(doc_texts)
        original_tokens = token_counter.count(concatenated_text)
        
        keywords = self._extract_keywords(query)
        
        start_time = time.time()
        
        try:
            results = self.compressor.compress_prompt(
                context=[concatenated_text],
                target_token=budget,
                force_tokens=keywords
            )
            compressed_text = results.get('compressed_prompt', "")
        except Exception as e:
            logger.error(f"Error during LLMLingua2 compression: {e}")
            compressed_text = ""
            
        latency = time.time() - start_time
        compressed_tokens = token_counter.count(compressed_text)
        
        _ = self.prompt_builder.build(query, compressed_text)
        
        compression_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 0.0
        reduction_percentage = max(0.0, 1.0 - (compressed_tokens / original_tokens)) if original_tokens > 0 else 0.0
        
        result = {
            "compressed_text": compressed_text,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compression_ratio,
            "reduction_percentage": reduction_percentage,
            "compression_latency": latency
        }
        
        return result
