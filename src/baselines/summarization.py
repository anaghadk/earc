import logging
import time
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.data.schemas import RetrievedDocument
from src.prompting.builder import PromptBuilder
from src.llm.base import TokenCounter

logger = logging.getLogger(__name__)

class SummarizationBaseline:
    """Summarization Baseline using Flan-T5."""
    
    def __init__(self, prompt_builder: PromptBuilder, model_name: str = 'google/flan-t5-base', device: str = 'cuda', max_summary_length: int = 200):
        self.prompt_builder = prompt_builder
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'
        self.max_summary_length = max_summary_length
        self.tokenizer = None
        self.model = None
        logger.info(f"Initialized SummarizationBaseline with model {model_name} on {self.device}")
        
    def load_model(self) -> None:
        if self.model is None:
            logger.info(f"Loading summarization model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully.")
            
    def summarize(self, text: str) -> str:
        self.load_model()
        
        # Max input length for T5 is 512
        max_input_length = 512
        prompt = f"Summarize the following text:\n{text}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=max_input_length, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_summary_length,
                num_beams=4,
                early_stopping=True
            )
            
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    def run(self, query: str, retrieved_docs: List[RetrievedDocument], token_counter: TokenCounter) -> Dict[str, Any]:
        """Runs the summarization baseline over retrieved documents."""
        
        doc_texts = []
        for doc in retrieved_docs:
            title_text = doc.title if doc.title else "Untitled"
            doc_texts.append(f"Title: {title_text}\n{doc.text}")
            
        full_text = "\n\n".join(doc_texts)
        original_tokens = token_counter.count(full_text)
        
        start_time = time.time()
        
        # Handle long inputs by chunking (~350 words per chunk)
        words = full_text.split()
        chunk_size = 350
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        summaries = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            summaries.append(self.summarize(chunk))
            
        compressed_text = "\n\n".join(summaries)
        
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
            "summarization_latency": latency
        }
        return result
