import unicodedata
import re
import logging
from typing import List

from src.data.schemas import QAExample, Document

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Normalize unicode to NFKC
    text = unicodedata.normalize('NFKC', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove HTML entities like &amp;
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'")
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_document(doc: Document) -> Document:
    doc.title = normalize_text(doc.title) if doc.title else ""
    doc.text = normalize_text(doc.text) if doc.text else ""
    return doc

def preprocess_example(example: QAExample) -> QAExample:
    example.question = normalize_text(example.question)
    example.answers = [normalize_text(a) for a in example.answers if a]
    
    if example.documents:
        example.documents = [normalize_document(doc) for doc in example.documents]
        
    return example

def preprocess_dataset(examples: List[QAExample]) -> List[QAExample]:
    logger.info(f"Preprocessing {len(examples)} examples...")
    processed = []
    for ex in examples:
        processed.append(preprocess_example(ex))
    logger.info("Preprocessing complete.")
    return processed
