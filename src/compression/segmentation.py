import logging
import re
from typing import List
from src.data.schemas import RetrievedDocument, CandidateSentence

logger = logging.getLogger(__name__)

class SentenceSegmenter:
    """Segments documents into sentences using spaCy with regex fallback."""

    def __init__(self, spacy_model: str = 'en_core_web_sm', use_regex_fallback: bool = True):
        self.spacy_model = spacy_model
        self.use_regex_fallback = use_regex_fallback
        self.nlp = None

    def _load_model(self):
        if self.nlp is None:
            try:
                import spacy
                self.nlp = spacy.load(self.spacy_model)
            except ImportError:
                logger.warning(f"spaCy not installed or model {self.spacy_model} not found. Falling back to regex.")
                self.use_regex_fallback = True
            except OSError:
                logger.warning(f"spaCy model {self.spacy_model} not found. Try `python -m spacy download {self.spacy_model}`. Falling back to regex.")
                self.use_regex_fallback = True

    def segment(self, documents: List[RetrievedDocument]) -> List[CandidateSentence]:
        self._load_model()
        candidates = []
        seen_texts = set()
        total_duplicates = 0
        
        for doc in documents:
            if not doc.text.strip():
                continue
                
            if self.nlp is not None and not self.use_regex_fallback:
                try:
                    sentences = self._spacy_segment(doc.text)
                except Exception as e:
                    logger.error(f"spaCy segmentation failed: {e}. Using regex fallback.")
                    sentences = self._regex_segment(doc.text)
            elif self.nlp is not None:
                 sentences = self._spacy_segment(doc.text)
            else:
                sentences = self._regex_segment(doc.text)
                
            doc_sentences_count = 0
            for text in sentences:
                clean_text = text.strip()
                if not clean_text:
                    continue
                    
                normalized_text = " ".join(clean_text.lower().split())
                if normalized_text in seen_texts:
                    total_duplicates += 1
                    continue
                    
                seen_texts.add(normalized_text)
                sentence_id = CandidateSentence.generate_id(doc.doc_id, clean_text)
                
                title = doc.title.strip() if doc.title else ""
                if title and not clean_text.lower().startswith(title.lower()):
                    candidate_text = f"{title}: {clean_text}"
                else:
                    candidate_text = clean_text

                candidates.append(CandidateSentence(
                    sentence_id=sentence_id,
                    document_id=doc.doc_id,
                    title=title,
                    text=candidate_text,
                    source_rank=doc.rank
                ))
                doc_sentences_count += 1
                
            logger.debug(f"Document {doc.doc_id} segmented into {doc_sentences_count} sentences.")
            
        logger.info(f"Segmented {len(documents)} documents into {len(candidates)} total candidates. Removed {total_duplicates} exact duplicates.")
        return candidates

    def _spacy_segment(self, text: str) -> List[str]:
        max_length = 1000000
        sentences = []
        for i in range(0, len(text), max_length):
            chunk = text[i:i + max_length]
            doc = self.nlp(chunk)
            sentences.extend([sent.text for sent in doc.sents])
        return sentences

    def _regex_segment(self, text: str) -> List[str]:
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Rev|Capt|Lt|Mt|St|e\.g|i\.e)\.', r'\1<DOT>', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        return sentences
