import logging
import hashlib
from typing import List
import numpy as np
import torch
from src.data.schemas import CandidateSentence
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: str = 'cuda', batch_size: int = 16):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.batch_size = batch_size
        self._model = None
        self._cache = {}

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model {self.model_name} to {self.device}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(f"{self.model_name}_{text}".encode('utf-8')).hexdigest()

    def embed_query(self, query: str) -> np.ndarray:
        self._load_model()
        cache_key = self._hash_text(query)
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        logger.debug("Embedding query.")
        with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
            emb = self._model.encode([query], show_progress_bar=False, convert_to_numpy=True)
            
        emb_normalized = normalize(emb, norm='l2', axis=1)[0]
        self._cache[cache_key] = emb_normalized
        return emb_normalized

    def embed_sentences(self, sentences: List[str], show_progress: bool = True) -> np.ndarray:
        self._load_model()
        uncached_indices = []
        uncached_texts = []
        embeddings = np.zeros((len(sentences), self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        
        for i, text in enumerate(sentences):
            cache_key = self._hash_text(text)
            if cache_key in self._cache:
                embeddings[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                
        if uncached_texts:
            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                new_embs = self._model.encode(
                    uncached_texts, 
                    batch_size=self.batch_size, 
                    show_progress_bar=show_progress, 
                    convert_to_numpy=True
                )
            new_embs_normalized = normalize(new_embs, norm='l2', axis=1)
            
            for idx, emb, text in zip(uncached_indices, new_embs_normalized, uncached_texts):
                embeddings[idx] = emb
                self._cache[self._hash_text(text)] = emb
                
        return embeddings

    def embed_candidates(self, candidates: List[CandidateSentence]) -> List[CandidateSentence]:
        if not candidates:
            return candidates
            
        texts = [c.text for c in candidates]
        embeddings = self.embed_sentences(texts)
        
        for i, candidate in enumerate(candidates):
            candidate.embedding = embeddings[i]
            
        logger.info(f"Embedded {len(candidates)} candidates.")
        return candidates
