import logging
import numpy as np
from typing import List
from src.data.schemas import CandidateSentence

logger = logging.getLogger(__name__)

class SemanticRelevanceScorer:
    def compute_relevance(self, query_embedding: np.ndarray, candidates: List[CandidateSentence]) -> List[CandidateSentence]:
        if not candidates:
            return candidates
            
        if query_embedding is None or len(query_embedding) == 0:
            logger.warning("Query embedding is empty or None.")
            return candidates
            
        logger.debug(f"Computing semantic relevance for {len(candidates)} candidates.")
        for candidate in candidates:
            if candidate.embedding is None:
                candidate.semantic_similarity = 0.0
                continue
                
            sim = self._cosine_similarity(query_embedding, candidate.embedding)
            sim = max(0.0, min(1.0, float(sim)))
            if np.isnan(sim):
                sim = 0.0
            candidate.semantic_similarity = sim
            
        return candidates

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b)
