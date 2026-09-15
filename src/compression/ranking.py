import logging
from typing import List
from src.data.schemas import CandidateSentence

logger = logging.getLogger(__name__)

class HybridRanker:
    def rank(self, candidates: List[CandidateSentence], alpha: float = 0.7, beta: float = 0.3) -> List[CandidateSentence]:
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be >= 0")
            
        if abs(alpha + beta - 1.0) > 1e-6:
            logger.warning(f"alpha ({alpha}) + beta ({beta}) != 1.0")
            
        for candidate in candidates:
            sim = candidate.semantic_similarity or 0.0
            ev = candidate.normalized_evidence or 0.0
            candidate.hybrid_score = alpha * sim + beta * ev
            
        candidates.sort(
            key=lambda c: (
                -(c.hybrid_score or 0.0),
                -(c.semantic_similarity or 0.0),
                -(c.normalized_evidence or 0.0),
                c.source_rank if c.source_rank is not None else float('inf'),
                c.sentence_id if c.sentence_id else ""
            )
        )
        
        return candidates
