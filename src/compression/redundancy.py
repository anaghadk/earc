"""
Cross-document redundancy filtering (paper §Stage 6).

Iterates through ranked candidates. For each candidate, computes cosine
similarity with ALL already-selected sentence embeddings (cross-document).
If max_similarity > τ, the candidate is discarded.

The paper explicitly emphasizes CROSS-DOCUMENT redundancy: comparing
sentences from different source documents, not just within the same doc.
"""

import logging
from typing import List

import numpy as np

from src.data.schemas import CandidateSentence, SentenceStatus

logger = logging.getLogger(__name__)


class RedundancyFilter:
    """
    Cross-document redundancy filter using embedding cosine similarity.

    Paper §Stage 6: "a candidate is discarded if its embedding cosine
    similarity to any already-kept sentence exceeds τ (default 0.85)."
    """

    def filter(
        self,
        candidates: List[CandidateSentence],
        threshold: float = 0.85,
    ) -> List[CandidateSentence]:
        """
        Filter redundant candidates using cosine similarity threshold.

        Args:
            candidates: Candidates sorted by hybrid_score (descending).
            threshold: Cosine similarity threshold τ (default 0.85).
                       Candidates with max_sim > threshold are discarded.
                       Note: strictly greater than, not >=.

        Returns:
            List of surviving candidates in rank order.
        """
        if not candidates:
            return []

        surviving: List[CandidateSentence] = []
        selected_embs: List[np.ndarray] = []

        for candidate in candidates:
            if candidate.embedding is None:
                logger.warning(
                    f"Candidate {candidate.sentence_id} has no embedding. "
                    f"Passing through redundancy filter."
                )
                surviving.append(candidate)
                continue

            # First candidate always survives
            if not selected_embs:
                surviving.append(candidate)
                selected_embs.append(candidate.embedding)
                continue

            max_sim = self._max_cosine_with_selected(
                candidate.embedding, selected_embs
            )

            if max_sim > threshold:
                candidate.status = SentenceStatus.REJECTED_REDUNDANCY
                logger.debug(
                    f"Rejected {candidate.sentence_id} (doc={candidate.document_id}): "
                    f"redundancy sim={max_sim:.4f} > τ={threshold}"
                )
            else:
                surviving.append(candidate)
                selected_embs.append(candidate.embedding)

        rejected_count = len(candidates) - len(surviving)
        logger.info(
            f"Redundancy filter: {len(surviving)}/{len(candidates)} survive "
            f"(τ={threshold}, {rejected_count} rejected)"
        )
        return surviving

    @staticmethod
    def _max_cosine_with_selected(
        candidate_emb: np.ndarray,
        selected_embs: List[np.ndarray],
    ) -> float:
        """
        Compute max cosine similarity between candidate and all selected embeddings.

        Since embeddings are L2-normalized, cosine = dot product.
        """
        if not selected_embs:
            return 0.0

        # Stack for vectorized computation
        selected_matrix = np.stack(selected_embs)
        similarities = selected_matrix @ candidate_emb
        return float(np.max(similarities))
