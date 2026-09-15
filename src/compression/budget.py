"""
Token-budget greedy sentence selection (paper §Stage 7).

Greedily packs sentences in rank order until the token budget is reached.
Whole sentences only — no truncation. Sentences exceeding the full
budget are skipped, not split.
"""

import logging
from typing import List

from src.data.schemas import CandidateSentence, SentenceStatus
from src.llm.base import TokenCounter

logger = logging.getLogger(__name__)


class SimpleTokenCounter(TokenCounter):
    """
    Approximate token counter that splits on whitespace.

    WARNING: This is an approximation only. For accurate budget accounting,
    use the backend-specific tokenizer (e.g., tiktoken for Ollama/Mistral).
    """

    def count(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        return len(text.split())

    @property
    def name(self) -> str:
        return "whitespace-split (approximate)"


import re
from typing import List, Optional, Set

_TEMPORAL_MARKER = re.compile(
    r"\b(1[0-9]{3}|20[0-2][0-9]|\d{1,2}/\d{1,2}/\d{2,4}|january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_LOCATIVE_MARKER = re.compile(
    r"\b(based in|located in|found in|native to|situated in|born in|headquartered in|"
    r"originates?\s+from|birthplace|place of birth|native of|citizen of|nationality|"
    r"serbia|serbian|croatia|croatian|austrian|austria|france|french|italy|italian|germany|german|america|american)\b",
    re.IGNORECASE,
)
_PERSON_MARKER = re.compile(
    r"\b(directed by|written by|composed by|founded by|performed by|played by|hosted by|"
    r"produced by|voiced by|invented by|established by|started by)\b",
    re.IGNORECASE,
)


class TokenBudgetSelector:
    """
    Greedy token-budget sentence selector (paper §Stage 7).

    Iterates through ranked, redundancy-filtered candidates in order.
    Supports question-type aware selection:
    - MULTI-HOP: Two-pass selection ensuring cross-document representation across hops
      before packing secondary sentences from the same document.
    - MULTI-PART FACTOID: Ensures sentences covering each distinct requested attribute
      (e.g. temporal + locative) are selected before filling remaining budget.
    - FACTOID & DESCRIPTIVE: Standard rank-ordered greedy packing.
    """

    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter

    def select(
        self,
        candidates: List[CandidateSentence],
        budget: int,
        q_type: str = "FACTOID",
        q_details: Optional[dict] = None,
    ) -> List[CandidateSentence]:
        """
        Select sentences within the token budget.

        Args:
            candidates: Ranked, redundancy-filtered candidates.
            budget: Maximum total tokens for the compressed context.
            q_type: Detected question type ("FACTOID", "DESCRIPTIVE", "MULTI-HOP").
            q_details: Optional details dictionary (e.g. is_multipart, requested_attributes).

        Returns:
            List of selected CandidateSentence objects.
        """
        if budget <= 0:
            logger.warning(f"Token budget is {budget}, returning empty selection.")
            for c in candidates:
                c.status = SentenceStatus.REJECTED_BUDGET
            return []

        for candidate in candidates:
            candidate.token_count = self.token_counter.count(candidate.text)

        selected: List[CandidateSentence] = []
        selected_ids: Set[str] = set()
        current_tokens = 0

        # Helper function to try adding a candidate
        def _try_add(candidate: CandidateSentence) -> bool:
            nonlocal current_tokens
            if candidate.sentence_id in selected_ids:
                return True
            tc = candidate.token_count or self.token_counter.count(candidate.text)
            if tc > budget:
                candidate.status = SentenceStatus.REJECTED_BUDGET
                return False
            if current_tokens + tc <= budget:
                candidate.status = SentenceStatus.SELECTED
                selected.append(candidate)
                selected_ids.add(candidate.sentence_id)
                current_tokens += tc
                return True
            else:
                return False

        is_multipart = bool(q_details and q_details.get("is_multipart"))
        req_attrs = (q_details or {}).get("requested_attributes", set()) if is_multipart else set()

        if q_type == "MULTI-HOP":
            # Pass 1: Cross-document coverage (take top candidate from each distinct source document)
            seen_docs: Set[str] = set()
            for candidate in candidates:
                doc_key = candidate.document_id or candidate.title
                if doc_key not in seen_docs:
                    if _try_add(candidate):
                        seen_docs.add(doc_key)

            # Pass 2: Greedily fill remaining budget with next best candidates
            for candidate in candidates:
                if candidate.sentence_id not in selected_ids:
                    _try_add(candidate)

        elif is_multipart and len(req_attrs) >= 2:
            # Pass 1: Ensure each requested attribute has at least one representative sentence
            for attr in sorted(req_attrs):
                for candidate in candidates:
                    if candidate.sentence_id in selected_ids:
                        continue
                    combined = f"{candidate.title}: {candidate.text}" if candidate.title else candidate.text
                    matches_attr = False
                    if attr == "temporal" and _TEMPORAL_MARKER.search(combined):
                        matches_attr = True
                    elif attr == "locative" and _LOCATIVE_MARKER.search(combined):
                        matches_attr = True
                    elif attr == "person" and _PERSON_MARKER.search(combined):
                        matches_attr = True

                    if matches_attr:
                        if _try_add(candidate):
                            break

            # Pass 2: Greedily fill remaining budget with next best candidates
            for candidate in candidates:
                if candidate.sentence_id not in selected_ids:
                    _try_add(candidate)

        else:
            # Standard greedy selection in rank order (FACTOID & DESCRIPTIVE)
            for candidate in candidates:
                _try_add(candidate)

        # Mark all unselected candidates as REJECTED_BUDGET
        for candidate in candidates:
            if candidate.sentence_id not in selected_ids:
                candidate.status = SentenceStatus.REJECTED_BUDGET

        logger.info(
            f"Budget selection ({q_type}): {len(selected)}/{len(candidates)} candidates, "
            f"{current_tokens}/{budget} tokens used."
        )
        return selected
