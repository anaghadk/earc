"""
Answer extraction for generative QA evaluation.

Extracts a concise answer string from a full LLM response so that
exact-match and F1 metrics are computed against the answer portion
rather than the entire natural-language explanation.

Design constraints:
  - NO LLM involvement.
  - NO hard-coded answers.
  - Conservative: if no clear pattern is found, returns the full
    cleaned response rather than guessing.
  - Normalizes whitespace only; does not alter casing or punctuation
    (normalization is handled downstream by exact_match/f1 utilities).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Patterns ordered from most-specific to least-specific.
# Each pattern must capture the answer in group 1.
# ---------------------------------------------------------------------------
_ANSWER_PATTERNS: list[re.Pattern[str]] = [
    # "The answer to the question is X."
    re.compile(
        r"[Tt]he\s+answer\s+to\s+the\s+question\s+is\s+[:\-]?\s*(.+?)(?:[.\n]|$)",
        re.IGNORECASE,
    ),
    # "The correct answer is X."
    re.compile(
        r"[Tt]he\s+correct\s+answer\s+is\s+[:\-]?\s*(.+?)(?:[.\n]|$)",
        re.IGNORECASE,
    ),
    # "The answer is X."
    re.compile(
        r"[Tt]he\s+answer\s+is\s+[:\-]?\s*(.+?)(?:[.\n]|$)",
        re.IGNORECASE,
    ),
    # "Final answer: X"  /  "Final Answer: X"  — highest priority for structured output format
    re.compile(
        r"[Ff]inal\s+[Aa]nswer\s*[:\-]\s*(.+?)(?:[.\n]|$)",
        re.IGNORECASE,
    ),
    # "Answer: X"  /  "Answer - X"
    re.compile(
        r"^[Aa]nswer\s*[:\-]\s*(.+?)(?:[.\n]|$)",
        re.MULTILINE,
    ),
    # "Therefore, the answer is X."
    re.compile(
        r"[Tt]herefore[,\s]+(?:the\s+)?answer\s+is\s+[:\-]?\s*(.+?)(?:[.\n]|$)",
        re.IGNORECASE,
    ),
    # "So the answer is X."  /  "Thus the answer is X."
    re.compile(
        r"(?:[Ss]o|[Tt]hus)[,\s]+(?:the\s+)?answer\s+is\s+[:\-]?\s*(.+?)(?:[.\n]|$)",
        re.IGNORECASE,
    ),
    # "Based on the evidence[,] X ..."
    re.compile(
        r"[Bb]ased\s+on\s+(?:the\s+)?evidence[,\s]+(?:the\s+answer\s+is\s+)?(.+?)(?:[.\n]|$)",
        re.IGNORECASE,
    ),
]


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including newlines) to a single space."""
    return " ".join(text.split())


def extract_answer(response: str) -> str:
    """
    Extract the concise answer from a generative QA model response.

    Tries each pattern in order of specificity. Returns the first
    non-empty match. If no pattern matches, returns the full response
    with whitespace normalized.

    Args:
        response: Raw text returned by the LLM.

    Returns:
        Extracted answer string with normalized whitespace. Never empty
        if response is non-empty; falls back to the full cleaned response.
    """
    if not response or not response.strip():
        return ""

    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(response)
        if match:
            candidate = _normalize_whitespace(match.group(1).strip())
            if candidate:
                return candidate

    # No pattern matched — return the full response, whitespace-normalized.
    return _normalize_whitespace(response.strip())
