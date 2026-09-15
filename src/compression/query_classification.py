"""
Deterministic query classification for the EARC pipeline.

Categorizes queries into exactly three broad types:
1. FACTOID: Direct-answer queries (person, place, object, date, number, etc.)
   including multi-part factoids (e.g. "when and where", "who and when").
2. DESCRIPTIVE: Explanation, mechanism, process, cause, reason, effect (e.g. "how does", "why did").
3. MULTI-HOP: Multi-document synthesis questions, determined by dataset metadata
   (e.g., HotpotQA) or explicit multi-hop bridge phrasing.

Ambiguous cases strictly default to FACTOID or DESCRIPTIVE, never forced to MULTI-HOP.
No ML models, no external network calls, zero gold answer leakage.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Set, Tuple


# Regex patterns for DESCRIPTIVE queries
_DESCRIPTIVE_PATTERNS = [
    # "how does/did/do/can/is/are/would/will/to..." (excluding numeric "how many/much/long/tall/far/old/big/fast")
    re.compile(r"^\s*how\s+(?:does|did|do|can|could|is|are|would|will|to)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(?:does|did|do|can|could|is|are|would|will)\s+.*(?:work|function|operate|occur|happen|form)\b", re.IGNORECASE),
    # "why did/does/do/was/were/is/are/would..."
    re.compile(r"^\s*why\s+(?:did|does|do|was|were|is|are|would|should|has|have|had)\b", re.IGNORECASE),
    # Explanatory commands & phrases
    re.compile(r"^\s*(?:explain|describe)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:causes|caused|happens\s+when|is\s+the\s+process\s+of|is\s+the\s+mechanism\s+of|are\s+the\s+reasons\s+for|was\s+the\s+reason\s+for|is\s+the\s+reason\s+for|are\s+the\s+consequences\s+of|are\s+the\s+effects\s+of)\b", re.IGNORECASE),
]

# Numeric/quantitative indicators (these are FACTOID even if they start with "how")
_HOW_QUANTITATIVE_PATTERN = re.compile(
    r"^\s*how\s+(?:many|much|long|old|far|tall|large|big|fast|high|heavy|deep|often)\b",
    re.IGNORECASE,
)

# Multi-hop linguistic indicators (used only if dataset metadata is absent)
_MULTIHOP_BRIDGE_PATTERNS = [
    re.compile(r"\bboth\s+.+\s+and\s+.+\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+(?:one|of\s+these)?\s*.*(?:\s+is\s+(?:older|younger|larger|smaller|taller|longer|shorter|more|less|first|earlier|later))\s*,\s*.+\s+or\s+.+\b", re.IGNORECASE),
    re.compile(r"\bwho\s+.*(?:directed|acted\s+in|starred\s+in|produced|wrote)\s+the\s+.*(?:in\s+which|where)\b", re.IGNORECASE),
    # Comparison and difference questions
    re.compile(r"\b(?:compare|comparison\s+(?:of|between)|differences?\s+between|similarities?\s+between|contrast\s+(?:between|with)|distinguish\s+between)\b", re.IGNORECASE),
    re.compile(r"\b(?:\w+)\s+(?:vs\.?|versus)\s+(?:\w+)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:compare|contrast)\b", re.IGNORECASE),
]


def classify_question(
    query: str,
    dataset: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Tuple[str, dict[str, Any]]:
    """
    Deterministically classify a user query into FACTOID, DESCRIPTIVE, or MULTI-HOP.

    Args:
        query: Raw query string.
        dataset: Dataset identifier (e.g. 'hotpot', 'hotpotqa', 'nq', 'trivia', 'triviaqa').
        metadata: Optional metadata dictionary associated with the query/example.

    Returns:
        (q_type, details) tuple where:
            q_type is one of "FACTOID", "DESCRIPTIVE", "MULTI-HOP".
            details contains auxiliary information (e.g. multi-part attributes for factoid).
    """
    q_str = query.strip()
    q_lower = q_str.lower()
    details: dict[str, Any] = {
        "is_multipart": False,
        "requested_attributes": set(),
    }

    # 1. Check for MULTI-HOP
    # Explicit dataset metadata check (HotpotQA is canonically multi-hop)
    ds_norm = (dataset or "").lower()
    meta_ds = ((metadata or {}).get("dataset") or "").lower()
    is_hotpot_dataset = ds_norm in ("hotpot", "hotpotqa") or meta_ds in ("hotpot", "hotpotqa")
    has_supporting_facts = bool((metadata or {}).get("supporting_facts")) or bool((metadata or {}).get("supporting_documents"))

    if is_hotpot_dataset or has_supporting_facts:
        return "MULTI-HOP", details

    # Strict multi-hop phrasing check (only if clearly comparative/bridge)
    for p in _MULTIHOP_BRIDGE_PATTERNS:
        if p.search(q_str):
            return "MULTI-HOP", details

    # 2. Check for DESCRIPTIVE
    # Exclude quantitative "how many/how long" which are FACTOID
    if not _HOW_QUANTITATIVE_PATTERN.search(q_str):
        for p in _DESCRIPTIVE_PATTERNS:
            if p.search(q_str):
                return "DESCRIPTIVE", details

    # 3. FACTOID (Default)
    # Check for multi-part factoids (e.g., "when and where", "who and when")
    requested_attributes: Set[str] = set()

    # Temporal attribute
    if re.search(r"\b(?:when|what\s+year|what\s+date|what\s+time)\b", q_lower):
        requested_attributes.add("temporal")

    # Locative attribute
    if re.search(r"\b(?:where|what\s+(?:country|city|state|place|location|town|region))\b", q_lower):
        requested_attributes.add("locative")

    # Person/Entity attribute
    if re.search(r"\b(?:who|whose|which\s+(?:person|author|actor|founder|singer|musician|inventor|director))\b", q_lower):
        requested_attributes.add("person")

    # Part/Structure/Definition attribute
    if re.search(r"\b(?:what|which)\s+(?:part|region|component|organ|area|section|structure)\s+of\b", q_lower):
        requested_attributes.add("part")

    # Multi-part detection: query asks for 2 or more distinct interrogative attributes
    # joined by conjunctions (e.g. "when and where", "who was ... and when did ...")
    if len(requested_attributes) >= 2 and re.search(r"\b(?:and|as\s+well\s+as)\b", q_lower):
        details["is_multipart"] = True
        details["requested_attributes"] = requested_attributes

    return "FACTOID", details
