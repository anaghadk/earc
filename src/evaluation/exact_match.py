"""
Exact Match metric for QA evaluation.

Computes normalized exact match between predicted and gold answers.
Supports multiple gold answers and answer aliases (TriviaQA, NQ).
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Optional


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for metric computation.

    Applies:
    - Unicode NFKD normalization
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Normalize whitespace

    Does NOT destroy meaningful numeric content.

    Args:
        answer: Raw answer string.

    Returns:
        Normalized answer string.
    """
    if not answer:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKD", answer)

    # Lowercase
    text = text.lower()

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Normalize whitespace
    text = " ".join(text.split())

    return text.strip()


def exact_match_score(
    prediction: str,
    gold_answers: list[str],
) -> float:
    """
    Compute exact match score.

    Returns 1.0 if the normalized prediction matches ANY of the
    normalized gold answers, 0.0 otherwise.

    Args:
        prediction: Model's predicted answer.
        gold_answers: List of acceptable gold answers (including aliases).

    Returns:
        1.0 or 0.0
    """
    if not prediction or not gold_answers:
        return 0.0

    norm_pred = normalize_answer(prediction)

    for gold in gold_answers:
        if normalize_answer(gold) == norm_pred:
            return 1.0

    return 0.0


def compute_exact_match(
    predictions: list[str],
    gold_answers_list: list[list[str]],
) -> dict[str, float]:
    """
    Compute exact match over a dataset.

    Args:
        predictions: List of predicted answer strings.
        gold_answers_list: List of lists of gold answers.

    Returns:
        Dict with 'mean', 'std', 'per_example' keys.
    """
    assert len(predictions) == len(gold_answers_list), (
        f"Predictions ({len(predictions)}) and gold answers "
        f"({len(gold_answers_list)}) must have the same length"
    )

    if not predictions:
        return {"mean": 0.0, "std": 0.0, "per_example": []}

    scores = [
        exact_match_score(pred, golds)
        for pred, golds in zip(predictions, gold_answers_list)
    ]

    import numpy as np
    scores_arr = np.array(scores)

    return {
        "mean": float(scores_arr.mean()),
        "std": float(scores_arr.std()),
        "per_example": scores,
    }
