"""
Token-level F1 metric for QA evaluation.

Computes precision, recall, and F1 over normalized tokens.
Supports multiple gold answers — returns the maximum F1
across all acceptable answers.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.exact_match import normalize_answer


def _tokenize(text: str) -> list[str]:
    """Tokenize normalized text by whitespace."""
    return normalize_answer(text).split()


def token_f1_score(
    prediction: str,
    gold_answer: str,
) -> dict[str, float]:
    """
    Compute token-level precision, recall, and F1
    between a prediction and a single gold answer.

    Args:
        prediction: Predicted answer string.
        gold_answer: A single gold answer string.

    Returns:
        Dict with 'precision', 'recall', 'f1'.
    """
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold_answer)

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Use multiset intersection for token overlap
    from collections import Counter

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = sum((pred_counter & gold_counter).values())

    if common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def max_token_f1_score(
    prediction: str,
    gold_answers: list[str],
) -> dict[str, float]:
    """
    Compute the maximum token F1 across all gold answers.

    Args:
        prediction: Predicted answer string.
        gold_answers: List of acceptable gold answers.

    Returns:
        Dict with 'precision', 'recall', 'f1' (best F1 answer).
    """
    if not prediction or not gold_answers:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    best_result = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for gold in gold_answers:
        result = token_f1_score(prediction, gold)
        if result["f1"] > best_result["f1"]:
            best_result = result

    return best_result


def compute_f1(
    predictions: list[str],
    gold_answers_list: list[list[str]],
) -> dict[str, float | list[float]]:
    """
    Compute token F1 over a dataset.

    For each example, takes the maximum F1 across all gold answers.

    Args:
        predictions: List of predicted answer strings.
        gold_answers_list: List of lists of gold answers.

    Returns:
        Dict with 'mean', 'std', 'per_example', 'precision_mean', 'recall_mean'.
    """
    assert len(predictions) == len(gold_answers_list), (
        f"Predictions ({len(predictions)}) and gold answers "
        f"({len(gold_answers_list)}) must have the same length"
    )

    if not predictions:
        return {
            "mean": 0.0,
            "std": 0.0,
            "per_example": [],
            "precision_mean": 0.0,
            "recall_mean": 0.0,
        }

    results = [
        max_token_f1_score(pred, golds)
        for pred, golds in zip(predictions, gold_answers_list)
    ]

    f1_scores = [r["f1"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]

    f1_arr = np.array(f1_scores)

    return {
        "mean": float(f1_arr.mean()),
        "std": float(f1_arr.std()),
        "per_example": f1_scores,
        "precision_mean": float(np.mean(precisions)),
        "recall_mean": float(np.mean(recalls)),
    }
