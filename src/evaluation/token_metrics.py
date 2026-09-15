"""
Token-level prompt metrics for compression evaluation.

Tracks retrieved tokens, compressed tokens, compression ratios,
and reduction percentages — the key compression metrics from the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TokenMetrics:
    """Aggregated token-level metrics for an experiment."""
    original_tokens: list[int] = field(default_factory=list)
    compressed_tokens: list[int] = field(default_factory=list)
    total_prompt_tokens: list[int] = field(default_factory=list)

    def add(
        self,
        original: int,
        compressed: int,
        total_prompt: Optional[int] = None,
    ) -> None:
        """Record token metrics for a single example."""
        self.original_tokens.append(original)
        self.compressed_tokens.append(compressed)
        if total_prompt is not None:
            self.total_prompt_tokens.append(total_prompt)

    @property
    def compression_ratios(self) -> list[float]:
        """Compression ratio = compressed / original for each example."""
        ratios = []
        for orig, comp in zip(self.original_tokens, self.compressed_tokens):
            if orig > 0:
                ratios.append(comp / orig)
            else:
                ratios.append(0.0)
        return ratios

    @property
    def reduction_percentages(self) -> list[float]:
        """Reduction % = 100 * (1 - compressed / original) for each example."""
        reductions = []
        for orig, comp in zip(self.original_tokens, self.compressed_tokens):
            if orig > 0:
                reductions.append(100.0 * (1.0 - comp / orig))
            else:
                reductions.append(0.0)
        return reductions

    def summary(self) -> dict[str, float]:
        """
        Compute summary statistics.

        Returns dict with mean, median, p95, min, max for:
        - original_tokens
        - compressed_tokens
        - compression_ratio
        - reduction_percentage
        """
        result = {}

        for name, values in [
            ("original_tokens", self.original_tokens),
            ("compressed_tokens", self.compressed_tokens),
            ("compression_ratio", self.compression_ratios),
            ("reduction_percentage", self.reduction_percentages),
        ]:
            if not values:
                for stat in ["mean", "median", "p95", "min", "max"]:
                    result[f"{name}_{stat}"] = 0.0
                continue

            arr = np.array(values)
            result[f"{name}_mean"] = float(arr.mean())
            result[f"{name}_median"] = float(np.median(arr))
            result[f"{name}_p95"] = float(np.percentile(arr, 95))
            result[f"{name}_min"] = float(arr.min())
            result[f"{name}_max"] = float(arr.max())

        if self.total_prompt_tokens:
            arr = np.array(self.total_prompt_tokens)
            result["total_prompt_tokens_mean"] = float(arr.mean())

        return result
