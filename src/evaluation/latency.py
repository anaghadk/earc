"""
Latency tracking for pipeline stages.

Records per-stage timing for: retrieval, segmentation, embedding,
evidence scoring, redundancy filtering, compression, LLM generation,
and end-to-end latency.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np


@dataclass
class LatencyTracker:
    """
    Track latency for each stage of the EARC pipeline.

    Usage:
        tracker = LatencyTracker()
        with tracker.track("retrieval"):
            results = retriever.retrieve(query)
        with tracker.track("compression"):
            compressed = compressor.compress(query, results)
    """

    _timings: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    @contextmanager
    def track(self, stage: str) -> Generator[None, None, None]:
        """Context manager to time a pipeline stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._timings[stage].append(elapsed)

    def record(self, stage: str, elapsed: float) -> None:
        """Manually record a timing."""
        self._timings[stage].append(elapsed)

    def get_last(self, stage: str) -> Optional[float]:
        """Get the most recent timing for a stage."""
        if stage in self._timings and self._timings[stage]:
            return self._timings[stage][-1]
        return None

    def get_all(self, stage: str) -> list[float]:
        """Get all timings for a stage."""
        return list(self._timings.get(stage, []))

    @property
    def stages(self) -> list[str]:
        """List of all tracked stages."""
        return list(self._timings.keys())

    def summary(self) -> dict[str, dict[str, float]]:
        """
        Compute summary statistics for all stages.

        Returns dict of stage -> {mean, median, p95, min, max, total, count}.
        """
        result = {}
        for stage, timings in self._timings.items():
            if not timings:
                continue
            arr = np.array(timings)
            result[stage] = {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "total": float(arr.sum()),
                "count": len(timings),
            }
        return result

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._timings.clear()

    def end_to_end(self) -> Optional[float]:
        """
        Compute end-to-end latency as sum of most recent stage timings.
        """
        if not self._timings:
            return None

        total = 0.0
        for stage, timings in self._timings.items():
            if timings:
                total += timings[-1]
        return total
