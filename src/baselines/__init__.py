"""Baselines module for EARC."""

from .standard_rag import StandardRAGBaseline
from .topk import TopKBaseline
from .summarization import SummarizationBaseline
from .llmlingua2 import LLMLingua2Baseline

__all__ = [
    'StandardRAGBaseline',
    'TopKBaseline',
    'SummarizationBaseline',
    'LLMLingua2Baseline'
]
