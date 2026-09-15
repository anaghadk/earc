"""
Abstract base class for all retrievers.

The EARC framework is retriever-agnostic (paper §Stage 1).
This interface allows BM25, dense, and hybrid retrievers to be
used interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.data.schemas import RetrievedDocument


class Retriever(ABC):
    """Abstract retriever interface."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """
        Retrieve the top-k most relevant documents for a query.

        Args:
            query: The user query string.
            top_k: Number of documents to retrieve (paper default: 5).

        Returns:
            List of RetrievedDocument, sorted by descending relevance.
        """
        ...

    @abstractmethod
    def index(self, documents: list[dict]) -> None:
        """
        Build or update the retrieval index from a list of documents.

        Args:
            documents: List of dicts with at minimum 'doc_id', 'title', 'text'.
        """
        ...

    @abstractmethod
    def save_index(self, path: str) -> None:
        """Persist the retrieval index to disk."""
        ...

    @abstractmethod
    def load_index(self, path: str) -> None:
        """Load a persisted retrieval index from disk."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the retriever name (e.g., 'bm25', 'dense')."""
        ...
