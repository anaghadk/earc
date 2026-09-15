import logging

from src.data.schemas import RetrievedDocument
from src.retrieval.base import Retriever

logger = logging.getLogger(__name__)


def min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 if max_score > 0 else 0.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


class HybridRetriever(Retriever):
    """
    Hybrid retriever that combines results from BM25 and Dense retrievers.
    """

    def __init__(
        self,
        bm25_retriever: Retriever,
        dense_retriever: Retriever,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        logger.info(f"Initialized HybridRetriever with bm25_weight={bm25_weight} and dense_weight={dense_weight}")

    def index(self, documents: list[dict]) -> None:
        logger.info("Indexing documents for Hybrid Retriever (delegating to sub-retrievers).")
        self.bm25_retriever.index(documents)
        self.dense_retriever.index(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        if not query:
            logger.warning("Empty query provided to Hybrid retrieve.")
            return []
            
        if top_k <= 0:
            return []

        # Retrieve from both
        # Retrieve more than top_k from each to ensure good overlap/coverage before combining
        fetch_k = max(top_k * 2, 20)
        bm25_results = self.bm25_retriever.retrieve(query, fetch_k)
        dense_results = self.dense_retriever.retrieve(query, fetch_k)
        
        if not bm25_results and not dense_results:
            return []

        # Normalize BM25 scores
        if bm25_results:
            bm25_scores = [doc.score for doc in bm25_results]
            norm_bm25_scores = min_max_normalize(bm25_scores)
            bm25_doc_scores = {doc.doc_id: (doc, norm_score) for doc, norm_score in zip(bm25_results, norm_bm25_scores)}
        else:
            bm25_doc_scores = {}

        # Normalize Dense scores
        if dense_results:
            dense_scores = [doc.score for doc in dense_results]
            norm_dense_scores = min_max_normalize(dense_scores)
            dense_doc_scores = {doc.doc_id: (doc, norm_score) for doc, norm_score in zip(dense_results, norm_dense_scores)}
        else:
            dense_doc_scores = {}

        # Combine
        combined_scores = {}
        all_docs = {}

        for doc_id, (doc, bm25_score) in bm25_doc_scores.items():
            all_docs[doc_id] = doc
            combined_scores[doc_id] = self.bm25_weight * bm25_score

        for doc_id, (doc, dense_score) in dense_doc_scores.items():
            if doc_id not in all_docs:
                all_docs[doc_id] = doc
                combined_scores[doc_id] = self.dense_weight * dense_score
            else:
                combined_scores[doc_id] += self.dense_weight * dense_score

        # Sort by combined score
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Create final RetrievedDocument objects
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], start=1):
            original_doc = all_docs[doc_id]
            results.append(RetrievedDocument(
                doc_id=doc_id,
                title=original_doc.title,
                text=original_doc.text,
                score=score,
                rank=rank,
                retriever=self.name,
                query=query,
                metadata=original_doc.metadata
            ))

        logger.info(f"Hybrid retrieved {len(results)} docs for query: '{query}'")
        return results

    def save_index(self, path: str) -> None:
        self.bm25_retriever.save_index(path + "_bm25")
        self.dense_retriever.save_index(path + "_dense")
        logger.info(f"Saved Hybrid index to {path}_bm25 and {path}_dense")

    def load_index(self, path: str) -> None:
        self.bm25_retriever.load_index(path + "_bm25")
        self.dense_retriever.load_index(path + "_dense")
        logger.info(f"Loaded Hybrid index from {path}_bm25 and {path}_dense")

    @property
    def name(self) -> str:
        return "hybrid"
