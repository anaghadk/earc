import logging
import pickle
import string
from typing import Any

from rank_bm25 import BM25Okapi

from src.data.schemas import RetrievedDocument
from src.retrieval.base import Retriever

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Lowercase and remove punctuation, then split by whitespace."""
    if not text:
        return []
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


class BM25Retriever(Retriever):
    """
    BM25 retriever using rank_bm25.BM25Okapi.
    """

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.corpus_metadata: list[dict[str, Any]] = []

    def index(self, documents: list[dict]) -> None:
        if not documents:
            logger.warning("Empty document list provided to BM25 index.")
            self.bm25 = None
            self.corpus_metadata = []
            return

        tokenized_corpus = []
        self.corpus_metadata = []

        for doc in documents:
            text = doc.get("text", "")
            tokenized_corpus.append(tokenize(text))
            self.corpus_metadata.append(doc)

        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Indexed {len(documents)} documents using BM25.")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        if not self.bm25 or not self.corpus_metadata:
            logger.warning("BM25 index is empty.")
            return []
        
        if not query:
            logger.warning("Empty query provided to BM25 retrieve.")
            return []

        if top_k <= 0:
            return []

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            doc = self.corpus_metadata[idx]
            score = float(scores[idx])
            results.append(RetrievedDocument(
                doc_id=doc.get("doc_id", str(idx)),
                title=doc.get("title", ""),
                text=doc.get("text", ""),
                score=score,
                rank=rank,
                retriever=self.name,
                query=query,
                metadata=doc.get("metadata", {})
            ))

        logger.info(f"BM25 retrieved {len(results)} docs for query: '{query}'")
        return results

    def save_index(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "metadata": self.corpus_metadata}, f)
        logger.info(f"Saved BM25 index to {path}")

    def load_index(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.corpus_metadata = data["metadata"]
        logger.info(f"Loaded BM25 index with {len(self.corpus_metadata)} docs from {path}")

    @property
    def name(self) -> str:
        return "bm25"
