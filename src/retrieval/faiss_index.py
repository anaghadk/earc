import logging
from typing import Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def build_index(embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP from normalized embeddings.
    Inner product on normalized vectors is equivalent to cosine similarity.
    """
    if embeddings.size == 0:
        raise ValueError("Cannot build index from empty embeddings.")

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("Using FAISS GPU index.")
        except Exception as e:
            logger.warning(f"Failed to use FAISS GPU: {e}. Falling back to CPU.")

    index.add(embeddings)
    logger.info(f"Built FAISS index with {index.ntotal} vectors of dimension {d}.")
    return index


def save_index(index: faiss.Index, path: str) -> None:
    """Save FAISS index to disk (CPU version)."""
    # If it's a GPU index, move to CPU first
    if hasattr(faiss, "index_gpu_to_cpu"):
        try:
            index = faiss.index_gpu_to_cpu(index)
        except Exception:
            pass
    faiss.write_index(index, path)
    logger.info(f"Saved FAISS index to {path}")


def load_index(path: str) -> faiss.Index:
    """Load FAISS index from disk."""
    index = faiss.read_index(path)
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors from {path}")
    return index


def validate_index(index: faiss.Index, expected_size: int) -> bool:
    """Validate that the index contains the expected number of vectors."""
    if index is None:
        return False
    return index.ntotal == expected_size


def search(index: faiss.Index, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index.
    
    Args:
        index: FAISS index object.
        query_embedding: Numpy array of shape (N, d).
        top_k: Number of top results to retrieve.
        
    Returns:
        Tuple of (distances, indices).
    """
    if top_k <= 0:
        return np.array([]), np.array([])
    if index.ntotal == 0:
        return np.array([]), np.array([])

    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_embedding, k)
    return distances, indices
