"""
RAGProjectRetriever — wraps the pre-built RAG_Project FAISS index.

Supports three retrieval modes via the `method` parameter:
  "dense"   – FAISS nearest-neighbour only (original behaviour)
  "bm25"    – BM25Okapi keyword search only
  "hybrid"  – Reciprocal Rank Fusion (RRF) of dense + BM25 candidates

Layout expected under <rag_dir>:
    faiss.index                         – FAISS index (957 220 vectors, 384-dim L2)
    bm25.pkl                            – rank_bm25.BM25Okapi, corpus_size=957 220
    tokenized_chunks.pkl                – list[list[str]], length=957 220 (same ID space)
    manifest.json                       – [{batch_idx, faiss_start, faiss_end}, ...]
    chunks/chunks_NNNNN.pkl             – list[str]  raw chunk texts per shard
    metadata/metadata_NNNNN.pkl         – list[dict] per-chunk metadata per shard

All indexes share the same ID space: position i in any of the above corresponds
to the same chunk.  This means FAISS IDs, BM25 corpus indices, and _lookup
indices are all identical.

Metadata dict keys: chunk_id, doc_id, dataset, title, char_start
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.data.schemas import RetrievedDocument
from src.retrieval.base import Retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion constant (standard: k=60)
# ---------------------------------------------------------------------------
_RRF_K = 60

# ---------------------------------------------------------------------------
# Module-level singleton cache
#
# Keyed by resolved rag_dir string.  Stores the four heavy artifact objects
# so that any RAGProjectRetriever constructed in the same Python process with
# the same rag_dir reuses already-loaded data rather than reloading from disk.
#
# Structure:  {rag_dir_str: {"lookup": list, "index": faiss.Index | None,
#                             "bm25": BM25Okapi | None,
#                             "tokenized": list | None}}
# ---------------------------------------------------------------------------
_ARTIFACT_CACHE: Dict[str, Dict[str, Any]] = {}


def _reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    k: int = _RRF_K,
) -> List[Tuple[int, float]]:
    """
    Combine multiple ranked lists of corpus IDs using Reciprocal Rank Fusion.

    RRF score for document d:  sum_r  1 / (k + rank_r(d))
    where rank_r(d) is the 1-based rank of d in list r (0 if absent).

    Returns a list of (corpus_id, rrf_score) sorted by descending score.
    """
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank_0, cid in enumerate(ranked):  # rank_0 is 0-based
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank_0 + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class RAGProjectRetriever(Retriever):
    """
    Dense / BM25 / Hybrid retriever backed by RAG_Project pre-built artifacts.

    All artifacts are loaded lazily on the first retrieve() call.

    Parameters
    ----------
    rag_dir   : path to the RAG_Project directory
    model_name: SentenceTransformer model (must match the one used to build
                faiss.index — confirmed all-MiniLM-L6-v2 / 384-dim)
    device    : 'cuda' or 'cpu'
    batch_size: embedding batch size
    method    : 'dense' | 'bm25' | 'hybrid'
    rrf_k     : RRF constant k (default 60)
    rrf_candidates : number of candidates fetched from each sub-retriever
                     before fusion (default max(top_k * 5, 20))
    """

    def __init__(
        self,
        rag_dir: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        method: str = "hybrid",
        rrf_k: int = _RRF_K,
        rrf_candidates: Optional[int] = None,
    ) -> None:
        self.rag_dir = Path(rag_dir)
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.method = method.lower()
        self.rrf_k = rrf_k
        self.rrf_candidates = rrf_candidates  # resolved at retrieve() time if None

        if self.method not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"method must be 'dense', 'bm25', or 'hybrid'; got '{method}'")

        # Load embedding model only when dense/hybrid is used
        if self.method in ("dense", "hybrid"):
            logger.info(
                f"RAGProjectRetriever [{method}]: loading SentenceTransformer "
                f"'{model_name}' on {self.device}"
            )
            self.model: Optional[SentenceTransformer] = SentenceTransformer(
                model_name, device=str(self.device)
            )
        else:
            self.model = None

        # Populated lazily on first retrieve()
        self._index: Optional[faiss.Index] = None
        self._bm25: Optional[Any] = None
        self._tokenized: Optional[List[List[str]]] = None
        # _lookup[id] = {"text": str, "meta": dict}  — shared by FAISS + BM25
        self._lookup: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_artifacts(self) -> None:
        """Load all required artifacts once per (rag_dir, method) combo per process.

        On the first call for a given rag_dir the artifacts are loaded from disk
        and stored in the module-level _ARTIFACT_CACHE.  Any subsequent
        RAGProjectRetriever instance with the same rag_dir (regardless of method)
        skips disk I/O entirely and borrows the cached references in < 1 ms.
        """
        if self._lookup is not None:
            return  # already bound to this instance

        cache_key = str(self.rag_dir.resolve())
        cached = _ARTIFACT_CACHE.get(cache_key)

        if cached is not None:
            # Warm cache hit — borrow references from cache
            self._lookup    = cached["lookup"]
            self._index     = cached["index"]
            self._bm25      = cached["bm25"]
            self._tokenized = cached["tokenized"]

            # Safety: detect if this method needs an artifact that wasn't loaded
            # by a prior method (e.g. dense ran first, now bm25 needs BM25 data).
            needs_faiss = self.method in ("dense", "hybrid") and self._index is None
            needs_bm25  = self.method in ("bm25", "hybrid") and self._bm25 is None

            if not needs_faiss and not needs_bm25:
                logger.info(
                    f"[cache hit] Reusing artifacts for '{cache_key}' "
                    f"(lookup={len(self._lookup)}, "
                    f"faiss={'ok' if self._index else 'n/a'}, "
                    f"bm25={'ok' if self._bm25 else 'n/a'})"
                )
                return

            # Partial reload — only fetch what is missing, update cache in-place
            logger.info(f"[cache partial] Loading missing artifacts: "
                        f"{'faiss ' if needs_faiss else ''}{'bm25+tokenized' if needs_bm25 else ''}")
            if needs_faiss:
                index_path = self.rag_dir / "faiss.index"
                self._index = faiss.read_index(str(index_path))
                cached["index"] = self._index
                logger.info(f"FAISS index ready: ntotal={self._index.ntotal}")
            if needs_bm25:
                with open(self.rag_dir / "bm25.pkl", "rb") as f:
                    self._bm25 = pickle.load(f)
                with open(self.rag_dir / "tokenized_chunks.pkl", "rb") as f:
                    self._tokenized = pickle.load(f)
                cached["bm25"]      = self._bm25
                cached["tokenized"] = self._tokenized
                logger.info(f"BM25 ready: corpus_size={self._bm25.corpus_size}")
            return

        # ----------------------------------------------------------------
        # Cold load — read from disk, then populate cache
        # ----------------------------------------------------------------
        logger.info(f"[cache miss] Loading artifacts from disk: '{cache_key}'")

        # 1. Chunk text + metadata lookup (always needed)
        manifest_path = self.rag_dir / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        self._lookup = []
        for shard in manifest["shards"]:
            idx = shard["batch_idx"]
            chunks_path = self.rag_dir / "chunks" / f"chunks_{idx:05d}.pkl"
            meta_path = self.rag_dir / "metadata" / f"metadata_{idx:05d}.pkl"

            logger.info(f"  Loading shard {idx}: {chunks_path.name}  {meta_path.name}")
            with open(chunks_path, "rb") as f:
                chunks: List[str] = pickle.load(f)
            with open(meta_path, "rb") as f:
                meta_list: List[Dict[str, Any]] = pickle.load(f)

            for text, meta in zip(chunks, meta_list):
                self._lookup.append({"text": text, "meta": meta})

        logger.info(f"Lookup table built: {len(self._lookup)} entries")

        # 2. FAISS index (dense / hybrid only)
        if self.method in ("dense", "hybrid"):
            index_path = self.rag_dir / "faiss.index"
            logger.info(f"Loading FAISS index from {index_path} …")
            self._index = faiss.read_index(str(index_path))
            logger.info(
                f"FAISS index ready: ntotal={self._index.ntotal}, d={self._index.d}, "
                f"metric={'L2' if self._index.metric_type == 0 else 'IP'}"
            )

        # 3. BM25 model + tokenized corpus (bm25 / hybrid only)
        if self.method in ("bm25", "hybrid"):
            bm25_path = self.rag_dir / "bm25.pkl"
            tok_path = self.rag_dir / "tokenized_chunks.pkl"
            logger.info(f"Loading BM25 model from {bm25_path} …")
            with open(bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)
            logger.info(f"Loading tokenized corpus from {tok_path} …")
            with open(tok_path, "rb") as f:
                self._tokenized = pickle.load(f)
            logger.info(
                f"BM25 ready: corpus_size={self._bm25.corpus_size}, "
                f"tokenized_chunks={len(self._tokenized)}"
            )

        # Populate cache so future instances in this process skip disk I/O
        _ARTIFACT_CACHE[cache_key] = {
            "lookup":    self._lookup,
            "index":     self._index,
            "bm25":      self._bm25,
            "tokenized": self._tokenized,
        }
        logger.info(f"[cache populated] '{cache_key}'")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into float32 embeddings (un-normalised, matches L2 index)."""
        all_embs: List[np.ndarray] = []
        use_amp = self.device.type == "cuda"
        batches = [texts[i: i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        with torch.no_grad():
            for batch in batches:
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        emb = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
                else:
                    emb = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
                all_embs.append(emb.cpu().numpy())
        if not all_embs:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        return np.vstack(all_embs).astype(np.float32)

    # ------------------------------------------------------------------
    # Sub-retrievers
    # ------------------------------------------------------------------

    def _dense_retrieve_ids(self, query: str, n: int) -> List[int]:
        """Return up to n corpus IDs ranked by FAISS L2 distance (closest first)."""
        query_emb = self._encode([query])
        k = min(n, self._index.ntotal)
        distances, indices = self._index.search(query_emb, k)
        return [int(i) for i in indices[0] if i >= 0]

    def _bm25_retrieve_ids(self, query: str, n: int) -> List[int]:
        """Return up to n corpus IDs ranked by BM25Okapi score (highest first)."""
        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)  # ndarray length=corpus_size
        top_indices = np.argsort(scores)[::-1][:n]
        return [int(i) for i in top_indices]

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        if not query:
            logger.warning("Empty query provided to RAGProjectRetriever.")
            return []
        if top_k <= 0:
            return []

        self._load_artifacts()  # no-op after first call

        # Number of candidates to fetch from each sub-retriever before fusion
        n_candidates = self.rrf_candidates or max(top_k * 5, 20)

        if self.method == "dense":
            final_ids = self._dense_retrieve_ids(query, top_k)
            logger.info(f"[dense] retrieved {len(final_ids)} docs for: '{query[:60]}'")

        elif self.method == "bm25":
            final_ids = self._bm25_retrieve_ids(query, top_k)
            logger.info(f"[bm25] retrieved {len(final_ids)} docs for: '{query[:60]}'")

        else:  # hybrid — RRF
            dense_ids = self._dense_retrieve_ids(query, n_candidates)
            bm25_ids = self._bm25_retrieve_ids(query, n_candidates)
            logger.info(
                f"[hybrid] dense_candidates={len(dense_ids)}, "
                f"bm25_candidates={len(bm25_ids)} for: '{query[:60]}'"
            )
            fused = _reciprocal_rank_fusion([dense_ids, bm25_ids], k=self.rrf_k)
            final_ids = [cid for cid, _ in fused[:top_k]]
            logger.info(f"[hybrid] RRF fusion → {len(final_ids)} final docs")

        # Build RetrievedDocument list from final_ids
        results: List[RetrievedDocument] = []
        for rank, cid in enumerate(final_ids[:top_k], start=1):
            if cid < 0 or cid >= len(self._lookup):
                continue
            entry = self._lookup[cid]
            text = entry["text"]
            meta = entry["meta"]
            results.append(RetrievedDocument(
                doc_id=str(meta.get("doc_id", cid)),
                title=str(meta.get("title", "")),
                text=text if isinstance(text, str) else str(text),
                score=float(rank),        # rank-based score (1=best)
                rank=rank,
                retriever=self.name,
                query=query,
                metadata={
                    "chunk_id": meta.get("chunk_id"),
                    "dataset": meta.get("dataset"),
                    "char_start": meta.get("char_start"),
                },
            ))

        return results

    # ------------------------------------------------------------------
    # Unused abstract methods — satisfied to avoid ABC errors
    # ------------------------------------------------------------------

    def index(self, documents: list) -> None:  # type: ignore[override]
        raise NotImplementedError(
            "RAGProjectRetriever uses a pre-built index. "
            "Run scripts/build_index.py to rebuild from scratch."
        )

    def save_index(self, path: str) -> None:
        raise NotImplementedError("RAGProjectRetriever does not save indexes.")

    def load_index(self, path: str) -> None:
        raise NotImplementedError(
            "Use __init__(rag_dir=...) to specify the RAG_Project directory."
        )

    @property
    def name(self) -> str:
        return f"rag_project_{self.method}"
