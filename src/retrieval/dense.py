import logging
import pickle
from typing import Any

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from src.data.schemas import RetrievedDocument
from src.retrieval.base import Retriever
from src.retrieval import faiss_index

logger = logging.getLogger(__name__)


class DenseRetriever(Retriever):
    """
    Dense retriever using SentenceTransformers and FAISS.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        logger.info(f"Loading SentenceTransformer model {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        self.index_faiss = None
        self.corpus_metadata: dict[int, dict[str, Any]] = {}

    def _encode(self, texts: list[str], show_progress: bool = False) -> torch.Tensor:
        """Encode texts into normalized embeddings using the model."""
        all_embeddings = []
        
        # Determine if we can use AMP
        use_amp = self.device.type == "cuda"
        
        # Batching
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        
        if show_progress:
            batches = tqdm(batches, desc="Encoding")
            
        with torch.no_grad():
            for batch in batches:
                # We can use AMP for inference if on CUDA
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        # encode returns numpy array by default, but we can tell it to return tensors
                        emb = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
                else:
                    emb = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
                
                # Normalize
                emb = F.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb.cpu())
                
        if not all_embeddings:
            return torch.empty((0, self.model.get_sentence_embedding_dimension()))
            
        return torch.cat(all_embeddings, dim=0)

    def index(self, documents: list[dict]) -> None:
        if not documents:
            logger.warning("Empty document list provided to Dense index.")
            self.index_faiss = None
            self.corpus_metadata = {}
            return

        texts = [doc.get("text", "") for doc in documents]
        
        logger.info(f"Encoding {len(texts)} documents...")
        embeddings = self._encode(texts, show_progress=True).numpy()
        
        logger.info("Building FAISS index...")
        use_gpu = self.device.type == "cuda"
        self.index_faiss = faiss_index.build_index(embeddings, use_gpu=use_gpu)
        
        self.corpus_metadata = {i: doc for i, doc in enumerate(documents)}
        logger.info(f"Indexed {len(documents)} documents using Dense retriever.")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        if not self.index_faiss or not self.corpus_metadata:
            logger.warning("Dense index is empty.")
            return []
            
        if not query:
            logger.warning("Empty query provided to Dense retrieve.")
            return []
            
        if top_k <= 0:
            return []

        query_emb = self._encode([query]).numpy()
        
        distances, indices = faiss_index.search(self.index_faiss, query_emb, top_k)
        
        if len(distances) == 0 or len(indices) == 0:
            return []

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx == -1:  # FAISS returns -1 if not enough results
                continue
                
            doc = self.corpus_metadata[idx]
            results.append(RetrievedDocument(
                doc_id=doc.get("doc_id", str(idx)),
                title=doc.get("title", ""),
                text=doc.get("text", ""),
                score=float(dist),
                rank=rank,
                retriever=self.name,
                query=query,
                metadata=doc.get("metadata", {})
            ))

        logger.info(f"Dense retrieved {len(results)} docs for query: '{query}'")
        return results

    def save_index(self, path: str) -> None:
        if not self.index_faiss:
            logger.warning("No index to save.")
            return
            
        faiss_index.save_index(self.index_faiss, path + ".faiss")
        with open(path + ".meta", "wb") as f:
            pickle.dump(self.corpus_metadata, f)
        logger.info(f"Saved Dense index to {path}.faiss and metadata to {path}.meta")

    def load_index(self, path: str) -> None:
        self.index_faiss = faiss_index.load_index(path + ".faiss")
        with open(path + ".meta", "rb") as f:
            self.corpus_metadata = pickle.load(f)
        logger.info(f"Loaded Dense index and metadata from {path}")

    @property
    def name(self) -> str:
        return "dense"
