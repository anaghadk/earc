import argparse
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import load_config
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.utils.logging import setup_logging
from src.utils.device import DeviceManager
from src.utils.io import ensure_dir, read_jsonl

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Character-level chunking, matching the old pipeline's scheme."""
    if not text:
        return []
    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def load_corpus(data_dir: str, datasets: list[str], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """
    Build ONE shared retrieval corpus across all datasets (matching the old
    data_builder.py design), rather than per-question isolated candidate
    sets. This is required for NQ, which has no per-question documents --
    its content only exists as a shared passage pool (nq_passages.jsonl).

    Sources combined here:
      - data/processed/nq_passages.jsonl        (flat NQ passage pool)
      - documents embedded in hotpotqa.jsonl     (per-question candidate paras)
      - documents embedded in triviaqa.jsonl     (per-question entity/search pages)

    Every document is then split into chunk_size/chunk_overlap character
    chunks (default 800/100) before indexing, so BM25/FAISS operate over
    passages, not whole (sometimes very long) documents.
    """
    seen_docs = {}

    # Flat NQ passage pool
    nq_passages_path = f"{data_dir.rstrip('/')}/nq_passages.jsonl"
    try:
        for doc in read_jsonl(nq_passages_path):
            doc_id = doc.get("doc_id")
            if doc_id and doc_id not in seen_docs:
                seen_docs[doc_id] = doc
    except FileNotFoundError:
        logger.warning(f"No NQ passage pool at {nq_passages_path} -- NQ will contribute 0 documents.")

    # Per-question documents from hotpotqa / triviaqa
    for ds_name in datasets:
        if ds_name == 'nq':
            continue  # handled above via the flat passage pool
        path = f"{data_dir.rstrip('/')}/{ds_name}.jsonl"
        try:
            examples = read_jsonl(path)
        except FileNotFoundError:
            logger.warning(f"No processed file for {ds_name} at {path}, skipping.")
            continue
        for ex in examples:
            for doc in ex.get("documents", []):
                doc_id = doc.get("doc_id")
                if doc_id and doc_id not in seen_docs:
                    seen_docs[doc_id] = doc

    logger.info(f"Loaded {len(seen_docs)} unique documents before chunking.")

    corpus = []
    for doc in seen_docs.values():
        text_chunks = chunk_text(doc.get("text", ""), chunk_size, chunk_overlap)
        for i, chunk in enumerate(text_chunks):
            corpus.append({
                "doc_id": f"{doc['doc_id']}::chunk_{i}",
                "title": doc.get("title", ""),
                "text": chunk,
                "metadata": {"source_doc_id": doc["doc_id"]},
            })

    logger.info(f"Built shared corpus of {len(corpus)} chunks from {len(seen_docs)} documents.")
    return corpus


def main():
    parser = argparse.ArgumentParser(description="Build index for EARC")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--index-dir", type=str, default="data/indexes", help="Index directory")
    parser.add_argument("--retriever", choices=["bm25", "dense", "both"], default="both", help="Retriever to build")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap in characters")
    args = parser.parse_args()

    setup_logging("INFO")
    config = load_config(args.config)

    if args.device != "auto":
        import torch
        device = torch.device(args.device)
    else:
        device = DeviceManager(fp16=config.hardware.fp16).get_device()

    ensure_dir(args.index_dir)

    logger.info(f"Building index with device={device}")

    try:
        corpus = load_corpus(args.data_dir, config.data.datasets, args.chunk_size, args.chunk_overlap)
        if not corpus:
            raise ValueError(
                f"No documents found under {args.data_dir}. Run scripts/prepare_data.py first."
            )

        if args.retriever in ["bm25", "both"]:
            logger.info("Building BM25 index...")
            bm25 = BM25Retriever()
            bm25.index(corpus)
            bm25.save_index(f"{args.index_dir.rstrip('/')}/bm25.pkl")

        if args.retriever in ["dense", "both"]:
            logger.info("Building Dense index...")
            dense = DenseRetriever(
                model_name=config.retrieval.embedding_model,
                device=str(device),
                batch_size=config.hardware.embedding_batch_size,
            )
            dense.index(corpus)
            dense.save_index(f"{args.index_dir.rstrip('/')}/dense")

        logger.info("Index building completed.")
    except Exception as e:
        logger.error(f"Error building index: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
