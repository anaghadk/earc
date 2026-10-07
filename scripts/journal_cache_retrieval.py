"""
Retrieval Caching Script for Journal Paper Additional Experiments.
Pre-computes and caches dense retrieval top-10 passages for all 3,000 locked evaluation questions
(1,000 NQ, 1,000 HotpotQA, 1,000 TriviaQA) from outputs/final_evaluation/selected_example_ids.json.
Guarantees 100% identical retrieval across all experiments (Exp 1, Exp 2, Exp 3, Exp 4).
"""

import sys
import os
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.config.settings import load_config
from src.data.loaders import load_dataset
from src.data.schemas import QAExample, RetrievedDocument
from src.retrieval.rag_project import RAGProjectRetriever
from src.utils.device import DeviceManager
from src.utils.logging import get_logger

logger = get_logger("journal_retrieval_cache")

def main():
    config = load_config("configs/default.yaml")
    device = DeviceManager().get_device().type
    rag_dir = config.retrieval.rag_project_dir or "RAG_Project"

    out_dir = Path("outputs/journal_experiments/retrieval_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_pkl = out_dir / "retrieved_passages.pkl"
    cache_json = out_dir / "retrieval_metadata.json"

    # Load locked IDs
    ids_file = Path("outputs/final_evaluation/selected_example_ids.json")
    with open(ids_file, "r", encoding="utf-8") as f:
        locked_ids = json.load(f)

    logger.info("Initializing Dense Retriever on %s...", device)
    t0 = time.time()
    retriever = RAGProjectRetriever(
        rag_dir=rag_dir,
        model_name=config.retrieval.embedding_model,
        device=device,
        method="dense",
    )
    logger.info("Retriever initialized in %.2fs", time.time() - t0)

    dataset_map = {
        "nq": "nq",
        "hotpotqa": "hotpotqa",
        "triviaqa": "triviaqa",
    }
    key_alias = {
        "nq": "nq",
        "hotpotqa": "hotpot",
        "triviaqa": "trivia",
    }

    all_cached: Dict[str, Dict[str, Any]] = {}
    if cache_pkl.exists():
        logger.info("Found existing cache at %s. Loading...", cache_pkl)
        with open(cache_pkl, "rb") as f:
            all_cached = pickle.load(f)

    total_queries = 0
    start_time = time.time()

    for ds_name, load_name in dataset_map.items():
        ds_alias = key_alias[ds_name]
        chosen_ids = locked_ids[ds_alias]
        full_examples = load_dataset(load_name)
        id_to_ex = {ex.id: ex for ex in full_examples}
        examples = [id_to_ex[eid] for eid in chosen_ids if eid in id_to_ex]

        logger.info("Caching retrieval for %s (N=%d)...", ds_name.upper(), len(examples))
        if ds_name not in all_cached:
            all_cached[ds_name] = {}

        for idx, ex in enumerate(examples, 1):
            if ex.id in all_cached[ds_name]:
                continue

            t_q0 = time.time()
            docs = retriever.retrieve(ex.question, top_k=10)
            q_lat = time.time() - t_q0

            # Store serialized dicts and docs
            all_cached[ds_name][ex.id] = {
                "question_id": ex.id,
                "dataset": ds_name,
                "question": ex.question,
                "gold_answers": ex.answers,
                "retrieval_latency_s": round(q_lat, 4),
                "docs": [
                    {
                        "doc_id": d.doc_id,
                        "title": d.title,
                        "text": d.text,
                        "rank": d.rank,
                        "score": d.score,
                    }
                    for d in docs
                ],
            }
            total_queries += 1
            if idx % 100 == 0 or idx == len(examples):
                logger.info("[%s] Cached %d/%d queries...", ds_name.upper(), idx, len(examples))

    logger.info("Saving full cache to %s...", cache_pkl)
    with open(cache_pkl, "wb") as f:
        pickle.dump(all_cached, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save summary metadata
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": config.retrieval.embedding_model,
        "retrieval_method": "dense",
        "top_k": 10,
        "dataset_counts": {k: len(v) for k, v in all_cached.items()},
        "total_examples": sum(len(v) for v in all_cached.items()),
    }
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Done! Caching completed in %.2fs. Total queries retrieved: %d", time.time() - start_time, total_queries)

if __name__ == "__main__":
    main()
