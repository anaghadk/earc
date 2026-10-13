"""
scripts/experiment_hybrid_v2_retrieval.py — Controlled Hybrid-v2 Retrieval Experiment.

Evaluates 5 retrieval configurations on 10 benchmark queries (5 NQ + 5 TriviaQA):
  1. Dense baseline (Top-K=20)
  2. BM25 baseline (Top-K=20)
  3. Current Hybrid (Standard RRF 50/50)
  4. Weighted Hybrid 70/30
  5. Weighted Hybrid 80/20

Measures Hit@1, Hit@5, Hit@10, Hit@20 and average latency.
Generates a detailed per-question comparison and markdown/JSON summary.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(str(_PROJECT_ROOT))

from src.config.settings import load_config
from src.data.loaders import load_dataset
from src.data.sampling import sample_dataset
from src.evaluation.exact_match import normalize_answer
from src.retrieval.rag_project import RAGProjectRetriever
from src.utils.device import DeviceManager


def compute_hit(docs: List[Dict[str, Any]], gold_answers: List[str], k: int) -> float:
    """Check if any normalized gold answer appears as a substring in top-k docs."""
    sub_docs = docs[:k]
    norm_golds = [normalize_answer(g) for g in gold_answers if normalize_answer(g)]
    if not norm_golds or not sub_docs:
        return 0.0
    for d in sub_docs:
        text = normalize_answer((d.get("title") or "") + " " + (d.get("text") or ""))
        for g in norm_golds:
            if g in text:
                return 1.0
    return 0.0


def rrf_fuse(
    dense_ids: List[int],
    bm25_ids: List[int],
    w_dense: float,
    w_bm25: float,
    k: int = 60,
) -> List[int]:
    """Reciprocal Rank Fusion with weights for dense and bm25."""
    scores: Dict[int, float] = {}
    for rank_0, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0.0) + w_dense / (k + rank_0 + 1)
    for rank_0, cid in enumerate(bm25_ids):
        scores[cid] = scores.get(cid, 0.0) + w_bm25 / (k + rank_0 + 1)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in sorted_items]


def build_doc_dicts(retriever: RAGProjectRetriever, cids: List[int], query: str) -> List[Dict[str, Any]]:
    docs = []
    for rank, cid in enumerate(cids, start=1):
        if 0 <= cid < len(retriever._lookup):
            entry = retriever._lookup[cid]
            docs.append({
                "rank": rank,
                "corpus_id": cid,
                "title": str(entry["meta"].get("title", "")),
                "text": str(entry.get("text", "")),
                "query": query,
            })
    return docs


def run_experiment():
    print("=" * 70)
    print("CONTROLLED HYBRID-V2 RETRIEVAL EXPERIMENT")
    print("=" * 70)

    config = load_config("configs/default.yaml")
    device = DeviceManager().get_device().type

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"outputs/hybrid_v2_retrieval_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # 1. Load and sample exact 5 NQ + 5 TriviaQA examples
    split_map = {
        "nq": config.data.nq_split,
        "triviaqa": getattr(config.data, "triviaqa_split", "train"),
    }
    datasets = ["nq", "triviaqa"]
    all_examples = []
    for ds in datasets:
        loaded = load_dataset(ds, split=split_map[ds], cache_dir=config.data.cache_dir)
        sampled = sample_dataset(loaded, n=20, seed=42)[:5]
        for ex in sampled:
            all_examples.append({
                "dataset": ds,
                "id": ex.id,
                "question": ex.question,
                "answers": ex.answers,
            })

    print(f"Loaded total {len(all_examples)} queries (5 NQ, 5 TriviaQA).")
    for idx, ex in enumerate(all_examples, 1):
        print(f"  [{idx:02d}/10] [{ex['dataset'].upper()}] {ex['question']}")

    # 2. Initialize retriever
    print("\nInitializing RAGProjectRetriever...")
    retriever = RAGProjectRetriever(
        rag_dir=config.retrieval.rag_project_dir,
        model_name=config.retrieval.embedding_model,
        device=device,
        method="hybrid",
    )
    retriever._load_artifacts()
    print("Retriever artifacts loaded successfully.")

    methods = [
        ("Dense", "dense"),
        ("BM25", "bm25"),
        ("RRF 50/50", "hybrid_50_50"),
        ("Weighted RRF 70/30", "hybrid_70_30"),
        ("Weighted RRF 80/20", "hybrid_80_20"),
    ]

    method_metrics: Dict[str, Dict[str, Any]] = {
        name: {
            "hit@1": [],
            "hit@5": [],
            "hit@10": [],
            "hit@20": [],
            "latencies": [],
        }
        for name, _ in methods
    }

    per_question_rows = []

    print("\nRunning retrieval evaluation across all 5 methods...")

    for q_idx, ex in enumerate(all_examples, 1):
        query = ex["question"]
        golds = ex["answers"]
        ds = ex["dataset"]

        q_row = {
            "q_idx": q_idx,
            "dataset": ds,
            "example_id": ex["id"],
            "question": query,
            "gold_answers": json.dumps(golds),
        }

        # 1. Dense candidates (Top-50)
        t_dense_start = time.time()
        dense_cids_50 = retriever._dense_retrieve_ids(query, n=50)
        dense_lat = time.time() - t_dense_start

        # 2. BM25 candidates (Top-50)
        t_bm25_start = time.time()
        bm25_cids_50 = retriever._bm25_retrieve_ids(query, n=50)
        bm25_lat = time.time() - t_bm25_start

        # Evaluate each method
        for m_name, m_key in methods:
            t_fuse_start = time.time()
            if m_key == "dense":
                top_cids = dense_cids_50[:20]
                m_lat = dense_lat
            elif m_key == "bm25":
                top_cids = bm25_cids_50[:20]
                m_lat = bm25_lat
            elif m_key == "hybrid_50_50":
                fused_cids = rrf_fuse(dense_cids_50, bm25_cids_50, w_dense=1.0, w_bm25=1.0, k=60)
                fuse_lat = time.time() - t_fuse_start
                top_cids = fused_cids[:20]
                m_lat = dense_lat + bm25_lat + fuse_lat
            elif m_key == "hybrid_70_30":
                fused_cids = rrf_fuse(dense_cids_50, bm25_cids_50, w_dense=0.7, w_bm25=0.3, k=60)
                fuse_lat = time.time() - t_fuse_start
                top_cids = fused_cids[:20]
                m_lat = dense_lat + bm25_lat + fuse_lat
            elif m_key == "hybrid_80_20":
                fused_cids = rrf_fuse(dense_cids_50, bm25_cids_50, w_dense=0.8, w_bm25=0.2, k=60)
                fuse_lat = time.time() - t_fuse_start
                top_cids = fused_cids[:20]
                m_lat = dense_lat + bm25_lat + fuse_lat
            else:
                top_cids = []
                m_lat = 0.0

            docs = build_doc_dicts(retriever, top_cids, query)

            h1 = compute_hit(docs, golds, 1)
            h5 = compute_hit(docs, golds, 5)
            h10 = compute_hit(docs, golds, 10)
            h20 = compute_hit(docs, golds, 20)

            method_metrics[m_name]["hit@1"].append(h1)
            method_metrics[m_name]["hit@5"].append(h5)
            method_metrics[m_name]["hit@10"].append(h10)
            method_metrics[m_name]["hit@20"].append(h20)
            method_metrics[m_name]["latencies"].append(m_lat)

            prefix = m_name.replace(" ", "_").replace("/", "_")
            q_row[f"{prefix}_hit@1"] = int(h1)
            q_row[f"{prefix}_hit@5"] = int(h5)
            q_row[f"{prefix}_hit@10"] = int(h10)
            q_row[f"{prefix}_hit@20"] = int(h20)
            q_row[f"{prefix}_lat_ms"] = round(m_lat * 1000, 1)
            top_titles = [d.get("title") or "Untitled" for d in docs[:5]]
            q_row[f"{prefix}_top5_titles"] = " | ".join(top_titles)

        per_question_rows.append(q_row)
        print(f"  Completed Q{q_idx:02d}: {query[:50]}...")

    # 3. Aggregate Summary Table
    summary_data = []
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS COMPARISON TABLE (N=10 queries)")
    print("=" * 70)
    print(f"{'Method':<22} | {'Hit@1':<7} | {'Hit@5':<7} | {'Hit@10':<7} | {'Hit@20':<7} | {'Avg Latency':<12}")
    print("-" * 70)

    for m_name, _ in methods:
        m_dict = method_metrics[m_name]
        mean_h1 = sum(m_dict["hit@1"]) / len(m_dict["hit@1"])
        mean_h5 = sum(m_dict["hit@5"]) / len(m_dict["hit@5"])
        mean_h10 = sum(m_dict["hit@10"]) / len(m_dict["hit@10"])
        mean_h20 = sum(m_dict["hit@20"]) / len(m_dict["hit@20"])
        avg_lat = sum(m_dict["latencies"]) / len(m_dict["latencies"])

        summary_data.append({
            "Method": m_name,
            "Hit@1": mean_h1,
            "Hit@5": mean_h5,
            "Hit@10": mean_h10,
            "Hit@20": mean_h20,
            "Avg_Latency_ms": avg_lat * 1000,
        })

        print(
            f"{m_name:<22} | {mean_h1*100:6.1f}% | {mean_h5*100:6.1f}% | {mean_h10*100:6.1f}% | {mean_h20*100:6.1f}% | {avg_lat*1000:9.1f} ms"
        )

    # 4. Save JSON and Markdown artifacts
    with open(out_dir / "summary_table.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(out_dir / "per_question_results.json", "w", encoding="utf-8") as f:
        json.dump(per_question_rows, f, indent=2)

    if per_question_rows:
        keys = list(per_question_rows[0].keys())
        with open(out_dir / "per_question_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_question_rows)

    md_path = out_dir / "summary_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Controlled Hybrid-v2 Retrieval Experiment Report\n\n")
        f.write(f"- **Timestamp**: {timestamp}\n")
        f.write(f"- **Queries Evaluated**: 10 (5 NQ + 5 TriviaQA)\n")
        f.write(f"- **FAISS Index**: RAG_Project (384-dim, 957,220 vectors)\n")
        f.write(f"- **BM25 Index**: RAG_Project BM25Okapi\n\n")
        f.write("## Comparison Table\n\n")
        f.write("| Method | Hit@1 | Hit@5 | Hit@10 | Hit@20 | Avg Latency |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for row in summary_data:
            f.write(
                f"| **{row['Method']}** | {row['Hit@1']*100:.1f}% | {row['Hit@5']*100:.1f}% | {row['Hit@10']*100:.1f}% | {row['Hit@20']*100:.1f}% | {row['Avg_Latency_ms']:.1f} ms |\n"
            )

        f.write("\n## Per-Question Breakdown\n\n")
        f.write("| # | Dataset | Question | Dense Hit@10 | BM25 Hit@10 | RRF 50/50 Hit@10 | Weighted 70/30 Hit@10 | Weighted 80/20 Hit@10 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for row in per_question_rows:
            f.write(
                f"| Q{row['q_idx']} | {row['dataset'].upper()} | {row['question']} | "
                f"{row['Dense_hit@10']} | {row['BM25_hit@10']} | {row['RRF_50_50_hit@10']} | "
                f"{row['Weighted_RRF_70_30_hit@10']} | {row['Weighted_RRF_80_20_hit@10']} |\n"
            )

    print(f"\nSaved artifacts to {out_dir}:")
    print(f"  - {out_dir / 'summary_table.json'}")
    print(f"  - {out_dir / 'summary_table.md'}")
    print(f"  - {out_dir / 'per_question_results.json'}")
    print(f"  - {out_dir / 'per_question_results.csv'}")


if __name__ == "__main__":
    run_experiment()
