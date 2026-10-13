"""
scripts/experiment_hybrid_v4_retrieval.py — Final Hybrid-v4 Retrieval Experiment.

Architecture: Dense-Anchored Hybrid (Option B variant)

=== Design rationale ===

The core failure mode in all previous Hybrid experiments (RRF 50/50, Weighted RRF,
Max-Rank+Agreement) was the SAME structural problem:

  BM25 candidates freely displace strong Dense candidates from the final Top-10.

When Dense ranks a gold document at positions 6-10 (semantically relevant but not
the absolute top), RRF-style fusion introduces BM25 candidates with superficial
keyword overlap that "fill in" the Top-10 list and push the Dense-only semantic
match beyond rank 10.

This happened on:
  - Q5 (Ottaviano Petrucci): Dense rank ~9, Hybrid pushed it to rank 14.
  - Q6 (woodwork joints): Dense rank ~8, Weighted RRF pushed it to rank 11-12.

Meanwhile, the ONE case where BM25 could help (Q1, Chick-fil-A → Mercedes-Benz
Stadium at BM25 rank 9) was also lost in RRF because BM25 also contributed many
irrelevant football candidates that outscored the gold document after fusion.

=== Hybrid-v4: Dense-Anchored with BM25 Supplement ===

Architecture:
  1. Start with Dense Top-10 as the ANCHOR set (never displaced).
  2. Retrieve BM25 Top-30 candidates.
  3. Find BM25 candidates NOT already in the Dense anchor set.
  4. Score BM25-only candidates: use raw BM25 score, min-max normalized to [0,1].
  5. Select the top BM25-only candidates with normalized_score >= threshold.
  6. APPEND (not interleave) these as supplemental results after the Dense Top-10.
  7. Final result: Dense Top-10 (unchanged ordering) + BM25 supplements → Top-20.
     For Hit@10 evaluation, only the Dense Top-10 matter.
     For Hit@20 evaluation, BM25 supplements can contribute additional recall.

Key properties:
  - Dense Top-10 results are NEVER displaced → no regressions.
  - BM25 can only ADD recall beyond position 10.
  - Hit@10 should be >= Dense baseline (never worse).
  - Hit@20 should be >= Dense baseline (potentially better from BM25 supplements).
  - Latency: Dense + BM25, but BM25 scoring is done on the smaller candidate set.

Additionally, we test a reranking variant:
  - Dense Top-12 + BM25 Top-30.
  - For documents in both sets, apply a small agreement boost to Dense rank score.
  - For the Dense anchor positions 1-10, preserve their order but allow agreed docs
    to float UP (never down).
  - BM25-only docs fill positions 11-20.
  → This can improve Hit@10 if BM25 agreement promotes a Dense-rank-11/12 doc
    into the Top-10.

We test TWO sub-variants:
  A) Dense-Anchored Append: Dense Top-10 frozen + BM25 supplements at 11-20.
  B) Dense-Anchored Promote: Dense Top-12 as candidates, docs agreed by BM25
     get a small rank boost, final Top-10 from reranked Dense + BM25 supplements
     at 11-20.
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

import numpy as np

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


# ── Hit@K metric (identical to existing evaluation logic) ─────────────────────
def compute_hit(docs: List[Dict[str, Any]], gold_answers: List[str], k: int) -> float:
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


def find_gold_rank(docs: List[Dict[str, Any]], gold_answers: List[str]) -> int:
    """Find the 1-based rank of the first document containing a gold answer. Returns 0 if not found."""
    norm_golds = [normalize_answer(g) for g in gold_answers if normalize_answer(g)]
    if not norm_golds:
        return 0
    for rank, d in enumerate(docs, 1):
        text = normalize_answer((d.get("title") or "") + " " + (d.get("text") or ""))
        for g in norm_golds:
            if g in text:
                return rank
    return 0


# ── Standard RRF 50/50 (for comparison) ──────────────────────────────────────
def rrf_50_50(dense_ids: List[int], bm25_ids: List[int], k: int = 60) -> List[int]:
    scores: Dict[int, float] = {}
    for rank_0, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank_0 + 1)
    for rank_0, cid in enumerate(bm25_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank_0 + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# ── Hybrid-v4A: Dense-Anchored Append ─────────────────────────────────────────
def hybrid_v4a_dense_anchored_append(
    dense_ids_10: List[int],
    bm25_ids_30: List[int],
    bm25_scores: Dict[int, float],
) -> List[int]:
    """
    Dense Top-10 as frozen anchor + BM25 supplements in positions 11-20.
    Dense ordering is never modified. BM25 supplements are sorted by raw BM25 score.
    """
    dense_set = set(dense_ids_10)
    # BM25 candidates not already in Dense Top-10, sorted by BM25 score descending
    bm25_supplements = [
        cid for cid in bm25_ids_30
        if cid not in dense_set
    ]
    # Take top-10 BM25 supplements to fill positions 11-20
    return dense_ids_10 + bm25_supplements[:10]


# ── Hybrid-v4B: Dense-Anchored Promote ────────────────────────────────────────
def hybrid_v4b_dense_anchored_promote(
    dense_ids_12: List[int],
    bm25_ids_30: List[int],
    bm25_scores: Dict[int, float],
) -> List[int]:
    """
    Dense Top-12 as candidates. Documents also found by BM25 get a rank promotion
    boost. Final Top-10 selected from the reranked Dense-12 candidates. BM25-only
    supplements fill positions 11-20.

    Scoring:
      dense_rank_score = 1 / (1 + dense_rank_0)   [range: 1.0 for rank 1 → 0.083 for rank 12]
      agreement_boost = 0.04 if in BM25 Top-30 else 0.0
      final_score = dense_rank_score + agreement_boost

    The boost of 0.04 is calibrated so that:
      - A Dense rank-12 doc (score=0.077) WITH BM25 agreement (score=0.117) can
        beat a Dense rank-10 doc (score=0.091) WITHOUT agreement.
      - But a Dense rank-1 doc (score=1.0) is never displaced by a Dense rank-12 doc.
      - This means only marginal Dense positions (11-12) can be promoted into Top-10,
        and only when BM25 independently confirms their relevance.
    """
    bm25_set = set(bm25_ids_30)
    dense_set = set(dense_ids_12)

    # Score and rerank Dense Top-12
    scored_dense = []
    for rank_0, cid in enumerate(dense_ids_12):
        base_score = 1.0 / (1 + rank_0)
        boost = 0.04 if cid in bm25_set else 0.0
        scored_dense.append((cid, base_score + boost))

    # Sort by score descending (stable sort preserves ties in original order)
    scored_dense.sort(key=lambda x: x[1], reverse=True)
    reranked_top10 = [cid for cid, _ in scored_dense[:10]]

    # BM25 supplements not in reranked top-10
    reranked_set = set(reranked_top10)
    bm25_supplements = [cid for cid in bm25_ids_30 if cid not in reranked_set]

    return reranked_top10 + bm25_supplements[:10]


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
    print("FINAL HYBRID-V4 RETRIEVAL EXPERIMENT")
    print("Architecture: Dense-Anchored Hybrid (Append + Promote variants)")
    print("=" * 70)

    config = load_config("configs/default.yaml")
    device = DeviceManager().get_device().type

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"outputs/hybrid_v4_retrieval_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # 1. Load the exact 5 NQ + 5 TriviaQA benchmark queries
    split_map = {
        "nq": config.data.nq_split,
        "triviaqa": getattr(config.data, "triviaqa_split", "train"),
    }
    all_examples = []
    for ds in ["nq", "triviaqa"]:
        loaded = load_dataset(ds, split=split_map[ds], cache_dir=config.data.cache_dir)
        sampled = sample_dataset(loaded, n=20, seed=42)[:5]
        for ex in sampled:
            all_examples.append({
                "dataset": ds,
                "id": ex.id,
                "question": ex.question,
                "answers": ex.answers,
            })

    print(f"Loaded {len(all_examples)} queries (5 NQ, 5 TriviaQA).")

    # 2. Initialize retriever (method=hybrid to load both FAISS and BM25)
    print("\nLoading RAGProjectRetriever artifacts (FAISS + BM25)...")
    retriever = RAGProjectRetriever(
        rag_dir=config.retrieval.rag_project_dir,
        model_name=config.retrieval.embedding_model,
        device=device,
        method="hybrid",
    )
    retriever._load_artifacts()
    print("Retriever artifacts ready.")

    methods = [
        ("Dense Top-10", "dense"),
        ("Current RRF 50/50", "rrf_50_50"),
        ("Hybrid-v4A (Dense-Anchored Append)", "v4a"),
        ("Hybrid-v4B (Dense-Anchored Promote)", "v4b"),
    ]

    method_metrics: Dict[str, Dict[str, Any]] = {
        name: {"hit@1": [], "hit@5": [], "hit@10": [], "hit@20": [], "latencies": []}
        for name, _ in methods
    }

    per_question_rows = []

    print("\nEvaluating retrieval across benchmark queries...\n")

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

        # ── Retrieve Dense Top-12 (need 12 for v4b promote variant) ────────
        t_dense_start = time.time()
        dense_cids_50 = retriever._dense_retrieve_ids(query, n=50)
        dense_lat = time.time() - t_dense_start
        dense_cids_10 = dense_cids_50[:10]
        dense_cids_12 = dense_cids_50[:12]

        # ── Retrieve BM25 Top-30 with raw scores ──────────────────────────
        t_bm25_start = time.time()
        query_tokens = query.lower().split()
        bm25_raw_scores = retriever._bm25.get_scores(query_tokens)
        bm25_top30_indices = np.argsort(bm25_raw_scores)[::-1][:30]
        bm25_cids_30 = [int(i) for i in bm25_top30_indices]
        bm25_scores_map = {int(i): float(bm25_raw_scores[i]) for i in bm25_top30_indices}
        bm25_lat = time.time() - t_bm25_start

        # Also need BM25 Top-50 for RRF comparison
        bm25_top50_indices = np.argsort(bm25_raw_scores)[::-1][:50]
        bm25_cids_50 = [int(i) for i in bm25_top50_indices]

        # ── Evaluate each method ──────────────────────────────────────────
        for m_name, m_key in methods:
            t_fuse = time.time()
            if m_key == "dense":
                top_cids = dense_cids_50[:20]
                m_lat = dense_lat
            elif m_key == "rrf_50_50":
                fused = rrf_50_50(dense_cids_50, bm25_cids_50, k=60)
                fuse_lat = time.time() - t_fuse
                top_cids = fused[:20]
                m_lat = dense_lat + bm25_lat + fuse_lat
            elif m_key == "v4a":
                top_cids = hybrid_v4a_dense_anchored_append(
                    dense_cids_10, bm25_cids_30, bm25_scores_map,
                )
                fuse_lat = time.time() - t_fuse
                m_lat = dense_lat + bm25_lat + fuse_lat
            elif m_key == "v4b":
                top_cids = hybrid_v4b_dense_anchored_promote(
                    dense_cids_12, bm25_cids_30, bm25_scores_map,
                )
                fuse_lat = time.time() - t_fuse
                m_lat = dense_lat + bm25_lat + fuse_lat
            else:
                top_cids = []
                m_lat = 0.0

            docs = build_doc_dicts(retriever, top_cids, query)

            h1 = compute_hit(docs, golds, 1)
            h5 = compute_hit(docs, golds, 5)
            h10 = compute_hit(docs, golds, 10)
            h20 = compute_hit(docs, golds, 20)
            gold_rank = find_gold_rank(docs, golds)

            method_metrics[m_name]["hit@1"].append(h1)
            method_metrics[m_name]["hit@5"].append(h5)
            method_metrics[m_name]["hit@10"].append(h10)
            method_metrics[m_name]["hit@20"].append(h20)
            method_metrics[m_name]["latencies"].append(m_lat)

            prefix = m_name.replace(" ", "_").replace("/", "_").replace("-", "_").replace("(", "").replace(")", "").replace("+", "")
            q_row[f"{prefix}_hit@1"] = int(h1)
            q_row[f"{prefix}_hit@5"] = int(h5)
            q_row[f"{prefix}_hit@10"] = int(h10)
            q_row[f"{prefix}_hit@20"] = int(h20)
            q_row[f"{prefix}_gold_rank"] = gold_rank
            q_row[f"{prefix}_lat_ms"] = round(m_lat * 1000, 1)
            top_titles = [d.get("title") or "Untitled" for d in docs[:5]]
            q_row[f"{prefix}_top5_titles"] = " | ".join(top_titles)

        per_question_rows.append(q_row)
        # Compact per-question report
        dense_gr = q_row.get("Dense_Top_10_gold_rank", 0)
        rrf_gr = q_row.get("Current_RRF_50_50_gold_rank", 0)
        v4a_gr = q_row.get("Hybrid_v4A_Dense_Anchored_Append_gold_rank", 0)
        v4b_gr = q_row.get("Hybrid_v4B_Dense_Anchored_Promote_gold_rank", 0)
        print(f"  Q{q_idx:02d} [{ds.upper():<8}] Gold ranks: Dense={dense_gr:>3}  RRF={rrf_gr:>3}  v4A={v4a_gr:>3}  v4B={v4b_gr:>3}  | {query[:55]}")

    # ── Summary Table ──────────────────────────────────────────────────────
    summary_data = []
    print("\n" + "=" * 80)
    print("HYBRID-V4 EXPERIMENT SUMMARY TABLE (N=10 queries)")
    print("=" * 80)
    print(f"{'Method':<42} | {'Hit@1':>6} | {'Hit@5':>6} | {'Hit@10':>6} | {'Hit@20':>6} | {'Latency':>10}")
    print("-" * 80)

    for m_name, _ in methods:
        m_dict = method_metrics[m_name]
        n = len(m_dict["hit@1"])
        mean_h1 = sum(m_dict["hit@1"]) / n
        mean_h5 = sum(m_dict["hit@5"]) / n
        mean_h10 = sum(m_dict["hit@10"]) / n
        mean_h20 = sum(m_dict["hit@20"]) / n
        avg_lat = sum(m_dict["latencies"]) / n

        summary_data.append({
            "Method": m_name,
            "Hit@1": mean_h1,
            "Hit@5": mean_h5,
            "Hit@10": mean_h10,
            "Hit@20": mean_h20,
            "Avg_Latency_ms": avg_lat * 1000,
        })
        print(
            f"{m_name:<42} | {mean_h1*100:5.1f}% | {mean_h5*100:5.1f}% | {mean_h10*100:5.1f}% | {mean_h20*100:5.1f}% | {avg_lat*1000:8.1f} ms"
        )

    # ── Displacement analysis ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DISPLACEMENT / RECOVERY ANALYSIS")
    print("=" * 80)
    for row in per_question_rows:
        d_gr = row.get("Dense_Top_10_gold_rank", 0)
        v4a_gr = row.get("Hybrid_v4A_Dense_Anchored_Append_gold_rank", 0)
        v4b_gr = row.get("Hybrid_v4B_Dense_Anchored_Promote_gold_rank", 0)

        d_h10 = row.get("Dense_Top_10_hit@10", 0)
        v4a_h10 = row.get("Hybrid_v4A_Dense_Anchored_Append_hit@10", 0)
        v4b_h10 = row.get("Hybrid_v4B_Dense_Anchored_Promote_hit@10", 0)

        status_a = ""
        if d_h10 == 0 and v4a_h10 == 1:
            status_a = "* RECOVERED by v4A"
        elif d_h10 == 1 and v4a_h10 == 0:
            status_a = "x DISPLACED by v4A"
        elif d_h10 == 1 and v4a_h10 == 1:
            status_a = "= preserved"
        else:
            status_a = "= missed by both"

        status_b = ""
        if d_h10 == 0 and v4b_h10 == 1:
            status_b = "* RECOVERED by v4B"
        elif d_h10 == 1 and v4b_h10 == 0:
            status_b = "x DISPLACED by v4B"
        elif d_h10 == 1 and v4b_h10 == 1:
            status_b = "= preserved"
        else:
            status_b = "= missed by both"

        print(f"  Q{row['q_idx']:02d}: Dense_rank={d_gr:>3}  v4A_rank={v4a_gr:>3} ({status_a})  v4B_rank={v4b_gr:>3} ({status_b})")

    # ── Save artifacts ────────────────────────────────────────────────────
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

    md_path = out_dir / "experiment_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Final Hybrid-v4 Retrieval Experiment Report\n\n")
        f.write(f"- **Timestamp**: {timestamp}\n")
        f.write(f"- **Queries Evaluated**: 10 (5 NQ + 5 TriviaQA)\n")
        f.write(f"- **Architecture**: Dense-Anchored Hybrid (two sub-variants)\n\n")
        f.write("## Design Rationale\n\n")
        f.write("All previous Hybrid experiments (RRF 50/50, Weighted RRF, Max-Rank+Agreement) "
                "shared the same structural failure: BM25 candidates freely displaced strong "
                "Dense candidates from the final Top-10.\n\n")
        f.write("**Hybrid-v4A (Dense-Anchored Append)**: Dense Top-10 is frozen as the anchor. "
                "BM25-only candidates fill positions 11-20 as supplements.\n\n")
        f.write("**Hybrid-v4B (Dense-Anchored Promote)**: Dense Top-12 candidates are reranked "
                "with a small agreement boost (+0.04) for docs also found by BM25, allowing "
                "Dense rank 11-12 docs to be promoted into the Top-10. BM25-only candidates "
                "fill remaining positions.\n\n")
        f.write("## Comparison Table\n\n")
        f.write("| Method | Hit@1 | Hit@5 | Hit@10 | Hit@20 | Avg Latency |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for row in summary_data:
            f.write(
                f"| **{row['Method']}** | {row['Hit@1']*100:.1f}% | {row['Hit@5']*100:.1f}% "
                f"| {row['Hit@10']*100:.1f}% | {row['Hit@20']*100:.1f}% | {row['Avg_Latency_ms']:.1f} ms |\n"
            )
        f.write("\n## Per-Question Gold Rank Comparison\n\n")
        f.write("| # | Dataset | Question | Dense Gold Rank | RRF Gold Rank | v4A Gold Rank | v4B Gold Rank |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for row in per_question_rows:
            f.write(
                f"| Q{row['q_idx']} | {row['dataset'].upper()} | {row['question'][:50]}... | "
                f"{row.get('Dense_Top_10_gold_rank', 0)} | "
                f"{row.get('Current_RRF_50_50_gold_rank', 0)} | "
                f"{row.get('Hybrid_v4A_Dense_Anchored_Append_gold_rank', 0)} | "
                f"{row.get('Hybrid_v4B_Dense_Anchored_Promote_gold_rank', 0)} |\n"
            )

    print(f"\nSaved artifacts to {out_dir}:")
    for p in sorted(out_dir.iterdir()):
        print(f"  - {p}")


if __name__ == "__main__":
    run_experiment()
