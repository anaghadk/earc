"""
Experiment 2: Budget Sweep and Budget-Matched Top-k (Journal Protocol - 2 October 2026).

Runs:
1. EARC Budget Sweep:
   - Grid budgets: 150, 300, 450, 600, 750, 900 tokens
   - Matched budgets:
     - NQ: 727 tokens (matches average Top-k context in main results)
     - HotpotQA: 598 tokens
     - TriviaQA: 843 tokens
   All with default alpha=0.7, beta=0.3, theta=0.85.

2. Top-k Passages Sweep:
   - k = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (top-k passages without sentence segmentation)
   - k=5 loaded directly from verified main results or re-verified.

Execution details:
- Shared dense retrieval cache (retrieved_passages.pkl)
- Compress once, generate twice (Ollama Llama 3.2 and Mistral API)
- Log format matches Protocol Section 1.2
"""

import os
import sys
import time
import json
import yaml
import pickle
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = "lIK71q2MhMrZXQwxYE0nVUvoNNUVjQzy"

import torch
import numpy as np

from src.config.settings import load_config
from src.data.schemas import RetrievedDocument, SentenceStatus
from src.compression.embeddings import EmbeddingEngine
from src.compression.compressor import EvidenceAwareCompressor
from src.prompting.builder import PromptBuilder
from src.prompting.templates import DEFAULT_QA_TEMPLATE
from src.llm.factory import create_provider
from src.evaluation.exact_match import exact_match_score
from src.evaluation.f1 import max_token_f1_score
from src.evaluation.answer_extraction import extract_answer
from src.utils.device import DeviceManager
from src.utils.logging import get_logger

logger = get_logger("journal_exp2_budget")

DATASET_MAP = {
    "nq": "nq",
    "hotpotqa": "hotpot",
    "triviaqa": "trivia",
}

GENERATOR_MAP = {
    "llama3.2:3b": "ollama",
    "ministral-8b": "mistral",
}

MATCHED_BUDGETS = {
    "nq": 727,
    "hotpotqa": 598,
    "triviaqa": 843,
}

GRID_BUDGETS = [150, 300, 450, 600, 750, 900]

def update_summary_csv(summary_csv: Path, row: Dict[str, Any]):
    existing_rows = []
    if summary_csv.exists():
        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    matched = False
    for i, r in enumerate(existing_rows):
        if r["config_id"] == str(row["config_id"]) and r["dataset"] == row["dataset"] and r["generator"] == row["generator"]:
            existing_rows[i] = {k: str(v) for k, v in row.items()}
            matched = True
            break
    if not matched:
        existing_rows.append({k: str(v) for k, v in row.items()})

    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(existing_rows)

def write_config_yaml(base_config, cfg_id, budget, k, out_path: Path):
    out_dict = {
        "data": {
            "seed": 42,
            "datasets": ["nq", "hotpotqa", "triviaqa"],
        },
        "retrieval": {
            "backend": "dense",
            "top_k": 10,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "compression": {
            "alpha": 0.7 if budget is not None else None,
            "beta": 0.3 if budget is not None else None,
            "use_redundancy": True if budget is not None else False,
            "redundancy_threshold": 0.85 if budget is not None else 1.0,
            "token_budget": budget,
            "spacy_model": "en_core_web_sm",
        },
        "experiment": {
            "experiment": "exp2",
            "config_id": cfg_id,
            "budget": budget,
            "k": k,
        }
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(out_dict, f, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description="Journal Exp 2 Budget Runner")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "earc", "topk"], help="Run 'earc', 'topk', or 'all'")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset (nq, hotpotqa, triviaqa) or 'all'")
    parser.add_argument("--generator", type=str, default="all", help="Generator (llama3.2:3b, ministral-8b) or 'all'")
    parser.add_argument("--budget", type=int, default=None, help="Specific budget to run for EARC")
    parser.add_argument("--k", type=int, default=None, help="Specific k to run for Top-k")
    parser.add_argument("--compress-only", action="store_true", help="Only run compression/context building")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries for test")
    args = parser.parse_args()

    base_config = load_config("configs/default.yaml")
    device = DeviceManager().get_device().type

    # Directories
    exp2_dir = Path("outputs/journal_experiments/exp2_budget")
    configs_dir = exp2_dir / "configs"
    contexts_dir = exp2_dir / "contexts"
    preds_dir = exp2_dir / "predictions"
    summary_csv = exp2_dir / "exp2_summary.csv"
    configs_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Load shared retrieval cache
    cache_pkl = Path("outputs/journal_experiments/retrieval_cache/retrieved_passages.pkl")
    if not cache_pkl.exists():
        raise FileNotFoundError(f"Missing retrieval cache {cache_pkl}. Run scripts/journal_cache_retrieval.py first.")
    
    logger.info("Loading cached retrieval from %s...", cache_pkl)
    with open(cache_pkl, "rb") as f:
        retrieval_cache = pickle.load(f)

    target_datasets = [args.dataset] if args.dataset != "all" else list(DATASET_MAP.keys())
    target_generators = [args.generator] if args.generator != "all" else list(GENERATOR_MAP.keys())

    # Initialize shared embedding engine & token counter
    logger.info("Initializing Embedding Engine on %s...", device)
    embedding_engine = EmbeddingEngine(base_config.retrieval.embedding_model, device=device)
    
    llms = {}
    if not args.compress_only:
        logger.info("Initializing LLM providers for generators: %s...", target_generators)
        for gen_id in target_generators:
            prov_name = GENERATOR_MAP[gen_id]
            llms[gen_id] = create_provider(base_config, prov_name)

    from src.llm.ollama_client import OllamaTokenCounter
    token_counter = OllamaTokenCounter("llama3.2")
    prompt_builder = PromptBuilder(DEFAULT_QA_TEMPLATE)

    # =========================================================================
    # PART 1: EARC Budget Sweep (Grid + Matched)
    # =========================================================================
    if args.mode in ["all", "earc"]:
        for ds_name in target_datasets:
            ds_alias = DATASET_MAP[ds_name]
            ds_queries = retrieval_cache[ds_name]
            if args.limit:
                q_keys = list(ds_queries.keys())[:args.limit]
                ds_queries = {k: ds_queries[k] for k in q_keys}

            matched_b = MATCHED_BUDGETS[ds_name]
            budgets_to_run = [args.budget] if args.budget is not None else sorted(list(set(GRID_BUDGETS + [matched_b])))

            for budget in budgets_to_run:
                cfg_id = f"earc_b{budget}" if budget in GRID_BUDGETS else f"earc_matched_{budget}"
                is_matched = (budget == matched_b)
                logger.info("\n" + "=" * 80)
                logger.info("EXP 2: EARC BUDGET %d TOKENS (%s) on %s", budget, "MATCHED" if is_matched else "GRID", ds_name.upper())
                logger.info("=" * 80)

                cfg_yaml_path = configs_dir / f"config_{cfg_id}_{ds_name}.yaml"
                write_config_yaml(base_config, cfg_id, budget, None, cfg_yaml_path)

                # Build compressor for this budget
                c_config = base_config.compression.model_copy(deep=True)
                c_config.alpha = 0.7
                c_config.beta = 0.3
                c_config.use_redundancy = True
                c_config.redundancy_threshold = 0.85
                c_config.token_budget = budget
                compressor = EvidenceAwareCompressor(c_config, embedding_engine, token_counter, device=device)

                # Compress once
                ctx_file = contexts_dir / f"ctx_earc_{ds_name}_b{budget}.json"
                compressed_contexts = {}
                if ctx_file.exists():
                    logger.info("Loading existing compressed contexts from %s...", ctx_file)
                    with open(ctx_file, "r", encoding="utf-8") as f:
                        compressed_contexts = json.load(f)
                else:
                    logger.info("Compressing %d queries for %s (budget=%d)...", len(ds_queries), ds_name, budget)
                    t_comp_start = time.time()
                    for q_idx, (q_id, q_data) in enumerate(ds_queries.items(), 1):
                        ret_docs = [
                            RetrievedDocument(
                                doc_id=d["doc_id"],
                                title=d["title"],
                                text=d["text"],
                                rank=d["rank"],
                                score=d["score"],
                                retriever="dense",
                            )
                            for d in q_data["docs"]
                        ]
                        t_c0 = time.time()
                        comp_res = compressor.compress(q_data["question"], ret_docs, dataset=ds_alias)
                        comp_time = time.time() - t_c0

                        n_candidates = len(comp_res.candidates) if comp_res.candidates else 0
                        n_rejected = sum(1 for c in comp_res.candidates if c.status == SentenceStatus.REJECTED_REDUNDANCY) if comp_res.candidates else 0

                        compressed_contexts[q_id] = {
                            "question_id": q_id,
                            "dataset": ds_name,
                            "question": q_data["question"],
                            "gold_answers": q_data["gold_answers"],
                            "compressed_text": comp_res.compressed_text,
                            "context_tokens": comp_res.compressed_tokens,
                            "compress_time_s": round(comp_time, 4),
                            "n_candidates": n_candidates,
                            "n_rejected_redundancy": n_rejected,
                        }
                        if q_idx % 200 == 0 or q_idx == len(ds_queries):
                            logger.info("[%s b%d] Compressed %d/%d (Elapsed: %.1fs)...",
                                        ds_name, budget, q_idx, len(ds_queries), time.time() - t_comp_start)

                    with open(ctx_file, "w", encoding="utf-8") as f:
                        json.dump(compressed_contexts, f, indent=2)
                    logger.info("Saved compressed contexts to %s", ctx_file)

                if args.compress_only:
                    continue

                # Generate twice
                for gen_id in target_generators:
                    llm_prov = llms[gen_id]
                    pred_file = preds_dir / f"pred_earc_{ds_name}_b{budget}_{gen_id.replace(':', '_')}.jsonl"
                    completed_recs = {}
                    if pred_file.exists():
                        with open(pred_file, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    r = json.loads(line)
                                    completed_recs[r["question_id"]] = r

                    logger.info("Generating for EARC b%d | %s | %s (Completed: %d/%d)...",
                                budget, ds_name, gen_id, len(completed_recs), len(ds_queries))

                    if len(completed_recs) < len(ds_queries):
                        with open(pred_file, "a", encoding="utf-8") as f_out:
                            for q_idx, (q_id, ctx) in enumerate(compressed_contexts.items(), 1):
                                if q_id in completed_recs:
                                    continue

                                prompt = prompt_builder.build(ctx["question"], ctx["compressed_text"])
                                t_g0 = time.time()
                                try:
                                    gen_res = llm_prov.generate(prompt)
                                    raw_pred = gen_res.text
                                except Exception as e:
                                    logger.error("Generation error on %s: %s", q_id, e)
                                    raw_pred = ""
                                gen_time = time.time() - t_g0

                                eval_ans = extract_answer(raw_pred)
                                em = exact_match_score(eval_ans, ctx["gold_answers"])
                                f1 = max_token_f1_score(eval_ans, ctx["gold_answers"])["f1"]

                                record = {
                                    "experiment": "exp2",
                                    "config_id": cfg_id,
                                    "dataset": ds_name,
                                    "generator": gen_id,
                                    "question_id": q_id,
                                    "gold_answers": ctx["gold_answers"],
                                    "prediction": raw_pred,
                                    "evaluated_answer": eval_ans,
                                    "em": int(em == 1.0),
                                    "f1": round(f1, 4),
                                    "context_tokens": ctx["context_tokens"],
                                    "compress_time_s": ctx["compress_time_s"],
                                    "gen_time_s": round(gen_time, 4),
                                    "alpha": 0.7,
                                    "beta": 0.3,
                                    "theta": 0.85,
                                    "budget": budget,
                                    "k": None,
                                }
                                f_out.write(json.dumps(record) + "\n")
                                f_out.flush()
                                completed_recs[q_id] = record

                                if q_idx % 100 == 0 or q_idx == len(compressed_contexts):
                                    logger.info("[%s b%d | %s] %d/%d generated...",
                                                ds_name, budget, gen_id, q_idx, len(compressed_contexts))

                    all_recs = list(completed_recs.values())
                    row = {
                        "experiment": "exp2",
                        "config_id": cfg_id,
                        "dataset": ds_name,
                        "generator": gen_id,
                        "method": f"EARC (budget={budget})",
                        "alpha": 0.7,
                        "beta": 0.3,
                        "theta": 0.85,
                        "budget": budget,
                        "k": None,
                        "avg_tokens": round(float(np.mean([r["context_tokens"] for r in all_recs])), 1),
                        "em": round(float(np.mean([r["em"] for r in all_recs])) * 100.0, 2),
                        "f1": round(float(np.mean([r["f1"] for r in all_recs])) * 100.0, 2),
                        "compress_time_s": round(float(np.mean([r["compress_time_s"] for r in all_recs])), 3),
                        "gen_time_s": round(float(np.mean([r["gen_time_s"] for r in all_recs])), 3),
                        "n_questions": len(all_recs),
                    }
                    update_summary_csv(summary_csv, row)
                    logger.info("RESULT: EARC %s | %s | %s -> Tokens: %.1f | EM: %.2f%% | F1: %.2f%%",
                                cfg_id, ds_name, gen_id, row["avg_tokens"], row["em"], row["f1"])

    # =========================================================================
    # PART 2: Top-k Passages Sweep (k = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    # =========================================================================
    if args.mode in ["all", "topk"]:
        k_list = [args.k] if args.k is not None else list(range(1, 11))

        for ds_name in target_datasets:
            ds_queries = retrieval_cache[ds_name]
            if args.limit:
                q_keys = list(ds_queries.keys())[:args.limit]
                ds_queries = {k: ds_queries[k] for k in q_keys}

            for k in k_list:
                cfg_id = f"topk_{k}"
                logger.info("\n" + "=" * 80)
                logger.info("EXP 2: TOP-k PASSAGES (k=%d) on %s", k, ds_name.upper())
                logger.info("=" * 80)

                cfg_yaml_path = configs_dir / f"config_{cfg_id}_{ds_name}.yaml"
                write_config_yaml(base_config, cfg_id, None, k, cfg_yaml_path)

                # Build contexts for top-k passages (without EARC sentence segmentation)
                ctx_file = contexts_dir / f"ctx_topk_{ds_name}_k{k}.json"
                topk_contexts = {}
                if ctx_file.exists():
                    with open(ctx_file, "r", encoding="utf-8") as f:
                        topk_contexts = json.load(f)
                else:
                    for q_id, q_data in ds_queries.items():
                        passages = [d["text"] for d in q_data["docs"][:k]]
                        merged_text = "\n\n".join(passages)
                        tok_count = token_counter.count(merged_text)
                        topk_contexts[q_id] = {
                            "question_id": q_id,
                            "dataset": ds_name,
                            "question": q_data["question"],
                            "gold_answers": q_data["gold_answers"],
                            "compressed_text": merged_text,
                            "context_tokens": tok_count,
                            "compress_time_s": 0.0,
                        }
                    with open(ctx_file, "w", encoding="utf-8") as f:
                        json.dump(topk_contexts, f, indent=2)

                if args.compress_only:
                    continue

                # Generate twice
                for gen_id in target_generators:
                    llm_prov = llms[gen_id]
                    pred_file = preds_dir / f"pred_topk_{ds_name}_k{k}_{gen_id.replace(':', '_')}.jsonl"
                    completed_recs = {}
                    if pred_file.exists():
                        with open(pred_file, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    r = json.loads(line)
                                    completed_recs[r["question_id"]] = r

                    logger.info("Generating for Top-%d | %s | %s (Completed: %d/%d)...",
                                k, ds_name, gen_id, len(completed_recs), len(ds_queries))

                    if len(completed_recs) < len(ds_queries):
                        with open(pred_file, "a", encoding="utf-8") as f_out:
                            for q_idx, (q_id, ctx) in enumerate(topk_contexts.items(), 1):
                                if q_id in completed_recs:
                                    continue

                                prompt = prompt_builder.build(ctx["question"], ctx["compressed_text"])
                                t_g0 = time.time()
                                try:
                                    gen_res = llm_prov.generate(prompt)
                                    raw_pred = gen_res.text
                                except Exception as e:
                                    logger.error("Generation error on %s: %s", q_id, e)
                                    raw_pred = ""
                                gen_time = time.time() - t_g0

                                eval_ans = extract_answer(raw_pred)
                                em = exact_match_score(eval_ans, ctx["gold_answers"])
                                f1 = max_token_f1_score(eval_ans, ctx["gold_answers"])["f1"]

                                record = {
                                    "experiment": "exp2",
                                    "config_id": cfg_id,
                                    "dataset": ds_name,
                                    "generator": gen_id,
                                    "question_id": q_id,
                                    "gold_answers": ctx["gold_answers"],
                                    "prediction": raw_pred,
                                    "evaluated_answer": eval_ans,
                                    "em": int(em == 1.0),
                                    "f1": round(f1, 4),
                                    "context_tokens": ctx["context_tokens"],
                                    "compress_time_s": 0.0,
                                    "gen_time_s": round(gen_time, 4),
                                    "alpha": None,
                                    "beta": None,
                                    "theta": None,
                                    "budget": None,
                                    "k": k,
                                }
                                f_out.write(json.dumps(record) + "\n")
                                f_out.flush()
                                completed_recs[q_id] = record

                                if q_idx % 100 == 0 or q_idx == len(topk_contexts):
                                    logger.info("[%s k%d | %s] %d/%d generated...",
                                                ds_name, k, gen_id, q_idx, len(topk_contexts))

                    all_recs = list(completed_recs.values())
                    row = {
                        "experiment": "exp2",
                        "config_id": cfg_id,
                        "dataset": ds_name,
                        "generator": gen_id,
                        "method": f"Top-{k} passages",
                        "alpha": None,
                        "beta": None,
                        "theta": None,
                        "budget": None,
                        "k": k,
                        "avg_tokens": round(float(np.mean([r["context_tokens"] for r in all_recs])), 1),
                        "em": round(float(np.mean([r["em"] for r in all_recs])) * 100.0, 2),
                        "f1": round(float(np.mean([r["f1"] for r in all_recs])) * 100.0, 2),
                        "compress_time_s": 0.0,
                        "gen_time_s": round(float(np.mean([r["gen_time_s"] for r in all_recs])), 3),
                        "n_questions": len(all_recs),
                    }
                    update_summary_csv(summary_csv, row)
                    logger.info("RESULT: TOP-K %s | %s | %s -> Tokens: %.1f | EM: %.2f%% | F1: %.2f%%",
                                cfg_id, ds_name, gen_id, row["avg_tokens"], row["em"], row["f1"])

    logger.info("\n" + "=" * 80)
    logger.info("EXP 2 EXECUTION STEP COMPLETE!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
