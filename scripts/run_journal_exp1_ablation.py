"""
Experiment 1: Component Ablation and Beta Sweep (Journal Protocol - 2 October 2026).

Configurations:
- A1: Similarity only          (alpha=1.0, beta=0.0, filter=off / tau=1.0)
- A2: Evidence only            (alpha=0.0, beta=1.0, filter=off / tau=1.0)
- A3: Similarity + evidence    (alpha=0.7, beta=0.3, filter=off / tau=1.0)
- A4: Similarity + redundancy  (alpha=1.0, beta=0.0, filter=on / tau=0.85) [beta=0.0 sweep]
- A5: Full EARC (default)      (alpha=0.7, beta=0.3, filter=on / tau=0.85) [beta=0.3 sweep / sanity check]
- B1: beta=0.1                 (alpha=0.9, beta=0.1, filter=on / tau=0.85)
- B2: beta=0.2                 (alpha=0.8, beta=0.2, filter=on / tau=0.85)
- B3: beta=0.5                 (alpha=0.5, beta=0.5, filter=on / tau=0.85)

Execution details:
- Shared dense retrieval cache (retrieved_passages.pkl)
- Compress once, generate twice (Ollama Llama 3.2 and Mistral API)
- Log format matches Protocol Section 1.2
- Separate config YAML per run written to disk
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

logger = get_logger("journal_exp1_ablation")

EXP1_CONFIGS = {
    "A1": {
        "name": "Similarity only",
        "alpha": 1.0,
        "beta": 0.0,
        "use_redundancy": False,
        "redundancy_threshold": 1.0,
        "budget": 300,
    },
    "A2": {
        "name": "Evidence only",
        "alpha": 0.0,
        "beta": 1.0,
        "use_redundancy": False,
        "redundancy_threshold": 1.0,
        "budget": 300,
    },
    "A3": {
        "name": "Similarity + evidence",
        "alpha": 0.7,
        "beta": 0.3,
        "use_redundancy": False,
        "redundancy_threshold": 1.0,
        "budget": 300,
    },
    "A4": {
        "name": "Similarity + redundancy",
        "alpha": 1.0,
        "beta": 0.0,
        "use_redundancy": True,
        "redundancy_threshold": 0.85,
        "budget": 300,
    },
    "A5": {
        "name": "Full EARC",
        "alpha": 0.7,
        "beta": 0.3,
        "use_redundancy": True,
        "redundancy_threshold": 0.85,
        "budget": 300,
    },
    "B1": {
        "name": "beta = 0.1",
        "alpha": 0.9,
        "beta": 0.1,
        "use_redundancy": True,
        "redundancy_threshold": 0.85,
        "budget": 300,
    },
    "B2": {
        "name": "beta = 0.2",
        "alpha": 0.8,
        "beta": 0.2,
        "use_redundancy": True,
        "redundancy_threshold": 0.85,
        "budget": 300,
    },
    "B3": {
        "name": "beta = 0.5",
        "alpha": 0.5,
        "beta": 0.5,
        "use_redundancy": True,
        "redundancy_threshold": 0.85,
        "budget": 300,
    },
}

DATASET_MAP = {
    "nq": "nq",
    "hotpotqa": "hotpot",
    "triviaqa": "trivia",
}

GENERATOR_MAP = {
    "llama3.2:3b": "ollama",
    "ministral-8b": "mistral",
}

def write_config_yaml(base_config, cfg_id, cfg_data, out_path: Path):
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
            "alpha": cfg_data["alpha"],
            "beta": cfg_data["beta"],
            "use_redundancy": cfg_data["use_redundancy"],
            "redundancy_threshold": cfg_data["redundancy_threshold"],
            "token_budget": cfg_data["budget"],
            "spacy_model": "en_core_web_sm",
        },
        "experiment": {
            "experiment": "exp1",
            "config_id": cfg_id,
            "config_name": cfg_data["name"],
        }
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(out_dict, f, default_flow_style=False)

def update_summary_csv(summary_csv: Path, row: Dict[str, Any]):
    existing_rows = []
    if summary_csv.exists():
        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    # Replace or append
    matched = False
    for i, r in enumerate(existing_rows):
        if r["config_id"] == row["config_id"] and r["dataset"] == row["dataset"] and r["generator"] == row["generator"]:
            existing_rows[i] = {k: str(v) for k, v in row.items()}
            matched = True
            break
    if not matched:
        existing_rows.append({k: str(v) for k, v in row.items()} )

    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(existing_rows)

def main():
    parser = argparse.ArgumentParser(description="Journal Exp 1 Ablation Runner")
    parser.add_argument("--config", type=str, default="all", help="Config ID (A1..B3) or 'all'")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset (nq, hotpotqa, triviaqa) or 'all'")
    parser.add_argument("--generator", type=str, default="all", help="Generator (llama3.2:3b, ministral-8b) or 'all'")
    parser.add_argument("--compress-only", action="store_true", help="Only run compression and save contexts")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries for test")
    args = parser.parse_args()

    base_config = load_config("configs/default.yaml")
    device = DeviceManager().get_device().type

    # Directories
    exp1_dir = Path("outputs/journal_experiments/exp1_ablation")
    configs_dir = exp1_dir / "configs"
    contexts_dir = exp1_dir / "contexts"
    preds_dir = exp1_dir / "predictions"
    summary_csv = exp1_dir / "exp1_summary.csv"
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

    # Filter configurations
    target_configs = [args.config] if args.config != "all" else list(EXP1_CONFIGS.keys())
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

    # Iterate over selected configurations
    for cfg_id in target_configs:
        cfg_data = EXP1_CONFIGS[cfg_id]
        logger.info("\n" + "=" * 80)
        logger.info("EXP 1 - CONFIG %s: %s (alpha=%.1f, beta=%.1f, red=%s, tau=%.2f)",
                    cfg_id, cfg_data["name"], cfg_data["alpha"], cfg_data["beta"],
                    cfg_data["use_redundancy"], cfg_data["redundancy_threshold"])
        logger.info("=" * 80)

        # 1. Save separate YAML configuration
        cfg_yaml_path = configs_dir / f"config_{cfg_id}.yaml"
        write_config_yaml(base_config, cfg_id, cfg_data, cfg_yaml_path)

        # Build compressor for this configuration
        c_config = base_config.compression.model_copy(deep=True)
        c_config.alpha = cfg_data["alpha"]
        c_config.beta = cfg_data["beta"]
        c_config.use_redundancy = cfg_data["use_redundancy"]
        c_config.redundancy_threshold = cfg_data["redundancy_threshold"]
        c_config.token_budget = cfg_data["budget"]
        compressor = EvidenceAwareCompressor(c_config, embedding_engine, token_counter, device=device)

        # Process each dataset
        for ds_name in target_datasets:
            ds_alias = DATASET_MAP[ds_name]
            ds_queries = retrieval_cache[ds_name]
            if args.limit:
                q_keys = list(ds_queries.keys())[:args.limit]
                ds_queries = {k: ds_queries[k] for k in q_keys}

            logger.info("--- Dataset: %s (N=%d queries) ---", ds_name.upper(), len(ds_queries))

            # Step 1: Compress Once (or load cached compressed contexts)
            ctx_file = contexts_dir / f"ctx_{cfg_id}_{ds_name}.json"
            compressed_contexts = {}
            if ctx_file.exists():
                logger.info("Loading existing compressed contexts from %s...", ctx_file)
                with open(ctx_file, "r", encoding="utf-8") as f:
                    compressed_contexts = json.load(f)
            else:
                logger.info("Compressing %d queries for %s - %s...", len(ds_queries), cfg_id, ds_name)
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
                        logger.info("[%s - %s] Compressed %d/%d queries (Elapsed: %.1fs)...",
                                    cfg_id, ds_name, q_idx, len(ds_queries), time.time() - t_comp_start)

                # Save compressed contexts
                with open(ctx_file, "w", encoding="utf-8") as f:
                    json.dump(compressed_contexts, f, indent=2)
                logger.info("Saved compressed contexts to %s", ctx_file)

            if args.compress_only:
                logger.info("Skipping generation (--compress-only specified).")
                continue

            # Step 2: Generate Twice (Llama 3.2 and Ministral-8B)
            for gen_id in target_generators:
                llm_prov = llms[gen_id]
                pred_file = preds_dir / f"pred_{cfg_id}_{ds_name}_{gen_id.replace(':', '_')}.jsonl"
                completed_recs = {}
                if pred_file.exists():
                    with open(pred_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                r = json.loads(line)
                                completed_recs[r["question_id"]] = r

                logger.info("Generating for %s | %s | %s (Completed: %d/%d)...",
                            cfg_id, ds_name, gen_id, len(completed_recs), len(ds_queries))

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
                                "experiment": "exp1",
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
                                "alpha": cfg_data["alpha"],
                                "beta": cfg_data["beta"],
                                "theta": cfg_data["redundancy_threshold"] if cfg_data["use_redundancy"] else None,
                                "budget": cfg_data["budget"],
                                "k": None,
                                "n_candidates": ctx["n_candidates"],
                                "n_rejected_redundancy": ctx["n_rejected_redundancy"],
                            }
                            f_out.write(json.dumps(record) + "\n")
                            f_out.flush()
                            completed_recs[q_id] = record

                            if q_idx % 100 == 0 or q_idx == len(compressed_contexts):
                                logger.info("[%s | %s | %s] %d/%d generated...",
                                            cfg_id, ds_name, gen_id, q_idx, len(compressed_contexts))

                # Compute aggregate metrics for this run
                all_recs = list(completed_recs.values())
                avg_tokens = np.mean([r["context_tokens"] for r in all_recs])
                em_pct = np.mean([r["em"] for r in all_recs]) * 100.0
                f1_pct = np.mean([r["f1"] for r in all_recs]) * 100.0
                avg_comp_t = np.mean([r["compress_time_s"] for r in all_recs])
                avg_gen_t = np.mean([r["gen_time_s"] for r in all_recs])

                row = {
                    "experiment": "exp1",
                    "config_id": cfg_id,
                    "dataset": ds_name,
                    "generator": gen_id,
                    "method": cfg_data["name"],
                    "alpha": cfg_data["alpha"],
                    "beta": cfg_data["beta"],
                    "theta": cfg_data["redundancy_threshold"] if cfg_data["use_redundancy"] else 1.0,
                    "budget": cfg_data["budget"],
                    "k": None,
                    "avg_tokens": round(avg_tokens, 1),
                    "em": round(em_pct, 2),
                    "f1": round(f1_pct, 2),
                    "compress_time_s": round(avg_comp_t, 3),
                    "gen_time_s": round(avg_gen_t, 3),
                    "n_questions": len(all_recs),
                }
                update_summary_csv(summary_csv, row)
                logger.info("RESULT: %s | %s | %s -> Tokens: %.1f | EM: %.2f%% | F1: %.2f%%",
                            cfg_id, ds_name, gen_id, avg_tokens, em_pct, f1_pct)

    logger.info("\n" + "=" * 80)
    logger.info("EXP 1 EXECUTION STEP COMPLETE!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
