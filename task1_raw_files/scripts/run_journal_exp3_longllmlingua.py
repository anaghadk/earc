"""
Experiment 3 — LongLLMLingua Question-Aware Compression Baseline
Protocol dated 2 October 2026.
Evaluates LongLLMLingua across 3 datasets x 2 generators = 6 runs.
"""

import os
import sys
import time
import json
import csv
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = "lIK71q2MhMrZXQwxYE0nVUvoNNUVjQzy"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp3_longllmlingua")

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
import tiktoken
from llmlingua import PromptCompressor
import llmlingua

def patched_get_ppl(self, text: str, granularity: str = 'sentence', input_ids=None, attention_mask=None,
                    past_key_values=None, return_kv=False, end=None, condition_mode: str = 'none', condition_pos_id: int = 0):
    if input_ids is None:
        tokenized_text = self.tokenizer(text, return_tensors='pt')
        input_ids = tokenized_text['input_ids'].to(self.device)
        attention_mask = tokenized_text['attention_mask'].to(self.device)
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[2]
        cache = DynamicCache()
        for item in past_key_values:
            layer = DynamicLayer()
            layer.keys = item[0]
            layer.values = item[1]
            layer.is_initialized = True
            cache.layers.append(layer)
        pkv_for_model = cache
    else:
        past_length = 0
        pkv_for_model = None

    if end is None:
        end = input_ids.shape[1]
    end = min(end, past_length + self.max_position_embeddings)
    with torch.no_grad():
        response = self.model(
            input_ids[:, past_length:end],
            attention_mask=attention_mask[:, :end],
            past_key_values=pkv_for_model,
            use_cache=True,
        )
        ret_pkv = [[layer.keys, layer.values] for layer in response.past_key_values.layers]

    shift_logits = response.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., past_length + 1 : end].contiguous()
    active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
    active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
    active_labels = shift_labels.view(-1)[active]
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(active_logits, active_labels)
    if condition_mode == 'before':
        loss = loss[:condition_pos_id]
    elif condition_mode == 'after':
        loss = loss[condition_pos_id:]
    res = loss.mean() if granularity == 'sentence' else loss
    return (res, ret_pkv) if return_kv else res

PromptCompressor.get_ppl = patched_get_ppl

from src.prompting.builder import PromptBuilder
from src.prompting.templates import DEFAULT_QA_TEMPLATE
from src.llm.factory import create_provider
from src.evaluation.exact_match import exact_match_score, normalize_answer
from src.evaluation.f1 import max_token_f1_score
from src.evaluation.answer_extraction import extract_answer
from src.config.settings import load_config

DATASET_MAP = {
    "nq": "nq",
    "hotpotqa": "hotpotqa",
    "triviaqa": "triviaqa",
}

GENERATOR_MAP = {
    "llama3.2:3b": "ollama",
    "ministral-8b": "mistral",
}

def main():
    parser = argparse.ArgumentParser(description="Journal Exp 3 LongLLMLingua Runner")
    parser.add_argument("--compress-only", action="store_true", help="Only run compression")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "nq", "hotpotqa", "triviaqa"])
    parser.add_argument("--generator", type=str, default="all", choices=["all", "llama3.2:3b", "ministral-8b"])
    args = parser.parse_args()

    # Directories
    exp3_dir = PROJECT_ROOT / "outputs/journal_experiments/exp3_longllmlingua"
    contexts_dir = exp3_dir / "contexts"
    preds_dir = exp3_dir / "predictions"
    exp3_summary_csv = exp3_dir / "exp3_summary.csv"
    contexts_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load locked question IDs
    with open("outputs/final_evaluation/selected_example_ids.json", "r", encoding="utf-8") as f:
        locked_example_ids = json.load(f)

    # 2. Load cached retrieval passages
    cache_pkl = PROJECT_ROOT / "outputs/journal_experiments/retrieval_cache/retrieved_passages.pkl"
    with open(cache_pkl, "rb") as f:
        retrieval_cache = pickle.load(f)

    enc = tiktoken.get_encoding("cl100k_base")
    prompt_builder = PromptBuilder(DEFAULT_QA_TEMPLATE)
    base_config = load_config("configs/default.yaml")

    datasets = ["nq", "hotpotqa", "triviaqa"] if args.dataset == "all" else [args.dataset]
    generators = ["llama3.2:3b", "ministral-8b"] if args.generator == "all" else [args.generator]

    # STAGE 1: COMPRESSION
    compressor = None
    for ds_name in datasets:
        ctx_file = contexts_dir / f"ctx_longllmlingua_{ds_name}.json"
        ds_queries = retrieval_cache[ds_name]

        compressed_contexts = {}
        if ctx_file.exists():
            logger.info("Loading existing compressed contexts from %s...", ctx_file)
            with open(ctx_file, "r", encoding="utf-8") as f:
                compressed_contexts = json.load(f)

        if len(compressed_contexts) < len(ds_queries):
            if compressor is None:
                logger.info("Initializing LongLLMLingua (llmlingua v%s) with microsoft/phi-2 on CUDA...", llmlingua.__version__)
                compressor = PromptCompressor(model_name="microsoft/phi-2", device_map="cuda")
            logger.info("Compressing %s (%d queries) with LongLLMLingua...", ds_name, len(ds_queries))
            t_comp_start = time.time()
            for q_idx, (q_id, q_data) in enumerate(ds_queries.items(), 1):
                if q_id in compressed_contexts:
                    continue

                question = q_data["question"]
                passages = [d["text"] for d in q_data["docs"]]

                t0 = time.time()
                res = compressor.compress_prompt(
                    context=passages,
                    question=question,
                    target_token=255,
                    rank_method="longllmlingua",
                    condition_in_question="after_condition",
                    reorder_context="sort",
                    dynamic_context_compression_ratio=0.3,
                    condition_compare=True,
                    context_budget="+100"
                )
                comp_time = time.time() - t0

                comp_text = res["compressed_prompt"]
                # If compressed prompt ends with the question, strip it per protocol
                if comp_text.strip().endswith(question.strip()):
                    comp_text = comp_text.strip()[:-len(question.strip())].strip()

                tok_count = len(enc.encode(comp_text))

                compressed_contexts[q_id] = {
                    "question_id": q_id,
                    "dataset": ds_name,
                    "question": question,
                    "gold_answers": q_data["gold_answers"],
                    "compressed_text": comp_text,
                    "context_tokens": tok_count,
                    "compress_time_s": round(comp_time, 4),
                    "compressor_model": "microsoft/phi-2",
                    "llmlingua_version": llmlingua.__version__,
                }

                if q_idx % 100 == 0 or q_idx == len(ds_queries):
                    elapsed = time.time() - t_comp_start
                    logger.info("[%s] Compressed %d/%d queries (Avg: %.2fs/query, Total: %.1fs)...",
                                ds_name, q_idx, len(ds_queries), elapsed / q_idx, elapsed)

            with open(ctx_file, "w", encoding="utf-8") as f:
                json.dump(compressed_contexts, f, indent=2)
            logger.info("Saved %d compressed contexts to %s", len(compressed_contexts), ctx_file)

    if compressor is not None:
        del compressor
        torch.cuda.empty_cache()

    if args.compress_only:
        logger.info("Compression-only mode complete.")
        return

    # STAGE 2: GENERATION & EVALUATION
    # Initialize generators
    llms = {}
    for gen_id in generators:
        prov_name = GENERATOR_MAP[gen_id]
        llms[gen_id] = create_provider(base_config, prov_name)

    for ds_name in datasets:
        ctx_file = contexts_dir / f"ctx_longllmlingua_{ds_name}.json"
        with open(ctx_file, "r", encoding="utf-8") as f:
            compressed_contexts = json.load(f)

        for gen_id in generators:
            llm_prov = llms[gen_id]
            gen_suffix = gen_id.replace(":", "_")
            pred_file = preds_dir / f"pred_longllmlingua_{ds_name}_{gen_suffix}.jsonl"

            completed_recs = {}
            if pred_file.exists():
                with open(pred_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            completed_recs[r["question_id"]] = r

            logger.info("Generating for LongLLMLingua | %s | %s (Completed: %d/%d)...",
                        ds_name, gen_id, len(completed_recs), len(compressed_contexts))

            if len(completed_recs) < len(compressed_contexts):
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

                        rec = {
                            "experiment": "exp3",
                            "config_id": "longllmlingua",
                            "dataset": ds_name,
                            "generator": gen_id,
                            "question_id": q_id,
                            "question": ctx["question"],
                            "gold_answers": ctx["gold_answers"],
                            "raw_prediction": raw_pred,
                            "predicted_answer": eval_ans,
                            "exact_match": em,
                            "f1": f1,
                            "context_tokens": ctx["context_tokens"],
                            "compress_time_s": ctx["compress_time_s"],
                            "gen_time_s": round(gen_time, 4),
                            "llmlingua_version": ctx.get("llmlingua_version", "0.2.2"),
                            "compressor_model": ctx.get("compressor_model", "microsoft/phi-2"),
                        }
                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f_out.flush()
                        completed_recs[q_id] = rec

                        if q_idx % 200 == 0 or q_idx == len(compressed_contexts):
                            logger.info("[%s | %s] Generated %d/%d questions...",
                                        ds_name, gen_id, len(completed_recs), len(compressed_contexts))

            # Sync summaries and deliverables
            logger.info("Syncing summaries after finishing %s | %s...", ds_name, gen_id)
            import subprocess
            subprocess.run([sys.executable, "scripts/sync_summaries.py"])
            subprocess.run([sys.executable, "scripts/journal_generate_deliverables.py"])

    logger.info("Experiment 3 completed all 6 runs!")

if __name__ == "__main__":
    main()
