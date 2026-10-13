"""
Comprehensive Diagnostic Evaluation Script
Evaluates the finalized EARC system vs Standard RAG across Ollama (Llama 3.2)
and Mistral (Ministral 8B) on the validated 20 examples per dataset (60 total).

Configuration:
- Dense retrieval, top_k=10
- EARC: alpha=0.7, beta=0.3, redundancy_threshold=0.85, token_budget=300
- Standard RAG: exact same retrieved documents, uncompressed
"""

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Fallback Mistral API key if not in env
if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = "lIK71q2MhMrZXQwxYE0nVUvoNNUVjQzy"

from src.config.settings import load_config
from src.data.loaders import load_dataset
from src.data.sampling import sample_dataset
from src.data.schemas import QAExample, RetrievedDocument
from src.retrieval.rag_project import RAGProjectRetriever
from src.compression.embeddings import EmbeddingEngine
from src.compression.compressor import EvidenceAwareCompressor
from src.prompting.builder import PromptBuilder
from src.prompting.templates import DEFAULT_QA_TEMPLATE
from src.llm.factory import create_provider
from src.evaluation.exact_match import exact_match_score
from src.evaluation.f1 import max_token_f1_score
from src.evaluation.answer_extraction import extract_answer
from src.evaluation.aggregation import compute_retrieval_hit
from src.utils.device import DeviceManager
from src.utils.logging import setup_logging, get_logger
from src.utils.io import ensure_dir

logger = get_logger("diagnostic")


def build_retrieval_pool(example: QAExample, retriever: RAGProjectRetriever, top_k: int) -> List[RetrievedDocument]:
    """Retrieve or load candidate documents for an example."""
    if example.documents and len(example.documents) > 0:
        pool = []
        for i, doc in enumerate(example.documents[:top_k]):
            pool.append(RetrievedDocument(
                doc_id=doc.doc_id or f"doc_{i}",
                title=doc.title,
                text=doc.text,
                score=1.0 - (i * 0.01),
                rank=i + 1,
                retriever="dataset",
                query=example.question,
            ))
        return pool
    else:
        return retriever.retrieve(example.question, top_k=top_k)


def format_standard_rag_context(retrieved_docs: List[RetrievedDocument]) -> str:
    """Format full uncompressed context for Standard RAG."""
    doc_texts = []
    for doc in retrieved_docs:
        title_text = doc.title if doc.title else ""
        if title_text:
            doc_texts.append(f"Title: {title_text}\n{doc.text}")
        else:
            doc_texts.append(doc.text)
    return "\n\n".join(doc_texts)


def safe_generate(llm: Any, prompt: str, max_retries: int = 5) -> Any:
    """Generate response with retries for network resilience."""
    for attempt in range(max_retries):
        try:
            return llm.generate(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Generation failed after {max_retries} attempts: {e}")
                raise
            delay = (2 ** attempt) + 1.5
            logger.warning(f"Generation error ({e}). Retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(delay)


def evaluate_single_example(
    example: QAExample,
    retriever: RAGProjectRetriever,
    compressor: EvidenceAwareCompressor,
    prompt_builder: PromptBuilder,
    providers: Dict[str, Any],
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Evaluates all 4 configurations on a single example ensuring identical retrieved documents.
    """
    # 1. Retrieval (shared across all configurations)
    t_ret0 = time.time()
    retrieved_docs = build_retrieval_pool(example, retriever, top_k=top_k)
    retrieval_latency = time.time() - t_ret0

    # Answer Hit@K
    docs_for_hit = [d.to_dict() if hasattr(d, "to_dict") else d for d in retrieved_docs]
    hit_1 = compute_retrieval_hit(docs_for_hit, example.answers, k=1)
    hit_5 = compute_retrieval_hit(docs_for_hit, example.answers, k=5)
    hit_10 = compute_retrieval_hit(docs_for_hit, example.answers, k=10)

    # 2. Evidence-Aware Compression
    t_comp0 = time.time()
    compressed = compressor.compress(example.question, retrieved_docs)
    compression_latency = time.time() - t_comp0

    earc_context = compressed.compressed_text
    earc_prompt = prompt_builder.build_from_compressed(example.question, compressed)
    earc_orig_tokens = compressed.original_tokens
    earc_comp_tokens = compressed.compressed_tokens
    earc_comp_pct = compressed.reduction_percentage

    # 3. Standard RAG uncompressed context
    standard_context = format_standard_rag_context(retrieved_docs)
    standard_prompt = prompt_builder.build(example.question, standard_context)
    # Token count for standard RAG
    token_counter = providers["ollama"].get_token_counter()
    standard_tokens = token_counter.count(standard_context)

    eval_result = {
        "dataset": example.dataset,
        "example_id": example.id,
        "question": example.question,
        "gold_answers": example.answers,
        "hit_at_1": hit_1,
        "hit_at_5": hit_5,
        "hit_at_10": hit_10,
        "retrieval_latency": retrieval_latency,
        "compression_latency": compression_latency,
        "original_context_tokens": earc_orig_tokens,
        "compressed_context_tokens": earc_comp_tokens,
        "compression_percentage": earc_comp_pct,
        "standard_context_tokens": standard_tokens,
        "provider_results": {},
    }

    # Evaluate across providers
    for provider_name in ["ollama", "mistral"]:
        llm = providers[provider_name]
        model_name = llm.model_name

        # EARC Generation
        t_gen_earc0 = time.time()
        gen_earc = safe_generate(llm, earc_prompt)
        gen_latency_earc = time.time() - t_gen_earc0
        ans_earc = extract_answer(gen_earc.text) or gen_earc.text.strip()
        em_earc = exact_match_score(ans_earc, example.answers)
        f1_earc = max_token_f1_score(ans_earc, example.answers)["f1"]
        e2e_latency_earc = retrieval_latency + compression_latency + gen_latency_earc

        # Standard RAG Generation
        t_gen_std0 = time.time()
        gen_std = safe_generate(llm, standard_prompt)
        gen_latency_std = time.time() - t_gen_std0
        ans_std = extract_answer(gen_std.text) or gen_std.text.strip()
        em_std = exact_match_score(ans_std, example.answers)
        f1_std = max_token_f1_score(ans_std, example.answers)["f1"]
        e2e_latency_std = retrieval_latency + gen_latency_std

        eval_result["provider_results"][provider_name] = {
            "model": model_name,
            "earc": {
                "final_answer": ans_earc,
                "raw_output": gen_earc.text,
                "exact_match": em_earc,
                "token_f1": f1_earc,
                "generation_latency": gen_latency_earc,
                "end_to_end_latency": e2e_latency_earc,
            },
            "standard_rag": {
                "final_answer": ans_std,
                "raw_output": gen_std.text,
                "exact_match": em_std,
                "token_f1": f1_std,
                "generation_latency": gen_latency_std,
                "end_to_end_latency": e2e_latency_std,
            },
        }

    return eval_result


def flatten_to_paired_records(eval_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten evaluation results into paired rows (one row per example per provider)."""
    rows = []
    for res in eval_results:
        for provider_name, p_data in res["provider_results"].items():
            row = {
                "dataset": res["dataset"],
                "example_id": res["example_id"],
                "question": res["question"],
                "gold_answers": " | ".join(res["gold_answers"]),
                "provider": provider_name,
                "model": p_data["model"],
                "final_answer_earc": p_data["earc"]["final_answer"],
                "final_answer_standard_rag": p_data["standard_rag"]["final_answer"],
                "raw_output_earc": p_data["earc"]["raw_output"].replace("\n", " ").strip(),
                "raw_output_standard_rag": p_data["standard_rag"]["raw_output"].replace("\n", " ").strip(),
                "exact_match_earc": p_data["earc"]["exact_match"],
                "token_f1_earc": round(p_data["earc"]["token_f1"], 4),
                "exact_match_standard_rag": p_data["standard_rag"]["exact_match"],
                "token_f1_standard_rag": round(p_data["standard_rag"]["token_f1"], 4),
                "answer_hit_at_1": res["hit_at_1"],
                "answer_hit_at_5": res["hit_at_5"],
                "answer_hit_at_10": res["hit_at_10"],
                "original_context_tokens": res["original_context_tokens"],
                "compressed_context_tokens": res["compressed_context_tokens"],
                "compression_percentage": round(res["compression_percentage"], 2),
                "retrieval_latency": round(res["retrieval_latency"], 4),
                "compression_latency": round(res["compression_latency"], 4),
                "generation_latency_earc": round(p_data["earc"]["generation_latency"], 4),
                "generation_latency_standard_rag": round(p_data["standard_rag"]["generation_latency"], 4),
                "end_to_end_latency_earc": round(p_data["earc"]["end_to_end_latency"], 4),
                "end_to_end_latency_standard_rag": round(p_data["standard_rag"]["end_to_end_latency"], 4),
            }
            rows.append(row)
    return rows


def flatten_to_config_records(eval_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten evaluation results into individual configuration rows (4 rows per example)."""
    rows = []
    for res in eval_results:
        for provider_name, p_data in res["provider_results"].items():
            # EARC
            rows.append({
                "dataset": res["dataset"],
                "example_id": res["example_id"],
                "question": res["question"],
                "gold_answers": " | ".join(res["gold_answers"]),
                "method": "EARC",
                "provider": provider_name,
                "model": p_data["model"],
                "final_answer": p_data["earc"]["final_answer"],
                "raw_output": p_data["earc"]["raw_output"].replace("\n", " ").strip(),
                "exact_match": p_data["earc"]["exact_match"],
                "token_f1": round(p_data["earc"]["token_f1"], 4),
                "answer_hit_at_1": res["hit_at_1"],
                "answer_hit_at_5": res["hit_at_5"],
                "answer_hit_at_10": res["hit_at_10"],
                "original_context_tokens": res["original_context_tokens"],
                "compressed_context_tokens": res["compressed_context_tokens"],
                "compression_percentage": round(res["compression_percentage"], 2),
                "retrieval_latency": round(res["retrieval_latency"], 4),
                "compression_latency": round(res["compression_latency"], 4),
                "generation_latency": round(p_data["earc"]["generation_latency"], 4),
                "end_to_end_latency": round(p_data["earc"]["end_to_end_latency"], 4),
            })
            # Standard RAG
            rows.append({
                "dataset": res["dataset"],
                "example_id": res["example_id"],
                "question": res["question"],
                "gold_answers": " | ".join(res["gold_answers"]),
                "method": "Standard RAG",
                "provider": provider_name,
                "model": p_data["model"],
                "final_answer": p_data["standard_rag"]["final_answer"],
                "raw_output": p_data["standard_rag"]["raw_output"].replace("\n", " ").strip(),
                "exact_match": p_data["standard_rag"]["exact_match"],
                "token_f1": round(p_data["standard_rag"]["token_f1"], 4),
                "answer_hit_at_1": res["hit_at_1"],
                "answer_hit_at_5": res["hit_at_5"],
                "answer_hit_at_10": res["hit_at_10"],
                "original_context_tokens": res["standard_context_tokens"],
                "compressed_context_tokens": res["standard_context_tokens"],
                "compression_percentage": 0.0,
                "retrieval_latency": round(res["retrieval_latency"], 4),
                "compression_latency": 0.0,
                "generation_latency": round(p_data["standard_rag"]["generation_latency"], 4),
                "end_to_end_latency": round(p_data["standard_rag"]["end_to_end_latency"], 4),
            })
    return rows


def compute_dataset_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics for a list of paired records."""
    import numpy as np

    summary = {}
    for provider in ["ollama", "mistral"]:
        p_recs = [r for r in records if r["provider"] == provider]
        if not p_recs:
            continue
        n = len(p_recs)
        summary[provider] = {
            "n_examples": n,
            "model": p_recs[0]["model"],
            "retrieval": {
                "hit_at_1": float(np.mean([r["answer_hit_at_1"] for r in p_recs])),
                "hit_at_5": float(np.mean([r["answer_hit_at_5"] for r in p_recs])),
                "hit_at_10": float(np.mean([r["answer_hit_at_10"] for r in p_recs])),
                "avg_latency": float(np.mean([r["retrieval_latency"] for r in p_recs])),
            },
            "compression": {
                "avg_original_tokens": float(np.mean([r["original_context_tokens"] for r in p_recs])),
                "avg_compressed_tokens": float(np.mean([r["compressed_context_tokens"] for r in p_recs])),
                "avg_compression_percentage": float(np.mean([r["compression_percentage"] for r in p_recs])),
                "avg_latency": float(np.mean([r["compression_latency"] for r in p_recs])),
            },
            "earc": {
                "exact_match": float(np.mean([r["exact_match_earc"] for r in p_recs])) * 100.0,
                "f1": float(np.mean([r["token_f1_earc"] for r in p_recs])) * 100.0,
                "avg_generation_latency": float(np.mean([r["generation_latency_earc"] for r in p_recs])),
                "avg_end_to_end_latency": float(np.mean([r["end_to_end_latency_earc"] for r in p_recs])),
            },
            "standard_rag": {
                "exact_match": float(np.mean([r["exact_match_standard_rag"] for r in p_recs])) * 100.0,
                "f1": float(np.mean([r["token_f1_standard_rag"] for r in p_recs])) * 100.0,
                "avg_generation_latency": float(np.mean([r["generation_latency_standard_rag"] for r in p_recs])),
                "avg_end_to_end_latency": float(np.mean([r["end_to_end_latency_standard_rag"] for r in p_recs])),
            },
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Diagnostic Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 example smoke test only")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to default config")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save/resume results")
    args = parser.parse_args()

    setup_logging("INFO")
    config = load_config(args.config)
    device = DeviceManager().get_device().type

    # Pin finalized hyperparameters
    config.retrieval.top_k = 10
    config.retrieval.retrieval_method = "dense"
    config.retrieval.backend = "dense"
    config.compression.alpha = 0.7
    config.compression.beta = 0.3
    config.compression.redundancy_threshold = 0.85
    config.compression.token_budget = 300

    print("=" * 80)
    print("COMPREHENSIVE FINAL DIAGNOSTIC EVALUATION")
    print("Configs to evaluate:")
    print("  1. EARC + Ollama (Llama 3.2)")
    print("  2. EARC + Mistral (Ministral 8B)")
    print("  3. Standard RAG + Ollama (Llama 3.2)")
    print("  4. Standard RAG + Mistral (Ministral 8B)")
    print(f"Hyperparameters: Dense top_k=10, alpha=0.7, beta=0.3, tau=0.85, budget=300")
    print("=" * 80)

    # Initialize shared components
    rag_dir = config.retrieval.rag_project_dir
    print("Initializing Dense retriever and Embedding engine...")
    retriever = RAGProjectRetriever(
        rag_dir=rag_dir,
        model_name=config.retrieval.embedding_model,
        device=device,
        method="dense",
    )
    embedding_engine = EmbeddingEngine(config.retrieval.embedding_model, device=device)

    print("Initializing LLM providers...")
    ollama_provider = create_provider(config, "ollama")
    mistral_provider = create_provider(config, "mistral")
    providers = {
        "ollama": ollama_provider,
        "mistral": mistral_provider,
    }

    token_counter = ollama_provider.get_token_counter()
    prompt_builder = PromptBuilder(DEFAULT_QA_TEMPLATE)
    compressor = EvidenceAwareCompressor(
        config.compression, embedding_engine, token_counter, device=device
    )

    datasets = ["nq", "hotpotqa", "triviaqa"]
    split_map = {
        "nq": config.data.nq_split,
        "hotpotqa": config.data.hotpotqa_split,
        "triviaqa": config.data.triviaqa_split,
    }

    if args.smoke_test:
        print("\n--- RUNNING 1-EXAMPLE SMOKE TEST (NQ Question 1) ---")
        loaded_nq = load_dataset("nq", split=split_map["nq"], cache_dir=config.data.cache_dir)
        sampled_nq = sample_dataset(loaded_nq, n=20, seed=42)
        test_example = sampled_nq[0]

        print(f"Example ID: {test_example.id}")
        print(f"Question:   {test_example.question}")
        print(f"Gold Ans:   {test_example.answers}")

        res = evaluate_single_example(
            example=test_example,
            retriever=retriever,
            compressor=compressor,
            prompt_builder=prompt_builder,
            providers=providers,
            top_k=10,
        )

        print("\nSMOKE TEST RESULTS:")
        print("-" * 80)
        print(f"Retrieval Latency: {res['retrieval_latency']:.3f}s | Compression Latency: {res['compression_latency']:.3f}s")
        print(f"Orig Tokens: {res['original_context_tokens']} -> Comp Tokens: {res['compressed_context_tokens']} ({res['compression_percentage']:.1f}% reduction)")
        print(f"Hit@1: {res['hit_at_1']} | Hit@5: {res['hit_at_5']} | Hit@10: {res['hit_at_10']}")
        print("-" * 80)

        for p_name, p_data in res["provider_results"].items():
            print(f"\nProvider: {p_name.upper()} ({p_data['model']})")
            print(f"  EARC Answer:         {p_data['earc']['final_answer']}")
            print(f"  Standard RAG Answer: {p_data['standard_rag']['final_answer']}")
            print(f"  EARC EM/F1:          EM={p_data['earc']['exact_match']} | F1={p_data['earc']['token_f1']:.4f} (Gen Latency: {p_data['earc']['generation_latency']:.2f}s, E2E: {p_data['earc']['end_to_end_latency']:.2f}s)")
            print(f"  Standard RAG EM/F1:  EM={p_data['standard_rag']['exact_match']} | F1={p_data['standard_rag']['token_f1']:.4f} (Gen Latency: {p_data['standard_rag']['generation_latency']:.2f}s, E2E: {p_data['standard_rag']['end_to_end_latency']:.2f}s)")

        print("\n" + "=" * 80)
        print("SMOKE TEST COMPLETE! Ready for full 60-example diagnostic.")
        print("=" * 80)
        return

    # FULL 60-EXAMPLE DIAGNOSTIC RUN
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"outputs/comprehensive_diagnostic_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting FULL 60-Example Diagnostic Run...")
    print(f"Output Directory: {out_dir}")

    all_eval_results = []
    dataset_eval_results = {ds: [] for ds in datasets}
    completed_keys = set()
    raw_file = out_dir / "eval_results_raw.jsonl"
    if raw_file.exists():
        with open(raw_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_keys.add((record["dataset"], record["example_id"]))
                        all_eval_results.append(record)
                        if record["dataset"] in dataset_eval_results:
                            dataset_eval_results[record["dataset"]].append(record)
                    except Exception:
                        pass
        print(f"Resuming from {raw_file}: found {len(completed_keys)} already completed examples.")

    for ds in datasets:
        print(f"\nLoading and sampling 20 examples for {ds.upper()}...")
        loaded = load_dataset(ds, split=split_map[ds], cache_dir=config.data.cache_dir)
        sampled = sample_dataset(loaded, n=20, seed=42)

        print(f"Evaluating {len(sampled)} examples for {ds.upper()} across 4 configurations...")
        ds_t0 = time.time()
        for i, ex in enumerate(sampled, 1):
            if (ds, ex.id) in completed_keys:
                print(f"  [{ds.upper()} {i:02d}/20] ID={ex.id[:16]}... (already completed, skipping)")
                continue

            ex_t0 = time.time()
            res = evaluate_single_example(
                example=ex,
                retriever=retriever,
                compressor=compressor,
                prompt_builder=prompt_builder,
                providers=providers,
                top_k=10,
            )
            dataset_eval_results[ds].append(res)
            all_eval_results.append(res)
            with open(out_dir / "eval_results_raw.jsonl", "a", encoding="utf-8") as f_raw:
                f_raw.write(json.dumps(res) + "\n")
            print(f"  [{ds.upper()} {i:02d}/20] ID={ex.id[:16]}... EARC Ollama={res['provider_results']['ollama']['earc']['exact_match']} Std Ollama={res['provider_results']['ollama']['standard_rag']['exact_match']} | EARC Mistral={res['provider_results']['mistral']['earc']['exact_match']} Std Mistral={res['provider_results']['mistral']['standard_rag']['exact_match']} ({time.time() - ex_t0:.1f}s)")

        ds_duration = time.time() - ds_t0
        print(f"Completed {ds.upper()} in {ds_duration:.1f}s")

    # 1. Flatten into paired records & config records
    paired_records_all = flatten_to_paired_records(all_eval_results)
    config_records_all = flatten_to_config_records(all_eval_results)

    # 2. Save JSONL & CSV for detailed per-example results
    paired_jsonl_path = out_dir / "predictions_paired.jsonl"
    with open(paired_jsonl_path, "w", encoding="utf-8") as f:
        for r in paired_records_all:
            f.write(json.dumps(r) + "\n")

    paired_csv_path = out_dir / "predictions_paired.csv"
    if paired_records_all:
        keys = list(paired_records_all[0].keys())
        with open(paired_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(paired_records_all)

    config_jsonl_path = out_dir / "predictions_all_configs.jsonl"
    with open(config_jsonl_path, "w", encoding="utf-8") as f:
        for r in config_records_all:
            f.write(json.dumps(r) + "\n")

    config_csv_path = out_dir / "predictions_all_configs.csv"
    if config_records_all:
        keys_cfg = list(config_records_all[0].keys())
        with open(config_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys_cfg)
            writer.writeheader()
            writer.writerows(config_records_all)

    # 3. Generate summaries separately for NQ, HotpotQA, TriviaQA
    summaries = {}
    for ds in datasets:
        ds_paired = flatten_to_paired_records(dataset_eval_results[ds])
        summaries[ds] = compute_dataset_summary(ds_paired)
        summary_file = out_dir / f"summary_{ds}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summaries[ds], f, indent=2)

    # 4. Generate combined 60-example summary by aggregating without rerunning
    combined_summary = compute_dataset_summary(paired_records_all)
    with open(out_dir / "summary_combined_60.json", "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)

    # 5. Build and print formatted report
    print("\n" + "=" * 90)
    print("FINAL COMPREHENSIVE DIAGNOSTIC SUMMARY")
    print(f"Output Directory: {out_dir}")
    print("=" * 90)

    report_lines = []
    report_lines.append("# Final Comprehensive Diagnostic Evaluation Report\n")
    report_lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Output Directory**: `{out_dir}`\n")
    report_lines.append("## Configuration")
    report_lines.append("- **Retrieval**: Dense (`sentence-transformers/all-MiniLM-L6-v2`), Top-K=10")
    report_lines.append("- **EARC**: alpha=0.7, beta=0.3, redundancy_threshold=0.85, token_budget=300")
    report_lines.append("- **Standard RAG**: Exact same Top-K=10 retrieved documents, uncompressed")
    report_lines.append("- **Models**: Ollama (`llama3.2`), Mistral (`ministral-8b-latest`)\n")

    for ds in datasets + ["combined"]:
        ds_name = "COMBINED (60 Examples)" if ds == "combined" else ds.upper() + " (20 Examples)"
        sum_data = combined_summary if ds == "combined" else summaries[ds]

        print(f"\n--- {ds_name} ---")
        header = f"{'Configuration':<32} | {'EM (%)':<8} | {'F1 (%)':<8} | {'Comp %':<8} | {'Tok In':<8} | {'Gen Lat':<8} | {'E2E Lat':<8}"
        print(header)
        print("-" * len(header))

        for prov in ["ollama", "mistral"]:
            p_res = sum_data[prov]
            model_tag = f"{prov.capitalize()} ({p_res['model']})"
            
            # EARC
            earc_em = p_res["earc"]["exact_match"]
            earc_f1 = p_res["earc"]["f1"]
            comp_pct = p_res["compression"]["avg_compression_percentage"]
            comp_tok = p_res["compression"]["avg_compressed_tokens"]
            earc_gen_lat = p_res["earc"]["avg_generation_latency"]
            earc_e2e_lat = p_res["earc"]["avg_end_to_end_latency"]
            print(f"EARC + {model_tag:<25} | {earc_em:<8.1f} | {earc_f1:<8.1f} | {comp_pct:<8.1f} | {comp_tok:<8.0f} | {earc_gen_lat:<8.2f} | {earc_e2e_lat:<8.2f}")

            # Standard RAG
            std_em = p_res["standard_rag"]["exact_match"]
            std_f1 = p_res["standard_rag"]["f1"]
            orig_tok = p_res["compression"]["avg_original_tokens"]
            std_gen_lat = p_res["standard_rag"]["avg_generation_latency"]
            std_e2e_lat = p_res["standard_rag"]["avg_end_to_end_latency"]
            print(f"Standard RAG + {model_tag:<17} | {std_em:<8.1f} | {std_f1:<8.1f} | {'0.0%':<8} | {orig_tok:<8.0f} | {std_gen_lat:<8.2f} | {std_e2e_lat:<8.2f}")

    print("\n" + "=" * 90)
    print("Files Saved:")
    print(f"  - Paired CSV:   {paired_csv_path}")
    print(f"  - Paired JSONL: {paired_jsonl_path}")
    print(f"  - Configs CSV:  {config_csv_path}")
    print(f"  - Configs JSONL:{config_jsonl_path}")
    print(f"  - Summary NQ:   {out_dir / 'summary_nq.json'}")
    print(f"  - Summary Hotpot:{out_dir / 'summary_hotpotqa.json'}")
    print(f"  - Summary Trivia:{out_dir / 'summary_triviaqa.json'}")
    print(f"  - Combined 60:  {out_dir / 'summary_combined_60.json'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
