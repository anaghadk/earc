"""
Final Comparative Evaluation Orchestrator for EARC.

Compares:
1. Standard RAG (uncompressed context, all retrieved passages)
2. Top-k passages (no EARC compression, top passages from retrieved set)
3. LLMLingua-2 (learned token classification compression)
4. Proposed EARC method (evidence-aware sentence compression)

Across 3 datasets:
- NQ (1000 examples)
- HotpotQA (1000 examples)
- TriviaQA (1000 examples)

Across 2 LLMs:
- Ollama (llama3.2)
- Mistral (ministral-8b-latest)

Locked Settings:
- Dense retrieval
- top_k = 10
- embedding = sentence-transformers/all-MiniLM-L6-v2
- token budget = 300 for EARC & LLMLingua-2
- alpha = 0.7, beta = 0.3, redundancy_threshold = 0.85
- temperature = 0
- seed = 42 for deterministic reproducible sampling
- Checkpoint/resume enabled via predictions.jsonl
"""

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
#added by anjana
from src.data.schemas import ExperimentResult

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = "lIK71q2MhMrZXQwxYE0nVUvoNNUVjQzy"

import torch
import numpy as np

from src.config.settings import load_config
from src.data.loaders import load_dataset
from src.data.sampling import sample_dataset
from src.data.schemas import QAExample, RetrievedDocument
from src.retrieval.rag_project import RAGProjectRetriever
from src.compression.embeddings import EmbeddingEngine
from src.compression.compressor import EvidenceAwareCompressor
from src.baselines.llmlingua2 import LLMLingua2Baseline
from src.baselines.topk import TopKBaseline
from src.baselines.standard_rag import StandardRAGBaseline
from src.prompting.builder import PromptBuilder
from src.prompting.templates import DEFAULT_QA_TEMPLATE
from src.llm.factory import create_provider
from src.evaluation.exact_match import exact_match_score
from src.evaluation.f1 import max_token_f1_score
from src.evaluation.answer_extraction import extract_answer
from src.evaluation.aggregation import compute_retrieval_hit
from src.utils.device import DeviceManager
from src.utils.logging import setup_logging, get_logger
from src.utils.io import ensure_dir, save_json

logger = get_logger("final_evaluation")


def get_gpu_memory_mb() -> float:
    """Returns peak GPU memory allocated in MB if CUDA is available, else 0.0."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def safe_generate(provider: Any, prompt: str, max_retries: int = 5) -> Any:
    """Invokes provider.generate with exponential backoff on transient failures."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return provider.generate(prompt)
        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                logger.error(f"Generation permanently failed after {max_retries} attempts: {e}")
                raise
            delay = (2 ** attempt) + 1.0
            logger.warning(f"Generation error ({e}). Retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(delay)
    raise last_err


def load_checkpoint_completed_ids(predictions_file: Path) -> set:
    """Read already completed example_ids from predictions.jsonl for resuming."""
    completed = set()
    if predictions_file.exists():
        with open(predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if line_s:
                    try:
                        data = json.loads(line_s)
                        if "example_id" in data:
                            completed.add(data["example_id"])
                    except Exception:
                        pass
    return completed


def compute_metrics_from_predictions(predictions_file: Path, dataset_name: str, provider_name: str, model_name: str, method_name: str) -> Dict[str, Any]:
    """Compute aggregate metrics from predictions.jsonl file."""
    if not predictions_file.exists():
        return {}

    records = []
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if line_s:
                try:
                    records.append(json.loads(line_s))
                except Exception:
                    pass

    if not records:
        return {}

    n = len(records)
    ems = [r.get("exact_match", 0.0) for r in records]
    f1s = [r.get("f1", 0.0) for r in records]
    h1s = [r.get("answer_hit_at_1", 0.0) for r in records]
    h5s = [r.get("answer_hit_at_5", 0.0) for r in records]
    h10s = [r.get("answer_hit_at_10", 0.0) for r in records]
    orig_toks = [r.get("original_tokens", 0) for r in records if isinstance(r.get("original_tokens"), (int, float))]
    proc_toks = [r.get("processed_tokens", 0) for r in records if isinstance(r.get("processed_tokens"), (int, float))]
    ret_lats = [r.get("retrieval_latency", 0.0) for r in records]
    gen_lats = [r.get("generation_latency", 0.0) for r in records]
    e2e_lats = [r.get("end_to_end_latency", 0.0) for r in records]
    peak_gpus = [r.get("peak_gpu_memory_mb", 0.0) for r in records if isinstance(r.get("peak_gpu_memory_mb"), (int, float))]

    comp_app = records[0].get("compression_applicable", False)

    if comp_app:
        comp_toks = [r.get("compressed_tokens", 0) for r in records if isinstance(r.get("compressed_tokens"), (int, float))]
        comp_pcts = [r.get("compression_percentage", 0.0) for r in records if isinstance(r.get("compression_percentage"), (int, float))]
        comp_rats = [r.get("compression_ratio", 0.0) for r in records if isinstance(r.get("compression_ratio"), (int, float))]
        comp_lats = [r.get("compression_latency", 0.0) for r in records if isinstance(r.get("compression_latency"), (int, float))]

        avg_comp_tokens = round(float(np.mean(comp_toks)), 2) if comp_toks else "N/A"
        avg_comp_pct = round(float(np.mean(comp_pcts)), 2) if comp_pcts else "N/A"
        avg_comp_ratio = round(float(np.mean(comp_rats)), 2) if comp_rats else "N/A"
        avg_comp_latency = round(float(np.mean(comp_lats)), 4) if comp_lats else "N/A"
    else:
        avg_comp_tokens = "N/A"
        avg_comp_pct = "N/A"
        avg_comp_ratio = "N/A"
        avg_comp_latency = "N/A"

    metrics = {
        "dataset": dataset_name,
        "provider": provider_name,
        "model": model_name,
        "method": method_name,
        "n_examples": n,
        "exact_match": round(float(np.mean(ems)) * 100.0, 2),
        "token_f1": round(float(np.mean(f1s)) * 100.0, 2),
        "answer_hit_at_1": round(float(np.mean(h1s)), 4),
        "answer_hit_at_5": round(float(np.mean(h5s)), 4),
        "answer_hit_at_10": round(float(np.mean(h10s)), 4),
        "avg_original_tokens": round(float(np.mean(orig_toks)), 1) if orig_toks else 0.0,
        "avg_processed_tokens": round(float(np.mean(proc_toks)), 1) if proc_toks else 0.0,
        "avg_compressed_tokens": avg_comp_tokens,
        "avg_compression_percentage": avg_comp_pct,
        "avg_compression_ratio": avg_comp_ratio,
        "compression_applicable": comp_app,
        "avg_retrieval_latency": round(float(np.mean(ret_lats)), 4),
        "avg_compression_latency": avg_comp_latency,
        "avg_generation_latency": round(float(np.mean(gen_lats)), 4),
        "avg_end_to_end_latency": round(float(np.mean(e2e_lats)), 4),
        "peak_gpu_memory_mb": round(float(np.max(peak_gpus)), 2) if peak_gpus else 0.0,
    }
    return metrics


def write_summary_files(all_metrics: Dict[str, Dict[str, Dict[str, Any]]], output_dir: Path):
    """Write FINAL_COMPARISON.json and FINAL_COMPARISON.csv."""
    json_path = output_dir / "FINAL_COMPARISON.json"
    save_json(json_path, all_metrics)

    csv_path = output_dir / "FINAL_COMPARISON.csv"
    rows = []
    for provider, p_datasets in all_metrics.items():
        for dataset, d_methods in p_datasets.items():
            for method, m_data in d_methods.items():
                if not m_data:
                    continue
                rows.append({
                    "Provider": provider,
                    "Model": m_data.get("model", ""),
                    "Dataset": dataset,
                    "Method": method,
                    "N_Examples": m_data.get("n_examples", 0),
                    "Exact_Match_Pct": m_data.get("exact_match", 0.0),
                    "Token_F1_Pct": m_data.get("token_f1", 0.0),
                    "Hit_At_1": m_data.get("answer_hit_at_1", 0.0),
                    "Hit_At_5": m_data.get("answer_hit_at_5", 0.0),
                    "Hit_At_10": m_data.get("answer_hit_at_10", 0.0),
                    "Avg_Orig_Tokens": m_data.get("avg_original_tokens", 0.0),
                    "Avg_Processed_Tokens": m_data.get("avg_processed_tokens", 0.0),
                    "Avg_Comp_Pct": m_data.get("avg_compression_percentage", "N/A"),
                    "Avg_Comp_Ratio": m_data.get("avg_compression_ratio", "N/A"),
                    "Avg_Retrieval_Latency_s": m_data.get("avg_retrieval_latency", 0.0),
                    "Avg_Compression_Latency_s": m_data.get("avg_compression_latency", "N/A"),
                    "Avg_Generation_Latency_s": m_data.get("avg_generation_latency", 0.0),
                    "Avg_E2E_Latency_s": m_data.get("avg_end_to_end_latency", 0.0),
                    "Peak_GPU_MB": m_data.get("peak_gpu_memory_mb", 0.0),
                })

    if rows:
        headers = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Updated summary CSV at {csv_path}")


def run_evaluation(
    limit: int = 1000,
    datasets_to_run: Optional[List[str]] = None,
    providers_to_run: Optional[List[str]] = None,
    methods_to_run: Optional[List[str]] = None,
    config_path: str = "configs/default.yaml",
    output_dir_str: str = "outputs/final_evaluation",
    seed: int = 42,
    smoke_test: bool = False,
):
    setup_logging("INFO")
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)

    # Lock fixed evaluation settings
    config.retrieval.backend = "dense"
    config.retrieval.retrieval_method = "dense"
    config.retrieval.top_k = 10
    config.retrieval.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.compression.token_budget = 300
    config.compression.alpha = 0.7
    config.compression.beta = 0.3
    config.compression.redundancy_threshold = 0.85
    config.ollama.model = "llama3.2"
    config.ollama.temperature = 0.0
    config.mistral.model = "ministral-8b-latest"
    config.mistral.temperature = 0.0

    target_n = 1 if smoke_test else limit

    datasets = datasets_to_run or ["nq", "hotpot", "trivia"]
    dataset_code_map = {
        "nq": "nq",
        "hotpot": "hotpotqa",
        "trivia": "triviaqa",
    }
    providers = providers_to_run or ["ollama", "mistral"]
    methods = methods_to_run or ["standard_rag", "topk", "llmlingua2", "earc"]

    logger.info("=" * 80)
    logger.info("FINAL COMPARATIVE EVALUATION")
    logger.info(f"Target: {target_n} examples per dataset across {datasets}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Providers: {providers}")
    logger.info(f"Settings: Dense top_k=10, token_budget=300, alpha=0.7, beta=0.3, tau=0.85, seed={seed}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("=" * 80)

    device = DeviceManager().get_device().type
    rag_dir = config.retrieval.rag_project_dir or "RAG_Project"

    # Initialize shared components
    logger.info("Loading shared Dense Retriever & Embedding Engine on CUDA...")
    retriever = RAGProjectRetriever(
        rag_dir=rag_dir,
        model_name=config.retrieval.embedding_model,
        device=device,
        method="dense",
    )
    embedding_engine = EmbeddingEngine(config.retrieval.embedding_model, device=device)

    # Initialize providers
    logger.info("Initializing LLM providers...")
    llm_instances = {}
    for p_name in providers:
        llm_instances[p_name] = create_provider(config, p_name)

    token_counter = llm_instances[providers[0]].get_token_counter()
    prompt_builder = PromptBuilder(DEFAULT_QA_TEMPLATE)

    # Initialize compression / baseline engines
    logger.info("Initializing EARC compressor (budget=300, alpha=0.7, beta=0.3, tau=0.85)...")
    earc_compressor = EvidenceAwareCompressor(
        config.compression, embedding_engine, token_counter, device=device
    )

    standard_rag_baseline = StandardRAGBaseline(prompt_builder)
    topk_baseline = TopKBaseline(prompt_builder, k=5)

    llmlingua2_baseline = None
    if "llmlingua2" in methods:
        logger.info("Initializing LLMLingua-2 baseline on CUDA...")
        try:
            llmlingua2_baseline = LLMLingua2Baseline(prompt_builder, device=device, target_token=300)
            logger.info("LLMLingua-2 baseline successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize LLMLingua-2: {e}")
            raise

    # 1. Sample datasets and save selected example IDs
    sampled_examples_by_dataset: Dict[str, List[QAExample]] = {}
    selected_ids_record: Dict[str, Any] = {"seed": seed}

    ids_file = output_dir / "selected_example_ids.json"
    existing_ids = {}
    if ids_file.exists():
        try:
            with open(ids_file, "r", encoding="utf-8") as f:
                existing_ids = json.load(f)
        except Exception:
            pass

    for ds_key in datasets:
        ds_load_name = dataset_code_map.get(ds_key, ds_key)
        logger.info(f"Loading full dataset for {ds_key.upper()} ({ds_load_name})...")
        full_examples = load_dataset(ds_load_name)
        id_to_example = {ex.id: ex for ex in full_examples}

        # Check if IDs were previously locked for this dataset
        if ds_key in existing_ids and len(existing_ids[ds_key]) >= target_n:
            chosen_ids = existing_ids[ds_key][:target_n]
            sampled = [id_to_example[eid] for eid in chosen_ids if eid in id_to_example]
            if len(sampled) == target_n:
                logger.info(f"Loaded {len(sampled)} previously locked example IDs for {ds_key}")
                sampled_examples_by_dataset[ds_key] = sampled
                selected_ids_record[ds_key] = chosen_ids
                continue

        logger.info(f"Sampling {target_n} examples with seed={seed} for {ds_key}...")
        sampled = sample_dataset(full_examples, n=target_n, seed=seed)
        sampled_examples_by_dataset[ds_key] = sampled
        selected_ids_record[ds_key] = [ex.id for ex in sampled]

    save_json(ids_file, selected_ids_record)
    logger.info(f"Saved reproducible example IDs to {ids_file}")

    # Initialize results container
    all_summary_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {
        p: {d: {} for d in datasets} for p in providers
    }

    # Main evaluation loop: dataset -> example -> shared retrieval/compression -> providers x methods
    for ds_key in datasets:
        dataset_examples = sampled_examples_by_dataset[ds_key]
        logger.info(f"\n=======================================================")
        logger.info(f"Processing Dataset: {ds_key.upper()} ({len(dataset_examples)} examples)")
        logger.info(f"=======================================================")

        # Prepare checkpoint sets and file handles for all provider x method combinations
        pred_files: Dict[str, Dict[str, Path]] = {}
        completed_ids: Dict[str, Dict[str, set]] = {}

        for p_name in providers:
            pred_files[p_name] = {}
            completed_ids[p_name] = {}
            for m_name in methods:
                m_dir = output_dir / p_name / ds_key / m_name
                m_dir.mkdir(parents=True, exist_ok=True)
                p_file = m_dir / "predictions.jsonl"
                pred_files[p_name][m_name] = p_file
                completed_ids[p_name][m_name] = load_checkpoint_completed_ids(p_file)

        for ex_idx, ex in enumerate(dataset_examples, 1):
            # Check if all provider x method combinations have already completed this example
            all_done = True
            for p_name in providers:
                for m_name in methods:
                    if ex.id not in completed_ids[p_name][m_name]:
                        all_done = False
                        break
                if not all_done:
                    break

            if all_done:
                if ex_idx % 50 == 0 or ex_idx == len(dataset_examples):
                    logger.info(f"[{ds_key.upper()}] Example {ex_idx}/{len(dataset_examples)} already completed across all methods.")
                continue

            t_ex_start = time.time()

            # 1. Retrieval (executed once per example)
            t_ret0 = time.time()
            retrieved_docs = retriever.retrieve(ex.question, top_k=10)
            retrieval_latency = time.time() - t_ret0

            # Compute answer hits
            docs_dict = [d.to_dict() if hasattr(d, "to_dict") else d for d in retrieved_docs]
            hit_1 = compute_retrieval_hit(docs_dict, ex.answers, k=1)
            hit_5 = compute_retrieval_hit(docs_dict, ex.answers, k=5)
            hit_10 = compute_retrieval_hit(docs_dict, ex.answers, k=10)

            # 2. Context Building & Compression (executed once per example)
            contexts: Dict[str, Dict[str, Any]] = {}

            # Standard RAG
            if "standard_rag" in methods:
                t_sr0 = time.time()
                sr_res = standard_rag_baseline.run(ex.question, retrieved_docs, token_counter)
                contexts["standard_rag"] = {
                    "text": sr_res["compressed_text"],
                    "original_tokens": sr_res["original_tokens"],
                    "processed_tokens": sr_res["compressed_tokens"],
                    "compressed_tokens": "N/A",
                    "compression_percentage": "N/A",
                    "compression_ratio": "N/A",
                    "compression_applicable": False,
                    "compression_latency": "N/A",
                }

            # Top-k Passages
            if "topk" in methods:
                topk_res = topk_baseline.run(ex.question, retrieved_docs, token_counter)
                contexts["topk"] = {
                    "text": topk_res["compressed_text"],
                    "original_tokens": topk_res["original_tokens"],
                    "processed_tokens": topk_res["compressed_tokens"],
                    "compressed_tokens": "N/A",
                    "compression_percentage": "N/A",
                    "compression_ratio": "N/A",
                    "compression_applicable": False,
                    "compression_latency": "N/A",
                }

            # LLMLingua-2
            if "llmlingua2" in methods and llmlingua2_baseline:
                lingua_res = llmlingua2_baseline.run(ex.question, retrieved_docs, token_counter, budget=300)
                contexts["llmlingua2"] = {
                    "text": lingua_res["compressed_text"],
                    "original_tokens": lingua_res["original_tokens"],
                    "processed_tokens": lingua_res["compressed_tokens"],
                    "compressed_tokens": lingua_res["compressed_tokens"],
                    "compression_percentage": round(lingua_res["reduction_percentage"] * 100.0, 2),
                    "compression_ratio": round(lingua_res["compression_ratio"], 2),
                    "compression_applicable": True,
                    "compression_latency": round(lingua_res["compression_latency"], 4),
                }

            # Proposed EARC
            if "earc" in methods:
                t_earc0 = time.time()
                # Should already look like this after previous change
                earc_comp = earc_compressor.compress(ex.question, retrieved_docs, dataset=ds_key)
                earc_lat = time.time() - t_earc0
                orig_t = earc_comp.original_tokens
                comp_t = earc_comp.compressed_tokens
                ratio = round(orig_t / comp_t, 2) if comp_t > 0 else 1.0
                contexts["earc"] = {
                    "text": earc_comp.compressed_text,
                    "original_tokens": orig_t,
                    "processed_tokens": comp_t,
                    "compressed_tokens": comp_t,
                    "compression_percentage": round(earc_comp.reduction_percentage, 2),
                    "compression_ratio": ratio,
                    "compression_applicable": True,
                    "compression_latency": round(earc_lat, 4),
                }

            # 3. LLM Generation and Evaluation across Providers & Methods
            for p_name in providers:
                llm = llm_instances[p_name]
                model_name = llm.model_name

                for m_name in methods:
                    if ex.id in completed_ids[p_name][m_name]:
                        continue

                    ctx_data = contexts[m_name]
                    prompt = prompt_builder.build(ex.question, ctx_data["text"])

                    t_gen0 = time.time()
                    gen_result = safe_generate(llm, prompt)
                    gen_latency = time.time() - t_gen0

                    ans = extract_answer(gen_result.text) or gen_result.text.strip()
                    em = exact_match_score(ans, ex.answers)
                    f1 = max_token_f1_score(ans, ex.answers)["f1"]

                    comp_lat_val = ctx_data["compression_latency"] if ctx_data["compression_applicable"] else 0.0
                    e2e_lat = retrieval_latency + comp_lat_val + gen_latency
                    gpu_mb = get_gpu_memory_mb()

                    rec = {
                        "example_id": ex.id,
                        "dataset": ds_key,
                        "question": ex.question,
                        "gold_answers": ex.answers,
                        "prediction": gen_result.text,
                        "evaluated_answer": ans,
                        "exact_match": em,
                        "f1": round(f1, 4),
                        "answer_hit_at_1": hit_1,
                        "answer_hit_at_5": hit_5,
                        "answer_hit_at_10": hit_10,
                        "original_tokens": ctx_data["original_tokens"],
                        "processed_tokens": ctx_data["processed_tokens"],
                        "compressed_tokens": ctx_data["compressed_tokens"],
                        "compression_percentage": ctx_data["compression_percentage"],
                        "compression_ratio": ctx_data["compression_ratio"],
                        "compression_applicable": ctx_data["compression_applicable"],
                        "retrieval_latency": round(retrieval_latency, 4),
                        "compression_latency": ctx_data["compression_latency"],
                        "generation_latency": round(gen_latency, 4),
                        "end_to_end_latency": round(e2e_lat, 4),
                        "peak_gpu_memory_mb": round(gpu_mb, 2),
                        "llm_provider": p_name,
                        "llm_model": model_name,
                        "method": m_name,
                    }

                    p_file = pred_files[p_name][m_name]
                    with open(p_file, "a", encoding="utf-8") as f_out:
                        f_out.write(json.dumps(rec) + "\n")
                    completed_ids[p_name][m_name].add(ex.id)

            if ex_idx % 10 == 0 or ex_idx == len(dataset_examples) or smoke_test:
                logger.info(
                    f"[{ds_key.upper()} {ex_idx:04d}/{len(dataset_examples)}] "
                    f"Q: '{ex.question[:35]}...' | Total Ex Latency: {time.time() - t_ex_start:.2f}s"
                )

        # Compute dataset metrics after completing dataset
        logger.info(f"\nComputing aggregate metrics for {ds_key.upper()}...")
        for p_name in providers:
            llm = llm_instances[p_name]
            model_name = llm.model_name
            for m_name in methods:
                p_file = pred_files[p_name][m_name]
                m_dir = output_dir / p_name / ds_key / m_name
                metrics_res = compute_metrics_from_predictions(p_file, ds_key, p_name, model_name, m_name)
                save_json(m_dir / "metrics.json", metrics_res)
                all_summary_metrics[p_name][ds_key][m_name] = metrics_res
                logger.info(
                    f"  [{p_name.upper()} | {ds_key.upper()} | {m_name}] "
                    f"N={metrics_res.get('n_examples')} EM={metrics_res.get('exact_match')}% "
                    f"F1={metrics_res.get('token_f1')}% E2E={metrics_res.get('avg_end_to_end_latency')}s"
                )

        # Update running summary files after each dataset completes
        write_summary_files(all_summary_metrics, output_dir)

    # Final summary update
    #significance testing
    # ── Significance tests ────────────────────────────────────────────────
    logger.info("\nRunning paired bootstrap significance tests...")

    from src.evaluation.aggregation import run_all_significance_tests
    from src.data.schemas import ExperimentResult
    def _load_results_from_jsonl(predictions_file: Path, method: str) -> list:
        """Load predictions.jsonl into ExperimentResult objects matching actual schema."""
        results = []
        if not predictions_file.exists():
            return results
        with open(predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    results.append(ExperimentResult(
                    example_id=r.get("example_id", ""),
                    dataset=r.get("dataset", ""),
                    question=r.get("question", ""),
                    gold_answers=r.get("gold_answers", []),
                    retrieved_documents=r.get("retrieved_documents", []),
                candidate_count=r.get("candidate_count", 0),
                    selected_sentences=r.get("selected_sentences", []),
                    original_tokens=r.get("original_tokens", 0),
                    compressed_tokens=r.get("compressed_tokens") or r.get("processed_tokens", 0),
                    compression_percentage=r.get("compression_percentage") or 0.0,
                    alpha=r.get("alpha", 0.7),
                    beta=r.get("beta", 0.3),
                    redundancy_threshold=r.get("redundancy_threshold", 0.85),
                    token_budget=r.get("token_budget", 300),
                    llm_provider=r.get("llm_provider", ""),
                    llm_model=r.get("llm_model", ""),
                    prediction=r.get("prediction", ""),
                    exact_match=r.get("exact_match", 0.0),
                    f1=r.get("f1", 0.0),
                    retrieval_latency=r.get("retrieval_latency", 0.0),
                    compression_latency=r.get("compression_latency") or 0.0,
                    generation_latency=r.get("generation_latency", 0.0),
                    end_to_end_latency=r.get("end_to_end_latency", 0.0),
                    peak_gpu_memory_mb=r.get("peak_gpu_memory_mb", 0.0),
                    method=method,
                    retriever_type=r.get("retriever_type", "dense"),
                    seed=r.get("seed", 42),
                    evaluated_answer=r.get("evaluated_answer", ""),
                    compressed_context_text=r.get("compressed_context_text", ""),
                ))
                except Exception as e:
                    logger.warning(f"Skipping malformed record: {e}")
        return results
    sig_results = {}
    baselines_to_test = ["standard_rag", "topk", "llmlingua2"]

    for p_name in providers:
        sig_results[p_name] = {}
        for ds_key in datasets:
            # Load EARC results for this provider + dataset
            earc_file = output_dir / p_name / ds_key / "earc" / "predictions.jsonl"
            earc_res = _load_results_from_jsonl(earc_file, "earc")
            if not earc_res:
                logger.warning(f"No EARC results for {p_name}/{ds_key} — skipping sig tests")
                continue

            # Load each baseline
            baseline_res_map = {}
            for b_name in baselines_to_test:
                if b_name not in methods:
                    continue
                b_file = output_dir / p_name / ds_key / b_name / "predictions.jsonl"
                b_res = _load_results_from_jsonl(b_file, b_name)
                if b_res:
                    baseline_res_map[b_name] = b_res

            if not baseline_res_map:
                continue

            # Run bootstrap tests
            tests = run_all_significance_tests(
                earc_results=earc_res,
                baseline_results=baseline_res_map,
                n_resamples=5000,
                seed=seed,
            )
            sig_results[p_name][ds_key] = tests

            # Log results
            for baseline_name, ds_tests in tests.items():
                for dataset_name, metric_tests in ds_tests.items():
                    f1_test = metric_tests.get("f1", {})
                    logger.info(
                        f"  Bootstrap [{p_name.upper()} | {ds_key.upper()} | "
                        f"EARC vs {baseline_name}] "
                        f"ΔF1={f1_test.get('observed_delta', 0):+.4f} "
                        f"p={f1_test.get('p_value', 1):.4f} "
                        f"{'✓ sig' if f1_test.get('significant') else '✗ n.s.'} "
                        f"95%CI=[{f1_test.get('ci_lower', 0):.4f}, {f1_test.get('ci_upper', 0):.4f}]"
                    )

    # Save significance test results
    sig_path = output_dir / "SIGNIFICANCE_TESTS.json"
    save_json(sig_path, sig_results)
    logger.info(f"Significance tests saved to {sig_path}")

    # Final summary update
    write_summary_files(all_summary_metrics, output_dir)
    logger.info("=" * 80)
    logger.info(f"FINAL COMPARATIVE EVALUATION COMPLETE.")
    logger.info(f"Significance tests: {sig_path}")
    
    logger.info(f"Results saved in {output_dir}")
    logger.info(f"Summary JSON: {output_dir / 'FINAL_COMPARISON.json'}")
    logger.info(f"Summary CSV:  {output_dir / 'FINAL_COMPARISON.csv'}")
    logger.info("=" * 80)
    return all_summary_metrics


def main():
    parser = argparse.ArgumentParser(description="Run Final Comparative Evaluation")
    parser.add_argument("--limit", type=int, default=1000, help="Number of examples per dataset (default: 1000)")
    parser.add_argument("--datasets", nargs="+", default=["nq", "hotpot", "trivia"], help="Datasets to evaluate")
    parser.add_argument("--providers", nargs="+", default=["ollama", "mistral"], help="Providers to evaluate")
    parser.add_argument("--methods", nargs="+", default=["standard_rag", "topk", "llmlingua2", "earc"], help="Methods to compare")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="outputs/final_evaluation", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 example smoke test")
    args = parser.parse_args()

    run_evaluation(
        limit=args.limit,
        datasets_to_run=args.datasets,
        providers_to_run=args.providers,
        methods_to_run=args.methods,
        config_path=args.config,
        output_dir_str=args.output_dir,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
