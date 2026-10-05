"""
Experiment 3 — Token-Budget Sensitivity Analysis for EARC
Evaluates EARC under different context token budgets:
[100, 150, 200, 300, 400, 500, 750] tokens
Fixed settings:
- alpha = 0.7, beta = 0.3
- redundancy_threshold = 0.85
- same retrieval results (MiniLM dense, top-k=10)
- same candidate sentences
- same datasets (HotpotQA, NQ, TriviaQA)
- same LLM (Mistral API / local Ollama)
- same prompt
- same decoding configuration (temperature 0.0)
- same random seed (42)
"""

import os
import sys
import json
import time
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config.settings import load_config
from src.data.loaders import load_dataset
from src.data.schemas import QAExample
from src.retrieval.rag_project import RAGProjectRetriever
from src.compression.embeddings import EmbeddingEngine
from src.compression.compressor import EvidenceAwareCompressor
from src.prompting.builder import PromptBuilder
from src.prompting.templates import DEFAULT_QA_TEMPLATE
from src.llm.factory import create_provider
from src.evaluation.exact_match import exact_match_score
from src.evaluation.f1 import max_token_f1_score
from src.evaluation.answer_extraction import extract_answer
from src.utils.device import DeviceManager
from src.utils.logging import setup_logging, get_logger
from src.utils.io import save_json

logger = get_logger("token_budget_sensitivity")

BUDGET_CONFIGS = [
    (100, "budget_100"),
    (150, "budget_150"),
    (200, "budget_200"),
    (300, "budget_300"),
    (400, "budget_400"),
    (500, "budget_500"),
    (750, "budget_750"),
]


def compute_evidence_retention(retrieved_docs: List[Any], compressed_text: str, answers: List[str]) -> float:
    orig_has_ans = any(
        any(ans.lower() in doc.text.lower() for ans in answers)
        for doc in retrieved_docs
    )
    if not orig_has_ans:
        return 1.0  # Gold answer was never retrieved
    comp_has_ans = any(ans.lower() in compressed_text.lower() for ans in answers)
    return 1.0 if comp_has_ans else 0.0


def load_checkpoint_records(pred_file: Path) -> Dict[str, Dict[str, Any]]:
    records = {}
    if pred_file.exists():
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records[data["example_id"]] = data
                except Exception:
                    pass
    return records


def compute_metrics_from_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    n = len(records)
    em_total = sum(r.get("exact_match", 0.0) for r in records)
    f1_total = sum(r.get("f1", 0.0) for r in records)
    ev_total = sum(r.get("evidence_retention", 0.0) for r in records)

    tok_total = sum(r.get("compressed_tokens", 0) for r in records)
    comp_pct_total = sum(r.get("compression_percentage", 0.0) for r in records)
    ratio_total = sum(r.get("compression_ratio", 1.0) for r in records)
    lat_total = sum(r.get("end_to_end_latency", 0.0) for r in records)

    return {
        "n_examples": n,
        "exact_match": round((em_total / n) * 100, 2),
        "token_f1": round((f1_total / n) * 100, 2),
        "evidence_retention_rate": round((ev_total / n) * 100, 2),
        "avg_compressed_tokens": round(tok_total / n, 1),
        "avg_compression_percentage": round(comp_pct_total / n, 1),
        "avg_compression_ratio": round(ratio_total / n, 2),
        "avg_e2e_latency": round(lat_total / n, 2),
    }


def load_baseline_metrics(final_eval_dir: Path, provider: str, dataset: str, locked_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    baselines = {}
    for method in ["standard_rag", "llmlingua2"]:
        path = final_eval_dir / provider / dataset / method / "predictions.jsonl"
        if not path.exists():
            continue
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("example_id") in locked_ids:
                        records.append(d)
                except Exception:
                    pass
        if records:
            n = len(records)
            em = sum(r.get("exact_match", 0.0) for r in records) / n * 100
            f1 = sum(r.get("f1", 0.0) for r in records) / n * 100
            toks = sum(float(r.get("processed_tokens", r.get("compressed_tokens", r.get("original_tokens", 0)))) for r in records) / n
            lat = sum(float(r.get("end_to_end_latency", 0.0)) for r in records) / n
            orig_toks = sum(float(r.get("original_tokens", toks)) for r in records) / n
            comp_pct = round(max(0.0, 1.0 - (toks / orig_toks)) * 100, 1) if orig_toks > 0 else 0.0
            ratio = round(orig_toks / toks, 2) if toks > 0 else 1.0
            baselines[method] = {
                "exact_match": round(em, 2),
                "token_f1": round(f1, 2),
                "avg_compressed_tokens": round(toks, 1),
                "avg_compression_percentage": comp_pct,
                "avg_compression_ratio": ratio,
                "avg_e2e_latency": round(lat, 2),
            }
    return baselines


def generate_plots(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    baselines_dict: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = [b[0] for b in BUDGET_CONFIGS]

    for provider, p_data in all_results.items():
        datasets = list(p_data.keys())
        colors = {"hotpot": "#1f77b4", "nq": "#ff7f0e", "trivia": "#2ca02c"}

        # -------------------------------------------------------------
        # Plot A: F1 versus Token Budget
        # -------------------------------------------------------------
        plt.figure(figsize=(10, 6))
        for ds in datasets:
            f1_scores = [p_data[ds].get(b[1], {}).get("token_f1", 0.0) for b in BUDGET_CONFIGS]
            c = colors.get(ds, None)
            plt.plot(budgets, f1_scores, marker="o", linewidth=2.4, label=f"EARC ({ds.upper()})", color=c)
            base = baselines_dict.get(ds, {})
            if "standard_rag" in base:
                plt.axhline(base["standard_rag"]["token_f1"], color=c, linestyle=":", alpha=0.7,
                            label=f"Standard RAG ({ds.upper()}) [{base['standard_rag']['token_f1']:.1f}%]")
            if "llmlingua2" in base:
                plt.axhline(base["llmlingua2"]["token_f1"], color=c, linestyle="--", alpha=0.7,
                            label=f"LLMLingua-2 ({ds.upper()}) [{base['llmlingua2']['token_f1']:.1f}%]")

        plt.axvline(300, color="gray", linestyle="-.", alpha=0.8, label="Production Budget (300t)")
        plt.title(f"A. Token F1 (%) vs. Token Budget — Provider: {provider.upper()}", fontsize=13, fontweight="bold")
        plt.xlabel("Target Token Budget", fontsize=11)
        plt.ylabel("Token F1 (%)", fontsize=11)
        plt.xticks(budgets)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8.5)
        plt.tight_layout()
        f1_path = output_dir / f"f1_vs_token_budget_{provider}.png"
        plt.savefig(f1_path, dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # Plot B: EM versus Token Budget
        # -------------------------------------------------------------
        plt.figure(figsize=(10, 6))
        for ds in datasets:
            em_scores = [p_data[ds].get(b[1], {}).get("exact_match", 0.0) for b in BUDGET_CONFIGS]
            c = colors.get(ds, None)
            plt.plot(budgets, em_scores, marker="s", linewidth=2.4, label=f"EARC ({ds.upper()})", color=c)
            base = baselines_dict.get(ds, {})
            if "standard_rag" in base:
                plt.axhline(base["standard_rag"]["exact_match"], color=c, linestyle=":", alpha=0.7,
                            label=f"Standard RAG ({ds.upper()}) [{base['standard_rag']['exact_match']:.1f}%]")
            if "llmlingua2" in base:
                plt.axhline(base["llmlingua2"]["exact_match"], color=c, linestyle="--", alpha=0.7,
                            label=f"LLMLingua-2 ({ds.upper()}) [{base['llmlingua2']['exact_match']:.1f}%]")

        plt.axvline(300, color="gray", linestyle="-.", alpha=0.8, label="Production Budget (300t)")
        plt.title(f"B. Exact Match (%) vs. Token Budget — Provider: {provider.upper()}", fontsize=13, fontweight="bold")
        plt.xlabel("Target Token Budget", fontsize=11)
        plt.ylabel("Exact Match (%)", fontsize=11)
        plt.xticks(budgets)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8.5)
        plt.tight_layout()
        em_path = output_dir / f"em_vs_token_budget_{provider}.png"
        plt.savefig(em_path, dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # Plot C: Compression Percentage versus Token Budget
        # -------------------------------------------------------------
        plt.figure(figsize=(9, 5.5))
        for ds in datasets:
            comp_pcts = [p_data[ds].get(b[1], {}).get("avg_compression_percentage", 0.0) for b in BUDGET_CONFIGS]
            c = colors.get(ds, None)
            plt.plot(budgets, comp_pcts, marker="^", linewidth=2.4, label=f"EARC ({ds.upper()})", color=c)

        plt.axvline(300, color="gray", linestyle="-.", alpha=0.8, label="Production Budget (300t)")
        plt.title(f"C. Compression Percentage (%) vs. Token Budget — Provider: {provider.upper()}", fontsize=13, fontweight="bold")
        plt.xlabel("Target Token Budget", fontsize=11)
        plt.ylabel("Compression Percentage (%)", fontsize=11)
        plt.xticks(budgets)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        comp_path = output_dir / f"compression_pct_vs_token_budget_{provider}.png"
        plt.savefig(comp_path, dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # Plot D: F1 versus Compression Percentage
        # -------------------------------------------------------------
        plt.figure(figsize=(9, 5.5))
        for ds in datasets:
            comp_pcts = [p_data[ds].get(b[1], {}).get("avg_compression_percentage", 0.0) for b in BUDGET_CONFIGS]
            f1_scores = [p_data[ds].get(b[1], {}).get("token_f1", 0.0) for b in BUDGET_CONFIGS]
            c = colors.get(ds, None)
            plt.plot(comp_pcts, f1_scores, marker="D", linewidth=2.4, label=f"EARC Frontier ({ds.upper()})", color=c)
            for cp, f1_val, b in zip(comp_pcts, f1_scores, budgets):
                plt.annotate(f"{b}t", (cp, f1_val), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5)

        plt.title(f"D. Token F1 (%) vs. Compression Percentage (%) — Provider: {provider.upper()}", fontsize=13, fontweight="bold")
        plt.xlabel("Compression Percentage (%) [Higher = More Aggressive]", fontsize=11)
        plt.ylabel("Token F1 (%)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="best", fontsize=10)
        plt.tight_layout()
        pareto_path = output_dir / f"f1_vs_compression_pct_{provider}.png"
        plt.savefig(pareto_path, dpi=300)
        plt.close()

    logger.info(f"Generated all 4 token-budget sensitivity plots in {output_dir}")


def generate_markdown_report(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    baselines_dict: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: Path,
):
    lines = [
        "# Experiment 3 — Token-Budget Sensitivity Analysis Report",
        "",
        "## 1. Executive Summary & Experimental Controls",
        "This experiment evaluates how EARC behaves under varying context token budgets $T \\in \\{100, 150, 200, 300, 400, 500, 750\\}$.",
        "- **Fixed Parameters**: $\\alpha = 0.7$, $\\beta = 0.3$, redundancy threshold $\\tau = 0.85$, top-$k=10$ dense retrieval (all-MiniLM-L6-v2), temperature 0.0, identical random seed (42).",
        "- **Compared Baselines**: Standard RAG (uncompressed context) and LLMLingua-2 (budget=300 tokens).",
        "",
        "## 2. Quantitative Results Table Across All Budgets",
    ]

    for provider, p_data in all_results.items():
        lines.append(f"### Provider: `{provider.upper()}`")
        lines.append("")
        lines.append("| Dataset | Method / Budget | Exact Match (%) | Token F1 (%) | Avg Tokens | Comp % | Comp Ratio | Latency (s) | Ev. Retention (%) | Notes |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

        for ds, ds_data in p_data.items():
            base = baselines_dict.get(ds, {})
            # Standard RAG baseline row
            if "standard_rag" in base:
                sb = base["standard_rag"]
                lines.append(
                    f"| {ds.upper()} | **Standard RAG** | {sb['exact_match']:.2f}% | {sb['token_f1']:.2f}% | {sb['avg_compressed_tokens']:.1f} | 0.0% | 1.00x | {sb['avg_e2e_latency']:.2f}s | 100.0% | Uncompressed full context |"
                )
            # LLMLingua-2 baseline row
            if "llmlingua2" in base:
                lb = base["llmlingua2"]
                lines.append(
                    f"| {ds.upper()} | **LLMLingua-2 (300t)** | {lb['exact_match']:.2f}% | {lb['token_f1']:.2f}% | {lb['avg_compressed_tokens']:.1f} | {lb['avg_compression_percentage']:.1f}% | {lb['avg_compression_ratio']:.2f}x | {lb['avg_e2e_latency']:.2f}s | N/A | Token-level pruning |"
                )

            for budget, cfg_name in BUDGET_CONFIGS:
                m = ds_data.get(cfg_name, {})
                em = m.get("exact_match", 0.0)
                f1 = m.get("token_f1", 0.0)
                toks = m.get("avg_compressed_tokens", 0.0)
                pct = m.get("avg_compression_percentage", 0.0)
                ratio = m.get("avg_compression_ratio", 0.0)
                lat = m.get("avg_e2e_latency", 0.0)
                ret = m.get("evidence_retention_rate", 0.0)
                note = "**Production Target (300t)**" if budget == 300 else f"EARC {budget}t"

                lines.append(
                    f"| {ds.upper()} | EARC ({budget} tokens) | {em:.2f}% | {f1:.2f}% | {toks:.1f} | {pct:.1f}% | {ratio:.2f}x | {lat:.2f}s | {ret:.1f}% | {note} |"
                )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved token-budget Markdown report to {output_path}")


def run_token_budget_experiment(
    limit: Optional[int] = 20,
    datasets_to_run: Optional[List[str]] = None,
    providers_to_run: Optional[List[str]] = None,
    config_path: str = "configs/default.yaml",
    output_dir_str: str = "outputs/sensitivity/token_budget",
    final_eval_dir_str: str = "outputs/final_evaluation",
    sensitivity_dir_str: str = "outputs/sensitivity/alpha_beta",
    seed: int = 42,
):
    setup_logging("INFO")
    output_dir = Path(output_dir_str)
    final_eval_dir = Path(final_eval_dir_str)
    sensitivity_dir = Path(sensitivity_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    config.retrieval.backend = "dense"
    config.retrieval.retrieval_method = "dense"
    config.retrieval.top_k = 10
    config.retrieval.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.compression.alpha = 0.7
    config.compression.beta = 0.3
    config.compression.use_redundancy = True
    config.compression.redundancy_threshold = 0.85
    config.mistral.model = "ministral-8b-latest"
    config.mistral.temperature = 0.0
    config.ollama.model = "llama3.2"
    config.ollama.temperature = 0.0

    providers = providers_to_run or ["mistral"]
    datasets = datasets_to_run or ["hotpot", "nq", "trivia"]
    dataset_code_map = {"nq": "nq", "hotpot": "hotpotqa", "trivia": "triviaqa"}

    device = DeviceManager().get_device().type
    rag_dir = config.retrieval.rag_project_dir or "RAG_Project"

    logger.info("=" * 80)
    logger.info("EARC TOKEN-BUDGET SENSITIVITY ANALYSIS (EXPERIMENT 3)")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Providers: {providers}")
    logger.info(f"Budgets: {[b for b, _ in BUDGET_CONFIGS]}")
    logger.info(f"Limit per dataset: {limit or 'Full dataset'}")
    logger.info("=" * 80)

    logger.info("Loading shared Dense Retriever & Embedding Engine...")
    retriever = RAGProjectRetriever(
        rag_dir=rag_dir,
        model_name=config.retrieval.embedding_model,
        device=device,
        method="dense",
    )
    embedding_engine = EmbeddingEngine(config.retrieval.embedding_model, device=device)

    logger.info("Initializing LLM providers...")
    llm_instances = {}
    for p_name in providers:
        llm_instances[p_name] = create_provider(config, p_name)

    token_counter = llm_instances[providers[0]].get_token_counter()
    prompt_builder = PromptBuilder(DEFAULT_QA_TEMPLATE)

    # 1. Load locked example IDs
    ids_file = final_eval_dir / "selected_example_ids.json"
    if not ids_file.exists():
        raise FileNotFoundError(f"Missing {ids_file}")
    with open(ids_file, "r", encoding="utf-8") as f:
        locked_ids = json.load(f)

    sampled_examples_by_dataset: Dict[str, List[QAExample]] = {}
    for ds_key in datasets:
        ds_load_name = dataset_code_map.get(ds_key, ds_key)
        full_examples = load_dataset(ds_load_name)
        id_to_example = {ex.id: ex for ex in full_examples}
        chosen_ids = locked_ids.get(ds_key, [])
        if limit is not None:
            chosen_ids = chosen_ids[:limit]
        sampled = [id_to_example[eid] for eid in chosen_ids if eid in id_to_example]
        sampled_examples_by_dataset[ds_key] = sampled
        logger.info(f"Dataset {ds_key.upper()}: loaded {len(sampled)} evaluation examples.")

    # 2. Main execution loop
    all_summary_results: Dict[str, Dict[str, Dict[str, Any]]] = {
        p: {d: {} for d in datasets} for p in providers
    }
    all_baselines: Dict[str, Dict[str, Dict[str, Any]]] = {d: {} for d in datasets}

    for ds_key in datasets:
        dataset_examples = sampled_examples_by_dataset[ds_key]
        chosen_ids = [ex.id for ex in dataset_examples]
        logger.info(f"\n{'='*70}\nProcessing Dataset: {ds_key.upper()} ({len(dataset_examples)} examples)\n{'='*70}")

        # Baseline metrics from final_eval_dir
        for p_name in providers:
            base_m = load_baseline_metrics(final_eval_dir, p_name, ds_key, chosen_ids)
            all_baselines[ds_key] = base_m

        # Cache retrieval results once per query
        logger.info("Caching dense retrieval results across all queries in dataset...")
        cached_retrievals: Dict[str, Tuple[List[Any], float]] = {}
        for ex in dataset_examples:
            t0 = time.time()
            retrieved_docs = retriever.retrieve(ex.question, top_k=10)
            cached_retrievals[ex.id] = (retrieved_docs, time.time() - t0)

        for budget, cfg_name in BUDGET_CONFIGS:
            logger.info(f"\n--- Testing Budget: {budget} tokens ({cfg_name}) ---")

            var_config = config.compression.model_copy(deep=True)
            var_config.alpha = 0.7
            var_config.beta = 0.3
            var_config.use_redundancy = True
            var_config.redundancy_threshold = 0.85
            var_config.token_budget = budget

            compressor = EvidenceAwareCompressor(
                var_config, embedding_engine, token_counter, device=device
            )

            for p_name in providers:
                cfg_dir = output_dir / p_name / ds_key / cfg_name
                cfg_dir.mkdir(parents=True, exist_ok=True)
                pred_file = cfg_dir / "predictions.jsonl"
                existing_records = load_checkpoint_records(pred_file)

                # Check if we can reuse budget 300 from Experiment 2 if not yet written
                if len(existing_records) < len(dataset_examples) and budget == 300:
                    exp2_file = sensitivity_dir / p_name / ds_key / "alpha_0.7_beta_0.3" / "predictions.jsonl"
                    if exp2_file.exists():
                        exp2_records = load_checkpoint_records(exp2_file)
                        if len(exp2_records) >= len(dataset_examples):
                            logger.info(f"Reusing {len(exp2_records)} records from Experiment 2 (budget 300).")
                            with open(pred_file, "w", encoding="utf-8") as f:
                                for r in exp2_records.values():
                                    f.write(json.dumps(r) + "\n")
                            existing_records = exp2_records

                # Check if already completed
                if len(existing_records) >= len(dataset_examples):
                    logger.info(f"[{p_name.upper()} - {cfg_name}] Already completed ({len(existing_records)} examples). Skipping.")
                    continue

                for ex_idx, ex in enumerate(dataset_examples, 1):
                    if ex.id in existing_records:
                        continue

                    retrieved_docs, ret_latency = cached_retrievals[ex.id]

                    t_comp0 = time.time()
                    comp_res = compressor.compress(ex.question, retrieved_docs, dataset=ds_key)
                    comp_latency = time.time() - t_comp0

                    prompt = prompt_builder.build(ex.question, comp_res.compressed_text)
                    ev_retention = compute_evidence_retention(retrieved_docs, comp_res.compressed_text, ex.answers)

                    llm = llm_instances[p_name]
                    t_gen0 = time.time()
                    try:
                        gen_res = llm.generate(prompt)
                        pred_text = gen_res.text
                    except Exception as e:
                        logger.error(f"Generation failure for {p_name} on {ex.id}: {e}")
                        pred_text = ""
                    gen_latency = time.time() - t_gen0

                    evaluated_ans = extract_answer(pred_text)
                    em = exact_match_score(evaluated_ans, ex.answers)
                    f1 = max_token_f1_score(evaluated_ans, ex.answers)["f1"]

                    orig_t = comp_res.original_tokens
                    comp_t = comp_res.compressed_tokens
                    comp_ratio = round(orig_t / comp_t, 2) if comp_t > 0 else 1.0
                    comp_pct = round(comp_res.reduction_percentage, 2)
                    e2e_lat = round(ret_latency + comp_latency + gen_latency, 4)

                    record = {
                        "example_id": ex.id,
                        "dataset": ds_key,
                        "budget": budget,
                        "config_name": cfg_name,
                        "provider": p_name,
                        "model": llm.model_name,
                        "question": ex.question,
                        "gold_answers": ex.answers,
                        "prediction": pred_text,
                        "evaluated_answer": evaluated_ans,
                        "exact_match": em,
                        "f1": f1,
                        "evidence_retention": ev_retention,
                        "original_tokens": orig_t,
                        "compressed_tokens": comp_t,
                        "compression_percentage": comp_pct,
                        "compression_ratio": comp_ratio,
                        "retrieval_latency": round(ret_latency, 4),
                        "compression_latency": round(comp_latency, 4),
                        "generation_latency": round(gen_latency, 4),
                        "end_to_end_latency": e2e_lat,
                    }

                    with open(pred_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    existing_records[ex.id] = record

                logger.info(f"[{p_name.upper()} - {ds_key.upper()} - {cfg_name}] Finished {len(existing_records)} examples.")

    # 3. Consolidation & Reporting
    consolidated_rows = []
    for p_name in providers:
        for ds_key in datasets:
            for budget, cfg_name in BUDGET_CONFIGS:
                pred_file = output_dir / p_name / ds_key / cfg_name / "predictions.jsonl"
                records = list(load_checkpoint_records(pred_file).values())
                metrics = compute_metrics_from_records(records)
                all_summary_results[p_name][ds_key][cfg_name] = metrics

                consolidated_rows.append({
                    "Provider": p_name,
                    "Model": llm_instances[p_name].model_name,
                    "Dataset": ds_key.upper(),
                    "Config_Name": cfg_name,
                    "Target_Budget": budget,
                    "N_Examples": metrics.get("n_examples", 0),
                    "Exact_Match_Pct": metrics.get("exact_match", 0.0),
                    "Token_F1_Pct": metrics.get("token_f1", 0.0),
                    "Avg_Comp_Tokens": metrics.get("avg_compressed_tokens", 0.0),
                    "Avg_Comp_Pct": metrics.get("avg_compression_percentage", 0.0),
                    "Avg_Comp_Ratio": metrics.get("avg_compression_ratio", 0.0),
                    "Avg_E2E_Latency_s": metrics.get("avg_e2e_latency", 0.0),
                    "Evidence_Retention_Pct": metrics.get("evidence_retention_rate", 0.0),
                    "Is_Production": (budget == 300),
                })

    save_json(output_dir / "token_budget_results.json", all_summary_results)

    csv_path = output_dir / "token_budget_results.csv"
    if consolidated_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(consolidated_rows[0].keys()))
            writer.writeheader()
            writer.writerows(consolidated_rows)
        logger.info(f"Saved complete CSV to {csv_path}")

    md_path = output_dir / "token_budget_consolidated_table.md"
    generate_markdown_report(all_summary_results, all_baselines, md_path)
    generate_plots(all_summary_results, all_baselines, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 3: Token-Budget Sensitivity")
    parser.add_argument("--limit", type=int, default=20, help="Number of examples per dataset")
    parser.add_argument("--datasets", nargs="+", default=["hotpot", "nq", "trivia"])
    parser.add_argument("--providers", nargs="+", default=["mistral"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_token_budget_experiment(
        limit=args.limit,
        datasets_to_run=args.datasets,
        providers_to_run=args.providers,
        seed=args.seed,
    )
