"""
Experiment 2 — Alpha-Beta Sensitivity Analysis Orchestrator for EARC.

Evaluates the hybrid scoring function:
    H(s) = alpha * Sim(q,s) + beta * Evidence(s)
across 10 grid configurations:
    (1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4),
    (0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.0, 1.0)

Strictly fixed controls:
- Identical dense retrieval (all-MiniLM-L6-v2, FAISS top_k=10)
- Identical candidate sentences
- Identical redundancy threshold = 0.85
- Identical token budget = 300
- Identical prompt template and decoding settings (temperature=0.0)
- Identical random seed and evaluation subsets from selected_example_ids.json
"""

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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

logger = get_logger("alpha_beta_sensitivity")

GRID_CONFIGS: List[Tuple[float, float, str]] = [
    (1.0, 0.0, "alpha_1.0_beta_0.0"),
    (0.9, 0.1, "alpha_0.9_beta_0.1"),
    (0.8, 0.2, "alpha_0.8_beta_0.2"),
    (0.7, 0.3, "alpha_0.7_beta_0.3"),
    (0.6, 0.4, "alpha_0.6_beta_0.4"),
    (0.5, 0.5, "alpha_0.5_beta_0.5"),
    (0.4, 0.6, "alpha_0.4_beta_0.6"),
    (0.3, 0.7, "alpha_0.3_beta_0.7"),
    (0.2, 0.8, "alpha_0.2_beta_0.8"),
    (0.0, 1.0, "alpha_0.0_beta_1.0"),
]


def load_checkpoint_records(predictions_file: Path) -> Dict[str, Dict[str, Any]]:
    records = {}
    if predictions_file.exists():
        with open(predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if line_s:
                    try:
                        data = json.loads(line_s)
                        if "example_id" in data:
                            records[data["example_id"]] = data
                    except Exception:
                        pass
    return records


def compute_evidence_retention(retrieved_docs: List[Any], compressed_text: str, gold_answers: List[str]) -> float:
    if not gold_answers:
        return 0.0
    retrieved_combined = " ".join(
        (d.text if hasattr(d, "text") else d.get("text", "")) for d in retrieved_docs
    ).lower()
    comp_lower = (compressed_text or "").lower()

    gold_in_retrieved = any(g.lower() in retrieved_combined for g in gold_answers if g)
    if not gold_in_retrieved:
        return 0.0

    gold_in_compressed = any(g.lower() in comp_lower for g in gold_answers if g)
    return 1.0 if gold_in_compressed else 0.0


def compute_metrics_from_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}

    n = len(records)
    ems = [r.get("exact_match", 0.0) for r in records]
    f1s = [r.get("f1", 0.0) for r in records]
    orig_toks = [r.get("original_tokens", 0) for r in records if isinstance(r.get("original_tokens"), (int, float))]
    comp_toks = [r.get("compressed_tokens", 0) for r in records if isinstance(r.get("compressed_tokens"), (int, float))]
    comp_pcts = [r.get("compression_percentage", 0.0) for r in records if isinstance(r.get("compression_percentage"), (int, float))]
    comp_rats = [r.get("compression_ratio", 0.0) for r in records if isinstance(r.get("compression_ratio"), (int, float))]
    e2e_lats = [r.get("end_to_end_latency", 0.0) for r in records]
    ev_ret = [r.get("evidence_retention", 0.0) for r in records if "evidence_retention" in r]

    avg_ev_ret = round(float(np.mean(ev_ret)) * 100.0, 2) if ev_ret else 0.0

    return {
        "n_examples": n,
        "exact_match": round(float(np.mean(ems)) * 100.0, 2),
        "token_f1": round(float(np.mean(f1s)) * 100.0, 2),
        "avg_original_tokens": round(float(np.mean(orig_toks)), 1) if orig_toks else 0.0,
        "avg_compressed_tokens": round(float(np.mean(comp_toks)), 1) if comp_toks else 0.0,
        "avg_compression_percentage": round(float(np.mean(comp_pcts)), 2) if comp_pcts else 0.0,
        "avg_compression_ratio": round(float(np.mean(comp_rats)), 2) if comp_rats else 0.0,
        "avg_e2e_latency": round(float(np.mean(e2e_lats)), 4) if e2e_lats else 0.0,
        "evidence_retention_rate": avg_ev_ret,
    }


def generate_plots(all_results: Dict[str, Dict[str, Dict[str, Any]]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    alphas = [cfg[0] for cfg in GRID_CONFIGS]
    labels = [f"{cfg[0]:.1f}/{cfg[1]:.1f}" for cfg in GRID_CONFIGS]

    for provider, p_data in all_results.items():
        datasets = list(p_data.keys())

        # 1. Alpha-Beta vs EM Plot
        plt.figure(figsize=(10, 6))
        for ds in datasets:
            em_scores = [p_data[ds].get(cfg[2], {}).get("exact_match", 0.0) for cfg in GRID_CONFIGS]
            plt.plot(alphas, em_scores, marker="o", linewidth=2.2, label=ds.upper())
        plt.axvline(0.7, color="gray", linestyle="--", alpha=0.8, label="Production (0.7 / 0.3)")
        plt.title(f"α–β Sensitivity vs. Exact Match (%) — Provider: {provider.upper()}")
        plt.xlabel("Alpha (Semantic Relevance Weight)")
        plt.ylabel("Exact Match (%)")
        plt.xticks(alphas, labels, rotation=30)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        em_plot_path = output_dir / f"alpha_beta_vs_em_{provider}.png"
        plt.savefig(em_plot_path, dpi=300)
        plt.close()

        # 2. Alpha-Beta vs F1 Plot
        plt.figure(figsize=(10, 6))
        for ds in datasets:
            f1_scores = [p_data[ds].get(cfg[2], {}).get("token_f1", 0.0) for cfg in GRID_CONFIGS]
            plt.plot(alphas, f1_scores, marker="s", linewidth=2.2, label=ds.upper())
        plt.axvline(0.7, color="gray", linestyle="--", alpha=0.8, label="Production (0.7 / 0.3)")
        plt.title(f"α–β Sensitivity vs. Token F1 (%) — Provider: {provider.upper()}")
        plt.xlabel("Alpha (Semantic Relevance Weight)")
        plt.ylabel("Token F1 (%)")
        plt.xticks(alphas, labels, rotation=30)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        f1_plot_path = output_dir / f"alpha_beta_vs_f1_{provider}.png"
        plt.savefig(f1_plot_path, dpi=300)
        plt.close()

    logger.info(f"Generated sensitivity plots in {output_dir}")


def generate_markdown_report(all_results: Dict[str, Dict[str, Dict[str, Any]]], output_path: Path):
    lines = [
        "# Experiment 2 — α–β Sensitivity Analysis Report",
        "",
        "## 1. Executive Summary & Experimental Formulation",
        "The EARC hybrid scoring function balances semantic relevance and factual evidentiality:",
        "$$H(s) = \\alpha \\cdot \\text{Sim}(q, s) + \\beta \\cdot \\text{Evidence}(s), \\quad \\text{with } \\alpha + \\beta = 1.0$$",
        "- **Evaluated Configurations**: 10 distinct $(\\alpha, \\beta)$ ratios from $(1.0, 0.0)$ to $(0.0, 1.0)$.",
        "- **Fixed Baseline Parameters**: Dense retrieval (MiniLM, $top\\_k=10$), redundancy threshold $\\tau=0.85$, token budget $T=300$, zero-shot temperature 0.0.",
        "",
        "## 2. Quantitative Results Table",
    ]

    for provider, p_data in all_results.items():
        lines.append(f"### Provider: `{provider.upper()}`")
        lines.append("")
        lines.append("| Dataset | Config (α / β) | Exact Match (%) | Token F1 (%) | Tokens | Comp % | Comp Ratio | Latency (s) | Ev. Retention (%) | Production Note |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

        for ds, ds_data in p_data.items():
            for alpha, beta, cfg_name in GRID_CONFIGS:
                m = ds_data.get(cfg_name, {})
                em = m.get("exact_match", 0.0)
                f1 = m.get("token_f1", 0.0)
                toks = m.get("avg_compressed_tokens", 0.0)
                pct = m.get("avg_compression_percentage", 0.0)
                ratio = m.get("avg_compression_ratio", 0.0)
                lat = m.get("avg_e2e_latency", 0.0)
                ret = m.get("evidence_retention_rate", 0.0)
                note = "**Current Default (0.7 / 0.3)**" if (abs(alpha - 0.7) < 1e-4 and abs(beta - 0.3) < 1e-4) else ""

                lines.append(
                    f"| {ds.upper()} | ({alpha:.1f} / {beta:.1f}) | {em:.2f}% | {f1:.2f}% | {toks:.1f} | {pct:.1f}% | {ratio:.2f}x | {lat:.2f}s | {ret:.1f}% | {note} |"
                )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved sensitivity Markdown table to {output_path}")


def run_sensitivity_study(
    limit: Optional[int] = 20,
    datasets_to_run: Optional[List[str]] = None,
    providers_to_run: Optional[List[str]] = None,
    config_path: str = "configs/default.yaml",
    output_dir_str: str = "outputs/sensitivity/alpha_beta",
    final_eval_dir_str: str = "outputs/final_evaluation",
    seed: int = 42,
):
    setup_logging("INFO")
    output_dir = Path(output_dir_str)
    final_eval_dir = Path(final_eval_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    config.retrieval.backend = "dense"
    config.retrieval.retrieval_method = "dense"
    config.retrieval.top_k = 10
    config.retrieval.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.compression.token_budget = 300
    config.compression.use_redundancy = True
    config.compression.redundancy_threshold = 0.85
    config.ollama.model = "llama3.2"
    config.ollama.temperature = 0.0
    config.mistral.model = "ministral-8b-latest"
    config.mistral.temperature = 0.0

    datasets = datasets_to_run or ["hotpot"]
    dataset_code_map = {"nq": "nq", "hotpot": "hotpotqa", "trivia": "triviaqa"}
    providers = providers_to_run or ["mistral", "ollama"]

    device = DeviceManager().get_device().type
    rag_dir = config.retrieval.rag_project_dir or "RAG_Project"

    logger.info("=" * 80)
    logger.info("EARC ALPHA-BETA SENSITIVITY ANALYSIS (EXPERIMENT 2)")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Providers: {providers}")
    logger.info(f"Grid Configurations ({len(GRID_CONFIGS)} total): {[(a, b) for a, b, _ in GRID_CONFIGS]}")
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

    # 1. Load exact evaluation sample IDs
    ids_file = final_eval_dir / "selected_example_ids.json"
    if not ids_file.exists():
        raise FileNotFoundError(f"Missing {ids_file}. Final evaluation must be initialized.")

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

    for ds_key in datasets:
        dataset_examples = sampled_examples_by_dataset[ds_key]
        logger.info(f"\n{'='*70}\nProcessing Dataset: {ds_key.upper()} ({len(dataset_examples)} examples)\n{'='*70}")

        # Cache retrieval results once per query
        logger.info("Caching dense retrieval results across all queries in dataset...")
        cached_retrievals: Dict[str, Tuple[List[Any], float]] = {}
        for ex in dataset_examples:
            t0 = time.time()
            retrieved_docs = retriever.retrieve(ex.question, top_k=10)
            cached_retrievals[ex.id] = (retrieved_docs, time.time() - t0)

        for alpha, beta, cfg_name in GRID_CONFIGS:
            logger.info(f"\n--- Testing Configuration: alpha={alpha:.1f}, beta={beta:.1f} ({cfg_name}) ---")

            var_config = config.compression.model_copy(deep=True)
            var_config.alpha = alpha
            var_config.beta = beta
            var_config.use_redundancy = True
            var_config.redundancy_threshold = 0.85
            var_config.token_budget = 300

            compressor = EvidenceAwareCompressor(
                var_config, embedding_engine, token_counter, device=device
            )

            for p_name in providers:
                cfg_dir = output_dir / p_name / ds_key / cfg_name
                cfg_dir.mkdir(parents=True, exist_ok=True)
                pred_file = cfg_dir / "predictions.jsonl"
                existing_records = load_checkpoint_records(pred_file)

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
                        "config_name": cfg_name,
                        "alpha": alpha,
                        "beta": beta,
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
            for alpha, beta, cfg_name in GRID_CONFIGS:
                pred_file = output_dir / p_name / ds_key / cfg_name / "predictions.jsonl"
                records = list(load_checkpoint_records(pred_file).values())
                metrics = compute_metrics_from_records(records)
                all_summary_results[p_name][ds_key][cfg_name] = metrics

                consolidated_rows.append({
                    "Provider": p_name,
                    "Model": llm_instances[p_name].model_name,
                    "Dataset": ds_key.upper(),
                    "Config_Name": cfg_name,
                    "Alpha": alpha,
                    "Beta": beta,
                    "N_Examples": metrics.get("n_examples", 0),
                    "Exact_Match_Pct": metrics.get("exact_match", 0.0),
                    "Token_F1_Pct": metrics.get("token_f1", 0.0),
                    "Avg_Comp_Tokens": metrics.get("avg_compressed_tokens", 0.0),
                    "Avg_Comp_Pct": metrics.get("avg_compression_percentage", 0.0),
                    "Avg_Comp_Ratio": metrics.get("avg_compression_ratio", 0.0),
                    "Avg_E2E_Latency_s": metrics.get("avg_e2e_latency", 0.0),
                    "Evidence_Retention_Pct": metrics.get("evidence_retention_rate", 0.0),
                    "Is_Production": (abs(alpha - 0.7) < 1e-4 and abs(beta - 0.3) < 1e-4),
                })

    save_json(output_dir / "sensitivity_results.json", all_summary_results)

    csv_path = output_dir / "sensitivity_results.csv"
    if consolidated_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(consolidated_rows[0].keys()))
            writer.writeheader()
            writer.writerows(consolidated_rows)
        logger.info(f"Saved complete CSV to {csv_path}")

    md_path = output_dir / "sensitivity_consolidated_table.md"
    generate_markdown_report(all_summary_results, md_path)

    generate_plots(all_summary_results, output_dir)
    logger.info("Alpha-Beta Sensitivity Study successfully completed!")


def main():
    parser = argparse.ArgumentParser(description="Run EARC Alpha-Beta Sensitivity Study (Experiment 2)")
    parser.add_argument("--limit", type=int, default=20, help="Limit number of examples per dataset (default=20)")
    parser.add_argument("--datasets", type=str, default="hotpot", help="Comma-separated datasets (default: hotpot)")
    parser.add_argument("--providers", type=str, default="mistral,ollama", help="Comma-separated providers (default: mistral,ollama)")
    parser.add_argument("--output-dir", type=str, default="outputs/sensitivity/alpha_beta", help="Output directory")
    args = parser.parse_args()

    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    pr_list = [p.strip() for p in args.providers.split(",") if p.strip()]

    run_sensitivity_study(
        limit=args.limit,
        datasets_to_run=ds_list,
        providers_to_run=pr_list,
        output_dir_str=args.output_dir,
    )


if __name__ == "__main__":
    main()
