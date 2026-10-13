"""
Component Ablation Study Orchestrator for EARC (Experiment 1).

Ablation Variants:
- A1: Semantic-only
      H(s) = Sim(q,s) without redundancy filtering (alpha=1.0, beta=0.0, use_redundancy=False)
- A2: Semantic + Evidence
      H(s) = 0.7*Sim(q,s) + 0.3*Evidence(s) without redundancy filtering (alpha=0.7, beta=0.3, use_redundancy=False)
- A3: Semantic + Redundancy
      Semantic relevance ranking + redundancy filtering tau=0.85, no evidence (alpha=1.0, beta=0.0, use_redundancy=True)
- A4: Evidence + Redundancy
      Evidence score ranking + redundancy filtering tau=0.85, no semantic (alpha=0.0, beta=1.0, use_redundancy=True)
- A5: Full EARC
      Semantic relevance + evidence scoring + redundancy filtering + budget selection (alpha=0.7, beta=0.3, use_redundancy=True)

Locked Experimental Controls:
- Shared identical dense retrieval pool: all-MiniLM-L6-v2, FAISS top_k=10
- Shared identical 1,000-example subsets from selected_example_ids.json
- Target token budget = 300
- LLMs: Ollama (llama3.2) and Mistral (ministral-8b-latest)
- Prompt template, extraction regex, seed=42
"""

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

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
from src.data.sampling import sample_dataset
from src.data.schemas import QAExample, RetrievedDocument, ExperimentResult
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
from src.utils.io import ensure_dir, save_json

logger = get_logger("component_ablation")

ABLATION_SPECS = {
    "A1_semantic_only": {
        "label": "A1: Semantic-only",
        "short_name": "A1 (Semantic)",
        "alpha": 1.0,
        "beta": 0.0,
        "use_redundancy": False,
        "redundancy_threshold": 0.85,
        "token_budget": 300,
        "description": "H(s) = Sim(q,s) without redundancy filtering",
    },
    "A2_semantic_evidence": {
        "label": "A2: Semantic + Evidence",
        "short_name": "A2 (Sem+Ev)",
        "alpha": 0.7,
        "beta": 0.3,
        "use_redundancy": False,
        "redundancy_threshold": 0.85,
        "token_budget": 300,
        "description": "H(s) = 0.7*Sim(q,s) + 0.3*Evidence(s) without redundancy filtering",
    },
    "A3_semantic_redundancy": {
        "label": "A3: Semantic + Redundancy",
        "short_name": "A3 (Sem+Red)",
        "alpha": 1.0,
        "beta": 0.0,
        "use_redundancy": True,
        "redundancy_threshold": 0.85,
        "token_budget": 300,
        "description": "Semantic relevance ranking with redundancy filtering (tau=0.85), no evidence",
    },
    "A4_evidence_redundancy": {
        "label": "A4: Evidence + Redundancy",
        "short_name": "A4 (Ev+Red)",
        "alpha": 0.0,
        "beta": 1.0,
        "use_redundancy": True,
        "redundancy_threshold": 0.85,
        "token_budget": 300,
        "description": "Evidence score ranking with redundancy filtering (tau=0.85), no semantic",
    },
    "A5_full_earc": {
        "label": "A5: Full EARC",
        "short_name": "A5 (Full EARC)",
        "alpha": 0.7,
        "beta": 0.3,
        "use_redundancy": True,
        "redundancy_threshold": 0.85,
        "token_budget": 300,
        "description": "Full EARC: Semantic relevance + evidence scoring + redundancy filtering + budget selection",
    },
}


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
    """Computes whether gold answer evidence is preserved in compressed text given it was present in retrieved docs."""
    if not gold_answers:
        return 0.0
    retrieved_combined = " ".join(
        (d.text if hasattr(d, "text") else d.get("text", "")) for d in retrieved_docs
    ).lower()
    comp_lower = (compressed_text or "").lower()

    gold_in_retrieved = any(g.lower() in retrieved_combined for g in gold_answers if g)
    if not gold_in_retrieved:
        return 0.0  # Not retrieved, cannot be retained

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

    avg_ev_ret = round(float(np.mean(ev_ret)) * 100.0, 2) if ev_ret else "N/A"

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


def plot_comparisons(all_results: Dict[str, Dict[str, Dict[str, Any]]], output_dir: Path):
    """Generates comparison plots for EM, F1, and Compressed Tokens across variants."""
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = list(ABLATION_SPECS.keys())
    var_labels = [ABLATION_SPECS[v]["short_name"] for v in variants]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]

    for provider, p_data in all_results.items():
        datasets = list(p_data.keys())

        # 1. EM Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_width = 0.25
        x = np.arange(len(variants))
        for idx, ds in enumerate(datasets):
            scores = [p_data[ds].get(v, {}).get("exact_match", 0.0) for v in variants]
            offset = (idx - len(datasets) / 2 + 0.5) * bar_width
            ax.bar(x + offset, scores, width=bar_width, label=ds.upper())
        ax.set_title(f"Ablation Study: Exact Match (%) — Provider: {provider.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(var_labels, rotation=15)
        ax.set_ylabel("Exact Match (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"em_comparison_{provider}.png", dpi=300)
        plt.close()

        # 2. F1 Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, ds in enumerate(datasets):
            scores = [p_data[ds].get(v, {}).get("token_f1", 0.0) for v in variants]
            offset = (idx - len(datasets) / 2 + 0.5) * bar_width
            ax.bar(x + offset, scores, width=bar_width, label=ds.upper())
        ax.set_title(f"Ablation Study: Token F1 (%) — Provider: {provider.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(var_labels, rotation=15)
        ax.set_ylabel("Token F1 (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"f1_comparison_{provider}.png", dpi=300)
        plt.close()

        # 3. Compressed Tokens Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, ds in enumerate(datasets):
            scores = [p_data[ds].get(v, {}).get("avg_compressed_tokens", 0.0) for v in variants]
            offset = (idx - len(datasets) / 2 + 0.5) * bar_width
            ax.bar(x + offset, scores, width=bar_width, label=ds.upper())
        ax.set_title(f"Ablation Study: Average Compressed Tokens — Provider: {provider.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(var_labels, rotation=15)
        ax.set_ylabel("Compressed Tokens")
        ax.axhline(300, color="red", linestyle="--", label="Target Budget (300)")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"compressed_tokens_comparison_{provider}.png", dpi=300)
        plt.close()

    logger.info(f"Generated comparison plots in {output_dir}")


def run_component_ablations(
    limit: Optional[int] = None,
    datasets_to_run: Optional[List[str]] = None,
    providers_to_run: Optional[List[str]] = None,
    variants_to_run: Optional[List[str]] = None,
    config_path: str = "configs/default.yaml",
    output_dir_str: str = "outputs/ablations",
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
    config.ollama.model = "llama3.2"
    config.ollama.temperature = 0.0
    config.mistral.model = "ministral-8b-latest"
    config.mistral.temperature = 0.0

    datasets = datasets_to_run or ["nq", "hotpot", "trivia"]
    dataset_code_map = {"nq": "nq", "hotpot": "hotpotqa", "trivia": "triviaqa"}
    providers = providers_to_run or ["ollama", "mistral"]
    variants = variants_to_run or list(ABLATION_SPECS.keys())

    device = DeviceManager().get_device().type
    rag_dir = config.retrieval.rag_project_dir or "RAG_Project"

    logger.info("=" * 80)
    logger.info("EARC COMPONENT ABLATION STUDY (EXPERIMENT 1)")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Providers: {providers}")
    logger.info(f"Variants: {variants}")
    logger.info(f"Limit: {limit or 'Full 1,000 evaluation subset'}")
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

    # 2. Main ablation execution loop
    all_summary_results: Dict[str, Dict[str, Dict[str, Any]]] = {
        p: {d: {} for d in datasets} for p in providers
    }

    for ds_key in datasets:
        dataset_examples = sampled_examples_by_dataset[ds_key]
        logger.info(f"\n{'='*70}\nProcessing Dataset: {ds_key.upper()} ({len(dataset_examples)} examples)\n{'='*70}")

        for var_key in variants:
            spec = ABLATION_SPECS[var_key]
            logger.info(f"\n--- Variant {var_key}: {spec['label']} ---")

            # Check if this is A5 and we can reuse existing final_evaluation predictions
            for p_name in providers:
                var_dir = output_dir / p_name / ds_key / var_key
                var_dir.mkdir(parents=True, exist_ok=True)
                pred_file = var_dir / "predictions.jsonl"
                existing_records = load_checkpoint_records(pred_file)

                # Check if A5 can be seeded from final_evaluation
                if var_key == "A5_full_earc" and len(existing_records) < len(dataset_examples):
                    final_pred_file = final_eval_dir / p_name / ds_key / "earc" / "predictions.jsonl"
                    if final_pred_file.exists():
                        final_recs = load_checkpoint_records(final_pred_file)
                        for ex in dataset_examples:
                            if ex.id in final_recs and ex.id not in existing_records:
                                r = final_recs[ex.id]
                                existing_records[ex.id] = r
                        with open(pred_file, "w", encoding="utf-8") as f:
                            for r in existing_records.values():
                                f.write(json.dumps(r) + "\n")
                        logger.info(f"Seeded {len(existing_records)} records for A5 from existing final_evaluation!")

            # Build compressor for this variant
            var_config = config.compression.model_copy(deep=True)
            var_config.alpha = spec["alpha"]
            var_config.beta = spec["beta"]
            var_config.use_redundancy = spec["use_redundancy"]
            var_config.redundancy_threshold = spec["redundancy_threshold"]
            var_config.token_budget = spec["token_budget"]

            compressor = EvidenceAwareCompressor(
                var_config, embedding_engine, token_counter, device=device
            )

            for ex_idx, ex in enumerate(dataset_examples, 1):
                # Check if already completed across all providers
                all_done = True
                for p_name in providers:
                    pred_file = output_dir / p_name / ds_key / var_key / "predictions.jsonl"
                    existing_records = load_checkpoint_records(pred_file)
                    if ex.id not in existing_records:
                        all_done = False
                        break

                if all_done:
                    continue

                # 1. Retrieval (shared identical pool)
                t_ret0 = time.time()
                retrieved_docs = retriever.retrieve(ex.question, top_k=10)
                ret_latency = time.time() - t_ret0

                # 2. Compression (shared across providers)
                t_comp0 = time.time()
                comp_res = compressor.compress(ex.question, retrieved_docs, dataset=ds_key)
                comp_latency = time.time() - t_comp0

                prompt = prompt_builder.build(ex.question, comp_res.compressed_text)
                ev_retention = compute_evidence_retention(retrieved_docs, comp_res.compressed_text, ex.answers)

                # 3. Generation across providers
                for p_name in providers:
                    pred_file = output_dir / p_name / ds_key / var_key / "predictions.jsonl"
                    existing_records = load_checkpoint_records(pred_file)
                    if ex.id in existing_records:
                        continue

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
                        "variant": var_key,
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

                    pred_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(pred_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")

                if ex_idx % 20 == 0 or ex_idx == len(dataset_examples):
                    logger.info(f"[{ds_key.upper()} - {var_key}] Progress: {ex_idx}/{len(dataset_examples)}")

    # 3. Compute Metrics and Consolidation
    consolidated_rows = []
    for p_name in providers:
        for ds_key in datasets:
            for var_key in variants:
                pred_file = output_dir / p_name / ds_key / var_key / "predictions.jsonl"
                records = list(load_checkpoint_records(pred_file).values())
                metrics = compute_metrics_from_records(records)
                all_summary_results[p_name][ds_key][var_key] = metrics

                consolidated_rows.append({
                    "Provider": p_name,
                    "Model": llm_instances[p_name].model_name,
                    "Dataset": ds_key.upper(),
                    "Variant_Key": var_key,
                    "Variant_Label": ABLATION_SPECS[var_key]["label"],
                    "Alpha": ABLATION_SPECS[var_key]["alpha"],
                    "Beta": ABLATION_SPECS[var_key]["beta"],
                    "Redundancy_Filter": ABLATION_SPECS[var_key]["use_redundancy"],
                    "N_Examples": metrics.get("n_examples", 0),
                    "Exact_Match_Pct": metrics.get("exact_match", 0.0),
                    "Token_F1_Pct": metrics.get("token_f1", 0.0),
                    "Avg_Comp_Tokens": metrics.get("avg_compressed_tokens", 0.0),
                    "Avg_Comp_Pct": metrics.get("avg_compression_percentage", 0.0),
                    "Avg_Comp_Ratio": metrics.get("avg_compression_ratio", 0.0),
                    "Avg_E2E_Latency_s": metrics.get("avg_e2e_latency", 0.0),
                    "Evidence_Retention_Pct": metrics.get("evidence_retention_rate", "N/A"),
                })

    # Save JSON and CSV
    save_json(output_dir / "ablation_results.json", all_summary_results)

    csv_path = output_dir / "ablation_results.csv"
    if consolidated_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(consolidated_rows[0].keys()))
            writer.writeheader()
            writer.writerows(consolidated_rows)
        logger.info(f"Saved complete CSV to {csv_path}")

    # Generate Consolidated Markdown Table with Improvements
    md_path = output_dir / "ablation_consolidated_table.md"
    generate_markdown_report(all_summary_results, md_path)

    # Generate Plots
    plot_comparisons(all_summary_results, output_dir)
    logger.info("Component Ablation Study successfully completed!")


def generate_markdown_report(all_results: Dict[str, Dict[str, Dict[str, Any]]], output_path: Path):
    lines = [
        "# EARC Component Ablation Study (Experiment 1) — Consolidated Report",
        "",
        "## 1. Executive Summary & Experimental Setup",
        "- **Retrieval**: Dense retrieval with `sentence-transformers/all-MiniLM-L6-v2`, FAISS `top_k=10`",
        "- **Target Token Budget**: 300 tokens",
        "- **Ablation Variants**:",
        "  - **A1: Semantic-only**: $H(s) = \\text{Sim}(q,s)$, no redundancy filter",
        "  - **A2: Semantic + Evidence**: $H(s) = 0.7\\text{Sim} + 0.3\\text{Ev}$, no redundancy filter",
        "  - **A3: Semantic + Redundancy**: $H(s) = \\text{Sim}(q,s)$, with redundancy filter ($\\tau=0.85$)",
        "  - **A4: Evidence + Redundancy**: $H(s) = \\text{Ev}(s)$, with redundancy filter ($\\tau=0.85$)",
        "  - **A5: Full EARC**: $H(s) = 0.7\\text{Sim} + 0.3\\text{Ev}$, with redundancy filter ($\\tau=0.85$)",
        "",
        "## 2. Consolidated Comparison Tables",
    ]

    for provider, p_data in all_results.items():
        lines.append(f"### Provider: `{provider.upper()}`")
        lines.append("")
        lines.append("| Dataset | Variant | EM (%) | F1 (%) | Tokens | Comp % | Comp Ratio | Latency (s) | Ev. Retention | Abs. ΔEM (A5 - Ax) | Rel. ΔEM (%) |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for ds, ds_data in p_data.items():
            a5_metrics = ds_data.get("A5_full_earc", {})
            a5_em = a5_metrics.get("exact_match", 0.0)

            for var_key, spec in ABLATION_SPECS.items():
                m = ds_data.get(var_key, {})
                em = m.get("exact_match", 0.0)
                f1 = m.get("token_f1", 0.0)
                toks = m.get("avg_compressed_tokens", 0.0)
                pct = m.get("avg_compression_percentage", 0.0)
                ratio = m.get("avg_compression_ratio", 0.0)
                lat = m.get("avg_e2e_latency", 0.0)
                ret = m.get("evidence_retention_rate", "N/A")

                abs_diff = round(a5_em - em, 2)
                rel_diff = round(((a5_em - em) / em) * 100.0, 1) if em > 0 else 0.0
                sign_str = f"+{abs_diff}" if abs_diff > 0 else f"{abs_diff}"
                rel_str = f"+{rel_diff}%" if rel_diff > 0 else f"{rel_diff}%"

                lines.append(
                    f"| {ds.upper()} | {spec['label']} | {em:.2f}% | {f1:.2f}% | {toks:.1f} | {pct:.1f}% | {ratio:.2f}x | {lat:.2f}s | {ret}% | {sign_str} | {rel_str} |"
                )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved Markdown report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run EARC Component Ablation Study (Experiment 1)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples per dataset (None = full subset)")
    parser.add_argument("--datasets", type=str, default="nq,hotpot,trivia", help="Comma-separated datasets (nq,hotpot,trivia)")
    parser.add_argument("--providers", type=str, default="ollama,mistral", help="Comma-separated providers (ollama,mistral)")
    parser.add_argument("--variants", type=str, default=None, help="Comma-separated variants (A1,A2,A3,A4,A5 or full names)")
    parser.add_argument("--output-dir", type=str, default="outputs/ablations", help="Output directory")
    parser.add_argument("--final-eval-dir", type=str, default="outputs/final_evaluation", help="Final evaluation directory")
    args = parser.parse_args()

    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    pr_list = [p.strip() for p in args.providers.split(",") if p.strip()]

    var_list = None
    if args.variants:
        var_map = {
            "A1": "A1_semantic_only",
            "A2": "A2_semantic_evidence",
            "A3": "A3_semantic_redundancy",
            "A4": "A4_evidence_redundancy",
            "A5": "A5_full_earc",
        }
        var_list = [var_map.get(v.strip(), v.strip()) for v in args.variants.split(",") if v.strip()]

    run_component_ablations(
        limit=args.limit,
        datasets_to_run=ds_list,
        providers_to_run=pr_list,
        variants_to_run=var_list,
        output_dir_str=args.output_dir,
        final_eval_dir_str=args.final_eval_dir,
    )


if __name__ == "__main__":
    main()
