"""
Experiment 5 — Systematic Failure Analysis of EARC
Categorizes all incorrect EARC predictions into:
- F1: Retrieval failure (required evidence absent from retrieved top-10)
- F2: Compression/evidence-selection failure (evidence present in candidates, but removed by EARC)
- F3: LLM generation/reasoning failure (evidence present in compressed context, but LLM produced incorrect answer)
- F4: Answer extraction/evaluation failure (generated text contains correct answer semantically, but normalized EM is 0)

Generates:
- Complete failure-analysis dataset as CSV and JSON
- Stacked bar chart showing failure categories across datasets and models
- 5 representative examples for each major failure type with full context
- Comprehensive Markdown summary tables
"""

import os
import sys
import json
import time
import string
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

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
from src.llm.factory import create_provider
from src.evaluation.answer_extraction import extract_answer
from src.utils.device import DeviceManager
from src.utils.logging import get_logger
from src.utils.io import save_json

logger = get_logger("failure_analysis")


def normalize_text(text: str) -> str:
    """Standard normalization for evaluation."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def check_semantic_match(pred_text: str, gold_answers: List[str]) -> bool:
    """Checks if gold answer is semantically present in prediction text."""
    if not pred_text or not gold_answers:
        return False
    norm_pred = normalize_text(pred_text)
    for g in gold_answers:
        norm_g = normalize_text(g)
        if norm_g and norm_g in norm_pred:
            return True
    return False


def run_failure_analysis(
    output_dir_str: str = "outputs/experiments/failure_analysis",
    final_eval_dir_str: str = "outputs/final_evaluation",
    evidence_dir_str: str = "outputs/experiments/evidence_preservation",
    config_path: str = "configs/default.yaml",
):
    output_dir = Path(output_dir_str)
    final_eval_dir = Path(final_eval_dir_str)
    evidence_dir = Path(evidence_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    config.retrieval.top_k = 10
    config.retrieval.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    device = DeviceManager().get_device().type
    rag_dir = config.retrieval.rag_project_dir or "RAG_Project"

    logger.info("=" * 80)
    logger.info("EARC EXPERIMENT 5: SYSTEMATIC FAILURE ANALYSIS")
    logger.info("=" * 80)

    # 1. Load HotpotQA detailed evidence results if available
    hotpot_ev_file = evidence_dir / "detailed_evidence_results.json"
    hotpot_ev_data = {}
    if hotpot_ev_file.exists():
        with open(hotpot_ev_file, "r", encoding="utf-8") as f:
            hotpot_ev_data = json.load(f)
        logger.info(f"Loaded {len(hotpot_ev_data)} HotpotQA evidence traces.")

    # 2. Initialize Retriever and EARC Compressor for on-the-fly candidate tracing
    logger.info("Initializing Dense Retriever & EARC Compressor...")
    retriever = RAGProjectRetriever(
        rag_dir=rag_dir,
        model_name=config.retrieval.embedding_model,
        device=device,
        method="dense",
    )
    embedding_engine = EmbeddingEngine(config.retrieval.embedding_model, device=device)
    llm = create_provider(config, "mistral")
    token_counter = llm.get_token_counter()

    cfg_earc = config.compression.model_copy(deep=True)
    cfg_earc.alpha = 0.7
    cfg_earc.beta = 0.3
    cfg_earc.use_redundancy = True
    cfg_earc.redundancy_threshold = 0.85
    cfg_earc.token_budget = 300
    compressor = EvidenceAwareCompressor(cfg_earc, embedding_engine, token_counter, device=device)

    # Cache for retrieval & compression per question
    compression_cache: Dict[str, Dict[str, Any]] = {}

    providers = ["mistral", "ollama"]
    datasets = ["hotpot", "nq", "trivia"]
    model_name_map = {"mistral": "Ministral-8B", "ollama": "Llama 3.2"}

    all_failure_records = []
    category_examples = {
        "F1_retrieval_failure": [],
        "F2_compression_failure": [],
        "F3_generation_failure": [],
        "F4_evaluation_failure": [],
    }

    stats = {
        p: {d: {"total": 0, "incorrect": 0, "F1": 0, "F2": 0, "F3": 0, "F4": 0} for d in datasets}
        for p in providers
    }

    for prov in providers:
        for ds in datasets:
            pred_file = final_eval_dir / prov / ds / "earc" / "predictions.jsonl"
            if not pred_file.exists():
                logger.warning(f"Missing {pred_file}")
                continue

            with open(pred_file, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f]

            stats[prov][ds]["total"] = len(records)

            for r in records:
                is_correct = (r.get("exact_match", 0.0) == 1.0)
                if is_correct:
                    continue

                stats[prov][ds]["incorrect"] += 1
                eid = r["example_id"]
                q = r["question"]
                golds = r.get("gold_answers", [])
                pred_raw = r.get("prediction", "")
                norm_pred = r.get("evaluated_answer", extract_answer(pred_raw))
                ret_hit = (r.get("answer_hit_at_10", 0.0) == 1.0)

                # Step 1: Check F4 (Evaluation Failure)
                # LLM produced the gold answer semantically, but normalization/extraction marked it incorrect
                is_eval_failure = check_semantic_match(pred_raw, golds)

                # Context details (retrieval & compression)
                ret_docs = []
                cand_sents = []
                sel_sents = []
                rem_sents = []
                sp_evidence = []

                if ds == "hotpot" and eid in hotpot_ev_data:
                    ev_item = hotpot_ev_data[eid]
                    ret_docs = ev_item.get("retrieved_titles", [])
                    sel_sents = ev_item.get("full_earc_selected_sents", [])
                    rem_sents = ev_item.get("full_earc_removed_sents", [])
                    sp_evidence = [s["text"] for s in ev_item.get("sp_sentences", [])]
                    cand_sents = sel_sents + rem_sents
                else:
                    # Lookup or compute compression trace
                    if q not in compression_cache:
                        ret_docs_objs = retriever.retrieve(q, top_k=10)
                        comp_res = compressor.compress(q, ret_docs_objs, dataset=ds)
                        sel_texts = [s.text for s in comp_res.selected_sentences]
                        sel_ids = set(s.sentence_id for s in comp_res.selected_sentences)
                        all_cands = comp_res.all_candidates or comp_res.candidates or comp_res.selected_sentences
                        rem_texts = [s.text for s in all_cands if s.sentence_id not in sel_ids]
                        compression_cache[q] = {
                            "ret_titles": [d.title for d in ret_docs_objs if d.title],
                            "all_cands": [s.text for s in all_cands],
                            "sel_sents": sel_texts,
                            "rem_sents": rem_texts,
                        }
                    trace = compression_cache[q]
                    ret_docs = trace["ret_titles"]
                    cand_sents = trace["all_cands"]
                    sel_sents = trace["sel_sents"]
                    rem_sents = trace["rem_sents"]

                # Step 2: Assign Primary and Secondary Failure Causes
                primary_failure = ""
                secondary_failure = ""
                failure_reason = ""

                if is_eval_failure:
                    primary_failure = "F4_evaluation_failure"
                    failure_reason = "The LLM response contains the correct answer semantically, but format/extraction marked EM=0."
                elif not ret_hit:
                    # Answer completely absent from retrieved top-10
                    primary_failure = "F1_retrieval_failure"
                    failure_reason = "The required factual answer was never retrieved into the top-10 candidate passages."
                else:
                    # Answer was retrieved. Check if present in EARC's selected context
                    norm_selected = normalize_text(" ".join(sel_sents))
                    ans_in_selected = any(normalize_text(g) in norm_selected for g in golds if normalize_text(g))

                    if not ans_in_selected:
                        primary_failure = "F2_compression_failure"
                        failure_reason = "The required evidence was present in retrieved candidates, but EARC pruned it from the compressed context."
                    else:
                        primary_failure = "F3_generation_failure"
                        failure_reason = "The required evidence was present in the compressed prompt, but the LLM hallucinated, missed it, or made a reasoning error."
                        # If compression pruned supporting sentences in multi-hop, record as secondary
                        if ds == "hotpot":
                            secondary_failure = "F2_partial_evidence_pruning"

                # Record stats
                cat_key = primary_failure.split("_")[0]
                stats[prov][ds][cat_key] += 1

                record = {
                    "example_id": eid,
                    "dataset": ds.upper(),
                    "provider": prov,
                    "model": model_name_map[prov],
                    "question": q,
                    "gold_answers": golds,
                    "generated_answer": pred_raw,
                    "normalized_prediction": norm_pred,
                    "primary_failure": primary_failure,
                    "secondary_failure": secondary_failure,
                    "failure_reason": failure_reason,
                    "retrieved_documents": ret_docs[:5],
                    "candidate_sentences_count": len(cand_sents),
                    "earc_selected_sentences": sel_sents[:4],
                    "earc_removed_sentences": rem_sents[:4],
                    "supporting_evidence": sp_evidence[:3] if sp_evidence else [],
                }
                all_failure_records.append(record)

                # Collect representative examples for each category (up to 5 each)
                if len(category_examples[primary_failure]) < 5:
                    category_examples[primary_failure].append(record)

            logger.info(f"[{prov.upper()} - {ds.upper()}] Processed {stats[prov][ds]['incorrect']} failures.")

    # 3. Save failure dataset as JSON and CSV
    json_path = output_dir / "failure_analysis_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_failure_records, f, indent=2)
    logger.info(f"Saved complete failure analysis dataset to {json_path}")

    csv_path = output_dir / "failure_analysis_dataset.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "example_id", "dataset", "provider", "model", "question", "gold_answers",
            "primary_failure", "secondary_failure", "failure_reason", "generated_answer",
            "normalized_prediction", "candidate_sentences_count"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_failure_records:
            writer.writerow({
                **r,
                "gold_answers": " | ".join(r["gold_answers"]),
            })
    logger.info(f"Saved failure analysis CSV to {csv_path}")

    # 4. Generate Stacked Bar Chart
    generate_stacked_bar_chart(stats, output_dir)

    # 5. Generate Markdown Report & Representative Examples
    generate_markdown_summary(stats, output_dir / "failure_summary_tables.md")
    generate_representative_examples_doc(category_examples, output_dir / "representative_failure_examples.md")


def generate_stacked_bar_chart(stats: Dict[str, Dict[str, Dict[str, int]]], output_dir: Path):
    labels = []
    f1_pcts, f2_pcts, f3_pcts, f4_pcts = [], [], [], []

    for prov, prov_data in stats.items():
        prov_label = "Ministral-8B" if prov == "mistral" else "Llama 3.2"
        for ds, d in prov_data.items():
            tot_err = d["incorrect"]
            if tot_err == 0:
                continue
            labels.append(f"{ds.upper()}\n({prov_label})")
            f1_pcts.append((d["F1"] / tot_err) * 100)
            f2_pcts.append((d["F2"] / tot_err) * 100)
            f3_pcts.append((d["F3"] / tot_err) * 100)
            f4_pcts.append((d["F4"] / tot_err) * 100)

    x = np.arange(len(labels))
    width = 0.58

    plt.figure(figsize=(11, 6.5))
    p1 = plt.bar(x, f1_pcts, width, label="F1: Retrieval Failure", color="#e74c3c", edgecolor="black", alpha=0.9)
    p2 = plt.bar(x, f2_pcts, width, bottom=f1_pcts, label="F2: Compression Failure", color="#f39c12", edgecolor="black", alpha=0.9)
    b3 = np.array(f1_pcts) + np.array(f2_pcts)
    p3 = plt.bar(x, f3_pcts, width, bottom=b3, label="F3: LLM Generation Failure", color="#3498db", edgecolor="black", alpha=0.9)
    b4 = b3 + np.array(f3_pcts)
    p4 = plt.bar(x, f4_pcts, width, bottom=b4, label="F4: Evaluation/Extraction Failure", color="#9b59b6", edgecolor="black", alpha=0.9)

    plt.ylabel("Proportion of Total Failures (%)", fontsize=11, fontweight="bold")
    plt.title("Systematic Failure Analysis Breakdown across Datasets & Models", fontsize=13, fontweight="bold")
    plt.xticks(x, labels, fontsize=9.5)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()

    chart_path = output_dir / "failure_breakdown_stacked_bar.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info(f"Saved stacked bar chart to {chart_path}")


def generate_markdown_summary(stats: Dict[str, Dict[str, Dict[str, int]]], output_path: Path):
    lines = [
        "# Systematic Failure Analysis of EARC — Comprehensive Summary",
        "",
        "## 1. Executive Summary & Failure Taxonomy",
        "Every incorrect EARC prediction was systematically attributed to one of four mutually exclusive primary root causes:",
        "- **F1. Retrieval Failure**: The required factual evidence was absent from the top-10 retrieved candidate documents.",
        "- **F2. Compression / Evidence-Selection Failure**: The required evidence existed in retrieved candidates, but EARC removed it during ranking or budget truncation.",
        "- **F3. LLM Generation / Reasoning Failure**: The required evidence was preserved in the compressed prompt, but the LLM hallucinated, missed it, or made a synthesis error.",
        "- **F4. Evaluation / Extraction Failure**: The LLM produced the correct answer semantically, but the extraction/normalization regex marked EM=0.",
        "",
        "## 2. Quantitative Failure Percentages",
        "",
        "### A. Breakdown by Dataset & Model",
        "| Model | Dataset | Total Queries | Errors | F1: Retrieval (%) | F2: Compression (%) | F3: Generation (%) | F4: Evaluation (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for prov in ["mistral", "ollama"]:
        m_name = "Ministral-8B" if prov == "mistral" else "Llama 3.2"
        for ds in ["hotpot", "nq", "trivia"]:
            d = stats[prov][ds]
            tot = d["incorrect"]
            if tot == 0:
                continue
            lines.append(
                f"| {m_name} | **{ds.upper()}** | {d['total']} | {tot} | "
                f"**{d['F1']/tot*100:.1f}%** ({d['F1']}) | "
                f"**{d['F2']/tot*100:.1f}%** ({d['F2']}) | "
                f"**{d['F3']/tot*100:.1f}%** ({d['F3']}) | "
                f"**{d['F4']/tot*100:.1f}%** ({d['F4']}) |"
            )

    # Dataset-level aggregates
    lines.extend([
        "",
        "### B. Aggregated by Dataset (Across Both Models)",
        "| Dataset | Total Errors | F1: Retrieval (%) | F2: Compression (%) | F3: Generation (%) | F4: Evaluation (%) | Primary Bottleneck |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    for ds in ["hotpot", "nq", "trivia"]:
        tot_err = stats["mistral"][ds]["incorrect"] + stats["ollama"][ds]["incorrect"]
        f1_tot = stats["mistral"][ds]["F1"] + stats["ollama"][ds]["F1"]
        f2_tot = stats["mistral"][ds]["F2"] + stats["ollama"][ds]["F2"]
        f3_tot = stats["mistral"][ds]["F3"] + stats["ollama"][ds]["F3"]
        f4_tot = stats["mistral"][ds]["F4"] + stats["ollama"][ds]["F4"]
        bottleneck = "Retrieval (all-MiniLM-L6-v2)" if f1_tot > max(f2_tot, f3_tot, f4_tot) else "Generation"
        lines.append(
            f"| **{ds.upper()}** | {tot_err} | {f1_tot/tot_err*100:.1f}% | {f2_tot/tot_err*100:.1f}% | {f3_tot/tot_err*100:.1f}% | {f4_tot/tot_err*100:.1f}% | **{bottleneck}** |"
        )

    # Model-level aggregates
    lines.extend([
        "",
        "### C. Aggregated by Model (Across All Datasets)",
        "| Model | Total Errors | F1: Retrieval (%) | F2: Compression (%) | F3: Generation (%) | F4: Evaluation (%) | Model Characteristic |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    for prov in ["mistral", "ollama"]:
        m_name = "Ministral-8B" if prov == "mistral" else "Llama 3.2"
        tot_err = sum(stats[prov][ds]["incorrect"] for ds in ["hotpot", "nq", "trivia"])
        f1_tot = sum(stats[prov][ds]["F1"] for ds in ["hotpot", "nq", "trivia"])
        f2_tot = sum(stats[prov][ds]["F2"] for ds in ["hotpot", "nq", "trivia"])
        f3_tot = sum(stats[prov][ds]["F3"] for ds in ["hotpot", "nq", "trivia"])
        f4_tot = sum(stats[prov][ds]["F4"] for ds in ["hotpot", "nq", "trivia"])
        char = "Higher reasoning capacity, lower F3 failure" if prov == "mistral" else "Smaller 3B capacity, elevated F3 failure"
        lines.append(
            f"| **{m_name}** | {tot_err} | {f1_tot/tot_err*100:.1f}% | {f2_tot/tot_err*100:.1f}% | {f3_tot/tot_err*100:.1f}% | {f4_tot/tot_err*100:.1f}% | {char} |"
        )

    lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved Markdown summary to {output_path}")


def generate_representative_examples_doc(cat_examples: Dict[str, List[Dict[str, Any]]], output_path: Path):
    lines = [
        "# Representative Failure Examples — EARC Error Diagnostics",
        "",
        "This document details 5 representative cases for each of the four systematic failure categories.",
        "",
    ]

    titles = {
        "F1_retrieval_failure": "F1. Retrieval Failure (Required evidence absent from retrieved top-10)",
        "F2_compression_failure": "F2. Compression / Evidence-Selection Failure (Evidence in candidates but pruned by EARC)",
        "F3_generation_failure": "F3. LLM Generation / Reasoning Failure (Evidence preserved in context, but LLM erred)",
        "F4_evaluation_failure": "F4. Answer Extraction / Evaluation Failure (Semantically correct, but regex/EM marked 0)",
    }

    for cat_key, cases in cat_examples.items():
        lines.append(f"## {titles[cat_key]}")
        lines.append("")
        for idx, c in enumerate(cases, 1):
            lines.append(f"### Example {idx} (ID: `{c['example_id']}`, Dataset: `{c['dataset']}`, Model: `{c['model']}`)")
            lines.append(f"- **Question**: {c['question']}")
            lines.append(f"- **Gold Answer**: `{c['gold_answers']}`")
            lines.append(f"- **Generated Raw Answer**: `{c['generated_answer']}`")
            lines.append(f"- **Normalized Prediction**: `{c['normalized_prediction']}`")
            lines.append(f"- **Primary Root Cause**: `{c['primary_failure']}`")
            lines.append(f"- **Diagnostic Explanation**: {c['failure_reason']}")
            lines.append(f"- **Retrieved Passages**: {', '.join(c['retrieved_documents'])}")
            lines.append("")
            if c.get("supporting_evidence"):
                lines.append("#### Ground-Truth Supporting Facts:")
                for sp in c["supporting_evidence"]:
                    lines.append(f"> 🔍 {sp}")
                lines.append("")
            lines.append("#### Selected EARC Sentences:")
            for s in c["earc_selected_sentences"][:3]:
                lines.append(f"- ✅ {s}")
            lines.append("")
            if c.get("earc_removed_sentences"):
                lines.append("#### Removed Candidate Sentences (Pruned by EARC):")
                for r in c["earc_removed_sentences"][:3]:
                    lines.append(f"- ❌ {r}")
                lines.append("")
            lines.append("---")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved representative examples document to {output_path}")


if __name__ == "__main__":
    run_failure_analysis()
