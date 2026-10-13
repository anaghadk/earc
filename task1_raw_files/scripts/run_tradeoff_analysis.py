"""
Experiment 7 — Compression–Quality–Cost Trade-off Analysis
Analyzes the multi-dimensional trade-offs between:
1. Context size (tokens)
2. Compression ratio
3. Answer quality (EM and F1)
4. Compression overhead (latency)
5. End-to-end latency

Calculates clearly defined derived efficiency measures:
1. F1 per 1,000 input tokens
2. EM per 1,000 input tokens
3. Token reduction per second of compression overhead

Generates:
- Publication-quality trade-off plots:
  A. F1 vs average input tokens
  B. EM vs average input tokens
  C. Compression ratio vs F1
  D. E2E latency vs compressed tokens
- Comprehensive Markdown comparison tables
- Trade-off regions identification and scientific discussion
"""

import os
import sys
import json
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging import get_logger

logger = get_logger("tradeoff_analysis")


def parse_float(val: Any, default: float = 0.0) -> float:
    if val is None or str(val).strip().upper() in ["N/A", "NONE", ""]:
        return default
    try:
        return float(val)
    except Exception:
        return default


def run_tradeoff_analysis(
    final_csv_path: str = "outputs/final_evaluation/FINAL_COMPARISON.csv",
    output_dir_str: str = "outputs/experiments/tradeoff_analysis",
):
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("EARC EXPERIMENT 7: COMPRESSION-QUALITY-COST TRADE-OFF ANALYSIS")
    logger.info("=" * 80)

    df = pd.read_csv(final_csv_path)

    processed_rows = []

    for _, row in df.iterrows():
        prov = str(row["Provider"]).lower()
        model = str(row["Model"])
        ds = str(row["Dataset"]).lower()
        method = str(row["Method"]).lower()
        n_ex = int(row["N_Examples"])

        em = float(row["Exact_Match_Pct"])
        f1 = float(row["Token_F1_Pct"])
        orig_t = float(row["Avg_Orig_Tokens"])
        proc_t = float(row["Avg_Processed_Tokens"])

        # Compression ratio & percentage
        if method == "standard_rag":
            comp_pct = 0.0
            comp_ratio = 1.00
            comp_lat = 0.0
        elif method == "topk":
            comp_pct = round(max(0.0, 1.0 - (proc_t / orig_t)) * 100, 2)
            comp_ratio = round(orig_t / proc_t, 2) if proc_t > 0 else 1.0
            comp_lat = 0.0
        else:
            comp_pct = parse_float(row["Avg_Comp_Pct"])
            comp_ratio = parse_float(row["Avg_Comp_Ratio"], default=1.0)
            comp_lat = parse_float(row["Avg_Compression_Latency_s"])

        ret_lat = parse_float(row["Avg_Retrieval_Latency_s"])
        gen_lat = parse_float(row["Avg_Generation_Latency_s"])
        e2e_lat = parse_float(row["Avg_E2E_Latency_s"])

        # -------------------------------------------------------------
        # Clearly Defined Derived Efficiency Measures:
        # -------------------------------------------------------------
        # 1. F1 per 1,000 input tokens
        f1_per_1k = round((f1 / (proc_t / 1000.0)), 2) if proc_t > 0 else 0.0

        # 2. EM per 1,000 input tokens
        em_per_1k = round((em / (proc_t / 1000.0)), 2) if proc_t > 0 else 0.0

        # 3. Token reduction per second of additional compression overhead
        tokens_saved = max(0.0, orig_t - proc_t)
        if comp_lat > 0.0:
            tokens_saved_per_sec = round(tokens_saved / comp_lat, 1)
        else:
            tokens_saved_per_sec = None  # Not applicable (zero compression overhead)

        processed_rows.append({
            "Provider": prov,
            "Model": "Ministral-8B" if prov == "mistral" else "Llama 3.2",
            "Dataset": ds.upper(),
            "Method": method,
            "Method_Display": "Standard RAG" if method == "standard_rag" else ("Top-k (top 5)" if method == "topk" else ("LLMLingua-2 (300t)" if method == "llmlingua2" else "EARC (300t)")),
            "N_Examples": n_ex,
            "Avg_Orig_Tokens": orig_t,
            "Avg_Processed_Tokens": proc_t,
            "Compression_Percentage": comp_pct,
            "Compression_Ratio": comp_ratio,
            "Exact_Match_Pct": em,
            "Token_F1_Pct": f1,
            "Avg_Retrieval_Latency_s": ret_lat,
            "Avg_Compression_Latency_s": comp_lat,
            "Avg_Generation_Latency_s": gen_lat,
            "Avg_E2E_Latency_s": e2e_lat,
            "F1_per_1k_tokens": f1_per_1k,
            "EM_per_1k_tokens": em_per_1k,
            "Tokens_saved_per_overhead_sec": tokens_saved_per_sec,
        })

    tradeoff_df = pd.DataFrame(processed_rows)

    # Save summary CSV & JSON
    summary_csv = output_dir / "tradeoff_summary.csv"
    tradeoff_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved tradeoff summary to {summary_csv}")

    summary_json = output_dir / "tradeoff_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(processed_rows, f, indent=2)
    logger.info(f"Saved tradeoff JSON to {summary_json}")

    # Generate Markdown comparison tables
    generate_markdown_report(processed_rows, output_dir / "tradeoff_analysis_tables.md")

    # Generate the 4 publication-quality plots
    generate_tradeoff_plots(tradeoff_df, output_dir)
    logger.info("Generated all 4 trade-off plots successfully.")


def generate_tradeoff_plots(df: pd.DataFrame, output_dir: Path):
    method_styles = {
        "standard_rag": {"label": "Standard RAG", "color": "#2ca02c", "marker": "o"},
        "topk": {"label": "Top-k", "color": "#1f77b4", "marker": "s"},
        "llmlingua2": {"label": "LLMLingua-2", "color": "#d62728", "marker": "^"},
        "earc": {"label": "EARC", "color": "#ff7f0e", "marker": "D"},
    }

    # Helper function for grid scatter plots
    def plot_scatter(x_col, y_col, x_label, y_label, title, filename):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
        models = [("Ministral-8B", "mistral"), ("Llama 3.2", "ollama")]

        for ax_idx, (m_label, p_key) in enumerate(models):
            ax = axes[ax_idx]
            sub = df[df["Provider"] == p_key]

            for ds in ["HOTPOT", "NQ", "TRIVIA"]:
                ds_sub = sub[sub["Dataset"] == ds]
                for method in ["standard_rag", "topk", "llmlingua2", "earc"]:
                    m_row = ds_sub[ds_sub["Method"] == method]
                    if m_row.empty:
                        continue
                    r = m_row.iloc[0]
                    style = method_styles[method]
                    ax.scatter(
                        r[x_col], r[y_col],
                        color=style["color"],
                        marker=style["marker"],
                        s=130,
                        edgecolor="black",
                        linewidth=1.2,
                        zorder=4,
                        label=style["label"] if ds == "HOTPOT" else None
                    )
                    ax.annotate(
                        f"{ds}",
                        (r[x_col], r[y_col]),
                        textcoords="offset points",
                        xytext=(0, 7),
                        ha="center",
                        fontsize=8.5,
                        fontweight="bold",
                        color="#333333"
                    )

            ax.set_title(f"{title} — {m_label}", fontsize=12, fontweight="bold")
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
            if ax_idx == 1:
                ax.legend(loc="best", fontsize=9.5)

        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300)
        plt.close()

    # Plot A: F1 vs Average Input Tokens
    plot_scatter(
        "Avg_Processed_Tokens", "Token_F1_Pct",
        "Average Input Context Tokens (Prompt Size)", "Token F1 (%)",
        "A. Token F1 vs. Input Context Tokens",
        "f1_vs_input_tokens.png"
    )

    # Plot B: EM vs Average Input Tokens
    plot_scatter(
        "Avg_Processed_Tokens", "Exact_Match_Pct",
        "Average Input Context Tokens (Prompt Size)", "Exact Match (%)",
        "B. Exact Match vs. Input Context Tokens",
        "em_vs_input_tokens.png"
    )

    # Plot C: Compression Ratio vs F1
    plot_scatter(
        "Compression_Ratio", "Token_F1_Pct",
        "Compression Ratio (Original / Compressed Tokens)", "Token F1 (%)",
        "C. Compression Ratio vs. Token F1",
        "compression_ratio_vs_f1.png"
    )

    # Plot D: E2E Latency vs Compressed Tokens
    plot_scatter(
        "Avg_Processed_Tokens", "Avg_E2E_Latency_s",
        "Average Processed Tokens", "End-to-End Latency (seconds)",
        "D. End-to-End Latency vs. Processed Tokens",
        "latency_vs_compressed_tokens.png"
    )


def generate_markdown_report(records: List[Dict[str, Any]], output_path: Path):
    lines = [
        "# Experiment 7 — Compression–Quality–Cost Trade-off Analysis Report",
        "",
        "## 1. Executive Summary & Evaluation Dimensions",
        "This experiment evaluates the multi-dimensional efficiency trade-offs between:",
        "1. **Context Size**: Average prompt tokens fed into the LLM.",
        "2. **Compression Ratio**: Factor of context reduction ($T_{\\text{orig}} / T_{\\text{compressed}}$).",
        "3. **Answer Quality**: Exact Match (EM) and Token F1.",
        "4. **Compression Overhead**: Execution latency added by candidate decomposition, scoring, and filtering.",
        "5. **End-to-End Latency**: Total pipeline time (Retrieval + Compression + Generation).",
        "",
        "### Clearly Defined Derived Efficiency Measures",
        "- **F1 per 1,000 Input Tokens**: $\\text{F1} / (\\text{Tokens} / 1000)$ — Measures factual information density per unit prompt cost.",
        "- **EM per 1,000 Input Tokens**: $\\text{EM} / (\\text{Tokens} / 1000)$ — Measures exact-match return on token spend.",
        "- **Token Reduction per Second of Overhead**: $(\\text{Orig Tokens} - \\text{Comp Tokens}) / \\text{Compression Latency (s)}$ — Measures the throughput of context pruning.",
        "",
        "## 2. Comprehensive Trade-off Tables",
    ]

    for prov_key, prov_name in [("mistral", "Ministral-8B"), ("ollama", "Llama 3.2")]:
        lines.append(f"### Provider / Model: `{prov_name}`")
        lines.append("")
        lines.append("| Dataset | Method | Orig Tokens | Comp Tokens | Comp % | Comp Ratio | EM (%) | F1 (%) | Comp Time (s) | Gen Time (s) | E2E Latency (s) | F1 / 1k Tokens | EM / 1k Tokens | Tokens Saved / Overhead Sec |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        sub_rows = [r for r in records if r["Provider"] == prov_key]
        for ds in ["HOTPOT", "NQ", "TRIVIA"]:
            ds_rows = [r for r in sub_rows if r["Dataset"] == ds]
            for r in ds_rows:
                comp_t_str = f"{r['Avg_Compression_Latency_s']:.3f}s" if r['Avg_Compression_Latency_s'] > 0 else "0.000s (None)"
                t_saved_str = f"{r['Tokens_saved_per_overhead_sec']:,.1f} t/s" if r['Tokens_saved_per_overhead_sec'] is not None else "N/A"
                is_earc = (r["Method"] == "earc")
                m_str = f"**{r['Method_Display']}**" if is_earc else r['Method_Display']

                lines.append(
                    f"| {ds} | {m_str} | {r['Avg_Orig_Tokens']:.1f} | {r['Avg_Processed_Tokens']:.1f} | "
                    f"{r['Compression_Percentage']:.1f}% | {r['Compression_Ratio']:.2f}x | "
                    f"**{r['Exact_Match_Pct']:.2f}%** | **{r['Token_F1_Pct']:.2f}%** | "
                    f"{comp_t_str} | {r['Avg_Generation_Latency_s']:.3f}s | {r['Avg_E2E_Latency_s']:.3f}s | "
                    f"**{r['F1_per_1k_tokens']:.1f}** | **{r['EM_per_1k_tokens']:.1f}** | {t_saved_str} |"
                )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved Markdown tables to {output_path}")


if __name__ == "__main__":
    run_tradeoff_analysis()
