"""
Experiment 6 — Statistical Validation of EARC Evaluation Results
Computes paired bootstrap 95% confidence intervals at the QUESTION level (N=5,000 resamples).
Evaluates:
- EARC vs Standard RAG
- EARC vs Top-k
- EARC vs LLMLingua-2

Separately across:
- NQ + Llama 3.2
- NQ + Ministral-8B
- HotpotQA + Llama 3.2
- HotpotQA + Ministral-8B
- TriviaQA + Llama 3.2
- TriviaQA + Ministral-8B

Outputs:
- Publication-ready Markdown tables
- Raw bootstrap distributions and summary JSON/CSV
- Publication-quality Forest Plots of Δ (with 95% CI error bars)
"""

import os
import sys
import json
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging import get_logger

logger = get_logger("statistical_validation")


def paired_bootstrap_test(
    scores_earc: np.ndarray,
    scores_base: np.ndarray,
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes paired bootstrap confidence interval for (EARC - Baseline)."""
    assert len(scores_earc) == len(scores_base)
    n = len(scores_earc)
    diff = scores_earc - scores_base
    obs_diff = float(np.mean(diff))

    rng = np.random.RandomState(seed)
    boot_diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_diffs[i] = np.mean(diff[idx])

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    contains_zero = bool(ci_lower <= 0.0 <= ci_upper)
    is_significant = not contains_zero

    # Two-sided empirical p-value under H0
    centered = boot_diffs - np.mean(boot_diffs)
    p_value = float(np.mean(np.abs(centered) >= np.abs(obs_diff)))

    return {
        "observed_earc_mean": float(np.mean(scores_earc)),
        "observed_base_mean": float(np.mean(scores_base)),
        "observed_diff": obs_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "contains_zero": contains_zero,
        "is_significant": is_significant,
        "p_value": p_value,
        "n_resamples": n_resamples,
        "n_examples": n,
    }


def run_statistical_validation(
    final_eval_dir_str: str = "outputs/final_evaluation",
    output_dir_str: str = "outputs/experiments/statistical_validation",
    n_resamples: int = 5000,
    seed: int = 42,
):
    final_eval_dir = Path(final_eval_dir_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"EARC STATISTICAL VALIDATION (N={n_resamples} Paired Bootstrap Resamples)")
    logger.info("=" * 80)

    providers = [("mistral", "Ministral-8B"), ("ollama", "Llama 3.2")]
    datasets = [("hotpot", "HotpotQA"), ("nq", "NQ"), ("trivia", "TriviaQA")]
    baselines = [("standard_rag", "Standard RAG"), ("topk", "Top-k"), ("llmlingua2", "LLMLingua-2")]

    all_comparisons = []
    raw_results = {}

    for prov_key, prov_name in providers:
        raw_results[prov_key] = {}
        for ds_key, ds_name in datasets:
            raw_results[prov_key][ds_key] = {}
            earc_file = final_eval_dir / prov_key / ds_key / "earc" / "predictions.jsonl"
            if not earc_file.exists():
                logger.warning(f"Missing {earc_file}")
                continue

            with open(earc_file, "r", encoding="utf-8") as f:
                earc_data = [json.loads(line) for line in f]

            earc_em = np.array([r["exact_match"] * 100 for r in earc_data], dtype=np.float64)
            earc_f1 = np.array([r["f1"] * 100 for r in earc_data], dtype=np.float64)

            for base_key, base_name in baselines:
                base_file = final_eval_dir / prov_key / ds_key / base_key / "predictions.jsonl"
                if not base_file.exists():
                    logger.warning(f"Missing {base_file}")
                    continue

                with open(base_file, "r", encoding="utf-8") as f:
                    base_data = [json.loads(line) for line in f]

                base_em = np.array([r["exact_match"] * 100 for r in base_data], dtype=np.float64)
                base_f1 = np.array([r["f1"] * 100 for r in base_data], dtype=np.float64)

                res_em = paired_bootstrap_test(earc_em, base_em, n_resamples=n_resamples, seed=seed)
                res_f1 = paired_bootstrap_test(earc_f1, base_f1, n_resamples=n_resamples, seed=seed)

                comp_entry = {
                    "provider": prov_key,
                    "model": prov_name,
                    "dataset_key": ds_key,
                    "dataset": ds_name,
                    "baseline_key": base_key,
                    "baseline": base_name,
                    "n_examples": len(earc_em),
                    "em": res_em,
                    "f1": res_f1,
                }
                all_comparisons.append(comp_entry)
                raw_results[prov_key][ds_key][base_key] = comp_entry

                logger.info(
                    f"[{prov_name} - {ds_name} vs {base_name}] "
                    f"ΔEM={res_em['observed_diff']:+.2f}% 95%CI=[{res_em['ci_lower']:+.2f}%, {res_em['ci_upper']:+.2f}%] (Zero in CI: {res_em['contains_zero']}) | "
                    f"ΔF1={res_f1['observed_diff']:+.2f}% 95%CI=[{res_f1['ci_lower']:+.2f}%, {res_f1['ci_upper']:+.2f}%] (Zero in CI: {res_f1['contains_zero']})"
                )

    # 1. Save JSON & CSV
    json_path = output_dir / "statistical_validation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)
    logger.info(f"Saved summary JSON to {json_path}")

    csv_path = output_dir / "statistical_validation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Dataset", "Baseline", "Metric", "EARC_Mean_Pct", "Baseline_Mean_Pct",
            "Observed_Diff_Pct", "CI_Lower_95", "CI_Upper_95", "Contains_Zero",
            "Is_Significant_p05", "p_value"
        ])
        for c in all_comparisons:
            for met_name, met_dict in [("EM", c["em"]), ("F1", c["f1"])]:
                writer.writerow([
                    c["model"], c["dataset"], c["baseline"], met_name,
                    f"{met_dict['observed_earc_mean']:.2f}",
                    f"{met_dict['observed_base_mean']:.2f}",
                    f"{met_dict['observed_diff']:+.2f}",
                    f"{met_dict['ci_lower']:+.2f}",
                    f"{met_dict['ci_upper']:+.2f}",
                    met_dict["contains_zero"],
                    met_dict["is_significant"],
                    f"{met_dict['p_value']:.4f}",
                ])
    logger.info(f"Saved summary CSV to {csv_path}")

    # 2. Generate Markdown Tables
    md_path = output_dir / "statistical_validation_tables.md"
    generate_markdown_report(all_comparisons, md_path)

    # 3. Generate Forest Plots
    generate_forest_plots(all_comparisons, output_dir)
    logger.info("Generated statistical validation plots successfully.")


def generate_markdown_report(comparisons: List[Dict[str, Any]], output_path: Path):
    lines = [
        "# Experiment 6 — Statistical Validation of EARC Evaluation Results",
        "",
        "## 1. Executive Summary & Methodology",
        "- **Resampling Procedure**: Paired bootstrap resampling at the **question level** with **5,000 resamples**.",
        "- **Evaluation Scale**: 1,000 matched examples per dataset (`seed=42`).",
        "- **Difference Metric**: $\\Delta = \\text{EARC} - \\text{Baseline}$.",
        "- **Significance Threshold**: Two-sided $\\alpha = 0.05$. A difference is declared statistically significant if and only if the **95% Bootstrap Confidence Interval strictly excludes zero** ($p < 0.05$).",
        "",
        "## 2. Statistical Comparison Tables",
    ]

    configs = [
        ("nq", "Llama 3.2"),
        ("nq", "Ministral-8B"),
        ("hotpot", "Llama 3.2"),
        ("hotpot", "Ministral-8B"),
        ("trivia", "Llama 3.2"),
        ("trivia", "Ministral-8B"),
    ]

    for ds_key, mod_name in configs:
        sub_comps = [c for c in comparisons if c["dataset_key"] == ds_key and c["model"] == mod_name]
        if not sub_comps:
            continue
        ds_title = sub_comps[0]["dataset"]
        lines.append(f"### Configuration: `{ds_title} + {mod_name}`")
        lines.append("")
        lines.append("| Metric | Comparison | EARC Mean (%) | Baseline Mean (%) | Absolute Difference (Δ) | 95% Bootstrap CI | Contains Zero? | Statistically Significant? |")
        lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for c in sub_comps:
            for m_label, m_data in [("EM", c["em"]), ("F1", c["f1"])]:
                sig_text = "**Yes (p < 0.05)**" if m_data["is_significant"] else "No (p ≥ 0.05)"
                zero_text = "No" if not m_data["contains_zero"] else "Yes"
                lines.append(
                    f"| **{m_label}** | EARC vs. {c['baseline']} | {m_data['observed_earc_mean']:.2f}% | "
                    f"{m_data['observed_base_mean']:.2f}% | **{m_data['observed_diff']:+.2f}%** | "
                    f"[{m_data['ci_lower']:+.2f}%, {m_data['ci_upper']:+.2f}%] | {zero_text} | {sig_text} |"
                )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_forest_plots(comparisons: List[Dict[str, Any]], output_dir: Path):
    for m_key, m_title in [("em", "Exact Match (EM)"), ("f1", "Token F1")]:
        plt.figure(figsize=(10, 8))
        labels = []
        diffs = []
        ci_lows = []
        ci_highs = []
        colors = []

        for c in comparisons:
            m = c[m_key]
            lbl = f"{c['dataset']} ({c['model']}) vs {c['baseline']}"
            labels.append(lbl)
            diffs.append(m["observed_diff"])
            ci_lows.append(m["ci_lower"])
            ci_highs.append(m["ci_upper"])
            colors.append("#2ca02c" if m["is_significant"] and m["observed_diff"] > 0 else ("#d62728" if m["is_significant"] else "#7f7f7f"))

        y_pos = np.arange(len(labels))[::-1]

        for y, d, cl, ch, col in zip(y_pos, diffs, ci_lows, ci_highs, colors):
            plt.errorbar(d, y, xerr=[[d - cl], [ch - d]], fmt="o", color=col, ecolor=col, elinewidth=2.2, capsize=4, markersize=6.5)
        plt.axvline(0.0, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Zero Effect Line (Δ = 0)")

        plt.yticks(y_pos, labels, fontsize=9)
        plt.xlabel(f"Absolute Difference Δ = EARC - Baseline (%) in {m_title}", fontsize=11, fontweight="bold")
        plt.title(f"Forest Plot: 95% Bootstrap Confidence Intervals for Δ ({m_title})", fontsize=12, fontweight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.legend(loc="lower right", fontsize=9.5)
        plt.tight_layout()

        plot_path = output_dir / f"delta_forest_plot_{m_key}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 6: Statistical Validation")
    parser.add_argument("--resamples", type=int, default=5000, help="Number of paired bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_statistical_validation(n_resamples=args.resamples, seed=args.seed)
