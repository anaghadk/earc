"""
Reporting module for generating tables and plots.

Produces publication-quality tables and figures for:
- Main comparison (method × dataset × LLM)
- Statistical significance
- Ablation study
- Hardware metrics
- Token budget curves
- Compression metrics
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_main_comparison_table(
    results: dict[str, dict[str, dict[str, Any]]],
    output_path: Path,
) -> str:
    """
    Generate main comparison table: Dataset × Method × LLM × metrics.

    Args:
        results: Nested dict: method -> dataset -> {avg_tokens, em, f1, compression_pct, latency}
        output_path: Path to save the table (CSV/markdown).

    Returns:
        Formatted markdown table string.
    """
    lines = [
        "| Dataset | Method | LLM | Avg Tokens | EM | F1 | Compression % | Latency (s) |",
        "|---------|--------|-----|------------|----|----|---------------|-------------|",
    ]

    for method, datasets in sorted(results.items()):
        for dataset, metrics in sorted(datasets.items()):
            llm = metrics.get("llm_provider", "—")
            tokens = metrics.get("avg_compressed_tokens", 0)
            em = metrics.get("em_mean", 0) * 100
            f1 = metrics.get("f1_mean", 0) * 100
            comp = metrics.get("avg_compression_pct", 0)
            lat = metrics.get("avg_latency", 0)
            lines.append(
                f"| {dataset} | {method} | {llm} | {tokens:.0f} | "
                f"{em:.1f} | {f1:.1f} | {comp:.1f} | {lat:.2f} |"
            )

    table = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table)

    logger.info(f"Main comparison table saved to {output_path}")
    return table


def generate_significance_table(
    results: list[dict[str, Any]],
    output_path: Path,
) -> str:
    """
    Generate significance table.

    Args:
        results: List of SignificanceResult.to_dict() dicts.
        output_path: Path to save.

    Returns:
        Formatted markdown table.
    """
    lines = [
        "| Comparison | Dataset | ΔF1 | p-value | Significant |",
        "|------------|---------|-----|---------|-------------|",
    ]

    for r in results:
        sig = "✓" if r.get("is_significant", False) else "✗ (n.s.)"
        p_str = f"p<0.001" if r["p_value"] < 0.001 else f"p={r['p_value']:.3f}"
        lines.append(
            f"| {r['comparison']} | {r['dataset']} | "
            f"{r['observed_delta']:+.1f} | {p_str} | {sig} |"
        )

    table = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table)

    logger.info(f"Significance table saved to {output_path}")
    return table


def generate_ablation_table(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> str:
    """Generate ablation study table."""
    lines = [
        "| Configuration | Dataset | Tokens | EM | F1 | Latency (s) |",
        "|---------------|---------|--------|----|----|-------------|",
    ]

    for config, val in results.items():
        if isinstance(val, dict) and any(isinstance(v, dict) for v in val.values()):
            for ds, metrics in val.items():
                if isinstance(metrics, dict) and ds in metrics:
                    metrics = metrics[ds]
                tokens = metrics.get("avg_compressed_tokens", 0) if isinstance(metrics, dict) else 0
                em = (metrics.get("em_mean", 0) * 100) if isinstance(metrics, dict) else 0
                f1 = (metrics.get("f1_mean", 0) * 100) if isinstance(metrics, dict) else 0
                lat = metrics.get("avg_latency", 0) if isinstance(metrics, dict) else 0
                lines.append(f"| {config} | {ds} | {tokens:.0f} | {em:.1f} | {f1:.1f} | {lat:.2f} |")
        else:
            metrics = val if isinstance(val, dict) else {}
            tokens = metrics.get("avg_compressed_tokens", 0)
            em = metrics.get("em_mean", 0) * 100
            f1 = metrics.get("f1_mean", 0) * 100
            lat = metrics.get("avg_latency", 0)
            lines.append(f"| {config} | all | {tokens:.0f} | {em:.1f} | {f1:.1f} | {lat:.2f} |")

    table = "\n".join(lines)

    target_file = output_path / "ablation_table.md" if (output_path.is_dir() or not output_path.suffix) else output_path
    target_file.parent.mkdir(parents=True, exist_ok=True)
    with open(target_file, "w") as f:
        f.write(table)

    return table


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _setup_plot_style():
    """Configure publication-quality plot defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_bar_comparison(
    data: dict[str, float],
    title: str,
    ylabel: str,
    output_dir: Path,
    filename: str,
    formats: list[str] = None,
    colors: Optional[list[str]] = None,
) -> None:
    """
    Generate a bar chart comparing methods.

    Args:
        data: Dict of method_name -> value.
        title: Plot title.
        ylabel: Y-axis label.
        output_dir: Directory to save figures.
        filename: Base filename (without extension).
        formats: Output formats (default: ['png', 'pdf']).
        colors: Optional list of colors.
    """
    if formats is None:
        formats = ["png", "pdf"]

    _setup_plot_style()

    if colors is None:
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0",
                   "#00BCD4", "#795548", "#607D8B"]

    methods = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        range(len(methods)), values,
        color=colors[:len(methods)],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{filename}.{fmt}"
        fig.savefig(path)
        logger.info(f"Plot saved: {path}")

    plt.close(fig)


def plot_token_budget_curve(
    budget_results: dict[int, dict[str, float]],
    output_dir: Path,
    filename: str = "token_budget_curve",
    formats: list[str] = None,
) -> None:
    """
    Plot F1/EM vs token budget.

    Args:
        budget_results: Dict of budget -> {em_mean, f1_mean, compression_pct_mean}.
        output_dir: Directory to save.
        filename: Base filename.
        formats: Output formats.
    """
    if formats is None:
        formats = ["png", "pdf"]

    _setup_plot_style()

    budgets = sorted(budget_results.keys())
    f1_scores = [budget_results[b].get("f1_mean", 0) * 100 for b in budgets]
    em_scores = [budget_results[b].get("em_mean", 0) * 100 for b in budgets]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_f1 = "#2196F3"
    color_em = "#FF9800"

    ax1.plot(budgets, f1_scores, "o-", color=color_f1, label="F1", linewidth=2, markersize=6)
    ax1.plot(budgets, em_scores, "s--", color=color_em, label="EM", linewidth=2, markersize=6)
    ax1.set_xlabel("Token Budget")
    ax1.set_ylabel("Score (%)")
    ax1.set_title("F1 and EM vs Token Budget")
    ax1.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{filename}.{fmt}"
        fig.savefig(path)

    plt.close(fig)


def plot_dataset_comparison(
    dataset_results: dict[str, dict[str, dict[str, float]]],
    metric: str,
    output_dir: Path,
    filename: str,
    formats: list[str] = None,
) -> None:
    """
    Plot grouped bar chart: methods grouped by dataset.

    Args:
        dataset_results: method -> dataset -> {metric: value}
        metric: Which metric to plot (e.g., 'f1_mean').
        output_dir: Directory to save.
        filename: Base filename.
    """
    if formats is None:
        formats = ["png", "pdf"]

    _setup_plot_style()

    methods = sorted(dataset_results.keys())
    datasets = sorted(
        set(d for m in dataset_results.values() for d in m.keys())
    )

    x = np.arange(len(datasets))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

    for i, method in enumerate(methods):
        values = [
            dataset_results[method].get(d, {}).get(metric, 0) * 100
            for d in datasets
        ]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=method, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(f"{metric} by Method and Dataset")
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{filename}.{fmt}"
        fig.savefig(path)

    plt.close(fig)


def plot_provider_comparison(
    provider_results: dict[str, dict[str, float]],
    output_dir: Path,
    filename: str = "ollama_vs_mistral",
    formats: list[str] = None,
) -> None:
    """Plot Ollama vs Mistral comparison."""
    if formats is None:
        formats = ["png", "pdf"]

    _setup_plot_style()

    providers = sorted(provider_results.keys())
    metrics = ["em_mean", "f1_mean"]
    metric_labels = ["EM", "F1"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5))

    colors = ["#2196F3", "#FF9800"]

    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [provider_results[p].get(metric, 0) * 100 for p in providers]
        ax.bar(providers, values, color=colors[:len(providers)])
        ax.set_ylabel(f"{label} (%)")
        ax.set_title(f"{label} by Provider")

        for i, v in enumerate(values):
            ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)

    plt.suptitle("Ollama vs Mistral Comparison")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{filename}.{fmt}"
        fig.savefig(path)

    plt.close(fig)


def generate_all_plots(
    results_summary: dict[str, Any],
    output_dir: Path,
    formats: list[str] = None,
) -> None:
    """
    Generate all 10+ required publication-quality figures.

    Args:
        results_summary: Complete experiment results dict.
        output_dir: Directory to save all figures.
        formats: Output formats.
    """
    if formats is None:
        formats = ["png", "pdf"]

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating plots to {figures_dir}")

    # Generate each plot if the data is available
    if "method_tokens" in results_summary:
        plot_bar_comparison(
            results_summary["method_tokens"],
            "Average Prompt Tokens by Method",
            "Tokens",
            figures_dir, "avg_tokens_by_method", formats,
        )

    if "method_f1" in results_summary:
        plot_bar_comparison(
            results_summary["method_f1"],
            "F1 Score by Method",
            "F1 (%)",
            figures_dir, "f1_by_method", formats,
        )

    if "method_em" in results_summary:
        plot_bar_comparison(
            results_summary["method_em"],
            "Exact Match by Method",
            "EM (%)",
            figures_dir, "em_by_method", formats,
        )

    if "method_compression" in results_summary:
        plot_bar_comparison(
            results_summary["method_compression"],
            "Compression Percentage by Method",
            "Compression (%)",
            figures_dir, "compression_by_method", formats,
        )

    if "method_latency" in results_summary:
        plot_bar_comparison(
            results_summary["method_latency"],
            "Latency by Method",
            "Latency (s)",
            figures_dir, "latency_by_method", formats,
        )

    if "budget_curve" in results_summary:
        plot_token_budget_curve(
            results_summary["budget_curve"],
            figures_dir, "token_budget_curve", formats,
        )

    if "provider_results" in results_summary:
        plot_provider_comparison(
            results_summary["provider_results"],
            figures_dir, "ollama_vs_mistral", formats,
        )

    logger.info(f"All plots generated in {figures_dir}")


def save_metrics_json(
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Save metrics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {output_path}")
