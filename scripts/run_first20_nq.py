#!/usr/bin/env python3
"""
Standalone temporary evaluation script for Natural Questions (NQ).

Loads NQ in its normal dataset order, takes exactly the first 20
examples using examples[:20] (no random sampling), runs the existing
EARC + Ollama pipeline, and produces all standard metrics and output files.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import load_config
from src.data.loaders import load_dataset
from src.experiments.proposed import ProposedExperiment
from src.utils.device import DeviceManager
from src.utils.io import ensure_dir
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run EARC + Ollama pipeline on the first 20 NQ examples without random sampling"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "mistral", "mock"],
        help="LLM provider (default: ollama)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory (default: outputs)",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Token budget override (default: from config)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Retrieval top-k override (default: from config)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="NQ split override (default: from config)",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    config = load_config(args.config)
    set_seed(config.data.seed)

    device_manager = DeviceManager()
    _ = device_manager.get_device()

    # Apply command-line overrides if given
    if args.token_budget is not None:
        config.compression.token_budget = args.token_budget
    if args.top_k is not None:
        config.retrieval.top_k = args.top_k

    # Configure provider
    config.generation.default_provider = args.provider

    # Timestamped output directory matching normal pipeline runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"run_{timestamp}"
    ensure_dir(out_dir)

    print("=== First 20 NQ Evaluation ===")
    print("Dataset: nq (exact first 20 examples, no random sampling)")
    print(f"Provider: {args.provider}")
    print(f"Token budget: {config.compression.token_budget}")
    print(f"Output directory: {out_dir}")
    print("===============================")

    try:
        split = args.split or config.data.nq_split
        logger.info(f"Loading dataset: nq (split={split})")
        examples = load_dataset("nq", split=split, cache_dir=config.data.cache_dir)
        logger.info(f"Loaded {len(examples)} examples in normal dataset order.")

        # Take exactly the first 20 examples (no random sampling)
        first_20 = examples[:20]
        logger.info(f"Taking first {len(first_20)} examples via examples[:20].")

        dataset_map = {"nq": first_20}

        # Run proposed EARC pipeline
        experiment = ProposedExperiment(config)
        results = experiment.run(dataset_map, out_dir)

        logger.info("Pipeline completed successfully.")

        # Display summary if available
        if "nq" in results and args.provider in results["nq"]:
            summary = results["nq"][args.provider]
            print("\n=== Aggregated Metrics ===")
            for metric, value in summary.items():
                print(f"  {metric}: {value}")
            print("==========================\n")

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
