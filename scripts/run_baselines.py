import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import load_config
from src.experiments.baselines import BaselineExperiments
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run baselines for EARC")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--dataset", choices=["nq", "hotpotqa", "triviaqa", "all"], default="all", help="Dataset")
    parser.add_argument("--provider", choices=["ollama", "mistral", "mock"], default="mock", help="Provider")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples")
    parser.add_argument("--output-dir", type=str, default="outputs/baselines", help="Output directory")
    parser.add_argument("--baselines", nargs="+", default=["standard_rag", "topk", "summarization", "llmlingua2"], help="Baselines to run")
    args = parser.parse_args()

    setup_logging()
    set_seed(42)
    
    config = load_config(args.config)
    ensure_dir(args.output_dir)

    try:
        datasets_to_run = ["nq", "hotpotqa", "triviaqa"] if args.dataset == "all" else [args.dataset]
        for ds in datasets_to_run:
            logger.info(f"Running baselines for {ds}")
            experiment = BaselineExperiments(config)
            experiment.run(
                dataset_name=ds,
                provider=args.provider,
                limit=args.limit,
                output_dir=args.output_dir,
                baselines=args.baselines
            )
        logger.info("Baselines completed successfully.")
    except Exception as e:
        logger.error(f"Error running baselines: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
