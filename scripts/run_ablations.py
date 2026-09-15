import argparse
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import load_config
from src.experiments.ablations import AblationExperiments
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run ablations for EARC")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--dataset", default="hotpotqa", help="Dataset per paper")
    parser.add_argument("--provider", choices=["ollama", "mistral", "mock"], default="mock", help="Provider")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples")
    parser.add_argument("--output-dir", type=str, default="outputs/ablations", help="Output directory")
    args = parser.parse_args()

    setup_logging()
    set_seed(42)
    
    config = load_config(args.config)
    ensure_dir(args.output_dir)

    try:
        logger.info(f"Running ablations for {args.dataset}")
        experiment = AblationExperiments(config)
        experiment.run(
            dataset_name=args.dataset,
            provider=args.provider,
            limit=args.limit,
            output_dir=args.output_dir
        )
        logger.info("Ablations completed successfully.")
    except Exception as e:
        logger.error(f"Error running ablations: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
