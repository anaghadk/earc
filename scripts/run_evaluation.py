import argparse
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.reporting import generate_all_plots, save_metrics_json
from src.utils.logging import setup_logging
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run evaluation for EARC")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing predictions")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation", help="Output directory")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000, help="Bootstrap resamples")
    parser.add_argument("--significance-resamples", type=int, default=1000, help="Significance resamples")
    args = parser.parse_args()

    setup_logging()
    ensure_dir(args.output_dir)

    try:
        logger.info(f"Evaluating results from {args.results_dir}")
        save_metrics_json(args.results_dir, args.output_dir, args.bootstrap_resamples)
        generate_all_plots(args.results_dir, args.output_dir)
        logger.info("Evaluation completed successfully.")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
