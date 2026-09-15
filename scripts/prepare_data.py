import argparse
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import load_config
from src.data.sampling import prepare_all_datasets
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Prepare data for EARC")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-examples", type=int, default=5000, help="Number of examples per dataset")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)
    
    config = load_config(args.config)
    ensure_dir(args.output_dir)

    try:
        logger.info(f"Preparing data with seed={args.seed}, n_examples={args.n_examples}")
        prepare_all_datasets(config.data, seed=args.seed, n_examples=args.n_examples, output_dir=args.output_dir)
        logger.info("Data preparation completed successfully.")
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
