import argparse
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging
from src.utils.io import ensure_dir
from datasets import load_dataset

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Download datasets for EARC")
    parser.add_argument("--cache-dir", default="data/cache", help="Cache directory")
    parser.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa", "triviaqa"], help="Datasets to download")
    args = parser.parse_args()

    setup_logging("INFO")
    
    ensure_dir(args.cache_dir)
    ensure_dir("data/raw")

    hf_mapping = {
        "nq": ("nq_open", None),
        "hotpotqa": ("hotpot_qa", "distractor"),
        "triviaqa": ("trivia_qa", "rc")
    }

    for ds in args.datasets:
        if ds not in hf_mapping:
            logger.warning(f"Unknown dataset {ds}, skipping.")
            continue
        
        hf_name, hf_subset = hf_mapping[ds]
        logger.info(f"Downloading {ds} ({hf_name} {hf_subset if hf_subset else ''})...")
        try:
            if hf_subset:
                dataset = load_dataset(hf_name, hf_subset, cache_dir=args.cache_dir)
            else:
                dataset = load_dataset(hf_name, cache_dir=args.cache_dir)
            
            dataset.save_to_disk(f"data/raw/{ds}")
            logger.info(f"Downloaded {ds} successfully. Saved to data/raw/{ds}.")
            
        except Exception as e:
            logger.error(f"Failed to download {ds}: {e}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
