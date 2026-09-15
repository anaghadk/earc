import argparse
import sys
from pathlib import Path
import logging
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

def run_cmd(cmd):
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run all experiments for EARC")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--provider", choices=["ollama", "mistral", "mock"], default="mock", help="Provider")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples")
    parser.add_argument("--output-dir", type=str, default="outputs/all_experiments", help="Output directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baselines")
    args = parser.parse_args()

    setup_logging()
    ensure_dir(args.output_dir)

    try:
        # Step 1: Download
        if not args.skip_download:
            run_cmd(["python", "scripts/download_data.py"])
            run_cmd(["python", "scripts/prepare_data.py", "--config", args.config])
            run_cmd(["python", "scripts/build_index.py", "--config", args.config])

        # Step 2: Proposed
        cmd_proposed = ["python", "scripts/run_pipeline.py", "--config", args.config, "--provider", args.provider, "--output-dir", f"{args.output_dir}/proposed"]
        if args.limit:
            cmd_proposed.extend(["--limit", str(args.limit)])
        run_cmd(cmd_proposed)

        # Step 3: Baselines
        if not args.skip_baselines:
            cmd_baselines = ["python", "scripts/run_baselines.py", "--config", args.config, "--provider", args.provider, "--output-dir", f"{args.output_dir}/baselines"]
            if args.limit:
                cmd_baselines.extend(["--limit", str(args.limit)])
            run_cmd(cmd_baselines)

        # Step 4: Ablations
        cmd_ablations = ["python", "scripts/run_ablations.py", "--config", args.config, "--provider", args.provider, "--output-dir", f"{args.output_dir}/ablations"]
        if args.limit:
            cmd_ablations.extend(["--limit", str(args.limit)])
        run_cmd(cmd_ablations)

        # Step 5: Evaluation
        run_cmd(["python", "scripts/run_evaluation.py", "--results-dir", args.output_dir, "--output-dir", f"{args.output_dir}/eval"])

        logger.info("All experiments completed successfully.")
    except Exception as e:
        logger.error(f"Error running all experiments: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
