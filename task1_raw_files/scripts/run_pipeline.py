import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import load_config
from src.data.loaders import load_dataset
from src.data.sampling import sample_dataset
from src.experiments.proposed import ProposedExperiment
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.utils.device import DeviceManager
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run main pipeline for EARC")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--dataset", choices=["nq", "hotpotqa", "triviaqa", "all"], default="all", help="Dataset to evaluate")
    parser.add_argument("--provider", choices=["ollama", "mistral", "mock"], default="mock", help="LLM provider")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples for debugging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--token-budget", type=int, default=None, help="Token budget override")
    parser.add_argument("--alpha", type=float, default=None, help="Alpha override")
    parser.add_argument("--beta", type=float, default=None, help="Beta override")
    parser.add_argument("--redundancy-threshold", type=float, default=None, help="Redundancy threshold override")
    parser.add_argument("--top-k", type=int, default=None, help="Top K override")
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default=None, help="Retriever override")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["dense", "bm25", "hybrid"],
        default=None,
        metavar="METHOD",
        help="Compare multiple retrieval methods in ONE process (e.g. --methods dense bm25 hybrid). "
             "Artifacts are loaded once and shared via the module-level cache.",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)
    
    config = load_config(args.config)
    device_manager = DeviceManager()
    device = device_manager.get_device()
    
    # Overrides
    if args.token_budget is not None:
        config.compression.token_budget = args.token_budget
    if args.alpha is not None:
        config.compression.alpha = args.alpha
    if args.beta is not None:
        config.compression.beta = args.beta
    if args.redundancy_threshold is not None:
        config.compression.redundancy_threshold = args.redundancy_threshold
    if args.top_k is not None:
        config.retrieval.top_k = args.top_k
    if args.retriever is not None:
        config.retrieval.backend = args.retriever
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"run_{timestamp}"
    ensure_dir(out_dir)

    # Set provider from CLI into config so ProposedExperiment picks it up
    config.generation.default_provider = args.provider

    print("=== EARC Pipeline ===")
    print(f"Dataset: {args.dataset}")
    print(f"Provider: {args.provider}")
    print(f"Budget: {config.compression.token_budget}")
    if args.limit:
        print(f"Examples limit: {args.limit}")
    print(f"Output dir: {out_dir}")
    print("=====================")

    try:
        datasets_to_run = ["nq", "hotpotqa", "triviaqa"] if args.dataset == "all" else [args.dataset]

        split_map = {
            "nq": config.data.nq_split,
            "hotpotqa": config.data.hotpotqa_split,
            "triviaqa": config.data.triviaqa_split,
        }

        examples = {}
        for ds in datasets_to_run:
            logger.info(f"Loading dataset: {ds}")
            loaded = load_dataset(ds, split=split_map[ds], cache_dir=config.data.cache_dir)
            if args.limit is not None:
                loaded = sample_dataset(loaded, args.limit, args.seed)
            examples[ds] = loaded

        experiment = ProposedExperiment(config)
        if args.methods:
            # Single-process multi-method comparison — artifacts loaded once, then cached
            logger.info(f"Running method comparison: {args.methods}")
            experiment.run_method_comparison(args.methods, examples, out_dir)
        else:
            experiment.run(examples, out_dir)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
