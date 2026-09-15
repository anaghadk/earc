import json
from pathlib import Path
from typing import List, Dict, Optional

from src.data.schemas import QAExample
from src.config.settings import EARCConfig
from src.experiments.runner import ExperimentRunner
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

class BaselineExperiments:
    def __init__(self, config: EARCConfig):
        self.config = config

    def _setup_runner(self, baseline_name: str):
        from src.retrieval.dense import DenseRetriever
        from src.retrieval.rag_project import RAGProjectRetriever
        from src.compression.compressor import EvidenceAwareCompressor
        from src.compression.embeddings import EmbeddingEngine
        from src.llm.factory import create_provider
        from src.prompting.builder import PromptBuilder
        from src.utils.device import DeviceManager
        
        config_copy = self.config.model_copy(deep=True)
        if baseline_name == "standard_rag":
            config_copy.compression.token_budget = 2000
            config_copy.compression.redundancy_threshold = 1.0
        elif baseline_name == "topk":
            config_copy.compression.alpha = 1.0
            config_copy.compression.beta = 0.0
            config_copy.compression.redundancy_threshold = 1.0
            
        device = DeviceManager().get_device().type
        rag_dir = getattr(config_copy.retrieval, "rag_project_dir", None)
        if rag_dir:
            retriever = RAGProjectRetriever(
                rag_dir=rag_dir,
                model_name=config_copy.retrieval.embedding_model,
                device=device,
                method=getattr(config_copy.retrieval, "retrieval_method", "hybrid"),
            )
        else:
            retriever = DenseRetriever(config_copy.retrieval.embedding_model, device=device)

        embedding_engine = EmbeddingEngine(config_copy.retrieval.embedding_model, device=device)
        provider_name = config_copy.generation.default_provider
        provider = create_provider(config_copy, provider_name)
        token_counter = provider.get_token_counter()
        compressor = EvidenceAwareCompressor(config_copy.compression, embedding_engine, token_counter, device=device)
        prompt_builder = PromptBuilder()
        
        return ExperimentRunner(
            config=config_copy,
            retriever=retriever,
            compressor=compressor,
            provider=provider,
            prompt_builder=prompt_builder
        )

    def run_all(self, examples: Dict[str, List[QAExample]], output_dir: Path) -> Dict:
        baselines = ["standard_rag", "topk", "summarization", "llmlingua2"]
        results = {}
        
        for baseline in baselines:
            results[baseline] = self._run_single_baseline(baseline, examples, output_dir)
                
        return results

    def run_standard_rag(self, examples: Dict[str, List[QAExample]], output_dir: Path):
        return self._run_single_baseline("standard_rag", examples, output_dir)
        
    def run_topk(self, examples: Dict[str, List[QAExample]], output_dir: Path):
        return self._run_single_baseline("topk", examples, output_dir)
        
    def run_summarization(self, examples: Dict[str, List[QAExample]], output_dir: Path):
        return self._run_single_baseline("summarization", examples, output_dir)
        
    def run_llmlingua2(self, examples: Dict[str, List[QAExample]], output_dir: Path):
        return self._run_single_baseline("llmlingua2", examples, output_dir)
        
    def _run_single_baseline(self, baseline_name: str, examples: Dict[str, List[QAExample]], output_dir: Path) -> Dict:
        runner = self._setup_runner(baseline_name)
        results = {}
        baseline_dir = output_dir / baseline_name
        ensure_dir(baseline_dir)
        
        for dataset_name, dataset_examples in examples.items():
            dataset_dir = baseline_dir / dataset_name
            provider_name = self.config.generation.default_provider
            provider_dir = dataset_dir / provider_name
            
            logger.info(f"Running baseline {baseline_name} on {dataset_name}")
            run_results = runner.run_dataset(dataset_examples, method=baseline_name, output_dir=provider_dir)
            
            from src.evaluation.aggregation import aggregate_by_dataset
            aggregated = aggregate_by_dataset(run_results)
            save_json(provider_dir / "metrics.json", aggregated)
            results[dataset_name] = aggregated
            
        return results

    def run(
        self,
        dataset_name: str,
        provider: str = "mock",
        limit: Optional[int] = None,
        output_dir: str | Path = "outputs/baselines",
        baselines: Optional[List[str]] = None,
    ) -> Dict:
        """Entry point for CLI or single-dataset runs."""
        from src.data.loaders import load_dataset
        from src.data.sampling import sample_dataset

        config_copy = self.config.model_copy(deep=True)
        config_copy.generation.default_provider = provider
        self.config = config_copy

        split_map = {
            "nq": self.config.data.nq_split,
            "hotpotqa": self.config.data.hotpotqa_split,
            "triviaqa": self.config.data.triviaqa_split,
        }
        loaded = load_dataset(dataset_name, split=split_map.get(dataset_name, "validation"), cache_dir=self.config.data.cache_dir)
        if limit is not None:
            loaded = sample_dataset(loaded, limit, self.config.seed)

        examples = {dataset_name: loaded}
        target_baselines = baselines or ["standard_rag", "topk"]

        out_path = Path(output_dir)
        results = {}
        for b in target_baselines:
            results[b] = self._run_single_baseline(b, examples, out_path)
        return results
