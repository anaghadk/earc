import json
from pathlib import Path
from typing import List, Dict, Optional

from src.data.schemas import QAExample
from src.config.settings import EARCConfig
from src.experiments.runner import ExperimentRunner
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger
from src.evaluation.reporting import generate_ablation_table

logger = get_logger(__name__)

class AblationExperiments:
    ABLATION_CONFIGS = {
        'similarity_only': (1.0, 0.0, False),
        'evidence_only': (0.0, 1.0, False),
        'similarity_evidence': (0.7, 0.3, False),
        'full_method': (0.7, 0.3, True)
    }

    def __init__(self, config: EARCConfig):
        self.config = config

    def _setup_runner(self, alpha: float, beta: float, use_redundancy: bool):
        from src.retrieval.dense import DenseRetriever
        from src.retrieval.rag_project import RAGProjectRetriever
        from src.compression.compressor import EvidenceAwareCompressor
        from src.compression.embeddings import EmbeddingEngine
        from src.llm.factory import create_provider
        from src.prompting.builder import PromptBuilder
        from src.utils.device import DeviceManager
        
        config_copy = self.config.model_copy(deep=True)
        config_copy.compression.alpha = alpha
        config_copy.compression.beta = beta
        config_copy.compression.redundancy_threshold = self.config.compression.redundancy_threshold if use_redundancy else 1.0
            
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
        ensure_dir(output_dir)
        results = {}
        
        for config_name, (alpha, beta, use_redundancy) in self.ABLATION_CONFIGS.items():
            logger.info(f"Running ablation: {config_name}")
            ablation_dir = output_dir / config_name
            runner = self._setup_runner(alpha, beta, use_redundancy)
            results[config_name] = {}
            
            for dataset_name, dataset_examples in examples.items():
                dataset_dir = ablation_dir / dataset_name
                provider_name = self.config.generation.default_provider
                provider_dir = dataset_dir / provider_name
                
                run_results = runner.run_dataset(dataset_examples, method=config_name, output_dir=provider_dir)
                
                from src.evaluation.aggregation import aggregate_by_dataset
                aggregated = aggregate_by_dataset(run_results)
                save_json(provider_dir / "metrics.json", aggregated)
                
                results[config_name][dataset_name] = aggregated
                
        # Generate ablation table
        generate_ablation_table(results, output_dir / "ablation_table.md")
        return results

    def run(
        self,
        dataset_name: str,
        provider: str = "mock",
        limit: Optional[int] = None,
        output_dir: str | Path = "outputs/ablations",
    ) -> Dict:
        """Entry point for CLI or single-dataset ablation runs."""
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
        return self.run_all(examples, Path(output_dir))
