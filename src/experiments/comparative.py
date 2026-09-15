import json
from pathlib import Path
from typing import List, Dict

from src.data.schemas import QAExample, LLMProviderName
from src.config.settings import EARCConfig
from src.experiments.runner import ExperimentRunner
from src.experiments.proposed import ProposedExperiment
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

class ComparativeExperiments:
    def __init__(self, config: EARCConfig):
        self.config = config

    def run_token_budget_curve(self, examples: Dict[str, List[QAExample]], output_dir: Path) -> Dict:
        """Run proposed method at different token budgets to generate a curve."""
        budgets = [50, 100, 150, 200, 250, 300, 400, 500]
        results = {}
        ensure_dir(output_dir)
        
        for budget in budgets:
            budget_dir = output_dir / f"budget_{budget}"
            logger.info(f"Running token budget curve: T={budget}")
            
            exp = ProposedExperiment(self.config)
            results[str(budget)] = exp.run_token_matched(examples, budget_dir, baseline_token_count=budget)
            
        return results

    def run_retriever_comparison(self, examples: Dict[str, List[QAExample]], output_dir: Path) -> Dict:
        """Compare BM25 vs Dense vs Hybrid retrieval with the same compression settings."""
        logger.info("Running retriever comparison (dense, bm25, hybrid)")
        exp = ProposedExperiment(self.config)
        return exp.run_method_comparison(["dense", "bm25", "hybrid"], examples, output_dir)

    def run_provider_comparison(self, examples: Dict[str, List[QAExample]], output_dir: Path) -> Dict:
        """Compare Ollama vs Mistral on the same examples."""
        ensure_dir(output_dir)
        results = {}
        
        from src.retrieval.dense import DenseRetriever
        from src.retrieval.rag_project import RAGProjectRetriever
        from src.compression.compressor import EvidenceAwareCompressor
        from src.compression.embeddings import EmbeddingEngine
        from src.llm.factory import create_provider
        from src.prompting.builder import PromptBuilder
        from src.utils.device import DeviceManager
        
        providers = ["ollama", "mistral"]
        device = DeviceManager().get_device().type
        
        for provider_name in providers:
            logger.info(f"Running provider comparison: {provider_name}")
            provider_dir = output_dir / provider_name
            
            config_copy = self.config.model_copy(deep=True)
            config_copy.generation.default_provider = provider_name
            
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
            provider = create_provider(config_copy, provider_name)
            token_counter = provider.get_token_counter()
            compressor = EvidenceAwareCompressor(config_copy.compression, embedding_engine, token_counter, device=device)
            prompt_builder = PromptBuilder()
            
            runner = ExperimentRunner(
                config=config_copy,
                retriever=retriever,
                compressor=compressor,
                provider=provider,
                prompt_builder=prompt_builder
            )
            
            results[provider_name] = {}
            for dataset_name, dataset_examples in examples.items():
                dataset_dir = provider_dir / dataset_name
                prov_dir = dataset_dir / provider_name
                
                run_results = runner.run_dataset(dataset_examples, method="proposed", output_dir=prov_dir)
                
                from src.evaluation.aggregation import aggregate_by_dataset
                aggregated = aggregate_by_dataset(run_results)
                save_json(prov_dir / "metrics.json", aggregated)
                
                results[provider_name][dataset_name] = aggregated
                
        return results
