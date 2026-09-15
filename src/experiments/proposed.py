import json
from pathlib import Path
from typing import List, Dict

from src.data.schemas import QAExample
from src.config.settings import EARCConfig
from src.retrieval.base import Retriever
from src.compression.compressor import EvidenceAwareCompressor
from src.llm.factory import create_provider
from src.prompting.builder import PromptBuilder
from src.experiments.runner import ExperimentRunner
from src.evaluation.aggregation import aggregate_by_dataset
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

class ProposedExperiment:
    def __init__(self, config: EARCConfig):
        self.config = config

    def _setup_components(self, provider_name: str, config: EARCConfig):
        from src.retrieval.dense import DenseRetriever
        from src.retrieval.rag_project import RAGProjectRetriever
        from src.llm.factory import create_provider
        from src.compression.embeddings import EmbeddingEngine
        from src.utils.device import DeviceManager

        device = DeviceManager().get_device().type  # str: "cuda", "mps", or "cpu"

        # Use pre-built RAG_Project index if configured; otherwise fall back to DenseRetriever
        rag_dir = getattr(config.retrieval, "rag_project_dir", None)
        if rag_dir:
            retriever = RAGProjectRetriever(
                rag_dir=rag_dir,
                model_name=config.retrieval.embedding_model,
                device=device,
                method=getattr(config.retrieval, "retrieval_method", "hybrid"),
            )
        else:
            retriever = DenseRetriever(config.retrieval.embedding_model, device=device)

        embedding_engine = EmbeddingEngine(config.retrieval.embedding_model, device=device)
        provider = create_provider(config, provider_name)
        token_counter = provider.get_token_counter()
        compressor = EvidenceAwareCompressor(config.compression, embedding_engine, token_counter, device=device)
        prompt_builder = PromptBuilder()

        return ExperimentRunner(
            config=config,
            retriever=retriever,
            compressor=compressor,
            provider=provider,
            prompt_builder=prompt_builder
        )

    def run(self, examples: Dict[str, List[QAExample]], output_dir: Path) -> Dict:
        """Run proposed method across datasets and providers."""
        ensure_dir(output_dir)
        all_results = {}
        
        for dataset_name, dataset_examples in examples.items():
            dataset_dir = output_dir / dataset_name
            all_results[dataset_name] = {}
            
            # Since the user config might have multiple providers, loop through them
            # For simplicity, let's use the one in config
            provider_name = self.config.generation.default_provider
            provider_dir = dataset_dir / provider_name
            
            runner = self._setup_components(provider_name, self.config)
            
            logger.info(f"Running proposed method on {dataset_name} with {provider_name}")
            results = runner.run_dataset(dataset_examples, method="proposed", output_dir=provider_dir)
            
            aggregated = aggregate_by_dataset(results)
            save_json(provider_dir / "metrics.json", aggregated)
            all_results[dataset_name][provider_name] = aggregated
            
        return all_results

    def run_method_comparison(
        self,
        methods: List[str],
        examples: Dict[str, List[QAExample]],
        output_dir: Path,
    ) -> Dict:
        """Run dense, bm25, and hybrid in ONE Python process, sharing cached RAG_Project artifacts.

        The module-level _ARTIFACT_CACHE in rag_project.py ensures that the heavy
        BM25 / FAISS / chunk artifacts are loaded from disk exactly once.  Each
        subsequent RAGProjectRetriever instance in the same process borrows the cached
        references in < 1 ms.

        Strategy:
          1. Build compressor, LLM provider, and PromptBuilder ONCE — they are
             independent of the retrieval method and are reused across all methods.
          2. Sort methods so "hybrid" executes first.  Hybrid loads all four artifacts
             (lookup + FAISS + BM25 + tokenized), filling the cache completely.
             Dense and BM25 then get a full cache hit at zero I/O cost.
          3. Create a new RAGProjectRetriever per method (cheap — only binds cached refs),
             then wrap it in an ExperimentRunner with the shared components.
          4. Write results to output_dir/{method}/{dataset}/{provider}/ so each method
             is clearly identifiable and comparable.

        Do NOT call this method without rag_project_dir configured in the config.
        """
        from src.retrieval.rag_project import RAGProjectRetriever
        from src.compression.embeddings import EmbeddingEngine
        from src.utils.device import DeviceManager

        ensure_dir(output_dir)
        provider_name = self.config.generation.default_provider
        device = DeviceManager().get_device().type

        rag_dir = getattr(self.config.retrieval, "rag_project_dir", None)
        if not rag_dir:
            raise ValueError(
                "retrieval.rag_project_dir must be set in config for --methods comparison. "
                "Add 'rag_project_dir: RAG_Project' under 'retrieval:' in your YAML."
            )

        # ── Build shared components ONCE ──────────────────────────────────────
        embedding_engine = EmbeddingEngine(self.config.retrieval.embedding_model, device=device)
        provider = create_provider(self.config, provider_name)
        token_counter = provider.get_token_counter()
        compressor = EvidenceAwareCompressor(
            self.config.compression, embedding_engine, token_counter, device=device
        )
        prompt_builder = PromptBuilder()

        # ── Sort so hybrid runs first (loads all artifacts → full cache for rest) ─
        # Priority: hybrid=0, dense=1, bm25=2  (anything else appended last)
        _PRIORITY = {"hybrid": 0, "dense": 1, "bm25": 2}
        ordered = sorted(set(methods), key=lambda m: _PRIORITY.get(m, 99))
        logger.info(f"Method comparison order: {ordered}  (hybrid first for cache warming)")

        # ── Run each method ───────────────────────────────────────────────────
        all_results: Dict[str, Dict] = {}
        for method in ordered:
            logger.info(f"=== Retrieval method: {method} ===")

            retriever = RAGProjectRetriever(
                rag_dir=rag_dir,
                model_name=self.config.retrieval.embedding_model,
                device=device,
                method=method,
            )
            runner = ExperimentRunner(
                config=self.config,
                retriever=retriever,
                compressor=compressor,
                provider=provider,
                prompt_builder=prompt_builder,
            )

            all_results[method] = {}
            for dataset_name, dataset_examples in examples.items():
                method_dir = output_dir / method / dataset_name / provider_name
                logger.info(f"  Running {method} on {dataset_name} -> {method_dir}")
                results = runner.run_dataset(
                    dataset_examples,
                    method=f"proposed_{method}",
                    output_dir=method_dir,
                )
                aggregated = aggregate_by_dataset(results)
                save_json(method_dir / "metrics.json", aggregated)
                all_results[method][dataset_name] = aggregated
                logger.info(f"  [{method}] {dataset_name}: EM={aggregated.get('em_mean', 'N/A')}")

        return all_results

    def run_aggressive(self, examples: Dict[str, List[QAExample]], output_dir: Path) -> Dict:
        """Run proposed method with aggressive token budget (T=200)."""
        logger.info("Running aggressive compression (T=200)")
        original_budget = self.config.compression.token_budget
        self.config.compression.token_budget = 200
        
        results = self.run(examples, output_dir / "aggressive")
        
        # Restore
        self.config.compression.token_budget = original_budget
        return results

    def run_token_matched(self, examples: Dict[str, List[QAExample]], output_dir: Path, baseline_token_count: int) -> Dict:
        """Run proposed method matching a baseline token count."""
        logger.info(f"Running token matched compression (T={baseline_token_count})")
        original_budget = self.config.compression.token_budget
        self.config.compression.token_budget = baseline_token_count
        
        results = self.run(examples, output_dir / "token_matched")
        
        # Restore
        self.config.compression.token_budget = original_budget
        return results
