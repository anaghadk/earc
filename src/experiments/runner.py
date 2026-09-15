import json
import traceback
from pathlib import Path
from typing import List
from tqdm import tqdm

from src.data.schemas import QAExample, RetrievedDocument, ExperimentResult
from src.config.settings import EARCConfig
from src.retrieval.base import Retriever
from src.compression.compressor import EvidenceAwareCompressor
from src.llm.base import LLMProvider
from src.prompting.builder import PromptBuilder
from src.evaluation.exact_match import exact_match_score
from src.evaluation.f1 import max_token_f1_score
from src.evaluation.latency import LatencyTracker
from src.evaluation.answer_extraction import extract_answer

from src.utils.io import append_jsonl, ensure_dir
from src.utils.logging import get_logger
from src.utils.monitoring import ResourceMonitor

logger = get_logger(__name__)

class ExperimentRunner:
    def __init__(
        self,
        config: EARCConfig,
        retriever: Retriever,
        compressor: EvidenceAwareCompressor,
        provider: LLMProvider,
        prompt_builder: PromptBuilder
    ):
        self.config = config
        self.retriever = retriever
        self.compressor = compressor
        self.provider = provider
        self.prompt_builder = prompt_builder
        self.resource_monitor = ResourceMonitor()

    def _build_retrieval_pool(self, example: QAExample) -> List[RetrievedDocument]:
        """Build retrieval pool from dataset documents or external retriever."""
        if example.documents and len(example.documents) > 0:
            pool = []
            for i, doc in enumerate(example.documents):
                pool.append(RetrievedDocument(
                    doc_id=doc.doc_id or f"doc_{i}",
                    title=doc.title,
                    text=doc.text,
                    score=1.0 - (i * 0.01),  # rank-order proxy score
                    rank=i + 1,
                    retriever="dataset",
                    query=example.question,
                ))
            return pool
        else:
            return self.retriever.retrieve(example.question, self.config.retrieval.top_k)

    def run_single(self, example: QAExample) -> ExperimentResult:
        """Run the full pipeline for a single example."""
        latency_tracker = LatencyTracker()

        try:
            # 1. Retrieval
            with latency_tracker.track("retrieval"):
                retrieved_docs = self._build_retrieval_pool(example)

            # 2. Compression
            with latency_tracker.track("compression"):
                compressed_context = self.compressor.compress(
                    example.question,
                    retrieved_docs,
                    dataset=example.dataset,
                    metadata=example.metadata,
                )

            # 3. Prompt Building
            with latency_tracker.track("prompt_building"):
                prompt = self.prompt_builder.build_from_compressed(example.question, compressed_context)

            # 4. Generation
            with latency_tracker.track("generation"):
                generation_result = self.provider.generate(prompt)

            # 5. Evaluation
            with latency_tracker.track("evaluation"):
                evaluated_answer = extract_answer(generation_result.text)
                em_score = exact_match_score(evaluated_answer, example.answers)
                f1_score = max_token_f1_score(evaluated_answer, example.answers)["f1"]

            # Memory tracking
            self.resource_monitor.track_gpu_memory()

            return ExperimentResult(
                example_id=example.id,
                dataset=example.dataset,
                question=example.question,
                gold_answers=example.answers,
                retrieved_documents=[d.to_dict() for d in retrieved_docs],
                candidate_count=len(compressed_context.all_candidates),
                selected_sentences=[s.to_dict() for s in compressed_context.selected_sentences],
                original_tokens=compressed_context.original_tokens,
                compressed_tokens=compressed_context.compressed_tokens,
                compression_percentage=compressed_context.reduction_percentage,
                alpha=self.config.compression.alpha,
                beta=self.config.compression.beta,
                redundancy_threshold=self.config.compression.redundancy_threshold,
                token_budget=self.config.compression.token_budget,
                llm_provider=self.provider.provider_name,
                llm_model=self.provider.model_name,
                prediction=generation_result.text,
                evaluated_answer=evaluated_answer,
                compressed_context_text=compressed_context.compressed_text,
                exact_match=em_score,
                f1=f1_score,
                retrieval_latency=latency_tracker.get_last("retrieval") or 0.0,
                compression_latency=latency_tracker.get_last("compression") or 0.0,
                generation_latency=latency_tracker.get_last("generation") or 0.0,
                end_to_end_latency=latency_tracker.end_to_end() or 0.0,
                peak_gpu_memory_mb=self.resource_monitor.peak_gpu_mb,
            )

        except Exception as e:
            logger.error(f"Error processing example {example.id}: {e}")
            logger.error(traceback.format_exc())
            self.resource_monitor.track_gpu_memory()
            return ExperimentResult(
                example_id=example.id,
                dataset=example.dataset,
                question=example.question,
                gold_answers=example.answers,
                retrieved_documents=[],
                candidate_count=0,
                selected_sentences=[],
                original_tokens=0,
                compressed_tokens=0,
                compression_percentage=0.0,
                alpha=self.config.compression.alpha,
                beta=self.config.compression.beta,
                redundancy_threshold=self.config.compression.redundancy_threshold,
                token_budget=self.config.compression.token_budget,
                llm_provider=self.provider.provider_name,
                llm_model=self.provider.model_name,
                prediction="",
                evaluated_answer="",
                compressed_context_text="",
                exact_match=0.0,
                f1=0.0,
                retrieval_latency=latency_tracker.get_last("retrieval") or 0.0,
                compression_latency=latency_tracker.get_last("compression") or 0.0,
                generation_latency=latency_tracker.get_last("generation") or 0.0,
                end_to_end_latency=latency_tracker.end_to_end() or 0.0,
                peak_gpu_memory_mb=self.resource_monitor.peak_gpu_mb,
                method="error",
            )

    def run_dataset(self, examples: List[QAExample], method: str, output_dir: Path, resume: bool = True) -> List[ExperimentResult]:
        """Run experiments on a dataset with checkpointing."""
        ensure_dir(output_dir)
        predictions_file = output_dir / "predictions.jsonl"
        
        completed_ids = set()
        if resume and predictions_file.exists():
            with open(predictions_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            completed_ids.add(data["example_id"])
                        except json.JSONDecodeError:
                            pass
                            
            logger.info(f"Resuming from checkpoint. Found {len(completed_ids)} completed examples.")

        results = []
        for i, example in enumerate(tqdm(examples, desc=f"Running {method}")):
            if example.id in completed_ids:
                continue
                
            result = self.run_single(example)
            result.method = method  # override method if specified
            
            append_jsonl(predictions_file, [result.to_dict()])
            results.append(result)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(examples)} examples.")
                
        return results
