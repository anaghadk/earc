import time
import logging
from typing import List
from src.data.schemas import RetrievedDocument, CompressedContext
from src.config.settings import CompressionConfig
from src.llm.base import TokenCounter
from src.compression.segmentation import SentenceSegmenter
from src.compression.embeddings import EmbeddingEngine
from src.compression.relevance import SemanticRelevanceScorer
from src.compression.evidence import EvidenceScorer
from src.compression.ranking import HybridRanker
from src.compression.redundancy import RedundancyFilter
from src.compression.budget import TokenBudgetSelector

logger = logging.getLogger(__name__)

class EvidenceAwareCompressor:
    def __init__(self, config: CompressionConfig, embedding_engine: EmbeddingEngine, token_counter: TokenCounter, device: str = 'cuda'):
        self.config = config
        self.embedding_engine = embedding_engine
        self.token_counter = token_counter
        self.device = device
        
        self.segmenter = SentenceSegmenter(spacy_model=getattr(config, 'spacy_model', 'en_core_web_sm'))
        self.relevance_scorer = SemanticRelevanceScorer()
        self.evidence_scorer = EvidenceScorer(spacy_model=getattr(config, 'spacy_model', 'en_core_web_sm'))
        self.ranker = HybridRanker()
        self.redundancy_filter = RedundancyFilter()
        self.budget_selector = TokenBudgetSelector(token_counter)

    def compress(
        self,
        query: str,
        retrieved_docs: List[RetrievedDocument],
        dataset: str | None = None,
        metadata: dict | None = None,
    ) -> CompressedContext:
        start_time = time.time()
        timings = {}
        
        if not retrieved_docs:
            logger.warning("No retrieved documents provided for compression.")
            return CompressedContext(
                original_text="",
                compressed_text="",
                candidates=[],
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=0.0,
                reduction_percentage=0.0
            )

        from src.compression.query_classification import classify_question
        q_type, q_details = classify_question(query, dataset=dataset, metadata=metadata)

        t0 = time.time()
        candidates = self.segmenter.segment(retrieved_docs)
        timings['segmentation'] = time.time() - t0
        
        if not candidates:
            logger.warning("No candidates generated from documents.")
            return CompressedContext(
                original_text="",
                compressed_text="",
                candidates=[],
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=0.0,
                reduction_percentage=0.0
            )
            
        t0 = time.time()
        query_embedding = self.embedding_engine.embed_query(query)
        candidates = self.embedding_engine.embed_candidates(candidates)
        timings['embedding'] = time.time() - t0
        
        t0 = time.time()
        candidates = self.relevance_scorer.compute_relevance(query_embedding, candidates)
        timings['semantic_relevance'] = time.time() - t0
        
        t0 = time.time()
        candidates = self.evidence_scorer.score(candidates, query, q_type=q_type, q_details=q_details)
        candidates = self.evidence_scorer.normalize(candidates)
        timings['evidence_scoring'] = time.time() - t0
        
        t0 = time.time()
        alpha = getattr(self.config, 'alpha', 0.7)
        beta = getattr(self.config, 'beta', 0.3)
        candidates = self.ranker.rank(candidates, alpha=alpha, beta=beta)
        timings['ranking'] = time.time() - t0
        
        t0 = time.time()
        threshold = getattr(self.config, 'redundancy_threshold', 0.85)
        surviving = self.redundancy_filter.filter(candidates, threshold=threshold)
        timings['redundancy_filtering'] = time.time() - t0
        
        t0 = time.time()
        token_budget = getattr(self.config, 'token_budget', 1000)
        selected = self.budget_selector.select(surviving, budget=token_budget, q_type=q_type, q_details=q_details)
        timings['budget_selection'] = time.time() - t0
        
        t0 = time.time()
        compressed_text = ' '.join(s.text for s in selected)
        
        original_text = ' '.join(doc.text for doc in retrieved_docs)
        original_tokens = self.token_counter.count(original_text)
        compressed_tokens = self.token_counter.count(compressed_text)
        
        compression_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else float('inf')
        reduction_percentage = max(0.0, (1.0 - (compressed_tokens / original_tokens)) * 100.0) if original_tokens > 0 else 0.0
        timings['output_construction'] = time.time() - t0
        
        total_time = time.time() - start_time
        logger.info(f"Compression completed in {total_time:.3f}s. Timings: {timings}")
        logger.info(f"Original tokens: {original_tokens}, Compressed tokens: {compressed_tokens}")
        logger.info(f"Compression ratio: {compression_ratio:.2f}, Reduction: {reduction_percentage:.2f}%")
        
        return CompressedContext(
            original_text=original_text,
            compressed_text=compressed_text,
            candidates=candidates,
            selected_sentences=selected,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            reduction_percentage=reduction_percentage
        )
