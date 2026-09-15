"""
src/ui/adapter.py — Adapter connecting the Streamlit UI to the finalized EARC backend.
"""

from __future__ import annotations

import re
import time
import numpy as np
from typing import Any, Dict, List, Optional

from src.config.settings import load_config, EARCConfig
from src.retrieval.rag_project import RAGProjectRetriever
from src.compression.embeddings import EmbeddingEngine
from src.compression.compressor import EvidenceAwareCompressor
from src.prompting.builder import PromptBuilder
from src.prompting.templates import DEFAULT_QA_TEMPLATE
from src.llm.factory import create_provider
from src.evaluation.answer_extraction import extract_answer
from src.evaluation.exact_match import normalize_answer
from src.data.schemas import SentenceStatus, RetrievedDocument
from src.utils.device import DeviceManager


class UIEARCPipeline:
    """
    Adapter that executes the finalized EARC pipeline (Dense top_k=10, budget=300,
    title-aware evidence scoring, redundancy filtering) and formats results
    for the Streamlit chat interface.
    """

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        llm_model: Optional[str] = None,
        provider_name: Optional[str] = None,
    ):
        self.config: EARCConfig = load_config(config_path)
        self.device = DeviceManager().get_device().type

        # Ensure finalized retrieval defaults are locked
        self.config.retrieval.top_k = 10
        self.config.retrieval.retrieval_method = "dense"
        self.config.retrieval.backend = "dense"
        self.config.compression.alpha = 0.7
        self.config.compression.beta = 0.3
        self.config.compression.redundancy_threshold = 0.85
        self.config.compression.token_budget = 300

        # Model / provider resolution
        if llm_model:
            model_lower = llm_model.lower().strip()
            if "mistral" in model_lower:
                self.provider_name = "mistral"
            else:
                self.provider_name = "ollama"
                # Map llama3 to llama3.2 if needed
                if model_lower in ["llama3", "llama3.2"]:
                    self.config.ollama.model = "llama3.2"
                else:
                    self.config.ollama.model = llm_model
        elif provider_name:
            self.provider_name = provider_name
        else:
            self.provider_name = self.config.generation.default_provider

        # Initialize shared backend components
        rag_dir = getattr(self.config.retrieval, "rag_project_dir", "RAG_Project")
        self.retriever = RAGProjectRetriever(
            rag_dir=rag_dir,
            model_name=self.config.retrieval.embedding_model,
            device=self.device,
            method="dense",
        )
        self.embedding_engine = EmbeddingEngine(
            self.config.retrieval.embedding_model, device=self.device
        )
        self.provider = create_provider(self.config, self.provider_name)
        self.token_counter = self.provider.get_token_counter()
        self.compressor = EvidenceAwareCompressor(
            self.config.compression,
            self.embedding_engine,
            self.token_counter,
            device=self.device,
        )
        self.prompt_builder = PromptBuilder(DEFAULT_QA_TEMPLATE)

    def _detect_query_type(self, query: str) -> str:
        from src.compression.query_classification import classify_question
        q_type, _ = classify_question(query)
        return q_type

    def _extract_keywords(self, query: str) -> List[str]:
        words = re.findall(r"\b[A-Za-z0-9_-]+\b", query)
        stopwords = {
            "what", "which", "who", "whom", "this", "that", "these", "those", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "a", "an", "the", "and", "but", "if", "or", "as", "of", "at", "by", "for", "with",
            "in", "out", "on", "off", "to", "from", "up", "down", "how", "why", "when", "where"
        }
        return [w for w in words if w.lower() not in stopwords]

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute full EARC pipeline on user query and return dictionary
        formatted for the Streamlit UI.
        """
        t0 = time.time()

        from src.compression.query_classification import classify_question
        q_type, q_details = classify_question(query)

        # 1. Retrieval
        retrieved_docs = self.retriever.retrieve(query, top_k=self.config.retrieval.top_k)

        # 2. Evidence-Aware Compression
        compressed = self.compressor.compress(query, retrieved_docs)

        # 3. Prompt Building
        prompt = self.prompt_builder.build_from_compressed(query, compressed, q_type=q_type, q_details=q_details)

        # 4. Generation
        gen_result = self.provider.generate(prompt)
        evaluated_answer = extract_answer(gen_result.text) or gen_result.text.strip()

        latency_ms = (time.time() - t0) * 1000.0

        # Candidate sentence lists
        all_candidates = compressed.all_candidates
        selected_candidates = [
            s for s in all_candidates if s.status == SentenceStatus.SELECTED
        ]
        leftover_candidates = [
            s for s in all_candidates if s.status != SentenceStatus.SELECTED
        ]
        redundant_candidates = [
            s for s in all_candidates if s.status == SentenceStatus.REJECTED_REDUNDANCY
        ]

        # Citations list for UI display
        citations = []
        for idx, s in enumerate(selected_candidates, 1):
            citations.append({
                "marker": f"[{idx}]",
                "text": s.text,
                "title": s.title,
                "score": float(s.hybrid_score),
                "is_bridge": False,
            })

        # Verification metrics
        norm_ans = normalize_answer(evaluated_answer)
        norm_ctx = normalize_answer(compressed.compressed_text)
        ans_tokens = set(norm_ans.split())
        ctx_tokens = set(norm_ctx.split())

        overlap_tokens = ans_tokens.intersection(ctx_tokens)
        faithfulness = len(overlap_tokens) / len(ans_tokens) if ans_tokens else 0.0
        mean_overlap = len(overlap_tokens) / len(ans_tokens.union(ctx_tokens)) if (ans_tokens or ctx_tokens) else 0.0

        refusal_triggers = [
            "cannot", "can't", "insufficient evidence", "not mentioned",
            "not provide", "does not mention", "unknown"
        ]
        is_refusal = any(trigger in evaluated_answer.lower() for trigger in refusal_triggers)
        grounded = (not is_refusal) and (len(overlap_tokens) > 0)

        verification = {
            "grounded": grounded,
            "is_refusal": is_refusal,
            "faithfulness": float(faithfulness),
            "mean_overlap": float(mean_overlap),
            "citation_count": len(citations),
            "invalid_citations": [],
        }

        query_info = {
            "query_type": self._detect_query_type(query),
            "keywords": self._extract_keywords(query),
        }

        scoring_stats = {
            "step4": {"total_embedded": len(all_candidates)},
            "step5": {
                "mean_score": float(np.mean([s.hybrid_score for s in all_candidates]))
                if all_candidates else 0.0
            },
            "step6": {
                "input_sentences": len(all_candidates),
                "output_sentences": len(all_candidates) - len(redundant_candidates),
                "removed": len(redundant_candidates),
            },
        }

        selection_stats = {
            "budget": {
                "tokens_used": compressed.compressed_tokens,
                "budget": self.config.compression.token_budget,
                "bridge_selected": 0,
            }
        }

        generation = {
            "backend": f"{self.provider.provider_name} ({self.provider.model_name})",
            "prompt": prompt,
            "citations": citations,
            "verification": verification,
        }

        return {
            "query": query,
            "answer": evaluated_answer,
            "raw_prediction": gen_result.text,
            "latency_ms": latency_ms,
            "original_tokens": compressed.original_tokens,
            "compressed_tokens": compressed.compressed_tokens,
            "compression_percentage": compressed.reduction_percentage,
            "query_info": query_info,
            "scoring_stats": scoring_stats,
            "selection_stats": selection_stats,
            "generation": generation,
            "selected_sentences": [s.to_dict() for s in selected_candidates],
            "candidate_sentences": [s.to_dict() for s in leftover_candidates],
            "retrieved_sentences": [s.to_dict() for s in all_candidates],
            "retrieved_documents": [d.to_dict() for d in retrieved_docs],
        }

    def generate_baseline(
        self,
        query: str,
        retrieved_docs: Optional[List[RetrievedDocument]] = None,
    ) -> Dict[str, Any]:
        """
        Generate uncompressed Standard RAG baseline for comparison.
        """
        if not retrieved_docs:
            retrieved_docs = self.retriever.retrieve(query, top_k=self.config.retrieval.top_k)

        full_context_parts = []
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                t = doc.get("title")
                txt = doc.get("text", "")
            else:
                t = getattr(doc, "title", None)
                txt = getattr(doc, "text", "")
            full_context_parts.append(f"{t}: {txt}" if t else txt)
        full_context = "\n\n".join(full_context_parts)
        base_prompt = self.prompt_builder.build(query, full_context)
        gen_result = self.provider.generate(base_prompt)
        evaluated_answer = extract_answer(gen_result.text) or gen_result.text.strip()

        return {
            "answer": evaluated_answer,
            "prompt": base_prompt,
            "raw_prediction": gen_result.text,
        }
