#!/usr/bin/env python3
"""
scripts/test_single_question.py

Standalone one-off testing script for the EARC pipeline.
Accepts a question from the command line, executes the locked Dense Top-10
retrieval and Evidence-Aware Compression with Ollama generation, and prints:
- Question & detected Question Type
- Retrieved evidence documents
- Compressed context passed to the LLM
- Final extracted answer & raw prediction
- Token counts & compression percentage
- Latency breakdown (retrieval, compression, generation, total)
- Faithfulness and grounding metrics (grounded, refusal, token overlap)
- Optional EM and F1 if a gold answer is provided

Usage:
  python scripts/test_single_question.py "Who invented the telephone?"
  python scripts/test_single_question.py --question "Where was Nikola Tesla born?" --gold "Smiljan"
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import load_config
from src.retrieval.rag_project import RAGProjectRetriever
from src.compression.embeddings import EmbeddingEngine
from src.compression.compressor import EvidenceAwareCompressor
from src.compression.query_classification import classify_question
from src.prompting.builder import PromptBuilder
from src.prompting.templates import DEFAULT_QA_TEMPLATE
from src.llm.factory import create_provider
from src.evaluation.answer_extraction import extract_answer
from src.evaluation.exact_match import exact_match_score, normalize_answer
from src.evaluation.f1 import max_token_f1_score
from src.data.schemas import SentenceStatus
from src.utils.device import DeviceManager
from src.utils.logging import setup_logging


def compute_faithfulness_and_grounding(evaluated_answer: str, compressed_text: str):
    """Compute faithfulness and grounding metrics between evaluated answer and context."""
    norm_ans = normalize_answer(evaluated_answer)
    norm_ctx = normalize_answer(compressed_text)
    ans_tokens = set(norm_ans.split())
    ctx_tokens = set(norm_ctx.split())

    overlap_tokens = ans_tokens.intersection(ctx_tokens)
    faithfulness = len(overlap_tokens) / len(ans_tokens) if ans_tokens else 0.0
    mean_overlap = (
        len(overlap_tokens) / len(ans_tokens.union(ctx_tokens))
        if (ans_tokens or ctx_tokens)
        else 0.0
    )

    refusal_triggers = [
        "cannot", "can't", "insufficient evidence", "not mentioned",
        "not provide", "does not mention", "unknown"
    ]
    is_refusal = any(trigger in evaluated_answer.lower() for trigger in refusal_triggers)
    grounded = (not is_refusal) and (len(overlap_tokens) > 0)

    return {
        "grounded": grounded,
        "is_refusal": is_refusal,
        "faithfulness": float(faithfulness),
        "mean_overlap": float(mean_overlap),
        "overlap_tokens": sorted(list(overlap_tokens)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run one-off question testing through the EARC + Dense + Ollama pipeline."
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help="The query/question to evaluate.",
    )
    parser.add_argument(
        "-q", "--query",
        dest="flag_question",
        type=str,
        default=None,
        help="Alternative flag to specify the question.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "mistral", "mock"],
        help="LLM provider (default: ollama).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model override (e.g., llama3.2, mistral).",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=300,
        help="Token budget for compression (default: 300).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of documents to retrieve via Dense (default: 10).",
    )
    parser.add_argument(
        "--gold",
        nargs="+",
        default=None,
        help="Optional gold answer(s) to compute EM and F1.",
    )

    args = parser.parse_args()

    # Determine query string
    question = args.question or args.flag_question
    if not question or not question.strip():
        parser.error("A question must be provided either as a positional argument or via -q/--query.")

    question = question.strip()
    setup_logging("WARNING")

    # Load and configure settings
    config = load_config(args.config)
    config.retrieval.top_k = args.top_k
    config.retrieval.backend = "dense"
    config.retrieval.retrieval_method = "dense"
    config.compression.token_budget = args.token_budget
    config.generation.default_provider = args.provider

    if args.model:
        if args.provider == "ollama":
            config.ollama.model = args.model
        elif args.provider == "mistral":
            config.mistral.model = args.model

    device = DeviceManager().get_device().type

    # Initialize components
    rag_dir = getattr(config.retrieval, "rag_project_dir", "RAG_Project")
    retriever = RAGProjectRetriever(
        rag_dir=rag_dir,
        model_name=config.retrieval.embedding_model,
        device=device,
        method="dense",
    )
    embedding_engine = EmbeddingEngine(config.retrieval.embedding_model, device=device)
    provider = create_provider(config, args.provider)
    token_counter = provider.get_token_counter()
    compressor = EvidenceAwareCompressor(
        config.compression,
        embedding_engine,
        token_counter,
        device=device,
    )
    prompt_builder = PromptBuilder(DEFAULT_QA_TEMPLATE)

    # 1. Question Classification
    q_type, q_details = classify_question(question)

    # 2. Retrieval
    t_start = time.perf_counter()
    t0 = time.perf_counter()
    retrieved_docs = retriever.retrieve(question, top_k=config.retrieval.top_k)
    t_retrieval = time.perf_counter() - t0

    # 3. Evidence-Aware Compression
    t0 = time.perf_counter()
    compressed = compressor.compress(question, retrieved_docs)
    t_compression = time.perf_counter() - t0

    # 4. Prompt Building
    prompt = prompt_builder.build_from_compressed(
        question, compressed, q_type=q_type, q_details=q_details
    )

    # 5. Generation
    t0 = time.perf_counter()
    gen_result = provider.generate(prompt)
    t_generation = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    # Answer extraction
    evaluated_answer = extract_answer(gen_result.text) or gen_result.text.strip()

    # Faithfulness & Grounding
    verification = compute_faithfulness_and_grounding(
        evaluated_answer, compressed.compressed_text
    )

    # Selected and candidate sentences
    selected_sentences = [
        s for s in compressed.all_candidates if s.status == SentenceStatus.SELECTED
    ]

    # Ensure stdout handles UTF-8 on Windows
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    # Display results
    print("\n" + "=" * 80)
    print("                    EARC SINGLE QUESTION TEST RESULT")
    print("=" * 80)
    print(f"\n[QUESTION]")
    print(f"    {question}")

    print(f"\n[QUESTION TYPE]")
    print(f"    Type:        {q_type}")
    if q_details:
        details_str = ", ".join(f"{k}={v}" for k, v in q_details.items())
        print(f"    Attributes:  {details_str}")

    print(f"\n[RETRIEVED EVIDENCE] (Dense Top-{len(retrieved_docs)})")
    for doc in retrieved_docs:
        rank = getattr(doc, "rank", "?")
        score = getattr(doc, "score", 0.0)
        title = getattr(doc, "title", "Untitled")
        text = getattr(doc, "text", "").strip().replace("\n", " ")
        snippet = text[:120] + "..." if len(text) > 120 else text
        print(f"    #{rank} [Score: {score:.4f}] {title}")
        print(f"       {snippet}")

    print(f"\n[COMPRESSED CONTEXT] (Budget: {config.compression.token_budget} tokens)")
    if compressed.compressed_text:
        for idx, sent in enumerate(selected_sentences, 1):
            stitle = sent.title or "Doc"
            print(f"    [{idx}] ({stitle} | score: {sent.hybrid_score:.4f}) {sent.text}")
    else:
        print("    (Empty context)")

    print(f"\n[FINAL ANSWER]")
    print(f"    Evaluated:  {evaluated_answer}")
    print(f"    Raw Output: {gen_result.text.strip()}")

    # Evaluation against gold answers if supplied
    if args.gold:
        em = exact_match_score(evaluated_answer, args.gold)
        f1 = max_token_f1_score(evaluated_answer, args.gold)["f1"]
        print(f"\n[EVALUATION AGAINST GOLD ANSWERS]")
        print(f"    Gold Answers:     {args.gold}")
        print(f"    Exact Match (EM): {em:.4f}")
        print(f"    F1 Score:         {f1:.4f}")

    print(f"\n[METRICS & EFFICIENCY]")
    print(f"    - Original Tokens:       {compressed.original_tokens}")
    print(f"    - Compressed Tokens:     {compressed.compressed_tokens} / {config.compression.token_budget}")
    print(f"    - Compression %:         {compressed.reduction_percentage:.2f}% reduction")
    print(f"    - Latency (Total):       {t_total:.3f}s")
    print(f"        * Retrieval:         {t_retrieval:.3f}s")
    print(f"        * Compression:       {t_compression:.3f}s")
    print(f"        * Generation:        {t_generation:.3f}s")

    print(f"\n[FAITHFULNESS & GROUNDING]")
    print(f"    - Grounded in Context:   {verification['grounded']}")
    print(f"    - Faithfulness Score:    {verification['faithfulness'] * 100:.1f}%")
    print(f"    - Mean Token Overlap:    {verification['mean_overlap']:.4f}")
    print(f"    - Refusal Detected:      {verification['is_refusal']}")
    print(f"    - Selected Sentences:    {len(selected_sentences)}")
    if verification["overlap_tokens"]:
        print(f"    - Overlapping Tokens:    {', '.join(verification['overlap_tokens'])}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
