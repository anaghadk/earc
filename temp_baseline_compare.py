import os
import sys
import json
import yaml
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = "lIK71q2MhMrZXQwxYE0nVUvoNNUVjQzy"

from src.config.settings import load_config
from src.data.loaders import load_dataset
from src.data.sampling import sample_dataset
from src.retrieval.rag_project import RAGProjectRetriever
from src.compression.embeddings import EmbeddingEngine
from src.compression.compressor import EvidenceAwareCompressor
from src.baselines.standard_rag import StandardRAGBaseline
from src.baselines.topk import TopKBaseline
from src.baselines.llmlingua2 import LLMLingua2Baseline
from src.prompting.builder import PromptBuilder
from src.prompting.templates import DEFAULT_QA_TEMPLATE
from src.llm.factory import create_provider
from src.evaluation.exact_match import exact_match_score
from src.evaluation.f1 import max_token_f1_score
from src.evaluation.answer_extraction import extract_answer
from src.utils.device import DeviceManager


def main():
    print("Initializing components for Ollama baseline comparison (k=20 documents, top-5 baseline)...")

    # Load configuration directly from configs/default.yaml without hardcoding token budgets
    config_yaml_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_yaml_path, "r", encoding="utf-8") as f:
        raw_yaml = yaml.safe_load(f)

    config = load_config(config_yaml_path)
    device = DeviceManager().get_device().type

    # Attach any custom config fields (e.g. hotpotqa_token_budget) from raw yaml into config.compression
    comp_raw = raw_yaml.get("compression", {})
    for key, val in comp_raw.items():
        object.__setattr__(config.compression, key, val)

    config.retrieval.top_k = 20
    config.retrieval.retrieval_method = "dense"

    rag_dir = config.retrieval.rag_project_dir or "RAG_Project"
    retriever = RAGProjectRetriever(
        rag_dir=rag_dir,
        model_name=config.retrieval.embedding_model,
        device=device,
        method="dense",
    )
    embedding_engine = EmbeddingEngine(config.retrieval.embedding_model, device=device)

    # Provider is Ollama llama3.2 only
    provider = create_provider(config, "ollama")
    token_counter = provider.get_token_counter()
    prompt_builder = PromptBuilder(DEFAULT_QA_TEMPLATE)

    # Baselines and compressor: top-k reduces 20 documents to top 5
    standard_rag_baseline = StandardRAGBaseline(prompt_builder)
    topk_baseline = TopKBaseline(prompt_builder, k=5)
    llmlingua2_baseline = LLMLingua2Baseline(prompt_builder, device=device, target_token=300)
    compressor = EvidenceAwareCompressor(config.compression, embedding_engine, token_counter, device=device)

    print(f"EARC config loaded: alpha={config.compression.alpha}, beta={config.compression.beta}, budget={config.compression.token_budget}, hotpotqa_budget={getattr(config.compression, 'hotpotqa_token_budget', 'N/A')}")

    # 3 datasets: (json_key, loader_name, dataset_param_for_compressor)
    dataset_configs = [
        ("nq", "nq", "nq"),
        ("hotpotqa", "hotpotqa", "hotpot"),
        ("triviaqa", "triviaqa", "triviaqa"),
    ]

    all_results = {}

    for json_key, load_name, ds_param in dataset_configs:
        print(f"\nLoading and sampling 20 examples from {json_key.upper()} with seed=42...")
        raw_examples = load_dataset(load_name)
        sampled_examples = sample_dataset(raw_examples, n=20, seed=42)

        ds_records = []

        for idx, ex in enumerate(sampled_examples, 1):
            # 1. Single dense retrieval with top_k=20, reused across all 4 methods
            retrieved_docs = retriever.retrieve(ex.question, top_k=20)

            # Method 1: Standard RAG (all 20 retrieved documents)
            sr_res = standard_rag_baseline.run(ex.question, retrieved_docs, token_counter)
            sr_prompt = prompt_builder.build(ex.question, sr_res["compressed_text"])
            sr_gen = provider.generate(sr_prompt)
            sr_eval = extract_answer(sr_gen.text)
            sr_em = float(exact_match_score(sr_eval, ex.answers))
            sr_f1 = float(max_token_f1_score(sr_eval, ex.answers)["f1"])
            sr_tokens = int(sr_res["compressed_tokens"])

            # Method 2: Top-k (reduces 20 documents to top 5 passages)
            topk_res = topk_baseline.run(ex.question, retrieved_docs, token_counter)
            topk_prompt = prompt_builder.build(ex.question, topk_res["compressed_text"])
            topk_gen = provider.generate(topk_prompt)
            topk_eval = extract_answer(topk_gen.text)
            topk_em = float(exact_match_score(topk_eval, ex.answers))
            topk_f1 = float(max_token_f1_score(topk_eval, ex.answers)["f1"])
            topk_tokens = int(topk_res["compressed_tokens"])

            # Method 3: LLMLingua-2 (target_token=300)
            lingua_res = llmlingua2_baseline.run(ex.question, retrieved_docs, token_counter, budget=300)
            lingua_prompt = prompt_builder.build(ex.question, lingua_res["compressed_text"])
            lingua_gen = provider.generate(lingua_prompt)
            lingua_eval = extract_answer(lingua_gen.text)
            lingua_em = float(exact_match_score(lingua_eval, ex.answers))
            lingua_f1 = float(max_token_f1_score(lingua_eval, ex.answers)["f1"])
            lingua_tokens = int(lingua_res["compressed_tokens"])

            # Method 4: EARC (reading budget from config directly, passing dataset=ds_param)
            earc_res = compressor.compress(ex.question, retrieved_docs, dataset=ds_param)
            earc_prompt = prompt_builder.build_from_compressed(ex.question, earc_res)
            earc_gen = provider.generate(earc_prompt)
            earc_eval = extract_answer(earc_gen.text)
            earc_em = float(exact_match_score(earc_eval, ex.answers))
            earc_f1 = float(max_token_f1_score(earc_eval, ex.answers)["f1"])
            earc_tokens = int(earc_res.compressed_tokens)
            earc_ratio = round(earc_res.original_tokens / earc_tokens, 2) if earc_tokens > 0 else 1.0

            # Print terminal output in requested format
            print(f"[{idx:02d}/20] [{json_key.upper()}] Q: {ex.question}")
            print(f"  standard_rag : EM={sr_em:.1f} F1={sr_f1:.2f} | tokens={sr_tokens}")
            print(f"  topk         : EM={topk_em:.1f} F1={topk_f1:.2f} | tokens={topk_tokens}")
            print(f"  llmlingua2   : EM={lingua_em:.1f} F1={lingua_f1:.2f} | tokens={lingua_tokens}")
            print(f"  earc         : EM={earc_em:.1f} F1={earc_f1:.2f} | tokens={earc_tokens}")

            ds_records.append({
                "question": ex.question,
                "gold_answers": ex.answers,
                "standard_rag": {
                    "raw_prediction": sr_gen.text,
                    "evaluated_answer": sr_eval,
                    "exact_match": sr_em,
                    "f1_score": round(sr_f1, 4),
                    "tokens": sr_tokens,
                },
                "topk": {
                    "raw_prediction": topk_gen.text,
                    "evaluated_answer": topk_eval,
                    "exact_match": topk_em,
                    "f1_score": round(topk_f1, 4),
                    "tokens": topk_tokens,
                },
                "llmlingua2": {
                    "raw_prediction": lingua_gen.text,
                    "evaluated_answer": lingua_eval,
                    "exact_match": lingua_em,
                    "f1_score": round(lingua_f1, 4),
                    "tokens": lingua_tokens,
                },
                "earc": {
                    "raw_prediction": earc_gen.text,
                    "evaluated_answer": earc_eval,
                    "exact_match": earc_em,
                    "f1_score": round(earc_f1, 4),
                    "tokens": earc_tokens,
                    "compression_ratio": earc_ratio,
                },
            })

        # Calculate summary for dataset
        sr_avg_em = round(float(np.mean([r["standard_rag"]["exact_match"] for r in ds_records])) * 100.0, 1)
        sr_avg_f1 = round(float(np.mean([r["standard_rag"]["f1_score"] for r in ds_records])) * 100.0, 1)
        sr_avg_tok = round(float(np.mean([r["standard_rag"]["tokens"] for r in ds_records])))

        topk_avg_em = round(float(np.mean([r["topk"]["exact_match"] for r in ds_records])) * 100.0, 1)
        topk_avg_f1 = round(float(np.mean([r["topk"]["f1_score"] for r in ds_records])) * 100.0, 1)
        topk_avg_tok = round(float(np.mean([r["topk"]["tokens"] for r in ds_records])))

        lingua_avg_em = round(float(np.mean([r["llmlingua2"]["exact_match"] for r in ds_records])) * 100.0, 1)
        lingua_avg_f1 = round(float(np.mean([r["llmlingua2"]["f1_score"] for r in ds_records])) * 100.0, 1)
        lingua_avg_tok = round(float(np.mean([r["llmlingua2"]["tokens"] for r in ds_records])))

        earc_avg_em = round(float(np.mean([r["earc"]["exact_match"] for r in ds_records])) * 100.0, 1)
        earc_avg_f1 = round(float(np.mean([r["earc"]["f1_score"] for r in ds_records])) * 100.0, 1)
        earc_avg_tok = round(float(np.mean([r["earc"]["tokens"] for r in ds_records])))
        earc_avg_ratio = round(float(np.mean([r["earc"]["compression_ratio"] for r in ds_records])), 2)

        all_results[json_key] = {
            "summary": {
                "standard_rag": {
                    "avg_exact_match": sr_avg_em,
                    "avg_f1": sr_avg_f1,
                    "avg_tokens": sr_avg_tok,
                },
                "topk": {
                    "avg_exact_match": topk_avg_em,
                    "avg_f1": topk_avg_f1,
                    "avg_tokens": topk_avg_tok,
                },
                "llmlingua2": {
                    "avg_exact_match": lingua_avg_em,
                    "avg_f1": lingua_avg_f1,
                    "avg_tokens": lingua_avg_tok,
                },
                "earc": {
                    "avg_exact_match": earc_avg_em,
                    "avg_f1": earc_avg_f1,
                    "avg_tokens": earc_avg_tok,
                    "avg_compression_ratio": earc_avg_ratio,
                },
            },
            "examples": ds_records,
        }

    # Write to temp_baseline_compare.json in project root
    out_file = PROJECT_ROOT / "temp_baseline_compare.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 57)
    print("BASELINE COMPARISON SUMMARY (20 examples each)")
    print("=" * 57)
    print(f"{'Dataset':<12}{'Method':<16}{'EM%':<8}{'F1%':<8}{'Tokens'}")
    print("-" * 57)

    display_names = [
        ("nq", "NQ"),
        ("hotpotqa", "HotpotQA"),
        ("triviaqa", "TriviaQA"),
    ]
    methods_order = [
        ("standard_rag", "Standard RAG"),
        ("topk", "Top-k"),
        ("llmlingua2", "LLMLingua-2"),
        ("earc", "EARC"),
    ]

    for key, d_label in display_names:
        for m_key, m_label in methods_order:
            m_stats = all_results[key]["summary"][m_key]
            em_val = f"{m_stats['avg_exact_match']:.1f}"
            f1_val = f"{m_stats['avg_f1']:.1f}"
            tok_val = f"{m_stats['avg_tokens']}"
            print(f"{d_label:<12}{m_label:<16}{em_val:<8}{f1_val:<8}{tok_val}")
        print("-" * 57)

    print(f"\nAll results saved to: {out_file}\n")


if __name__ == "__main__":
    main()
