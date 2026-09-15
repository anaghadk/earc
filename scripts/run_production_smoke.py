"""
Production Smoke Test Script
Verifies dataset-independent pipeline execution via the UI adapter entry point (UIEARCPipeline).
Tests both Ollama (llama3.2) and Mistral (ministral-8b-latest) across 5 natural-language questions
retrieved directly from the RAG_Project corpus.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = "lIK71q2MhMrZXQwxYE0nVUvoNNUVjQzy"

from src.ui.adapter import UIEARCPipeline
from src.utils.logging import setup_logging, get_logger

logger = get_logger("production_smoke")

QUESTIONS = [
    "Where do the ilium, ischium, and pubis meet?",
    "What is the capital of Australia?",
    "What organ is primarily responsible for pumping blood throughout the human body?",
    "In woodwork, what are butt, dovetail, and mitre?",
    "Who wrote the play Romeo and Juliet?",
]


def run_test():
    setup_logging("INFO")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / f"production_smoke_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FINAL PRODUCTION-STYLE DATASET-INDEPENDENT SMOKE TEST")
    print(f"Output Directory: {out_dir}")
    print("Configuration: configs/default.yaml (Dense top_k=10, budget=300, alpha=0.7, beta=0.3, tau=0.85)")
    print("Providers: Ollama (llama3.2) & Mistral (ministral-8b-latest)")
    print("Entry Point: src.ui.adapter.UIEARCPipeline")
    print("=" * 80)

    # Initialize pipelines for both providers using the UI adapter
    print("\n[1/2] Initializing Ollama pipeline (UIEARCPipeline)...")
    ollama_pipeline = UIEARCPipeline(config_path="configs/default.yaml", provider_name="ollama")

    print("[2/2] Initializing Mistral pipeline (UIEARCPipeline)...")
    mistral_pipeline = UIEARCPipeline(config_path="configs/default.yaml", provider_name="mistral")

    pipelines = {
        "Ollama (llama3.2)": ollama_pipeline,
        "Mistral (ministral-8b-latest)": mistral_pipeline,
    }

    results = []
    all_passed = True

    for q_idx, question in enumerate(QUESTIONS, 1):
        print(f"\n" + "=" * 80)
        print(f"QUESTION {q_idx}/5: \"{question}\"")
        print("=" * 80)

        q_record = {
            "question_index": q_idx,
            "question": question,
            "runs": {},
        }

        for provider_label, pipeline in pipelines.items():
            print(f"\n--- Running: {provider_label} ---")
            t_start = time.time()
            success = False
            error_msg = None
            run_data = {}

            try:
                out = pipeline.run(question)
                e2e_latency = time.time() - t_start
                success = True

                ret_docs = out.get("retrieved_documents", [])
                doc_titles = [d.get("title") for d in ret_docs if d.get("title")]
                # unique titles preserved in order
                seen_titles = set()
                unique_titles = []
                for t in doc_titles:
                    if t not in seen_titles:
                        seen_titles.add(t)
                        unique_titles.append(t)

                sel_sents = out.get("selected_sentences", [])
                sel_evidence_texts = [s.get("text") for s in sel_sents]

                orig_tokens = out.get("original_tokens", 0)
                comp_tokens = out.get("compressed_tokens", 0)
                comp_pct = out.get("compression_percentage", 0.0)
                raw_output = out.get("raw_prediction", "").replace("\n", " ").strip()
                final_answer = out.get("answer", "").strip()

                run_data = {
                    "provider": provider_label,
                    "completed_successfully": True,
                    "retrieved_doc_count": len(ret_docs),
                    "top_doc_titles": unique_titles[:5],
                    "original_context_tokens": orig_tokens,
                    "compressed_tokens": comp_tokens,
                    "compression_percentage": round(comp_pct, 2),
                    "selected_evidence_count": len(sel_sents),
                    "selected_evidence": sel_evidence_texts,
                    "raw_output": raw_output,
                    "final_answer": final_answer,
                    "end_to_end_latency_s": round(e2e_latency, 3),
                }

                print(f"  Status:               SUCCESS (E2E: {e2e_latency:.2f}s)")
                print(f"  Retrieved Docs:       {len(ret_docs)} docs -> Top Titles: {unique_titles[:3]}")
                print(f"  Token Compression:    {orig_tokens} -> {comp_tokens} tokens ({comp_pct:.1f}% reduction)")
                print(f"  Selected Sentences:   {len(sel_sents)} evidence sentences")
                print(f"  Raw Model Output:     {raw_output[:120]}...")
                print(f"  Extracted Answer:     {final_answer}")

            except Exception as e:
                e2e_latency = time.time() - t_start
                all_passed = False
                error_msg = str(e)
                print(f"  Status:               FAILED ({e})")
                run_data = {
                    "provider": provider_label,
                    "completed_successfully": False,
                    "error": error_msg,
                    "end_to_end_latency_s": round(e2e_latency, 3),
                }

            q_record["runs"][provider_label] = run_data

        results.append(q_record)

    # Save results as JSON
    results_json_path = out_dir / "production_smoke_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Report
    report_md_path = out_dir / "production_smoke_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Production Smoke Test Report\n\n")
        f.write(f"- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Overall Verdict**: {'PASS' if all_passed else 'FAIL'}\n")
        f.write(f"- **Configuration**: Dense retrieval, top_k=10, budget=300, alpha=0.7, beta=0.3, tau=0.85\n")
        f.write(f"- **Corpus**: `RAG_Project` (Wikipedia indexed FAISS)\n")
        f.write(f"- **Entry Point**: `src.ui.adapter.UIEARCPipeline`\n\n")

        for item in results:
            f.write(f"### Q{item['question_index']}: {item['question']}\n\n")
            for prov, r in item["runs"].items():
                f.write(f"#### Provider: {prov}\n")
                if r["completed_successfully"]:
                    f.write(f"- **Completed Successfully**: Yes\n")
                    f.write(f"- **Retrieved Docs Count**: {r['retrieved_doc_count']}\n")
                    f.write(f"- **Top Document Titles**: {', '.join(r['top_doc_titles'])}\n")
                    f.write(f"- **Original Context Tokens**: {r['original_context_tokens']}\n")
                    f.write(f"- **Compressed Tokens**: {r['compressed_tokens']} ({r['compression_percentage']}% reduction)\n")
                    f.write(f"- **Extracted Final Answer**: `{r['final_answer']}`\n")
                    f.write(f"- **Raw Output**: {r['raw_output']}\n")
                    f.write(f"- **End-to-End Latency**: {r['end_to_end_latency_s']}s\n")
                    f.write(f"- **Selected Evidence Sample**:\n")
                    for s in r["selected_evidence"][:3]:
                        f.write(f"  - \"{s}\"\n")
                else:
                    f.write(f"- **Completed Successfully**: No\n")
                    f.write(f"- **Error**: {r.get('error')}\n")
                f.write("\n")

    print("\n" + "=" * 80)
    print("PRODUCTION SMOKE TEST SUMMARY")
    print(f"Verdict: {'ALL 5 QUESTIONS PASSED' if all_passed else 'FAILURES DETECTED'}")
    print(f"Results JSON: {results_json_path}")
    print(f"Report MD:    {report_md_path}")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
