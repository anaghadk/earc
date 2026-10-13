"""
scripts/run_task3_rerun.py

Task 3 Rerun Script for Default EARC and Top-k (k=5).
Generates:
  runs/rerun_default/earc_default_rerun__{dataset}__{generator}.jsonl
  runs/rerun_default/topk5_rerun__{dataset}__{generator}.jsonl

Carries all Section 1.2 fields plus:
  - context (the exact text given to the generator)
  - retrieved_ids (the IDs of the cached top-K passages)
"""

import os
import sys
import time
import json
import pickle
import argparse
from pathlib import Path

PROJECT_ROOT = Path("C:/earc")
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# Check Windows User registry as fallback if not in os.environ
if not os.environ.get("MISTRAL_API_KEY") and sys.platform == "win32":
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
            val, _ = winreg.QueryValueEx(k, "MISTRAL_API_KEY")
            if val:
                os.environ["MISTRAL_API_KEY"] = val
    except Exception:
        pass

from src.config.settings import load_config
from src.data.schemas import RetrievedDocument
from src.compression.embeddings import EmbeddingEngine
from src.compression.compressor import EvidenceAwareCompressor
from src.prompting.builder import PromptBuilder
from src.prompting.templates import DEFAULT_QA_TEMPLATE
from src.llm.factory import create_provider
from src.llm.ollama_client import OllamaTokenCounter
from src.evaluation.answer_extraction import extract_answer
from scripts.rescore_strict import em as strict_em, f1 as strict_f1

GENERATORS = {
    "llama3.2_3b": "ollama",
    "ministral-8b": "mistral",
    "ministral_8b": "mistral",
}

DATASETS = ["hotpotqa", "nq", "triviaqa"]

def get_topk_context(docs, k=5):
    """Concatenate top-k passage texts matching stored ctx_topk_*_k5.json."""
    return "\n\n".join(d["text"] for d in docs[:k])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries per run")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "nq", "hotpotqa", "triviaqa"])
    parser.add_argument("--generator", type=str, default="all", choices=["all", "llama3.2_3b", "ministral-8b", "ministral_8b"])
    parser.add_argument("--method", type=str, default="all", choices=["all", "earc", "topk5"])
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "runs/rerun_default"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_pkl = PROJECT_ROOT / "task1_raw_files/retrieval_cache/retrieved_passages.pkl"
    with open(cache_pkl, "rb") as f:
        retrieval_cache = pickle.load(f)

    cfg = load_config("configs/default.yaml")
    engine = EmbeddingEngine(cfg.retrieval.embedding_model, device="cuda" if sys.platform != "darwin" else "cpu")
    counter = OllamaTokenCounter("llama3.2")
    prompt_builder = PromptBuilder(DEFAULT_QA_TEMPLATE)

    # EARC compressor
    c_cfg = cfg.compression.model_copy(deep=True)
    c_cfg.alpha = 0.7
    c_cfg.beta = 0.3
    c_cfg.use_redundancy = True
    c_cfg.redundancy_threshold = 0.85
    c_cfg.token_budget = 300
    compressor = EvidenceAwareCompressor(c_cfg, engine, counter, device="cuda" if sys.platform != "darwin" else "cpu")

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    generators = list(GENERATORS.keys()) if args.generator == "all" else [args.generator]
    methods = ["earc", "topk5"] if args.method == "all" else [args.method]

    # Pre-initialize providers
    providers = {}
    for gen in generators:
        providers[gen] = create_provider(cfg, GENERATORS[gen])

    for m in methods:
        for ds in datasets:
            ds_queries = retrieval_cache[ds]
            q_ids = list(ds_queries.keys())
            if args.limit:
                q_ids = q_ids[:args.limit]

            for gen in generators:
                llm = providers[gen]
                if m == "earc":
                    fname = f"earc_default_rerun__{ds}__{gen}.jsonl"
                    cfg_id = "earc_default"
                else:
                    fname = f"topk5_rerun__{ds}__{gen}.jsonl"
                    cfg_id = "topk_5"

                fpath = out_dir / fname
                completed = {}
                if fpath.exists():
                    for line in open(fpath, "r", encoding="utf-8"):
                        if line.strip():
                            r = json.loads(line)
                            completed[r["question_id"]] = r

                remaining_ids = [qid for qid in q_ids if qid not in completed]
                print(f"[{m.upper()}] {ds} | {gen} -> {fname}: {len(completed)} completed, {len(remaining_ids)} remaining")

                if not remaining_ids:
                    continue

                with open(fpath, "a", encoding="utf-8") as f_out:
                    for i, qid in enumerate(remaining_ids, 1):
                        q_data = ds_queries[qid]
                        raw_docs = q_data["docs"]
                        retrieved_ids = [d["doc_id"] for d in raw_docs]

                        t0 = time.time()
                        if m == "earc":
                            ret_docs = [
                                RetrievedDocument(
                                    doc_id=d["doc_id"],
                                    title=d["title"],
                                    text=d["text"],
                                    rank=d["rank"],
                                    score=d["score"],
                                    retriever="dense"
                                )
                                for d in raw_docs
                            ]
                            c_res = compressor.compress(q_data["question"], ret_docs, dataset=ds)
                            context_text = c_res.compressed_text
                            ctx_tokens = c_res.compressed_tokens
                        else:
                            context_text = get_topk_context(raw_docs, k=5)
                            ctx_tokens = counter.count(context_text)
                        comp_time = time.time() - t0

                        # Generate
                        prompt = prompt_builder.build(q_data["question"], context_text)
                        t_g0 = time.time()
                        raw_pred = ""
                        for attempt in range(5):
                            try:
                                g_res = llm.generate(prompt)
                                raw_pred = g_res.text
                                break
                            except Exception as e:
                                print(f"Error generating for {qid} (attempt {attempt+1}/5): {e}")
                                time.sleep(2.0 * (attempt + 1))
                        gen_time = time.time() - t_g0
                        if gen.startswith("ministral"):
                            time.sleep(0.1)

                        eval_ans = extract_answer(raw_pred)
                        golds = q_data["gold_answers"]
                        em_val = strict_em(eval_ans, golds)
                        f1_val = strict_f1(eval_ans, golds)

                        rec = {
                            "experiment": "rerun_default",
                            "config_id": cfg_id,
                            "dataset": ds,
                            "generator": gen,
                            "question_id": qid,
                            "gold_answers": golds,
                            "prediction": raw_pred,
                            "evaluated_answer": eval_ans,
                            "em": em_val,
                            "f1": round(f1_val, 4),
                            "em_strict": em_val,
                            "f1_strict": round(f1_val, 4),
                            "context_tokens": ctx_tokens,
                            "compress_time_s": round(comp_time, 4),
                            "gen_time_s": round(gen_time, 4),
                            "alpha": 0.7 if m == "earc" else None,
                            "beta": 0.3 if m == "earc" else None,
                            "theta": 0.85 if m == "earc" else None,
                            "budget": 300 if m == "earc" else None,
                            "k": 5 if m == "topk5" else None,
                            "context": context_text,
                            "retrieved_ids": retrieved_ids,
                        }
                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f_out.flush()

                        if i % 100 == 0 or i == len(remaining_ids):
                            print(f"  Processed {i}/{len(remaining_ids)}...")

                if "ministral" in gen and fpath.exists():
                    alt_fname = fname.replace("ministral_8b", "ministral-8b") if "ministral_8b" in fname else fname.replace("ministral-8b", "ministral_8b")
                    alt_fpath = out_dir / alt_fname
                    import shutil
                    shutil.copyfile(fpath, alt_fpath)

if __name__ == "__main__":
    main()
