"""
scripts/run_hotpot_reps.py

Runs default EARC on HotpotQA twice more with each generator:
  - Llama 3.2 3B rep2 & rep3 -> runs/hotpot_reps/earc_default__hotpotqa__llama__rep{2|3}.jsonl
  - Ministral-8B rep2 & rep3 -> runs/hotpot_reps/earc_default__hotpotqa__ministral__rep{2|3}.jsonl

Settings:
  alpha = 0.7, beta = 0.3, theta = 0.85, budget = 300 tokens
  Same 1,000 question IDs, cached top-10 retrieval, prompt and decoding as main runs.
  Adds 'prompt' and 'model_id' to each JSONL line.
"""

import os
import sys
import time
import json
import pickle
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path("C:/earc")
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Load Windows environment key if needed
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
from src.prompting.builder import PromptBuilder
from src.prompting.templates import DEFAULT_QA_TEMPLATE
from src.llm.factory import create_provider
from src.evaluation.answer_extraction import extract_answer
from scripts.rescore_strict import em as strict_em, f1 as strict_f1

OUT_DIR = PROJECT_ROOT / "runs/hotpot_reps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_SPECS = [
    {
        "name": "earc_default__hotpotqa__ministral__rep2.jsonl",
        "generator_key": "mistral",
        "gen_label": "ministral-8b",
        "model_id": "ministral-8b-latest",
        "rep": 2
    },
    {
        "name": "earc_default__hotpotqa__ministral__rep3.jsonl",
        "generator_key": "mistral",
        "gen_label": "ministral-8b",
        "model_id": "ministral-8b-latest",
        "rep": 3
    },
    {
        "name": "earc_default__hotpotqa__llama__rep2.jsonl",
        "generator_key": "ollama",
        "gen_label": "llama3.2_3b",
        "model_id": "llama3.2",
        "rep": 2
    },
    {
        "name": "earc_default__hotpotqa__llama__rep3.jsonl",
        "generator_key": "ollama",
        "gen_label": "llama3.2_3b",
        "model_id": "llama3.2",
        "rep": 3
    },
]

def load_canonical_data():
    cache_pkl = PROJECT_ROOT / "task1_raw_files/retrieval_cache/retrieved_passages.pkl"
    with open(cache_pkl, "rb") as f:
        retrieval_cache = pickle.load(f)
    hotpot_queries = retrieval_cache["hotpotqa"]

    # Pre-load verified compressed contexts from rep1
    rep1_path = PROJECT_ROOT / "runs/rerun_default/predictions/earc_default_rerun__hotpotqa__llama3.2_3b.jsonl"
    ctx_data = {}
    with open(rep1_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ctx_data[r["question_id"]] = {
                    "context": r["context"],
                    "context_tokens": r["context_tokens"],
                    "retrieved_ids": r["retrieved_ids"],
                    "gold_answers": r["gold_answers"],
                    "compress_time_s": r.get("compress_time_s", 0.6145)
                }
    return hotpot_queries, ctx_data

def run_single_experiment(spec, hotpot_queries, ctx_data, cfg, prompt_builder, limit=None):
    fpath = OUT_DIR / spec["name"]
    print(f"\n==================================================")
    print(f"Starting run: {spec['name']}")
    print(f"Generator: {spec['generator_key']} ({spec['model_id']}), Rep: {spec['rep']}")
    print(f"==================================================")

    completed = {}
    if fpath.exists():
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed[r["question_id"]] = r
        print(f"Found {len(completed)} existing records in {spec['name']} (resuming safely).")

    q_ids = list(hotpot_queries.keys())
    if limit:
        q_ids = q_ids[:limit]

    remaining_ids = [qid for qid in q_ids if qid not in completed]
    print(f"Remaining questions to process: {len(remaining_ids)} / {len(q_ids)}")
    if not remaining_ids:
        print(f"Run {spec['name']} is already complete!")
        return

    provider = create_provider(cfg, spec["generator_key"])

    def process_one(qid):
        q_info = hotpot_queries[qid]
        c_info = ctx_data[qid]
        question = q_info["question"]
        context_text = c_info["context"]
        prompt = prompt_builder.build(question, context_text)

        t0 = time.time()
        raw_pred = ""
        for attempt in range(5):
            try:
                g_res = provider.generate(prompt)
                raw_pred = g_res.text
                break
            except Exception as e:
                time.sleep(1.5 * (attempt + 1))
        gen_time = time.time() - t0

        eval_ans = extract_answer(raw_pred)
        golds = c_info["gold_answers"]
        em_val = strict_em(eval_ans, golds)
        f1_val = strict_f1(eval_ans, golds)

        record = {
            "experiment": "hotpot_reps",
            "config_id": "earc_default",
            "dataset": "hotpotqa",
            "generator": spec["gen_label"],
            "model_id": spec["model_id"],
            "rep": spec["rep"],
            "question_id": qid,
            "gold_answers": golds,
            "prompt": prompt,
            "prediction": raw_pred,
            "evaluated_answer": eval_ans,
            "em": em_val,
            "f1": round(f1_val, 4),
            "em_strict": em_val,
            "f1_strict": round(f1_val, 4),
            "context_tokens": c_info["context_tokens"],
            "compress_time_s": c_info["compress_time_s"],
            "gen_time_s": round(gen_time, 4),
            "alpha": 0.7,
            "beta": 0.3,
            "theta": 0.85,
            "budget": 300,
            "k": 10,
            "context": context_text,
            "retrieved_ids": c_info["retrieved_ids"],
        }
        return record

    # Process records
    if spec["generator_key"] == "ministral":
        # API can use concurrent threads
        max_workers = 6
        with open(fpath, "a", encoding="utf-8") as f_out:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_one, qid): qid for qid in remaining_ids}
                count = 0
                for fut in as_completed(futures):
                    rec = fut.result()
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_out.flush()
                    count += 1
                    if count % 100 == 0 or count == len(remaining_ids):
                        print(f"[{spec['name']}] Completed {count}/{len(remaining_ids)}...")
    else:
        # Ollama local inference: sequential to prevent GPU VRAM thrashing
        with open(fpath, "a", encoding="utf-8") as f_out:
            for count, qid in enumerate(remaining_ids, 1):
                rec = process_one(qid)
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                if count % 50 == 0 or count == len(remaining_ids):
                    print(f"[{spec['name']}] Completed {count}/{len(remaining_ids)}...")

    print(f"Finished {spec['name']} successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit questions for testing")
    parser.add_argument("--target", type=str, default="all", choices=["all", "ministral", "llama", "rep2", "rep3"])
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml")
    prompt_builder = PromptBuilder(DEFAULT_QA_TEMPLATE)
    hotpot_queries, ctx_data = load_canonical_data()

    for spec in RUN_SPECS:
        if args.target == "ministral" and spec["generator_key"] != "ministral":
            continue
        if args.target == "llama" and spec["generator_key"] != "ollama":
            continue
        if args.target == "rep2" and spec["rep"] != 2:
            continue
        if args.target == "rep3" and spec["rep"] != 3:
            continue

        run_single_experiment(spec, hotpot_queries, ctx_data, cfg, prompt_builder, limit=args.limit)

if __name__ == "__main__":
    main()
