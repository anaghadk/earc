"""
scripts/enrich_contexts_in_runs.py

Enriches recoverable .strict.jsonl files with:
  - context (from task1_raw_files/contexts/)
  - retrieved_ids (from task1_raw_files/retrieval_cache/retrieved_passages.pkl)
"""

import os
import glob
import json
import pickle
from pathlib import Path

PROJECT_ROOT = Path("C:/earc")
ctx_dir = PROJECT_ROOT / "task1_raw_files/contexts"
cache_pkl = PROJECT_ROOT / "task1_raw_files/retrieval_cache/retrieved_passages.pkl"

with open(cache_pkl, "rb") as f:
    retrieval_cache = pickle.load(f)

ctx_cache = {}

def get_ctx_dict(cname):
    if not cname:
        return None
    cpath = ctx_dir / cname
    if not cpath.exists():
        return None
    if cname not in ctx_cache:
        with open(cpath, "r", encoding="utf-8") as f:
            ctx_cache[cname] = json.load(f)
    return ctx_cache[cname]

def main():
    runs = sorted(glob.glob(str(PROJECT_ROOT / "runs/**/*.strict.jsonl"), recursive=True))
    runs = [r for r in runs if "rerun_default" not in r]

    print(f"Processing {len(runs)} strict JSONL files...")
    updated_files = 0

    for path in runs:
        basename = os.path.basename(path)
        ctx_name = None
        if "exp1_ablation" in path:
            parts = basename.split("_")
            cfg_tag = parts[1]
            ds_tag = parts[2]
            ctx_name = f"ctx_{cfg_tag}_{ds_tag}.json"
        elif "exp2_budget" in path:
            parts = basename.split("_")
            method_type = parts[1]
            ds_tag = parts[2]
            param_tag = parts[3]
            ctx_name = f"ctx_{method_type}_{ds_tag}_{param_tag}.json"
        elif "baseline" in path:
            parts = basename.split("_")
            ds_tag = parts[2]
            method_tag = parts[3]
            if method_tag == "earc":
                ctx_name = f"ctx_A5_{ds_tag}.json"
            elif method_tag == "topk":
                ctx_name = f"ctx_topk_{ds_tag}_k5.json"
            elif method_tag == "standard":
                ctx_name = f"ctx_topk_{ds_tag}_k10.json"
            else:
                ctx_name = None

        ctx_dict = get_ctx_dict(ctx_name)
        
        lines = [l for l in open(path, "r", encoding="utf-8") if l.strip()]
        new_lines = []
        for line in lines:
            r = json.loads(line)
            qid = str(r["question_id"])
            ds = r["dataset"].lower()
            if ds.startswith("hotpot"): ds = "hotpotqa"
            elif ds.startswith("trivia"): ds = "triviaqa"
            elif ds.startswith("nq"): ds = "nq"

            # Context
            if ctx_dict and qid in ctx_dict:
                item = ctx_dict[qid]
                if isinstance(item, dict):
                    r["context"] = item.get("compressed_text", "")
                else:
                    r["context"] = str(item)
            elif "context" not in r:
                r["context"] = ""

            # Retrieved ids
            if ds in retrieval_cache and qid in retrieval_cache[ds]:
                r["retrieved_ids"] = [d["doc_id"] for d in retrieval_cache[ds][qid]["docs"]]

            new_lines.append(json.dumps(r, ensure_ascii=False))

        with open(path, "w", encoding="utf-8") as f:
            for nl in new_lines:
                f.write(nl + "\n")

        if ctx_dict:
            updated_files += 1

    print(f"Successfully enriched {updated_files}/{len(runs)} files with exact context and retrieved_ids.")

if __name__ == "__main__":
    main()
