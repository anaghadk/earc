"""
Prepare runs/ and task1_raw_files/ directories from recovered outputs.
"""

import os
import shutil
import json
from pathlib import Path

PROJECT_ROOT = Path("C:/earc")
REC_ROOT = PROJECT_ROOT / "recovered_outputs/outputs"
RUNS_DIR = PROJECT_ROOT / "runs"
RAW_DIR = PROJECT_ROOT / "task1_raw_files"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 1. Exp 1
exp1_src = REC_ROOT / "journal_experiments/exp1_ablation/predictions"
exp1_dst = RUNS_DIR / "exp1_ablation"
exp1_dst.mkdir(parents=True, exist_ok=True)
for f in exp1_src.glob("*.jsonl"):
    shutil.copy2(f, exp1_dst / f.name)
print(f"Copied {len(list(exp1_dst.glob('*.jsonl')))} Exp1 runs")

# 2. Exp 2
exp2_src = REC_ROOT / "journal_experiments/exp2_budget/predictions"
exp2_dst = RUNS_DIR / "exp2_budget"
exp2_dst.mkdir(parents=True, exist_ok=True)
for f in exp2_src.glob("*.jsonl"):
    shutil.copy2(f, exp2_dst / f.name)
print(f"Copied {len(list(exp2_dst.glob('*.jsonl')))} Exp2 runs")

# 3. Baselines from final_evaluation
baseline_dst = RUNS_DIR / "baseline"
baseline_dst.mkdir(parents=True, exist_ok=True)
gen_map = {"ollama": "llama3.2_3b", "mistral": "ministral-8b"}
ds_map = {"hotpot": "hotpotqa", "nq": "nq", "trivia": "triviaqa"}

count_base = 0
for prov, gen in gen_map.items():
    for ds_short, ds_full in ds_map.items():
        for m in ["standard_rag", "topk", "llmlingua2", "earc"]:
            src = REC_ROOT / f"final_evaluation/{prov}/{ds_short}/{m}/predictions.jsonl"
            if src.exists():
                dst_name = f"pred_baseline_{ds_full}_{m}_{gen}.jsonl"
                dst_path = baseline_dst / dst_name
                with open(src, "r", encoding="utf-8") as f_in, open(dst_path, "w", encoding="utf-8") as f_out:
                    for line in f_in:
                        if line.strip():
                            r = json.loads(line)
                            if "question_id" not in r and "example_id" in r:
                                r["question_id"] = r["example_id"]
                            if "em" not in r and "exact_match" in r:
                                r["em"] = int(r["exact_match"] == 1.0)
                            if "context_tokens" not in r and "processed_tokens" in r:
                                r["context_tokens"] = r["processed_tokens"]
                            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                count_base += 1
print(f"Prepared and copied {count_base} Baseline runs")
total_runs = len(list(RUNS_DIR.glob("**/*.jsonl")))
print(f"Total runs in C:/earc/runs: {total_runs}")

# =============================================================================
# Now prepare task1_raw_files package
# =============================================================================
print("\nPreparing task1_raw_files package...")

# Deliverable CSVs
deliv_src = REC_ROOT / "journal_experiments/deliverables"
for csv_name in ["results.csv", "hitk.csv", "retention.csv", "attribution.csv"]:
    p = deliv_src / csv_name
    if p.exists():
        shutil.copy2(p, RAW_DIR / csv_name)

# Exp summaries
exp1_sum = REC_ROOT / "journal_experiments/exp1_ablation/exp1_summary.csv"
if exp1_sum.exists():
    shutil.copy2(exp1_sum, RAW_DIR / "exp1_summary.csv")

exp2_sum = REC_ROOT / "journal_experiments/exp2_budget/exp2_summary.csv"
if exp2_sum.exists():
    shutil.copy2(exp2_sum, RAW_DIR / "exp2_summary.csv")

# Contexts
contexts_dst = RAW_DIR / "contexts"
contexts_dst.mkdir(parents=True, exist_ok=True)
for cdir in [REC_ROOT / "journal_experiments/exp1_ablation/contexts", REC_ROOT / "journal_experiments/exp2_budget/contexts"]:
    if cdir.exists():
        for cf in cdir.glob("*.json"):
            shutil.copy2(cf, contexts_dst / cf.name)
print(f"Copied {len(list(contexts_dst.glob('*.json')))} context files")

# Retrieval cache
ret_dst = RAW_DIR / "retrieval_cache"
ret_dst.mkdir(parents=True, exist_ok=True)
ret_src = REC_ROOT / "journal_experiments/retrieval_cache"
if ret_src.exists():
    for rf in ret_src.glob("*"):
        shutil.copy2(rf, ret_dst / rf.name)
print(f"Copied retrieval cache files")

# Question IDs
qid_dst = RAW_DIR / "question_ids"
qid_dst.mkdir(parents=True, exist_ok=True)
sel_p = REC_ROOT / "final_evaluation/selected_example_ids.json"
if sel_p.exists():
    shutil.copy2(sel_p, qid_dst / "selected_example_ids.json")
    with open(sel_p, "r", encoding="utf-8") as f:
        sel_ids = json.load(f)
    for k, ids in sel_ids.items():
        if isinstance(ids, list):
            fname = f"{k}_question_ids.txt"
            with open(qid_dst / fname, "w", encoding="utf-8") as f:
                for q in ids:
                    f.write(f"{q}\n")
print(f"Saved locked question ID files")

# Configs
cfg_dst = RAW_DIR / "configs"
cfg_dst.mkdir(parents=True, exist_ok=True)
for cdir in [REC_ROOT / "journal_experiments/exp1_ablation/configs", REC_ROOT / "journal_experiments/exp2_budget/configs"]:
    if cdir.exists():
        for cf in cdir.glob("*.yaml"):
            shutil.copy2(cf, cfg_dst / cf.name)
if (PROJECT_ROOT / "configs/default.yaml").exists():
    shutil.copy2(PROJECT_ROOT / "configs/default.yaml", cfg_dst / "default.yaml")
print(f"Copied {len(list(cfg_dst.glob('*.yaml')))} config files")

# Copy runs to task1_raw_files/runs
raw_runs_dst = RAW_DIR / "runs"
if raw_runs_dst.exists():
    shutil.rmtree(raw_runs_dst)
shutil.copytree(RUNS_DIR, raw_runs_dst)
print(f"Copied runs/ into task1_raw_files/runs")

# Scripts
scripts_dst = RAW_DIR / "scripts"
scripts_dst.mkdir(parents=True, exist_ok=True)
for sf in (PROJECT_ROOT / "scripts").glob("*.py"):
    shutil.copy2(sf, scripts_dst / sf.name)
print(f"Copied scripts into task1_raw_files/scripts")

# Commits.txt
with open(RAW_DIR / "commits.txt", "w", encoding="utf-8") as f:
    f.write("607566f1a2739f9fce6c2cbf4bff34b5a409269c all the 4 journal experiments\n")
    f.write("681fa3d6515fc3f63116a4f932bf26dc31bb99c3 after 1st set of tests\n")
    f.write("416c14692d96887cd0f3088ee87ef69ad2de6c67 LAtest updates\n")
    f.write("90d67f378456a163d2acbed87f8734d50865c5a9 Initial commit\n")

print("task1_raw_files preparation complete!")
