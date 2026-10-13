"""
Master Pipeline Runner for Experiment 2 (Budget Sweep and Top-k Sweep).
Iterates through all budgets and k values:
1. Runs context construction (compress-once) for any missing contexts.
2. Runs generation for Ministral-8B and Llama 3.2.
3. Automatically updates exp2_summary.csv incrementally.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

DATASETS = ["hotpotqa", "nq", "triviaqa"]
GRID_BUDGETS = [150, 450, 600, 750, 900] # 300 is already seeded
MATCHED_BUDGETS = {
    "nq": 727,
    "hotpotqa": 598,
    "triviaqa": 843,
}
K_LIST = [1, 2, 3, 4, 6, 7, 8, 9, 10] # k=5 is already seeded
GENERATORS = ["ministral-8b", "llama3.2:3b"]

def run_step(cmd):
    print(f"\n[RUNNING] {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[ERROR] Command failed with returncode {res.returncode}")
    else:
        print(f"[COMPLETED] in {elapsed:.1f}s")
    return res.returncode == 0

def main():
    print("=" * 80)
    print("STARTING EXPERIMENT 2 MASTER PIPELINE (EARC Budget Sweep + Top-k Sweep)")
    print("=" * 80)

    # Phase 1: Pre-compress all EARC budget contexts
    print("\n--- PHASE 1: PRE-COMPRESSING ALL EARC BUDGET CONTEXTS ---")
    for ds in DATASETS:
        all_b = GRID_BUDGETS + [MATCHED_BUDGETS[ds]]
        for b in all_b:
            ctx_file = Path(f"outputs/journal_experiments/exp2_budget/contexts/ctx_earc_{ds}_b{b}.json")
            if ctx_file.exists():
                print(f"[CACHE HIT] Contexts already exist: {ctx_file}")
                continue
            cmd = [sys.executable, "scripts/run_journal_exp2_budget.py", "--mode", "earc", "--dataset", ds, "--budget", str(b), "--compress-only"]
            run_step(cmd)

    # Phase 2: Pre-build all Top-k contexts
    print("\n--- PHASE 2: PRE-BUILDING ALL TOP-K CONTEXTS ---")
    for ds in DATASETS:
        for k in K_LIST:
            ctx_file = Path(f"outputs/journal_experiments/exp2_budget/contexts/ctx_topk_{ds}_k{k}.json")
            if ctx_file.exists():
                print(f"[CACHE HIT] Top-k contexts already exist: {ctx_file}")
                continue
            cmd = [sys.executable, "scripts/run_journal_exp2_budget.py", "--mode", "topk", "--dataset", ds, "--k", str(k), "--compress-only"]
            run_step(cmd)

    # Phase 3: Run generation for EARC budgets
    print("\n--- PHASE 3: GENERATION FOR EARC BUDGETS ---")
    for ds in DATASETS:
        all_b = GRID_BUDGETS + [MATCHED_BUDGETS[ds]]
        for b in all_b:
            for gen in GENERATORS:
                pred_file = Path(f"outputs/journal_experiments/exp2_budget/predictions/pred_earc_{ds}_b{b}_{gen.replace(':', '_')}.jsonl")
                if pred_file.exists():
                    lines = open(pred_file, "r", encoding="utf-8").readlines()
                    if len(lines) >= 1000:
                        print(f"[COMPLETED] Already finished 1000 lines for EARC b{b} | {ds} | {gen}")
                        continue
                cmd = [sys.executable, "scripts/run_journal_exp2_budget.py", "--mode", "earc", "--dataset", ds, "--budget", str(b), "--generator", gen]
                run_step(cmd)

    # Phase 4: Run generation for Top-k
    print("\n--- PHASE 4: GENERATION FOR TOP-K PASSAGES ---")
    for ds in DATASETS:
        for k in K_LIST:
            for gen in GENERATORS:
                pred_file = Path(f"outputs/journal_experiments/exp2_budget/predictions/pred_topk_{ds}_k{k}_{gen.replace(':', '_')}.jsonl")
                if pred_file.exists():
                    lines = open(pred_file, "r", encoding="utf-8").readlines()
                    if len(lines) >= 1000:
                        print(f"[COMPLETED] Already finished 1000 lines for Top-{k} | {ds} | {gen}")
                        continue
                cmd = [sys.executable, "scripts/run_journal_exp2_budget.py", "--mode", "topk", "--dataset", ds, "--k", str(k), "--generator", gen]
                run_step(cmd)

    print("\n" + "=" * 80)
    print("EXPERIMENT 2 MASTER PIPELINE FINISHED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
