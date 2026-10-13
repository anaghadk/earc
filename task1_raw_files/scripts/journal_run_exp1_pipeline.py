"""
Master Pipeline Runner for Experiment 1 (Ablation and Beta Sweep).
Iterates through all configurations and datasets:
1. Runs compression (compress-once) for any missing contexts.
2. Runs generation for Ministral-8B and Llama 3.2.
3. Automatically updates exp1_summary.csv incrementally.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

CONFIGS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3"] # A5 is already seeded
DATASETS = ["hotpotqa", "nq", "triviaqa"]
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
    print("STARTING EXPERIMENT 1 FULL PIPELINE (7 configs x 3 datasets x 2 generators)")
    print("=" * 80)

    # Step 1: Pre-compress all contexts first (if missing)
    print("\n--- PHASE 1: PRE-COMPRESSING ALL CONTEXTS ---")
    for cfg in CONFIGS:
        for ds in DATASETS:
            ctx_file = Path(f"outputs/journal_experiments/exp1_ablation/contexts/ctx_{cfg}_{ds}.json")
            if ctx_file.exists():
                print(f"[CACHE HIT] Contexts already exist: {ctx_file}")
                continue
            cmd = [sys.executable, "scripts/run_journal_exp1_ablation.py", "--config", cfg, "--dataset", ds, "--compress-only"]
            run_step(cmd)

    # Step 2: Run generation
    print("\n--- PHASE 2: GENERATION ACROSS PROVIDERS ---")
    for cfg in CONFIGS:
        for ds in DATASETS:
            for gen in GENERATORS:
                pred_file = Path(f"outputs/journal_experiments/exp1_ablation/predictions/pred_{cfg}_{ds}_{gen.replace(':', '_')}.jsonl")
                if pred_file.exists():
                    lines = open(pred_file, "r", encoding="utf-8").readlines()
                    if len(lines) >= 1000:
                        print(f"[COMPLETED] Already finished 1000 lines for {cfg} | {ds} | {gen}")
                        continue
                cmd = [sys.executable, "scripts/run_journal_exp1_ablation.py", "--config", cfg, "--dataset", ds, "--generator", gen]
                run_step(cmd)

    print("\n" + "=" * 80)
    print("EXPERIMENT 1 MASTER PIPELINE FINISHED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
