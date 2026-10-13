"""
Autonomous generator-specific runner for Journal Experiments.
Runs through all required runs for a given generator sequentially.
Resumes existing runs, handles in-progress tasks, and auto-updates summary CSVs.
"""

import sys
import os
import time
import json
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def is_file_growing(path: Path, wait_sec: float = 3.0) -> bool:
    if not path.exists():
        return False
    sz1 = path.stat().st_size
    time.sleep(wait_sec)
    sz2 = path.stat().st_size
    return sz2 > sz1

def run_exp1_for_generator(generator: str):
    print("=" * 80)
    print(f"STARTING EXPERIMENT 1 SEQUENTIAL RUNNER FOR GENERATOR: {generator}")
    print("=" * 80)

    configs = ["A1", "A2", "A3", "A4", "B1", "B2", "B3"]
    # Run HotpotQA first, then NQ, then TriviaQA as specified in Protocol Section 6
    datasets = ["hotpotqa", "nq", "triviaqa"]

    preds_dir = PROJECT_ROOT / "outputs/journal_experiments/exp1_ablation/predictions"
    gen_file_suffix = generator.replace(":", "_")

    for cfg in configs:
        for ds in datasets:
            pred_file = preds_dir / f"pred_{cfg}_{ds}_{gen_file_suffix}.jsonl"

            # Check if already completed
            if pred_file.exists():
                lines = [l for l in open(pred_file, "r", encoding="utf-8") if l.strip()]
                if len(lines) >= 1000:
                    print(f"[ALREADY COMPLETED] {cfg} | {ds} | {generator} ({len(lines)} lines)")
                    continue
                elif is_file_growing(pred_file):
                    print(f"[IN PROGRESS DETECTED] File {pred_file.name} is currently growing. Waiting for it to finish...")
                    while True:
                        time.sleep(10)
                        lines = [l for l in open(pred_file, "r", encoding="utf-8") if l.strip()]
                        print(f"[{pred_file.name}] Current lines: {len(lines)}/1000...")
                        if len(lines) >= 1000:
                            print(f"[COMPLETED] {pred_file.name} reached 1000 lines!")
                            break
                        if not is_file_growing(pred_file, wait_sec=15.0):
                            print(f"[STALLED DETECTED] {pred_file.name} stopped growing at {len(lines)} lines. Resuming via runner...")
                            break
                    if len(lines) >= 1000:
                        continue

            print(f"\n[LAUNCHING] Exp 1 | Config: {cfg} | Dataset: {ds} | Generator: {generator}")
            cmd = [
                sys.executable,
                "scripts/run_journal_exp1_ablation.py",
                "--config", cfg,
                "--dataset", ds,
                "--generator", generator
            ]
            t0 = time.time()
            res = subprocess.run(cmd)
            elapsed = time.time() - t0
            if res.returncode != 0:
                print(f"[ERROR] Run failed with returncode {res.returncode}")
            else:
                print(f"[FINISHED] {cfg} | {ds} | {generator} in {elapsed:.1f}s")
                subprocess.run([sys.executable, "scripts/sync_summaries.py"])
                subprocess.run([sys.executable, "scripts/journal_generate_deliverables.py"])

    print("\n" + "=" * 80)
    print(f"EXPERIMENT 1 RUNNER FOR {generator} COMPLETED ALL RUNS!")
    print("=" * 80)

def run_exp2_for_generator(generator: str):
    print("=" * 80)
    print(f"STARTING EXPERIMENT 2 SEQUENTIAL RUNNER FOR GENERATOR: {generator}")
    print("=" * 80)

    datasets = ["hotpotqa", "nq", "triviaqa"]
    grid_budgets = [150, 450, 600, 750, 900]
    matched_budgets = {"nq": 727, "hotpotqa": 598, "triviaqa": 843}
    k_list = [1, 2, 3, 4, 6, 7, 8, 9, 10]

    preds_dir = PROJECT_ROOT / "outputs/journal_experiments/exp2_budget/predictions"
    gen_file_suffix = generator.replace(":", "_")

    # Part 1: EARC budgets (HotpotQA first)
    print("\n--- PART 1: EARC BUDGETS ---")
    for ds in datasets:
        all_b = grid_budgets + [matched_budgets[ds]]
        for b in all_b:
            pred_file = preds_dir / f"pred_earc_{ds}_b{b}_{gen_file_suffix}.jsonl"
            if pred_file.exists():
                lines = [l for l in open(pred_file, "r", encoding="utf-8") if l.strip()]
                if len(lines) >= 1000:
                    print(f"[ALREADY COMPLETED] EARC b{b} | {ds} | {generator}")
                    continue

            print(f"\n[LAUNCHING] Exp 2 EARC | Budget: {b} | Dataset: {ds} | Generator: {generator}")
            cmd = [
                sys.executable,
                "scripts/run_journal_exp2_budget.py",
                "--mode", "earc",
                "--dataset", ds,
                "--budget", str(b),
                "--generator", generator
            ]
            t0 = time.time()
            res = subprocess.run(cmd)
            elapsed = time.time() - t0
            print(f"[FINISHED] EARC b{b} | {ds} | {generator} in {elapsed:.1f}s")
            subprocess.run([sys.executable, "scripts/sync_summaries.py"])
            subprocess.run([sys.executable, "scripts/journal_generate_deliverables.py"])

    # Part 2: Top-k passages (HotpotQA first)
    print("\n--- PART 2: TOP-K PASSAGES ---")
    for ds in datasets:
        for k in k_list:
            pred_file = preds_dir / f"pred_topk_{ds}_k{k}_{gen_file_suffix}.jsonl"
            if pred_file.exists():
                lines = [l for l in open(pred_file, "r", encoding="utf-8") if l.strip()]
                if len(lines) >= 1000:
                    print(f"[ALREADY COMPLETED] Top-{k} | {ds} | {generator}")
                    continue

            print(f"\n[LAUNCHING] Exp 2 Top-k | k: {k} | Dataset: {ds} | Generator: {generator}")
            cmd = [
                sys.executable,
                "scripts/run_journal_exp2_budget.py",
                "--mode", "topk",
                "--dataset", ds,
                "--k", str(k),
                "--generator", generator
            ]
            t0 = time.time()
            res = subprocess.run(cmd)
            elapsed = time.time() - t0
            print(f"[FINISHED] Top-{k} | {ds} | {generator} in {elapsed:.1f}s")
            subprocess.run([sys.executable, "scripts/sync_summaries.py"])
            subprocess.run([sys.executable, "scripts/journal_generate_deliverables.py"])

    print("\n" + "=" * 80)
    print(f"EXPERIMENT 2 RUNNER FOR {generator} COMPLETED ALL RUNS!")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Journal Experiments Autonomous Runner")
    parser.add_argument("--exp", type=str, choices=["exp1", "exp2", "all"], default="exp1", help="Experiment to run")
    parser.add_argument("--generator", type=str, required=True, choices=["ministral-8b", "llama3.2:3b"], help="Target generator")
    args = parser.parse_args()

    if args.exp in ["exp1", "all"]:
        run_exp1_for_generator(args.generator)
    
    if args.exp in ["exp2", "all"]:
        flag_file = PROJECT_ROOT / "outputs/journal_experiments/exp2_allowed.flag"
        if args.exp == "all" and not flag_file.exists():
            print("\n" + "=" * 80)
            print(f"[PAUSED] Experiment 1 completed for {args.generator}.")
            print("Awaiting approval before starting Experiment 2 (create outputs/journal_experiments/exp2_allowed.flag to proceed).")
            print("=" * 80)
            return

        run_exp2_for_generator(args.generator)

if __name__ == "__main__":
    main()

