"""
Rebuild results_strict.csv, exp1_summary_strict.csv, exp2_summary_strict.csv
from the .strict.jsonl files, using em_strict and f1_strict as the em and f1 columns.
"""

import json
import csv
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("C:/earc")
DELIV_DIR = PROJECT_ROOT / "recovered_outputs/outputs/journal_experiments/deliverables"
RUNS_DIR = PROJECT_ROOT / "runs"

# 1. Rebuild exp1_summary_strict.csv
exp1_old_csv = PROJECT_ROOT / "recovered_outputs/outputs/journal_experiments/exp1_ablation/exp1_summary.csv"
if exp1_old_csv.exists():
    df_exp1 = pd.read_csv(exp1_old_csv)
    for idx, row in df_exp1.iterrows():
        cfg = row["config_id"]
        ds = row["dataset"]
        gen = row["generator"].replace(":", "_")
        strict_file = RUNS_DIR / f"exp1_ablation/pred_{cfg}_{ds}_{gen}.strict.jsonl"
        if strict_file.exists():
            rows = [json.loads(l) for l in open(strict_file, encoding="utf-8") if l.strip()]
            n = len(rows)
            em_st = round(100.0 * sum(r["em_strict"] for r in rows) / n, 2)
            f1_st = round(100.0 * sum(r["f1_strict"] for r in rows) / n, 2)
            df_exp1.at[idx, "em"] = em_st
            df_exp1.at[idx, "f1"] = f1_st
    df_exp1.to_csv("exp1_summary_strict.csv", index=False)
    print("Saved exp1_summary_strict.csv")

# 2. Rebuild exp2_summary_strict.csv
exp2_old_csv = PROJECT_ROOT / "recovered_outputs/outputs/journal_experiments/exp2_budget/exp2_summary.csv"
if exp2_old_csv.exists():
    df_exp2 = pd.read_csv(exp2_old_csv)
    for idx, row in df_exp2.iterrows():
        cfg = row["config_id"]
        ds = row["dataset"]
        gen = row["generator"].replace(":", "_")
        # Find matching file in exp2_budget
        budget_str = cfg.replace("earc_", "") if "earc_" in cfg else "k5"
        method_prefix = "earc" if "earc" in cfg else "topk"
        candidates = list((RUNS_DIR / "exp2_budget").glob(f"pred_{method_prefix}_{ds}_{budget_str}*{gen}.strict.jsonl"))
        if candidates and candidates[0].exists():
            matched_file = candidates[0]
            rows = [json.loads(l) for l in open(matched_file, encoding="utf-8") if l.strip()]
            n = len(rows)
            em_st = round(100.0 * sum(r["em_strict"] for r in rows) / n, 2)
            f1_st = round(100.0 * sum(r["f1_strict"] for r in rows) / n, 2)
            df_exp2.at[idx, "em"] = em_st
            df_exp2.at[idx, "f1"] = f1_st
    df_exp2.to_csv("exp2_summary_strict.csv", index=False)
    df_exp2.to_csv("runs/rerun_default/tables/exp2_summary_strict.csv", index=False)
    print("Saved exp2_summary_strict.csv")

# 3. Rebuild results_strict.csv
results_old_csv = DELIV_DIR / "results.csv"
if results_old_csv.exists():
    df_res = pd.read_csv(results_old_csv)
    for idx, row in df_res.iterrows():
        exp = row["experiment"]
        cfg = row["config_id"]
        ds = row["dataset"]
        gen = row["generator"].replace(":", "_")
        
        strict_file = None
        if exp == "exp1":
            strict_file = RUNS_DIR / f"exp1_ablation/pred_{cfg}_{ds}_{gen}.strict.jsonl"
        elif exp == "exp2":
            budget_str = cfg.replace("earc_", "") if "earc_" in cfg else "k5"
            method_prefix = "earc" if "earc" in cfg else "topk"
            candidates = list((RUNS_DIR / "exp2_budget").glob(f"pred_{method_prefix}_{ds}_{budget_str}*{gen}.strict.jsonl"))
            if candidates:
                strict_file = candidates[0]
        elif exp == "baseline":
            strict_file = RUNS_DIR / f"baseline/pred_baseline_{ds}_{cfg}_{gen}.strict.jsonl"
            
        if strict_file and strict_file.exists():
            rows = [json.loads(l) for l in open(strict_file, encoding="utf-8") if l.strip()]
            n = len(rows)
            em_st = round(100.0 * sum(r["em_strict"] for r in rows) / n, 2)
            f1_st = round(100.0 * sum(r["f1_strict"] for r in rows) / n, 2)
            df_res.at[idx, "em"] = em_st
            df_res.at[idx, "f1"] = f1_st
            
    df_res.to_csv("results_strict.csv", index=False)
    df_res.to_csv("runs/rerun_default/tables/results_strict.csv", index=False)
    print("Saved results_strict.csv")

# Also copy exp1_summary_strict.csv to runs/rerun_default/tables/
if Path("exp1_summary_strict.csv").exists():
    df_exp1.to_csv("runs/rerun_default/tables/exp1_summary_strict.csv", index=False)

