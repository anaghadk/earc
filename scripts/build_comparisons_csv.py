"""
scripts/build_comparisons_csv.py

Builds comparisons.csv containing all valid comparisons where both file_a and file_b exist.
Omits comparisons where files were missing from outputs.zip recovery (e.g. LongLLMLingua, matched budgets, k=2).
"""

import csv
import os
from pathlib import Path

DATASETS = ["nq", "hotpotqa", "triviaqa"]
GENERATORS = ["llama3.2_3b", "ministral-8b"]

MATCHED_BUDGET = {
    "nq": "b727",
    "hotpotqa": "b598",
    "triviaqa": "b843",
}

def main():
    rows = []
    omitted = []
    for ds in DATASETS:
        for gen in GENERATORS:
            candidates = [
                ("full_vs_sim_only", f"runs/exp1_ablation/pred_A5_{ds}_{gen}.strict.jsonl", f"runs/exp1_ablation/pred_A1_{ds}_{gen}.strict.jsonl"),
                ("full_vs_evidence_only", f"runs/exp1_ablation/pred_A5_{ds}_{gen}.strict.jsonl", f"runs/exp1_ablation/pred_A2_{ds}_{gen}.strict.jsonl"),
                ("full_vs_no_redundancy", f"runs/exp1_ablation/pred_A5_{ds}_{gen}.strict.jsonl", f"runs/exp1_ablation/pred_A3_{ds}_{gen}.strict.jsonl"),
                ("full_vs_sim_redundancy", f"runs/exp1_ablation/pred_A5_{ds}_{gen}.strict.jsonl", f"runs/exp1_ablation/pred_A4_{ds}_{gen}.strict.jsonl"),
                ("redundancy_alone", f"runs/exp1_ablation/pred_A4_{ds}_{gen}.strict.jsonl", f"runs/exp1_ablation/pred_A1_{ds}_{gen}.strict.jsonl"),
                ("evidence_alone", f"runs/exp1_ablation/pred_A3_{ds}_{gen}.strict.jsonl", f"runs/exp1_ablation/pred_A1_{ds}_{gen}.strict.jsonl"),
                ("earc300_vs_top5", f"runs/baseline/pred_baseline_{ds}_earc_{gen}.strict.jsonl", f"runs/baseline/pred_baseline_{ds}_topk_{gen}.strict.jsonl"),
                ("earc_matched_vs_top5", f"runs/exp2_budget/pred_earc_{ds}_{MATCHED_BUDGET[ds]}_{gen}.strict.jsonl", f"runs/baseline/pred_baseline_{ds}_topk_{gen}.strict.jsonl"),
                ("earc300_vs_top2", f"runs/baseline/pred_baseline_{ds}_earc_{gen}.strict.jsonl", f"runs/exp2_budget/pred_topk_{ds}_k2_{gen}.strict.jsonl"),
                ("earc_vs_longllmlingua", f"runs/baseline/pred_baseline_{ds}_earc_{gen}.strict.jsonl", f"runs/exp3_longllmlingua/pred_longllmlingua_{ds}_{gen}.strict.jsonl"),
                ("earc_vs_llmlingua2", f"runs/baseline/pred_baseline_{ds}_earc_{gen}.strict.jsonl", f"runs/baseline/pred_baseline_{ds}_llmlingua2_{gen}.strict.jsonl"),
            ]

            for label, f_a, f_b in candidates:
                if os.path.exists(f_a) and os.path.exists(f_b):
                    rows.append({"label": label, "dataset": ds, "generator": gen, "file_a": f_a, "file_b": f_b})
                else:
                    omitted.append((label, ds, gen, f_a if not os.path.exists(f_a) else f_b))

    out_csv = "comparisons.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "dataset", "generator", "file_a", "file_b"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} valid comparisons to {out_csv}")
    print(f"Omitted {len(omitted)} comparisons due to missing prediction files in recovery:")
    for label, ds, gen, missing in omitted:
        print(f"  - {label:25s} | {ds:8s} | {gen:15s} -> missing: {missing}")

if __name__ == "__main__":
    main()
