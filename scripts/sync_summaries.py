"""
Sync completed JSONL predictions into summary CSVs.
"""

import json
import csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def sync_exp(exp_name: str):
    exp_dir = PROJECT_ROOT / f"outputs/journal_experiments/{exp_name}"
    preds_dir = exp_dir / "predictions"
    summary_csv = exp_dir / f"{exp_name.split('_')[0]}_summary.csv"

    if not preds_dir.exists():
        return

    rows = []
    for f in sorted(preds_dir.glob("*.jsonl")):
        with open(f, "r", encoding="utf-8") as fp:
            lines = [l for l in fp if l.strip()]
        if len(lines) == 1000:
            recs = [json.loads(l) for l in lines]
            r0 = recs[0]
            em_val = [r.get("em", r.get("exact_match", 0.0)) for r in recs]
            f1_val = [r.get("f1", 0.0) for r in recs]
            rows.append({
                "experiment": exp_name.split("_")[0],
                "config_id": r0.get("config_id", f.stem),
                "dataset": r0.get("dataset"),
                "generator": r0.get("generator"),
                "method": f"{r0.get('config_id', f.stem)}",
                "alpha": r0.get("alpha"),
                "beta": r0.get("beta"),
                "theta": r0.get("theta"),
                "budget": r0.get("budget", 300 if "longllmlingua" in f.stem else None),
                "k": r0.get("k"),
                "avg_tokens": round(float(np.mean([r["context_tokens"] for r in recs])), 1),
                "em": round(float(np.mean(em_val)) * 100.0, 2),
                "f1": round(float(np.mean(f1_val)) * 100.0, 2),
                "compress_time_s": round(float(np.mean([r["compress_time_s"] for r in recs])), 3),
                "gen_time_s": round(float(np.mean([r["gen_time_s"] for r in recs])), 3),
                "n_questions": len(recs),
            })

    if rows:
        with open(summary_csv, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[{exp_name}] Synced {len(rows)} completed runs to {summary_csv}")

def main():
    sync_exp("exp1_ablation")
    sync_exp("exp2_budget")
    sync_exp("exp3_longllmlingua")

if __name__ == "__main__":
    main()
