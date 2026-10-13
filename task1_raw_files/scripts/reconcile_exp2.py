import json
import csv
from pathlib import Path

def reconcile():
    with open("outputs/final_evaluation/selected_example_ids.json", "r") as f:
        locked = json.load(f)
    ds_map = {"hotpotqa": "hotpot", "nq": "nq", "triviaqa": "trivia"}

    datasets = ["hotpotqa", "nq", "triviaqa"]
    generators = ["llama3.2:3b", "ministral-8b"]
    grid_budgets = [150, 300, 450, 600, 750, 900]
    matched_budgets = {"hotpotqa": 598, "nq": 727, "triviaqa": 843}
    topk_ks = list(range(1, 11))

    preds_dir = Path("outputs/journal_experiments/exp2_budget/predictions")

    configs_list = []
    for ds in datasets:
        for gen in generators:
            gen_suffix = gen.replace(":", "_")
            # Grid budgets
            for b in grid_budgets:
                configs_list.append({
                    "dataset": ds,
                    "generator": gen,
                    "method": "earc",
                    "config_id": f"earc_b{b}",
                    "budget": b,
                    "k": None,
                    "jsonl": preds_dir / f"pred_earc_{ds}_b{b}_{gen_suffix}.jsonl"
                })
            # Matched budget
            mb = matched_budgets[ds]
            configs_list.append({
                "dataset": ds,
                "generator": gen,
                "method": "earc",
                "config_id": f"earc_b{mb}",
                "budget": mb,
                "k": None,
                "jsonl": preds_dir / f"pred_earc_{ds}_b{mb}_{gen_suffix}.jsonl"
            })
            # Top-k
            for k in topk_ks:
                configs_list.append({
                    "dataset": ds,
                    "generator": gen,
                    "method": "topk",
                    "config_id": f"topk_k{k}",
                    "budget": None,
                    "k": k,
                    "jsonl": preds_dir / f"pred_topk_{ds}_k{k}_{gen_suffix}.jsonl"
                })

    complete_list = []
    partial_list = []
    missing_list = []

    for c in configs_list:
        locked_set = set(locked[ds_map[c["dataset"]]])
        p = c["jsonl"]
        if not p.exists():
            c["valid_count"] = 0
            c["dup_count"] = 0
            c["missing_count"] = 1000
            c["status"] = "MISSING"
            missing_list.append(c)
            continue
        
        with open(p, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        
        seen = set()
        dups = 0
        valid = 0
        for l in lines:
            try:
                r = json.loads(l)
                qid = r.get("question_id")
                if qid in locked_set:
                    if qid in seen:
                        dups += 1
                    else:
                        seen.add(qid)
                        valid += 1
            except Exception:
                pass
        
        missing_ids = len(locked_set - seen)
        c["valid_count"] = valid
        c["dup_count"] = dups
        c["missing_count"] = missing_ids
        
        if valid == 1000 and dups == 0 and missing_ids == 0:
            c["status"] = "COMPLETE"
            complete_list.append(c)
        elif valid > 0:
            c["status"] = "PARTIAL"
            partial_list.append(c)
        else:
            c["status"] = "MISSING"
            missing_list.append(c)

    print("=" * 80)
    print("EXPERIMENT 2 RECONCILIATION REPORT")
    print("=" * 80)
    print(f"Total Target Combinations: {len(configs_list)}")
    print(f"  COMPLETE : {len(complete_list)}")
    print(f"  PARTIAL  : {len(partial_list)}")
    print(f"  MISSING  : {len(missing_list)}")
    print(f"  TOTAL    : {len(complete_list) + len(partial_list) + len(missing_list)}")
    print("=" * 80)

    if partial_list:
        print("\n--- PARTIAL RUNS ---")
        for c in partial_list:
            print(f"  {c['dataset']} | {c['generator']} | {c['config_id']} | budget={c['budget']} | k={c['k']} -> Valid: {c['valid_count']}, Dups: {c['dup_count']}, Missing: {c['missing_count']}, Path: {c['jsonl'].name}")

    print("\n--- SUMMARY BY DATASET & GENERATOR ---")
    for ds in datasets:
        for gen in generators:
            c_comp = [c for c in complete_list if c["dataset"] == ds and c["generator"] == gen]
            c_part = [c for c in partial_list if c["dataset"] == ds and c["generator"] == gen]
            c_miss = [c for c in missing_list if c["dataset"] == ds and c["generator"] == gen]
            print(f"  {ds:<10} | {gen:<14} -> Complete: {len(c_comp):2d}/17 | Partial: {len(c_part):2d} | Missing: {len(c_miss):2d}")

    # Output detailed JSON for downstream automation
    report_data = {
        "summary": {
            "total_target": len(configs_list),
            "complete": len(complete_list),
            "partial": len(partial_list),
            "missing": len(missing_list),
        },
        "complete": [{k: str(v) if isinstance(v, Path) else v for k, v in c.items()} for c in complete_list],
        "partial": [{k: str(v) if isinstance(v, Path) else v for k, v in c.items()} for c in partial_list],
        "missing": [{k: str(v) if isinstance(v, Path) else v for k, v in c.items()} for c in missing_list],
    }
    with open("outputs/journal_experiments/exp2_reconciliation.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print("\nSaved report to outputs/journal_experiments/exp2_reconciliation.json")

if __name__ == "__main__":
    reconcile()
