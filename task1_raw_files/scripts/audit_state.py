"""
Detailed audit of Experiment 1 and Experiment 2 state.
Checks line counts, unique IDs, duplicate detection, and alignment with locked IDs.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def main():
    locked_ids_file = PROJECT_ROOT / "outputs/final_evaluation/selected_example_ids.json"
    with open(locked_ids_file, "r", encoding="utf-8") as f:
        locked_ids = json.load(f)

    for ds in ["nq", "hotpotqa", "triviaqa"]:
        ds_key = "hotpot" if ds == "hotpotqa" else ("trivia" if ds == "triviaqa" else "nq")
        print(f"Locked IDs for {ds} ({ds_key}): {len(locked_ids[ds_key])}")

    # =========================================================================
    # EXPERIMENT 1 AUDIT
    # =========================================================================
    exp1_dir = PROJECT_ROOT / "outputs/journal_experiments/exp1_ablation/predictions"
    configs = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3"]
    datasets = ["hotpotqa", "nq", "triviaqa"]
    generators = ["ministral-8b", "llama3.2_3b"]

    print("\n" + "="*80)
    print("EXPERIMENT 1 AUDIT")
    print("="*80)
    exp1_status = {"complete": [], "partial": [], "missing": []}

    for cfg in configs:
        for ds in datasets:
            ds_key = "hotpot" if ds == "hotpotqa" else ("trivia" if ds == "triviaqa" else "nq")
            target_locked = set(locked_ids[ds_key])
            for gen in generators:
                fname = f"pred_{cfg}_{ds}_{gen}.jsonl"
                fpath = exp1_dir / fname
                if not fpath.exists():
                    exp1_status["missing"].append((cfg, ds, gen, fname))
                else:
                    lines = [l for l in open(fpath, "r", encoding="utf-8") if l.strip()]
                    q_ids = []
                    for l in lines:
                        try:
                            r = json.loads(l)
                            q_ids.append(r["question_id"])
                        except:
                            pass
                    unique_ids = set(q_ids)
                    has_dups = len(q_ids) != len(unique_ids)
                    matches_locked = unique_ids.issubset(target_locked)
                    
                    if len(q_ids) == 1000 and not has_dups and len(unique_ids) == 1000 and matches_locked:
                        exp1_status["complete"].append((cfg, ds, gen, fname))
                    else:
                        exp1_status["partial"].append((cfg, ds, gen, fname, len(q_ids), len(unique_ids), has_dups, matches_locked))

    print(f"TOTAL COMPLETE: {len(exp1_status['complete'])} / 48")
    for item in exp1_status["complete"]:
        print(f"  [DONE] {item[0]} | {item[1]} | {item[2]} -> {item[3]}")

    print(f"\nTOTAL PARTIAL: {len(exp1_status['partial'])}")
    for item in exp1_status["partial"]:
        print(f"  [PARTIAL] {item[0]} | {item[1]} | {item[2]} -> {item[3]}: lines={item[4]}, unique={item[5]}, dups={item[6]}, matches_locked={item[7]}")

    print(f"\nTOTAL MISSING: {len(exp1_status['missing'])}")
    for item in exp1_status["missing"]:
        print(f"  [MISSING] {item[0]} | {item[1]} | {item[2]} -> {item[3]}")

    # =========================================================================
    # EXPERIMENT 2 AUDIT
    # =========================================================================
    exp2_dir = PROJECT_ROOT / "outputs/journal_experiments/exp2_budget/predictions"
    print("\n" + "="*80)
    print("EXPERIMENT 2 AUDIT")
    print("="*80)
    exp2_files = sorted([f for f in exp2_dir.glob("*.jsonl")])
    exp2_complete = []
    exp2_partial = []
    for fpath in exp2_files:
        lines = [l for l in open(fpath, "r", encoding="utf-8") if l.strip()]
        if len(lines) == 1000:
            exp2_complete.append((fpath.name, len(lines)))
        else:
            exp2_partial.append((fpath.name, len(lines)))

    print(f"Exp 2 Complete files (1000 lines): {len(exp2_complete)}")
    for item in exp2_complete:
        print(f"  [DONE] {item[0]}")

    print(f"\nExp 2 Partial files: {len(exp2_partial)}")
    for item in exp2_partial:
        print(f"  [PARTIAL] {item[0]} ({item[1]} lines)")

if __name__ == "__main__":
    main()
