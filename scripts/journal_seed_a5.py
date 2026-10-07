"""
Seed A5 (Default EARC, budget 300) for Experiment 1 and Experiment 2 from verified final evaluation outputs.
Matches Protocol Section 1.1 sanity check and Section 1.2 logging format.
"""

import json
from pathlib import Path

def main():
    final_eval_dir = Path("outputs/final_evaluation")
    exp1_preds_dir = Path("outputs/journal_experiments/exp1_ablation/predictions")
    exp2_preds_dir = Path("outputs/journal_experiments/exp2_budget/predictions")
    exp1_preds_dir.mkdir(parents=True, exist_ok=True)
    exp2_preds_dir.mkdir(parents=True, exist_ok=True)

    dataset_map = {
        "nq": "nq",
        "hotpot": "hotpotqa",
        "trivia": "triviaqa",
    }
    generator_map = {
        "ollama": "llama3.2:3b",
        "mistral": "ministral-8b",
    }

    count = 0
    for prov_key, gen_id in generator_map.items():
        gen_clean = gen_id.replace(":", "_")
        for ds_key, ds_name in dataset_map.items():
            src_file = final_eval_dir / prov_key / ds_key / "earc" / "predictions.jsonl"
            if not src_file.exists():
                print(f"Warning: Missing {src_file}")
                continue

            exp1_out = exp1_preds_dir / f"pred_A5_{ds_name}_{gen_clean}.jsonl"
            exp2_out = exp2_preds_dir / f"pred_earc_{ds_name}_b300_{gen_clean}.jsonl"

            records_exp1 = []
            records_exp2 = []

            with open(src_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    rec1 = {
                        "experiment": "exp1",
                        "config_id": "A5",
                        "dataset": ds_name,
                        "generator": gen_id,
                        "question_id": r["example_id"],
                        "gold_answers": r["gold_answers"],
                        "prediction": r["prediction"],
                        "evaluated_answer": r.get("evaluated_answer", ""),
                        "em": int(r["exact_match"] == 1.0),
                        "f1": round(r["f1"], 4),
                        "context_tokens": r["processed_tokens"],
                        "compress_time_s": r.get("compression_latency", 0.0),
                        "gen_time_s": r.get("generation_latency", 0.0),
                        "alpha": 0.7,
                        "beta": 0.3,
                        "theta": 0.85,
                        "budget": 300,
                        "k": None,
                        "n_candidates": None,
                        "n_rejected_redundancy": None,
                    }
                    records_exp1.append(rec1)

                    rec2 = {
                        "experiment": "exp2",
                        "config_id": "earc_b300",
                        "dataset": ds_name,
                        "generator": gen_id,
                        "question_id": r["example_id"],
                        "gold_answers": r["gold_answers"],
                        "prediction": r["prediction"],
                        "evaluated_answer": r.get("evaluated_answer", ""),
                        "em": int(r["exact_match"] == 1.0),
                        "f1": round(r["f1"], 4),
                        "context_tokens": r["processed_tokens"],
                        "compress_time_s": r.get("compression_latency", 0.0),
                        "gen_time_s": r.get("generation_latency", 0.0),
                        "alpha": 0.7,
                        "beta": 0.3,
                        "theta": 0.85,
                        "budget": 300,
                        "k": None,
                    }
                    records_exp2.append(rec2)

            with open(exp1_out, "w", encoding="utf-8") as f:
                for rec in records_exp1:
                    f.write(json.dumps(rec) + "\n")

            with open(exp2_out, "w", encoding="utf-8") as f:
                for rec in records_exp2:
                    f.write(json.dumps(rec) + "\n")

            print(f"Seeded {len(records_exp1)} records for {ds_name} | {gen_id} (A5 / b300)")
            count += len(records_exp1)

    print(f"Total seeded records across Exp 1 & Exp 2: {count}")

if __name__ == "__main__":
    main()
