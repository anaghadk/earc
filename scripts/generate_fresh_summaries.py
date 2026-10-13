import json, csv, os, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("C:/earc")
sys.path.insert(0, str(PROJECT_ROOT))

fresh_files = [
    ('NQ', 'Llama 3.2 3B', 'EARC default', 'runs/rerun_default/earc_default_rerun__nq__llama3.2_3b.jsonl'),
    ('HotpotQA', 'Llama 3.2 3B', 'EARC default', 'runs/rerun_default/earc_default_rerun__hotpotqa__llama3.2_3b.jsonl'),
    ('TriviaQA', 'Llama 3.2 3B', 'EARC default', 'runs/rerun_default/earc_default_rerun__triviaqa__llama3.2_3b.jsonl'),
    ('NQ', 'Llama 3.2 3B', 'Top-k (k=5)', 'runs/rerun_default/topk5_rerun__nq__llama3.2_3b.jsonl'),
    ('HotpotQA', 'Llama 3.2 3B', 'Top-k (k=5)', 'runs/rerun_default/topk5_rerun__hotpotqa__llama3.2_3b.jsonl'),
    ('TriviaQA', 'Llama 3.2 3B', 'Top-k (k=5)', 'runs/rerun_default/topk5_rerun__triviaqa__llama3.2_3b.jsonl'),
    ('NQ', 'Ministral 8B', 'EARC default', 'runs/rerun_default/earc_default_rerun__nq__ministral_8b.jsonl'),
    ('HotpotQA', 'Ministral 8B', 'EARC default', 'runs/rerun_default/earc_default_rerun__hotpotqa__ministral_8b.jsonl'),
    ('TriviaQA', 'Ministral 8B', 'EARC default', 'runs/rerun_default/earc_default_rerun__triviaqa__ministral_8b.jsonl'),
    ('NQ', 'Ministral 8B', 'Top-k (k=5)', 'runs/rerun_default/topk5_rerun__nq__ministral_8b.jsonl'),
    ('HotpotQA', 'Ministral 8B', 'Top-k (k=5)', 'runs/rerun_default/topk5_rerun__hotpotqa__ministral_8b.jsonl'),
    ('TriviaQA', 'Ministral 8B', 'Top-k (k=5)', 'runs/rerun_default/topk5_rerun__triviaqa__ministral_8b.jsonl'),
]

rows = []
for ds, gen, method, fpath in fresh_files:
    lines = [json.loads(l) for l in open(fpath, encoding='utf-8') if l.strip()]
    n = len(lines)
    em_strict = 100 * np.mean([r['em_strict'] for r in lines])
    f1_strict = 100 * np.mean([r['f1_strict'] for r in lines])
    avg_tokens = np.mean([r['context_tokens'] for r in lines])
    avg_comp_time = np.mean([r['compress_time_s'] for r in lines])
    avg_gen_time = np.mean([r['gen_time_s'] for r in lines])
    rows.append({
        'dataset': ds,
        'generator': gen,
        'method': method,
        'n': n,
        'strict_em': round(em_strict, 2),
        'strict_f1': round(f1_strict, 2),
        'avg_context_tokens': round(avg_tokens, 1),
        'avg_compress_time_s': round(avg_comp_time, 4),
        'avg_gen_time_s': round(avg_gen_time, 4),
        'file': os.path.basename(fpath)
    })

out_p = PROJECT_ROOT / "fresh_runs_summary_strict.csv"
with open(out_p, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Created {out_p} with {len(rows)} rows:")
for r in rows:
    print(f"{r['dataset']:10s} | {r['generator']:14s} | {r['method']:14s} | EM: {r['strict_em']:5.2f}% | F1: {r['strict_f1']:5.2f}% | Tokens: {r['avg_context_tokens']:5.1f}")
