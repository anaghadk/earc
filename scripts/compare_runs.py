# scripts/compare_runs.py usage: python scripts/compare_runs.py stored.jsonl fresh.jsonl
import json, sys

def load(p):
    rows = {}
    for l in open(p, encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            qid = str(r.get('question_id', r.get('example_id', '')))
            rows[qid] = r
    return rows

a, b = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(a) & set(b))
print(f'stored {len(a)} fresh {len(b)} common ids {len(common)}')

def share(key):
    n = sum(1 for q in common if key in a[q] and key in b[q] and a[q][key] == b[q][key])
    return 100 * n / len(common) if common else 0.0

def mean(rows, key):
    vals = [rows[q][key] for q in common if key in rows[q]]
    return 100 * sum(vals) / len(vals) if vals else float('nan')

for key in ['context', 'prediction', 'context_tokens']:
    print(f'identical {key:15s}: {share(key):6.2f}% of questions')

for key in ['em', 'f1', 'em_strict', 'f1_strict']:
    print(f'mean {key:10s} stored {mean(a, key):6.2f} fresh {mean(b, key):6.2f}')
