# scripts/rescore_strict.py
# Standard SQuAD v1.1 EM and token F1 (Rajpurkar et al., 2016). Max over gold answers.
import json, re, string, sys, csv, collections, glob, os

def normalize(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def em(pred, golds):
    p = normalize(pred)
    return max(int(p == normalize(g)) for g in golds)

def f1(pred, golds):
    best = 0.0
    p_toks = normalize(pred).split()
    for g in golds:
        g_toks = normalize(g).split()
        common = collections.Counter(p_toks) & collections.Counter(g_toks)
        n = sum(common.values())
        if n == 0:
            continue
        prec, rec = n / len(p_toks), n / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best

def rescore_file(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    out_path = path.replace('.jsonl', '.strict.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in rows:
            golds = r['gold_answers'] if isinstance(r['gold_answers'], list) else [r['gold_answers']]
            r['em_strict'] = em(r['prediction'] or '', golds)
            r['f1_strict'] = f1(r['prediction'] or '', golds)
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    n = len(rows)
    return dict(file=os.path.basename(path), n=n,
        em_old=100 * sum(r.get('em', 0) for r in rows) / n,
        f1_old=100 * sum(r.get('f1', 0) for r in rows) / n,
        em_strict=100 * sum(r['em_strict'] for r in rows) / n,
        f1_strict=100 * sum(r['f1_strict'] for r in rows) / n)

if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else 'runs'
    files = sorted(glob.glob(os.path.join(root, '**', '*.jsonl'), recursive=True))
    files = [f for f in files if not f.endswith('.strict.jsonl')]
    with open('rescore_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'n', 'em_old', 'f1_old', 'em_strict', 'f1_strict'])
        w.writeheader()
        for p in files:
            s = rescore_file(p)
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})
            flag = ' <-- EM above F1, check' if s['em_strict'] > s['f1_strict'] + 1e-9 else ''
            print(f"{s['file']:50s} n={s['n']:5d} old EM/F1 {s['em_old']:6.2f}/{s['f1_old']:6.2f} strict EM/F1 {s['em_strict']:6.2f}/{s['f1_strict']:6.2f}{flag}")
