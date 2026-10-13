# scripts/retention_attribution.py usage: python scripts/retention_attribution.py runs/**/*.strict.jsonl
import json, sys, glob, csv, re, string, os, pickle

_CACHE = None
_NORM_CACHE = {}
_IN_RET_CACHE = {}

def normalize(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def contains_gold(text, golds):
    t = normalize(text)
    return any(normalize(g) and normalize(g) in t for g in golds)

def _get_cache():
    global _CACHE
    if _CACHE is None:
        cache_path = os.path.join("task1_raw_files", "retrieval_cache", "retrieved_passages.pkl")
        if not os.path.exists(cache_path):
            cache_path = os.path.join("outputs", "journal_experiments", "retrieval_cache", "retrieved_passages.pkl")
        with open(cache_path, "rb") as f:
            _CACHE = pickle.load(f)
    return _CACHE

def RETRIEVAL_CACHE(dataset, question_id):
    cache = _get_cache()
    ds = dataset.lower()
    if ds.startswith("hotpot"): ds = "hotpotqa"
    elif ds.startswith("trivia"): ds = "triviaqa"
    elif ds.startswith("nq"): ds = "nq"
    return [d["text"] for d in cache[ds][question_id]["docs"]]

def is_yes_no(golds):
    return all(normalize(g) in ('yes', 'no') for g in golds)

def main():
    _get_cache()
    patterns = sys.argv[1:] if len(sys.argv) > 1 else ['runs/**/*.strict.jsonl']
    files = sorted(p for arg in patterns for p in glob.glob(arg, recursive=True))

    out_file = 'retention_attribution_strict.csv'
    fieldnames = ['file', 'dataset', 'generator', 'config_id', 'n', 'excluded',
                  'retention', 'correct', 'retrieval_miss', 'compression_loss', 'generation_error']

    with open(out_file, 'w', newline='', encoding='utf-8') as f_out:
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
        f_out.flush()

        for idx, path in enumerate(files, 1):
            if "rerun_default" in path:
                continue
            rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
            if not rows:
                continue
            n = correct = rmiss = closs = gerr = retained = excluded = 0
            for r in rows:
                golds = r['gold_answers'] if isinstance(r['gold_answers'], list) else [r['gold_answers']]
                if r['dataset'].lower().startswith('hotpot') and is_yes_no(golds):
                    excluded += 1
                    continue
                n += 1
                ds = r['dataset'].lower()
                if ds.startswith("hotpot"): ds = "hotpotqa"
                elif ds.startswith("trivia"): ds = "triviaqa"
                elif ds.startswith("nq"): ds = "nq"
                qid = str(r['question_id'])

                cache_key = (ds, qid)
                if cache_key in _IN_RET_CACHE:
                    in_ret = _IN_RET_CACHE[cache_key]
                else:
                    norm_golds = [normalize(g) for g in golds if normalize(g)]
                    norm_docs = _NORM_CACHE.get(cache_key)
                    if norm_docs is None:
                        norm_docs = [normalize(t) for t in RETRIEVAL_CACHE(ds, qid)]
                        _NORM_CACHE[cache_key] = norm_docs
                    in_ret = any(any(ng in nd for ng in norm_golds) for nd in norm_docs)
                    _IN_RET_CACHE[cache_key] = in_ret

                in_ctx = contains_gold(r.get('context', ''), golds)
                retained += in_ctx
                if r['em_strict'] == 1:
                    correct += 1
                elif not in_ret:
                    rmiss += 1
                elif not in_ctx:
                    closs += 1
                else:
                    gerr += 1

            pct = lambda x: round(100 * x / n, 2)
            row_dict = dict(
                file=os.path.basename(path),
                dataset=rows[0]['dataset'],
                generator=rows[0].get('generator', ''),
                config_id=rows[0].get('config_id', ''),
                n=n,
                excluded=excluded,
                retention=pct(retained),
                correct=pct(correct),
                retrieval_miss=pct(rmiss),
                compression_loss=pct(closs),
                generation_error=pct(gerr)
            )
            w.writerow(row_dict)
            f_out.flush()
            print(f"[{idx:2d}/{len(files)}] {row_dict}", flush=True)

if __name__ == "__main__":
    main()
