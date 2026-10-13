# scripts/paired_tests.py usage: python scripts/paired_tests.py comparisons.csv
# comparisons.csv columns: label,dataset,generator,file_a,file_b (test = A minus B)
import json, csv, sys, numpy as np
from scipy.stats import wilcoxon

rng = np.random.default_rng(12345)
B = 10000

def load(p, key='f1_strict'):
    d = {}
    for l in open(p, encoding='utf-8'):
        if l.strip():
            r = json.loads(l); d[str(r['question_id'])] = float(r[key])
    return d

rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8')))
with open('paired_tests_strict.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['label', 'dataset', 'generator', 'n', 'mean_a', 'mean_b', 'delta',
                'ci_low', 'ci_high', 'boot_p', 'wilcoxon_p'])
    for c in rows:
        a, b = load(c['file_a']), load(c['file_b'])
        ids = sorted(set(a) & set(b))
        assert len(ids) >= 950, f"{c['label']}: only {len(ids)} shared question ids"
        d = np.array([a[q] - b[q] for q in ids])
        boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(B)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p_boot = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        p_wil = wilcoxon(d, zero_method='zsplit').pvalue if np.any(d != 0) else 1.0
        w.writerow([c['label'], c['dataset'], c['generator'], len(ids),
                    round(100 * np.mean([a[q] for q in ids]), 2), round(100 * np.mean([b[q] for q in ids]), 2),
                    round(100 * d.mean(), 2), round(100 * lo, 2), round(100 * hi, 2),
                    round(p_boot, 4), round(p_wil, 4)])
        print(c['label'], c['dataset'], c['generator'], f'delta {100*d.mean():+.2f} CI [{100*lo:+.2f}, {100*hi:+.2f}] p_boot {p_boot:.4f} p_wilcoxon {p_wil:.4f}')
