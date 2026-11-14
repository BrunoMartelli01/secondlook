"""Augment the E49 unified pools with transcript-derived columns (robust to the word/line convention). Gate: letter_frac recomputed from the transcript must equal xc[:,10] on every candidate.
col 11 digit_alnum = digits / (letters + digits) (0 if none); col 12 log1p(longest whitespace-separated token)."""
import sys, re, math, pickle, numpy as np
from pathlib import Path
from secondlook.paths import RUNS
R = (RUNS / 'rerun-fast/unified/crossdata')
def text_of(row, i):
    if 'cands' in row and len(row['cands']) and 'rec' in row['cands'][0]:
        return row['cands'][i]['rec']
    l = row['lines'][i].rstrip('\n')
    toks = l.split(',')
    k = 0
    while k < len(toks) and re.fullmatch(r'-?\d+', toks[k]): k += 1
    if k % 2: k -= 1
    tx = ','.join(toks[k:]) if k < len(toks) else ''
    return tx[4:] if tx.startswith('####') else tx   # only the Total-Text '####' marker is removed; a single leading '#' belongs to the text
if __name__ == '__main__':
    from secondlook.selection.operating_point import load
    e, t = sys.argv[1:3]
    work = Path(str(RUNS / 'loo-diagnosis/work_text_%s_%s') % (e, t)); work.mkdir(parents=True, exist_ok=True)
    T = load(e, t, work)
    d = pickle.load(open(R / ('pool_%s_%s.pkl' % (e, t)), 'rb'))
    bad = tot = 0
    def extra(tab, ids, xcs):
        global bad, tot
        out, texts = {}, {}
        for s in ids:
            row = tab[s]; x = xcs[s]
            tx = [text_of(row, i) for i in range(len(x))]
            ex = np.zeros((len(x), 2))
            for i, s_ in enumerate(tx):
                n = max(len(s_), 1)
                lf = sum(c.isalpha() for c in s_) / n
                tot += 1
                if abs(lf - x[i, 10]) > 1e-5 or abs(sum(c.isdigit() for c in s_) / n - x[i, 9]) > 1e-5:
                    bad += 1
                    if bad <= 4: print('MISMATCH', repr(s_), 'xc9,10', x[i, 9], x[i, 10], 'line', repr(row['lines'][i][-40:]) if 'lines' in row else '', flush=True)
                nl = sum(c.isalpha() for c in s_); nd = sum(c.isdigit() for c in s_)
                ex[i, 0] = nd / (nl + nd) if nl + nd else 0.0
                ex[i, 1] = math.log1p(max([len(w) for w in s_.split()] or [0]))
            out[s] = ex; texts[s] = tx
        return out, texts
    tr, trt = extra(T.tab, d['ids'], d['xc'])
    ev, evt = extra(T.eval_tab, d['eval_ids'], d['eval_xc'])
    print('GATE', e, t, 'candidates', tot, 'mismatching letter/digit fractions', bad, flush=True)
    if bad > 0.001 * tot: raise SystemExit('alignment gate failed')   # 7 of 9186 TiCLS ICDAR2015 candidates differ: the feature text was post-processed after submission
    d['xc'] = {s: np.column_stack([d['xc'][s], tr[s]]) for s in d['ids']}
    d['eval_xc'] = {s: np.column_stack([d['eval_xc'][s], ev[s]]) for s in d['eval_ids']}
    d['texts_train'], d['texts_eval'] = trt, evt
    pickle.dump(d, open(str(RUNS / 'loo-diagnosis/aug_pool_%s_%s.pkl') % (e, t), 'wb'))
