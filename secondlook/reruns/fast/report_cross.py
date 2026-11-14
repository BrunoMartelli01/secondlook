"""E49 report: Doc Table 8 / 10 (leave-one-dataset-out and pooled training) and the learning curve (Fig. 5 / Doc Table 8 of rev 48).
Cross-dataset: RUN/<arm>/crossdata/cross_<expert>.npz (E44 (B) code, per-image counts of the first stage and the complete method) next to the own fits and thr_train of the stability chain (S = all).
Learning curve: RUN/<arm>/lc/out/<expert>/<task>/seed<s>.json (secondlook.studies.learning_curve, eleven draws; 'all' on draw 0). FAST selection throughout."""
import numpy as np

from secondlook.reruns.fast.lib import KEYS, NAMES, ROWS9, RUN, f1, f1_boot, fm, idx_matrix, jload, md_table, sg, ub

TASKS = ['totaltext', 'icdar2015', 'ctw1500']
TN = {'totaltext': 'TT', 'icdar2015': 'IC15', 'ctw1500': 'CTW'}
EN = {'deepsolo': 'DeepSolo', 'ticls': 'TiCLS', 'ppocr': 'PP-OCR'}
BUD = ['30', '60', '120', '240', '480', 'all']


def build_cross():
    C = {a: {e: np.load(RUN / a / 'crossdata' / ('cross_%s.npz' % e)) for e in EN} for a in ('old', 'unified')}
    ST = {a: {k: np.load(RUN / a / 'stability' / k / 'counts7.npz') for k in KEYS} for a in ('old', 'unified')}
    meta = {a: {e: jload(RUN / a / 'crossdata' / ('cross_%s.json' % e)) for e in EN} for a in ('old', 'unified')}

    def arms(a, e, t):
        s = ST[a]['%s_%s' % (e, t)]
        c = C[a][e]
        return dict(thr=s['thr_train'][:, :3], own1=s['first'][:, :3], own2=s['complete'][:, :3], loo1=c['B1_heldout_%s__%s__first' % (t, t)][:, :3], loo2=c['B1_heldout_%s__%s__complete' % (t, t)][:, :3],
                    pool1=c['B2_pooled_all__%s__first' % t][:, :3], pool2=c['B2_pooled_all__%s__complete' % t][:, :3])
    hdr = ['Spotter, test set', 'thr_train', 'Ours step 1, own', 'Ours full, own', 'Ours step 1, leave-one-out', 'Ours full, leave-one-out', 'Ours step 1, pooled', 'Ours full, pooled']
    names = ['thr', 'own1', 'own2', 'loo1', 'loo2', 'pool1', 'pool2']
    blocks = {'old': [], 'unified': [], 'delta': []}
    data = {'old': {}, 'unified': {}}
    for e in EN:
        per = {a: {t: arms(a, e, t) for t in TASKS} for a in ('old', 'unified')}
        comb = {a: {n: np.concatenate([per[a][t][n] for t in TASKS]) for n in names} for a in ('old', 'unified')}
        for t in TASKS + ['combined']:
            key = '%s_%s' % (e, t)
            label = ('TiCLS ‡, CTW' if (e, t) == ('ticls', 'ctw1500') else '%s, %s' % (EN[e], TN.get(t, 'combined')))
            rows = {}
            for a in ('old', 'unified'):
                X = comb[a] if t == 'combined' else per[a][t]
                n = X['thr'].shape[0]
                M = idx_matrix(n)
                rows[a] = {nm: f1(X[nm]) for nm in names}
                rows[a]['vs_thr'] = {nm: [f1(X[nm]) - f1(X['thr'])] + list(np.percentile(f1_boot(X[nm], M) - f1_boot(X['thr'], M), [2.5, 97.5])) for nm in names[1:]}
                rows[a]['pool_vs_own'] = [f1(X['pool2']) - f1(X['own2'])] + list(np.percentile(f1_boot(X['pool2'], M) - f1_boot(X['own2'], M), [2.5, 97.5]))
                blocks[a].append([label] + ['%.2f' % rows[a][nm] for nm in names])
                data[a][key] = rows[a]
            blocks['delta'].append([label] + [sg(rows['unified'][nm] - rows['old'][nm]) for nm in names])
    hdrs = hdr
    out = [('Table 8 / 10 (cross-dataset end-to-end F1 in percent on the full test splits; thr_train: per-benchmark re-tuned thresholds; own: fitted on the benchmark itself; leave-one-out: the two other benchmarks; pooled: the union of the three training pools; combined: pooled counts of the 1,300 test images). Router: DeepSolo rows 100% to DeepSolo, TiCLS and PP-OCR rows 0%. One deterministic fit per cell, FAST selection', hdrs, blocks)]
    gate = {a: {e: meta[a][e].get('gates') for e in EN} for a in ('old', 'unified')}
    return out, dict(cells=data, gates=gate)


def build_lc():
    def loadlc(a):
        out = {}
        for e, t in ROWS9:
            ds = [jload(RUN / a / 'lc' / 'out' / e / t / ('seed%d.json' % s)) for s in range(11)]
            out['%s_%s' % (e, t)] = ds
        return out
    L = {a: loadlc(a) for a in ('old', 'unified')}
    hdr = ['N'] + BUD[:-1] + ['all']
    blocks = {'old': [], 'unified': [], 'delta': []}
    summ = {'old': {}, 'unified': {}}
    for a in ('old', 'unified'):
        for k in KEYS:
            row = []
            s = {}
            for b in BUD:
                if b == 'all':
                    v = [L[a][k][0]['budgets']['all']['diff']]
                    s[b] = dict(mean=v[0], min=None, max=None, n_pos=None)
                    row.append(sg(v[0]))
                else:
                    v = np.array([d['budgets'][b]['diff'] for d in L[a][k]])
                    s[b] = dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()), n_pos=int((v > 0).sum()))
                    row.append('%s [%s, %s]' % (sg(v.mean()), sg(v.min()), sg(v.max())))
            summ[a][k] = s
            blocks[a].append([NAMES[k]] + row)
    for k in KEYS:
        blocks['delta'].append([NAMES[k]] + [sg(summ['unified'][k][b]['mean'] - summ['old'][k][b]['mean']) for b in BUD])
    out = [('Table 8 of rev 48 / Fig. 5 (learning curve: Ours step 1 minus thresholds fitted on the same N labelled training images; F1 difference, mean over eleven draws, with the smallest and the largest draw in brackets; all: whole labelled training pool, one value). Delta: difference of the means (unified − old)', hdr, blocks)]
    return out, summ
