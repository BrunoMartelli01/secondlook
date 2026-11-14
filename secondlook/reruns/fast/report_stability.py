"""E49 report: Doc Table 3 (detection-only P / R / F with the standard deviation over the eleven 80% subsets) and Table 9 (stability of the gain of the complete method over the re-tuned
thresholds), from RUN/<arm>/stability/<row>/{stability.json, counts7.npz}. FAST selection protocol for c, k* and the baseline thresholds (label-based F1), identical for both."""
import numpy as np

from secondlook.reruns.fast.lib import KEYS, NAMES, RUN, det_boot, det_rates, f1, f1_boot, fm, idx_matrix, jload, md_table, router_text, sg, ub

METHODS = [('released', 'Released'), ('thr_train', 'thr_train'), ('first', 'Ours, step 1'), ('complete', 'Ours, full')]
TASKCOL = {'totaltext': 'Total-Text', 'icdar2015': 'ICDAR2015', 'ctw1500': 'CTW1500'}


def load(arm, k):
    d = RUN / arm / 'stability' / k
    return jload(d / 'stability.json'), np.load(d / 'counts7.npz')


def sd_f(Z, name):
    v = [det_rates(Z['%s_s%d' % (name, s)])['f'] for s in range(11)]
    return float(np.std(v, ddof=1))


def build():
    out, data = [], {}
    hdr = ['Spotter, data'] + [m[1] for m in METHODS]
    blocks = {'old': [], 'unified': [], 'delta': []}
    claims3 = {'old': {}, 'unified': {}}
    for k in KEYS:
        S = {a: load(a, k) for a in ('old', 'unified')}
        n = S['old'][1]['released'].shape[0]
        M = idx_matrix(n)
        cells = {}
        for a in ('old', 'unified'):
            Z = S[a][1]
            row = [NAMES[k]]
            cl = {}
            for m, _ in METHODS:
                r = det_rates(Z[m])
                sd = '' if m == 'released' else ' (%.2f)' % sd_f(Z, m)
                row.append('%.2f / %.2f / %.2f%s' % (r['p'], r['r'], r['f'], sd))
                if m != 'released':
                    b, bt = det_boot(Z[m], M), det_boot(Z['thr_train'], M)
                    cl[m] = dict(P=[r['p'] - det_rates(Z['thr_train'])['p']] + list(np.percentile(b['p'] - bt['p'], [2.5, 97.5])), R=[r['r'] - det_rates(Z['thr_train'])['r']] + list(np.percentile(b['r'] - bt['r'], [2.5, 97.5])),
                                 F=[r['f'] - det_rates(Z['thr_train'])['f']] + list(np.percentile(b['f'] - bt['f'], [2.5, 97.5])), n=r['n'])
            blocks[a].append(row)
            cells[a] = row
            claims3[a][k] = cl
        dl = [NAMES[k]]
        for m, _ in METHODS:
            ro, ru = det_rates(S['old'][1][m]), det_rates(S['unified'][1][m])
            bo, bu = det_boot(S['old'][1][m], M), det_boot(S['unified'][1][m], M)
            dF = list(np.percentile(bu['f'] - bo['f'], [2.5, 97.5]))
            dl.append('%s / %s / %s [%s, %s]' % (sg(ru['p'] - ro['p']), sg(ru['r'] - ro['r']), sg(ru['f'] - ro['f']), ub(dF[0]), ub(dF[1])) + ('' if m == 'released' else ' sd %s' % sg(sd_f(S['unified'][1], m) - sd_f(S['old'][1], m))))
        blocks['delta'].append(dl)
        data[k] = dict(t3=dict(old=cells['old'], unified=cells['unified']))
    data['claims3'] = claims3
    out.append(('Table 3 (detection-only precision / recall / F in percent, polygon IoU 0.5, transcription ignored; in parentheses the standard deviation of F over the eleven random 80% training subsets). Delta cells: ΔP / ΔR / ΔF [95% paired CI of ΔF] and Δsd. '
                'Router: DeepSolo rows 100% to DeepSolo, TiCLS and PP-OCR 0%', hdr, blocks))
    # ---- Table 5 add-on: detection P, R, n ---- (end-to-end P/R/n is in secondlook.reruns.fast.report_main)
    # ---- Table 9 (stability) ----
    stab = {a: {k: load(a, k)[0] for k in KEYS} for a in ('old', 'unified')}
    hdr = ['Spotter', 'Total-Text', 'ICDAR2015', 'CTW1500']
    blocks = {'old': [], 'unified': [], 'delta': []}
    claims9 = {'old': {}, 'unified': {}}
    first9 = {'old': {}, 'unified': {}}
    for a in ('old', 'unified'):
        for k in KEYS:
            Z = load(a, k)[1]
            n = Z['released'].shape[0]
            M = idx_matrix(n)
            rows = [r for r in stab[a][k]['rows'] if r['subset'] != 'all']
            g = np.array([r['f1_final'] - r['f1_thr'] for r in rows])
            h = np.array([r['f1_head'] - r['f1_thr'] for r in rows])
            ci_g = [np.percentile(f1_boot(Z['complete_s%d' % s], M) - f1_boot(Z['thr_train_s%d' % s], M), 2.5) for s in range(11)]
            ci_h = [np.percentile(f1_boot(Z['first_s%d' % s], M) - f1_boot(Z['thr_train_s%d' % s], M), 2.5) for s in range(11)]
            allr = [r for r in stab[a][k]['rows'] if r['subset'] == 'all'][0]
            claims9[a][k] = dict(mean=float(g.mean()), sd=float(g.std(ddof=1)), min=float(g.min()), max=float(g.max()), n_ci_above_zero=int(sum(c > 0 for c in ci_g)), single_fit=float(allr['f1_final'] - allr['f1_thr']),
                                 k_star=[r['k_star'] for r in rows], c=[r['c'] for r in rows])
            first9[a][k] = dict(mean=float(h.mean()), sd=float(h.std(ddof=1)), min=float(h.min()), max=float(h.max()), n_ci_above_zero=int(sum(c > 0 for c in ci_h)), single_fit=float(allr['f1_head'] - allr['f1_thr']))
    for a in ('old', 'unified'):
        for sp in ('deepsolo', 'ticls', 'ppocr'):
            blocks[a].append([sp.replace('deepsolo', 'DeepSolo').replace('ticls', 'TiCLS').replace('ppocr', 'PP-OCR')] +
                             ['%s (%.2f; %.2f to %.2f)' % (sg(claims9[a]['%s_%s' % (sp, t)]['mean']), claims9[a]['%s_%s' % (sp, t)]['sd'], claims9[a]['%s_%s' % (sp, t)]['min'], claims9[a]['%s_%s' % (sp, t)]['max']) for t in ('totaltext', 'icdar2015', 'ctw1500')])
    for sp in ('deepsolo', 'ticls', 'ppocr'):
        blocks['delta'].append([sp] + ['mean %s, sd %s, min %s, max %s' % tuple(sg(claims9['unified']['%s_%s' % (sp, t)][q] - claims9['old']['%s_%s' % (sp, t)][q]) for q in ('mean', 'sd', 'min', 'max')) for t in ('totaltext', 'icdar2015', 'ctw1500')])
    data['claims9'] = claims9
    data['first9'] = first9
    out.append(('Table 9 (stability: complete method minus the re-tuned thresholds over the eleven random 80% subsets of the training images: mean (standard deviation; range), F1 points; each result evaluated once on the full test split; the whole pipeline is re-run on each subset, FAST selection for the classifier and the baseline alike). Delta: unified − old of mean, sd, min, max', hdr, blocks))
    return out, data
