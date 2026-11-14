"""E49 report, S = all chain: Doc Table 4 (first stage), 5 (precision / recall / n), 6 (second stage), 7 (same-reading rules), 13 (fixed cut: from the ablation stage, see e49_rep_ablation),
14 (ECE and arg max), 15 (band sweep). OLD and UNIFIED side by side; paired 95% image bootstrap (2000 draws, seed 20261106, one index matrix per row for both arms).
Selection protocol of every number here: FAST (label-based F1) for c, k* and the baseline thresholds alike."""
import numpy as np

from secondlook.reruns.fast.lib import (E18, KEYS, NAMES, ROWS9, RUN, diff, f1, f1_boot, fm, idx_matrix, jload, md_table, prf, router_text, sg, ub)

SWK = ['1.2', '1.4', '1.6', '1.9', '2.2', '2.6', '3.0', 'all']
SHOW_K = ['1.2', '1.6', '1.9', '2.2', '2.6', 'all']
RULES = ['thr_train+agree_in_band', 'head+agree_in_band', 'agree_all', 'head_AND_agree_all', 'thr_train_AND_agree_all']
RULE_HDR = ['thr_train, agree in band', 'Ours step 1, agree in band', 'agree, all words', 'Ours step 1 and agree, all words', 'thr_train and agree, all words']


def load_arm(arm, e, t):
    k = '%s_%s' % (e, t)
    d = RUN / arm / 'main'
    A = np.load(d / 'p3' / ('counts_%s_clean.npz' % k))
    B = np.load(d / 'tables/baselines' / ('counts_%s.npz' % k))
    X = dict(released=np.load(E18 / e / t / 'grid.npz')['author'].astype(np.int64), thr=B['thr_train'].astype(np.int64), s1=A['head'].astype(np.int64), s2=A['verifier'].astype(np.int64),
             ctrl=A['control'].astype(np.int64), s1A=np.load(d / 'p3' / ('counts_%s_targmax.npz' % k))['head'].astype(np.int64), s2A=np.load(d / 'p3' / ('counts_%s_targmax.npz' % k))['verifier'].astype(np.int64))
    X['rules'] = {r: B[r].astype(np.int64) for r in RULES}
    X['final_baselines'] = B['final'].astype(np.int64)
    sw = {}
    for kk in SWK:
        S = np.load(d / 'p3' / ('counts_%s_sw%s.npz' % (k, kk)))
        sw[kk] = dict(head=S['head'].astype(np.int64), ver=S['verifier'].astype(np.int64), res=jload(d / 'p3' / ('result_%s_sw%s.json' % (k, kk))))
    return dict(c=X, sw=sw, res=jload(d / 'p3' / ('result_%s_clean.json' % k)), band=jload(d / 'audit' / ('train_band_%s.json' % k)), aud=jload(d / 'audit' / ('op_audit_%s.json' % k)), n=X['released'].shape[0])


def kfmt(b):
    k = b['k_star']
    return '%s (%.2f)' % ('all' if k == 'all' else '%.1f' % float(k), b['band_hi_star'])


def build():
    D = {k: {a: load_arm(a, *k.split('_')) for a in ('old', 'unified')} for k in KEYS}
    out, data = [], {}
    # ---------------- Table 4 (first stage) ----------------
    hdr = ['Spotter, data', 'c', 'Released', 'thr_train', 'thr_train − released', 'Ours, step 1', 'Ours, step 1 − released', 'Ours, step 1 − thr_train']
    blocks = {'old': [], 'unified': [], 'delta': []}
    for k in KEYS:
        n = D[k]['old']['n']
        M = idx_matrix(n)
        row = {}
        for a in ('old', 'unified'):
            X = D[k][a]['c']
            row[a] = [NAMES[k], '%.2f' % D[k][a]['band']['c'], '%.2f' % f1(X['released']), '%.2f' % f1(X['thr']), fm(diff(X['thr'], X['released'], M)), '%.2f' % f1(X['s1']), fm(diff(X['s1'], X['released'], M)), fm(diff(X['s1'], X['thr'], M))]
            blocks[a].append(row[a])
        o, u = D[k]['old']['c'], D[k]['unified']['c']
        blocks['delta'].append([NAMES[k], sg(D[k]['unified']['band']['c'] - D[k]['old']['band']['c']), '+0.00 (same)', sg(f1(u['thr']) - f1(o['thr'])) + ' ' + fm(diff(u['thr'], o['thr'], M))[5:], '', sg(f1(u['s1']) - f1(o['s1'])) + ' ' + fm(diff(u['s1'], o['s1'], M))[5:],
                                '', 'old %s → unified %s' % (fm(diff(o['s1'], o['thr'], M)), fm(diff(u['s1'], u['thr'], M)))])
        data.setdefault(k, {})['t4'] = dict(old=row['old'], unified=row['unified'])
    out.append(('Table 4 (first stage; "Ours, step 1"): end-to-end F1 on the full test split. Delta block: unified − old, `diff [95% paired CI]` of the F1 (thr_train: unified uses rec = geometric mean)', hdr, blocks))
    # ---------------- Table 5 (P, R, n) ----------------
    hdr = ['Spotter, data (GT)', 'Released P / R (words)', 'thr_train P / R (words)', 'Ours, step 1 P / R (words)', 'Ours, full P / R (words)']
    blocks = {'old': [], 'unified': [], 'delta': []}
    for k in KEYS:
        cells = {}
        for a in ('old', 'unified'):
            X = D[k][a]['c']
            f = lambda c: '%.2f / %.2f (%s)' % (prf(c)[0], prf(c)[1], format(prf(c)[3], ','))
            gt = prf(X['released'])[4]
            cells[a] = ['%s (%s)' % (NAMES[k], format(gt, ','))] + [f(X[m]) for m in ('released', 'thr', 's1', 's2')]
            blocks[a].append(cells[a])
        o, u = D[k]['old']['c'], D[k]['unified']['c']
        dl = lambda m: '%s / %s (%+d)' % (sg(prf(u[m])[0] - prf(o[m])[0]), sg(prf(u[m])[1] - prf(o[m])[1]), prf(u[m])[3] - prf(o[m])[3])
        blocks['delta'].append([NAMES[k]] + [dl(m) for m in ('released', 'thr', 's1', 's2')])
        data[k]['t5'] = dict(old=cells['old'], unified=cells['unified'])
    out.append(('Table 5 (precision / recall in percent and number of output words, full test split). Router: DeepSolo rows 100% to DeepSolo, TiCLS and PP-OCR rows 0% (stored-output decisions). Delta = unified − old (P, R in points; words as signed count)', hdr, blocks))
    # ---------------- Table 6 (second stage) ----------------
    hdr = ['Spotter, data', 'k (band edge)', 'Band', 'Crops per image', 'AUC in band: step 1 / full (without reading)', 'Ours, step 1', 'Ours, full', 'Gain of the second stage', 'Ours, full − thr_train']
    blocks = {'old': [], 'unified': [], 'delta': []}
    claims6 = {}
    for k in KEYS:
        n = D[k]['old']['n']
        M = idx_matrix(n)
        cells = {}
        for a in ('old', 'unified'):
            X, r, b = D[k][a]['c'], D[k][a]['res'], D[k][a]['band']
            cells[a] = [NAMES[k], kfmt(b), '%.1f%%' % (100 * r['band_share']), '%.1f' % (r['band_candidates_test'] / n), '%.3f / %.3f (%.3f)' % (r['auc_ranker'], r['auc_band_with_verifier'], r['auc_band_control_no_verifier']),
                        '%.2f' % f1(X['s1']), '%.2f' % f1(X['s2']), fm(diff(X['s2'], X['s1'], M)), fm(diff(X['s2'], X['thr'], M))]
            blocks[a].append(cells[a])
        o, u = D[k]['old'], D[k]['unified']
        ro, ru = o['res'], u['res']
        dd = lambda ca, cb, cc, cd: (lambda d: [f1(ca) - f1(cb) - (f1(cc) - f1(cd))] + [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])((f1_boot(ca, M) - f1_boot(cb, M)) - (f1_boot(cc, M) - f1_boot(cd, M)))
        blocks['delta'].append([NAMES[k], '%s → %s' % (kfmt(o['band']), kfmt(u['band'])), sg(100 * (ru['band_share'] - ro['band_share']), 1) + ' pts', sg((ru['band_candidates_test'] - ro['band_candidates_test']) / n, 1),
                                '%s / %s (%s)' % (sg(ru['auc_ranker'] - ro['auc_ranker'], 3), sg(ru['auc_band_with_verifier'] - ro['auc_band_with_verifier'], 3), sg(ru['auc_band_control_no_verifier'] - ro['auc_band_control_no_verifier'], 3)),
                                sg(f1(u['c']['s1']) - f1(o['c']['s1'])), sg(f1(u['c']['s2']) - f1(o['c']['s2'])), fm(dd(u['c']['s2'], u['c']['s1'], o['c']['s2'], o['c']['s1'])), fm(dd(u['c']['s2'], u['c']['thr'], o['c']['s2'], o['c']['thr']))])
        data[k]['t6'] = dict(old=cells['old'], unified=cells['unified'])
        for a in ('old', 'unified'):
            X, r = D[k][a]['c'], D[k][a]['res']
            g = diff(X['s2'], X['s1'], M); gt_ = diff(X['s2'], X['thr'], M); gc = diff(X['ctrl'], X['s1'], M); gr = diff(X['s2'], X['ctrl'], M)
            claims6.setdefault(a, {})[k] = dict(gain=g, final_minus_thr=gt_, ctrl_minus_s1=gc, reading_over_ctrl=gr, auc_gain_vs_ctrl=r['auc_gain_vs_control'], agree_auc_test=r['auc_verifier_agree_alone'], agree_auc_train=r['auc_verifier_agree_alone_train'],
                                                final_minus_released=diff(X['s2'], X['released'], M), share=r['band_share'], crops=r['band_candidates_test'] / n, k=D[k][a]['band']['k_star'])
    data['claims6'] = claims6
    out.append(('Table 6 (second stage with PARSeq-S*; band factor k chosen by the nested five-fold cross-fit on the training split). Delta: unified − old (AUC differences, F1 differences with paired CI; the "gain" delta is the difference of the two gains)', hdr, blocks))
    # ---------------- Table 7 (rules with the same reading) ----------------
    hdr = ['Spotter, data'] + RULE_HDR
    blocks = {'old': [], 'unified': [], 'delta': []}
    claims7 = {'old': {}, 'unified': {}}
    for k in KEYS:
        n = D[k]['old']['n']
        M = idx_matrix(n)
        cells = {}
        for a in ('old', 'unified'):
            X = D[k][a]['c']
            vals = [diff(X['rules'][r], X['s2'], M) for r in RULES]
            cells[a] = [NAMES[k]] + [fm(v) for v in vals]
            blocks[a].append(cells[a])
            claims7[a][k] = dict(zip(RULES, vals))
        o, u = D[k]['old']['c'], D[k]['unified']['c']
        dl = []
        for r in RULES:
            d = (f1_boot(u['rules'][r], M) - f1_boot(u['s2'], M)) - (f1_boot(o['rules'][r], M) - f1_boot(o['s2'], M))
            dl.append(fm([claims7['unified'][k][r][0] - claims7['old'][k][r][0], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]))
        blocks['delta'].append([NAMES[k]] + dl)
        data[k]['t7'] = dict(old=cells['old'], unified=cells['unified'])
    data['claims7'] = claims7
    out.append(('Table 7 (rules with the same second reading; F1 difference from the complete method, 95% intervals; "in band" uses the band of Table 6). Delta: change of the difference (unified − old) with paired CI', hdr, blocks))
    # ---------------- Table 14 (ECE, F/2 versus argmax) ----------------
    hdr = ['Spotter, data', 'ECE (train)', 'c', 'Arg max: c, F1 − main']
    blocks = {'old': [], 'unified': [], 'delta': []}
    claims14 = {'old': {}, 'unified': {}}
    for k in KEYS:
        n = D[k]['old']['n']
        M = idx_matrix(n)
        cells = {}
        for a in ('old', 'unified'):
            X, au, b = D[k][a]['c'], D[k][a]['aud'], D[k][a]['band']
            dm = diff(X['s2A'], X['s2'], M)
            cells[a] = [NAMES[k], '%.3f' % au['calibration_train_cf']['ece'], '%.2f' % b['c'], '%.2f, %s' % (au['c_argmax'], fm(dm))]
            blocks[a].append(cells[a])
            claims14[a][k] = dict(ece=au['calibration_train_cf']['ece'], c=b['c'], c_argmax=au['c_argmax'], argmax_minus_main=dm)
        o, u = claims14['old'][k], claims14['unified'][k]
        blocks['delta'].append([NAMES[k], sg(u['ece'] - o['ece'], 3), sg(u['c'] - o['c']), '%s, %s' % (sg(u['c_argmax'] - o['c_argmax']), sg(u['argmax_minus_main'][0] - o['argmax_minus_main'][0]))])
        data[k]['t14'] = dict(old=cells['old'], unified=cells['unified'])
    data['claims14'] = claims14
    out.append(('Table 14 (calibration and the arg max variant of the threshold). ECE of the cross-fitted training candidates; arg max: threshold at the maximum of the training F1 and F1 of the complete method minus the main configuration', hdr, blocks))
    # ---------------- Table 15 (band sweep) ----------------
    hdr = ['Spotter, data (chosen k)'] + ['k = %s' % x if x != 'all' else 'All candidates' for x in SHOW_K]
    blocks = {'old': [], 'unified': [], 'delta': []}
    sweep = {'old': {}, 'unified': {}}
    for k in KEYS:
        n = D[k]['old']['n']
        M = idx_matrix(n)
        cells = {}
        for a in ('old', 'unified'):
            b = D[k][a]['band']
            ch = 'all' if b['k_star'] == 'all' else '%.1f' % float(b['k_star'])
            row = [NAMES[k] + ' (%s)' % ch]
            sweep[a][k] = {}
            for kk in SWK:
                S = D[k][a]['sw'][kk]
                g = diff(S['ver'], S['head'], M)
                sweep[a][k][kk] = dict(share=S['res']['band_share'], crops=S['res']['band_candidates_test'] / n, gain=g)
                if kk in SHOW_K:
                    cell = '%.1f%%, %s' % (100 * S['res']['band_share'], fm(g))
                    row.append('**%s**' % cell if kk == ch else cell)
            cells[a] = row
            blocks[a].append(row)
        blocks['delta'].append([NAMES[k]] + ['%s pts, %s' % (sg(100 * (sweep['unified'][k][kk]['share'] - sweep['old'][k][kk]['share']), 1), sg(sweep['unified'][k][kk]['gain'][0] - sweep['old'][k][kk]['gain'][0])) for kk in SHOW_K])
        data[k]['t15'] = dict(old=cells['old'], unified=cells['unified'])
    data['sweep'] = sweep
    out.append(('Table 15 / Fig. 6 (sensitivity to the band factor k; share of test candidates read again and gain over the first stage at c). Bold: value chosen on the training split. Delta: share (points) and gain (F1 points) unified − old', hdr, blocks))
    return out, data


def render(out):
    md = []
    for title, hdr, blocks in out:
        md.append('#### ' + title + '\n')
        for name, label in (('old', 'OLD features (reproduced with the same code; gate against the stored manuscript files passed)'), ('unified', 'UNIFIED features'), ('delta', 'DIFFERENCE (unified − old)')):
            md.append('**%s**\n\n%s\n' % (label, md_table(hdr, blocks[name])))
    return '\n'.join(md)
