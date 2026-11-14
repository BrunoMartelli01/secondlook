"""E49 report: Doc Table 11 (feature ablation of the first step), 12 (learners and calibration, also the arg max variant), 13 (fixed cuts), and Section 4.6 controls (negative controls,
single signals, hyper-parameters), from RUN/<arm>/ablations/<row>/counts_all.npz and counts_table10_argmax.npz (secondlook.studies.ablations, unchanged, through the E49 launcher).
Every variant is one fit on the whole training pool with its own cross-fitted c = round(F_cf*/2, 2) (FAST selection); thr_train of the baseline is the same label-based rule. Counts are per image;
differences carry paired 95% intervals (2000 draws, seed 20261106, one index matrix per row, shared by both arms)."""
import numpy as np

from secondlook.reruns.fast.lib import KEYS, NAMES, RUN, f1, f1_boot, fm, idx_matrix, md_table, sg, ub

FEAT = [('common8_det_rec_geo', 'Detection, recognition and geometry (8)'), ('common5_det_rec', 'Detection and recognition (5)'), ('no_detection', 'Without detection score'), ('only_detection', 'Detection score only'),
        ('no_recognition', 'Without recognition statistics'), ('only_recognition', 'Recognition statistics only'), ('no_geometry', 'Without geometry'), ('only_geometry', 'Geometry only'), ('only_transcript', 'Transcript features only')]
LEARN = [('hgb', 'Gradient-boosted trees (ours)'), ('rf', 'Random forest'), ('mlp', 'Multilayer perceptron'), ('extratrees', 'Extremely randomised trees'), ('logistic', 'Logistic regression'),
         ('vote_soft', 'Soft vote of the above'), ('vote_hard', 'Hard vote of the above'), ('lr5', 'Logistic regression on the 5 confidence signals')]
ORDER4 = ['deepsolo_totaltext', 'deepsolo_icdar2015', 'ticls_totaltext', 'ticls_icdar2015']
ORDER_T12 = ['ticls_totaltext', 'ticls_icdar2015', 'deepsolo_totaltext', 'deepsolo_icdar2015']
OTHER5 = ['ppocr_totaltext', 'ppocr_icdar2015', 'deepsolo_ctw1500', 'ticls_ctw1500', 'ppocr_ctw1500']
CONTROLS = ['sig_logit', 'sig_geo', 'sig_minp', 'sig_entropy', 'sig_margin', 'prod_det_geo', 'neg_shuffled_labels', 'neg_noise_features', 'hp_iter30', 'hp_iter120', 'hp_iter240', 'hp_leaves3', 'hp_leaves15',
            'hp_leaves31', 'hp_l2_1', 'hp_l2_20', 'hp_minleaf10', 'hp_minleaf50', 'hp_lr005', 'hp_sklearn_default']


def load(arm, k):
    d = RUN / arm / 'ablations' / k
    A = np.load(d / 'counts_all.npz')
    G = np.load(d / 'counts_table10_argmax.npz') if (d / 'counts_table10_argmax.npz').exists() else None
    return {n: A[n].astype(np.int64) for n in A.files}, ({n: G[n].astype(np.int64) for n in G.files} if G is not None else {})


def build():
    keys = ORDER4 + OTHER5
    L = {a: {k: load(a, k) for k in keys} for a in ('old', 'unified')}
    out, data = [], {'old': {}, 'unified': {}}
    # ---- Table 11 ----
    hdr = ['Features'] + [NAMES[k] for k in keys]
    blocks = {'old': [], 'unified': [], 'delta': []}
    Ms = {k: idx_matrix(L['old'][k][0]['common11'].shape[0]) for k in keys}
    row0 = {a: ['Eleven shared signals (F1)'] + ['%.2f' % f1(L[a][k][0]['common11']) for k in keys] for a in ('old', 'unified')}
    for a in ('old', 'unified'):
        blocks[a].append(row0[a])
    blocks['delta'].append(['Eleven shared signals (F1)'] + [sg(f1(L['unified'][k][0]['common11']) - f1(L['old'][k][0]['common11'])) for k in keys])
    t11 = {'old': {}, 'unified': {}}
    for name, label in FEAT:
        rows = {}
        for a in ('old', 'unified'):
            cells = []
            for k in keys:
                X = L[a][k][0]
                v = [f1(X[name]) - f1(X['common11'])] + list(np.percentile(f1_boot(X[name], Ms[k]) - f1_boot(X['common11'], Ms[k]), [2.5, 97.5]))
                t11[a].setdefault(name, {})[k] = v
                cells.append(fm(v))
            blocks[a].append([label] + cells)
        blocks['delta'].append([label] + [sg(t11['unified'][name][k][0] - t11['old'][name][k][0]) for k in keys])
    data['t11'] = t11
    data['t11_f1'] = {a: {k: f1(L[a][k][0]['common11']) for k in keys} for a in ('old', 'unified')}
    out.append(('Table 9 (user numbering; Doc Table 11): feature ablation of the first step on the eleven shared signals. F1 of the full set and change in F1 for each subset of it, with the paired 95% interval (one fit per variant, full test split; FAST selection). '
                'The four manuscript settings are the first four columns, the other five rows follow. Delta: unified − old (change of the cell)', hdr, blocks))
    # ---- Table 12 ----
    keys12 = ORDER_T12 + OTHER5
    hdr = ['Model'] + [NAMES[k] for k in keys12]
    blocks = {'old': [], 'unified': [], 'delta': []}
    t12 = {'old': {}, 'unified': {}}
    for suffix, tag in (('', ''), ('@argmax', ' (cut at the arg max of the training F1)')):
        for name, label in LEARN:
            nm = name + suffix
            if suffix and name in ('vote_soft', 'vote_hard'):
                continue
            row = {}
            ok = True
            for a in ('old', 'unified'):
                cells = []
                for k in keys12:
                    X, G = L[a][k]
                    src = G if suffix else X
                    if nm not in src:
                        ok = False
                        break
                    v = [f1(src[nm]) - f1(X['thr_train'])] + list(np.percentile(f1_boot(src[nm], Ms[k]) - f1_boot(X['thr_train'], Ms[k]), [2.5, 97.5]))
                    t12[a].setdefault(nm, {})[k] = v
                    cells.append(fm(v))
                if not ok:
                    break
                row[a] = cells
            if not ok:
                continue
            for a in ('old', 'unified'):
                blocks[a].append([label + tag] + row[a])
            blocks['delta'].append([label + tag] + [sg(t12['unified'][nm][k][0] - t12['old'][nm][k][0]) for k in keys12])
    data['t12'] = t12
    out.append(('Table 12 (user numbering: learners and calibration, Doc Table 12; "alternative tree ensembles"): difference in F1 from thr_train with the paired 95% interval, one fit per learner on all training images, threshold half of the best cross-fitted F1 of that learner. Delta = unified − old', hdr, blocks))
    # ---- Table 13 (fixed cut) ----
    hdr = ['Spotter, data', 'Our c', 'F1 at c', '0.30', '0.40', '0.50', '0.60']
    blocks = {'old': [], 'unified': [], 'delta': []}
    t13 = {'old': {}, 'unified': {}}
    from secondlook.reruns.fast.lib import jload
    for k in KEYS:
        if k not in L['old']:
            L['old'][k], L['unified'][k] = load('old', k), load('unified', k)
        Ms.setdefault(k, idx_matrix(L['old'][k][0]['common11'].shape[0]))
    cvals = {a: {k: jload(RUN / a / 'main' / 'audit' / ('train_band_%s.json' % k))['c'] for k in KEYS} for a in ('old', 'unified')}
    for k in KEYS:
        rows = {}
        for a in ('old', 'unified'):
            X = L[a][k][0]
            ch = [f1(X['fixed_%.2f' % c]) - f1(X['common11']) for c in (0.3, 0.4, 0.5, 0.6)]
            t13[a][k] = dict(c=cvals[a][k], f1_at_c=f1(X['common11']), change=ch)
            rows[a] = [NAMES[k], '%.2f' % cvals[a][k], '%.2f' % f1(X['common11'])] + [sg(x) for x in ch]
            blocks[a].append(rows[a])
        blocks['delta'].append([NAMES[k], sg(cvals['unified'][k] - cvals['old'][k]), sg(t13['unified'][k]['f1_at_c'] - t13['old'][k]['f1_at_c'])] + [sg(t13['unified'][k]['change'][i] - t13['old'][k]['change'][i]) for i in range(4)])
    data['t13'] = t13
    out.append(('Table 13 (Doc; sensitivity to the threshold): change in first-stage F1 at a fixed threshold relative to F1 at our c (positive: the fixed threshold is better). Delta: unified − old', hdr, blocks))
    # ---- controls ----
    hdr = ['Variant (difference from thr_train)'] + [NAMES[k] for k in ORDER4]
    blocks = {'old': [], 'unified': [], 'delta': []}
    t63 = {'old': {}, 'unified': {}}
    for nm in CONTROLS:
        for a in ('old', 'unified'):
            cells = []
            for k in ORDER4:
                X = L[a][k][0]
                v = [f1(X[nm]) - f1(X['thr_train'])] + list(np.percentile(f1_boot(X[nm], Ms[k]) - f1_boot(X['thr_train'], Ms[k]), [2.5, 97.5]))
                t63[a].setdefault(nm, {})[k] = v
                cells.append(fm(v))
            blocks[a].append([nm] + cells)
        blocks['delta'].append([nm] + [sg(t63['unified'][nm][k][0] - t63['old'][nm][k][0]) for k in ORDER4])
    data['t63'] = t63
    out.append(('Section 4.6 controls (single-signal rules, product rule, negative controls, hyper-parameters): difference in F1 from thr_train with paired 95% interval. Delta: unified − old', hdr, blocks))
    return out, data
