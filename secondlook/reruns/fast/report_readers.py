"""E49 report: second-reader comparison (Doc Tables 16 and 17; user numbering Table 10), four settings, PARSeq-S*, ABINet, ViTSTR-S and the shuffled-reading control.
The readings are the cached E42 files (they read crops of the submitted polygons and do not depend on the classifier features: reused, sha256 in readers/reading_files_sha256.json);
for every reader the second stage (k*, band model, second classifier) is fitted again on the arm's own first stage (secondlook.selection.verification_band, secondlook.second_stage.band_classifier, FAST selection)."""
import numpy as np

from secondlook.reruns.fast.lib import KEYS, NAMES, ROWS4, RUN, diff, f1, f1_boot, fm, idx_matrix, jload, md_table, sg, ub

ROWS = ['deepsolo_totaltext', 'deepsolo_icdar2015', 'ticls_totaltext', 'ticls_icdar2015']
READERS = [('abinet', 'ABINet, synthetic data'), ('vitstr', 'ViTSTR-S, synthetic data'), ('shuffled', 'Control: readings of PARSeq-S* shuffled among the candidates')]
TAG = {'clean': 'clean', 'vitstr': 'vitstr', 'abinet': 'abinet', 'shuffled': 'shuffled'}


def get(arm, name, k):
    d = RUN / arm / ('main' if name == 'clean' else 'readers/%s' % name)
    C = np.load(d / 'p3' / ('counts_%s_%s.npz' % (k, TAG[name])))
    return dict(head=C['head'].astype(np.int64), ver=C['verifier'].astype(np.int64), ctrl=C['control'].astype(np.int64), res=jload(d / 'p3' / ('result_%s_%s.json' % (k, TAG[name]))), band=jload(d / 'audit' / ('train_band_%s.json' % k)))


def build():
    arms = ('old', 'unified')
    G = {a: {n: {k: get(a, n, k) for k in ROWS} for n in ['clean', 'vitstr', 'abinet', 'shuffled']} for a in arms}
    thr = {a: {k: np.load(RUN / a / 'main/tables/baselines' / ('counts_%s.npz' % k))['thr_train'].astype(np.int64) for k in ROWS} for a in arms}
    acc = {a: {k: jload(RUN / a / 'accuracy' / ('acc.%s.json' % k)) for k in ROWS} for a in arms}
    M = {k: idx_matrix(G['old']['clean'][k]['ver'].shape[0]) for k in ROWS}
    hdr = ['Second reader (training data)', 'Accuracy'] + [NAMES[k] for k in ROWS] + ['Mean']
    blocks = {'old': [], 'unified': [], 'delta': []}
    data = {'t16': {'old': {}, 'unified': {}}, 't17': {'old': {}, 'unified': {}}}
    for a in arms:
        rows = [['Two thresholds re-tuned (thr_train), F1', ''] + ['%.2f' % f1(thr[a][k]) for k in ROWS] + ['%.2f' % np.mean([f1(thr[a][k]) for k in ROWS])],
                ['First stage alone, F1', ''] + ['%.2f' % f1(G[a]['clean'][k]['head']) for k in ROWS] + ['%.2f' % np.mean([f1(G[a]['clean'][k]['head']) for k in ROWS])]]
        macc = np.mean([acc[a][k]['readers']['clean']['acc_matched'] for k in ROWS])
        rows.append(['PARSeq-S*, F1', '%.1f' % macc] + ['%.2f' % f1(G[a]['clean'][k]['ver']) for k in ROWS] + ['%.2f' % np.mean([f1(G[a]['clean'][k]['ver']) for k in ROWS])])
        for n, label in READERS:
            ds = {k: diff(G[a][n][k]['ver'], G[a]['clean'][k]['ver'], M[k]) for k in ROWS}
            accs = '–' if n == 'shuffled' else '%.1f' % np.mean([acc[a][k]['readers'][n]['acc_matched'] for k in ROWS])
            rows.append([label, accs] + [fm(ds[k]) for k in ROWS] + [sg(np.mean([ds[k][0] for k in ROWS]))])
            data['t16'][a][n] = dict(diff=ds, mean=float(np.mean([ds[k][0] for k in ROWS])), final={k: f1(G[a][n][k]['ver']) for k in ROWS})
        data['t16'][a]['clean'] = dict(final={k: f1(G[a]['clean'][k]['ver']) for k in ROWS}, thr={k: f1(thr[a][k]) for k in ROWS}, first={k: f1(G[a]['clean'][k]['head']) for k in ROWS})
        blocks[a] = rows
    # delta block: change of each cell (point) unified − old
    dl = []
    for i in range(len(blocks['old'])):
        r = [blocks['old'][i][0], '']
        for j, k in enumerate(ROWS):
            if i < 3:
                vals = [lambda a: f1(thr[a][k]), lambda a: f1(G[a]['clean'][k]['head']), lambda a: f1(G[a]['clean'][k]['ver'])][i]
                r.append(sg(vals('unified') - vals('old')))
            else:
                n = READERS[i - 3][0]
                r.append(sg(data['t16']['unified'][n]['diff'][k][0] - data['t16']['old'][n]['diff'][k][0]))
        r.append('')
        dl.append(r)
    blocks['delta'] = dl
    out = [('Table 10 (user numbering; Doc Table 16): final F1 of the complete method with each second reader, second stage re-tuned for each. Top rows: F1; other rows: difference in final F1 from the main configuration (PARSeq-S*) with the paired 95% interval (positive: better than the main configuration). Accuracy: word accuracy on matched training candidates, mean of the four settings. Delta: unified − old of the cell', hdr, blocks)]
    # ---- Table 17 ----
    hdr = ['Second reader, configuration', 'k (band edge)', 'Candidates read', 'Crops per image', 'Accuracy in band', 'Final − first stage', 'Reading features − same classifier without them']
    blocks = {'old': [], 'unified': [], 'delta': []}
    order = [('vitstr', 'ViTSTR-S'), ('abinet', 'ABINet'), ('clean', 'PARSeq-S*'), ('shuffled', 'Control (shuffled)')]
    for n, label in order:
        for k in ROWS:
            row = {}
            for a in arms:
                g = G[a][n][k]
                nimg = g['ver'].shape[0]
                ab = acc[a][k]['readers'][n]['acc_in_band'] if n != 'shuffled' else None
                r = [('%s, %s' % (label, NAMES[k])), '%s (%.2f)' % ('all' if g['band']['k_star'] == 'all' else '%.1f' % float(g['band']['k_star']), g['band']['band_hi_star']), '%.1f%%' % (100 * g['res']['band_share']), '%.1f' % (g['res']['band_candidates_test'] / nimg),
                     '–' if ab is None else '%.1f' % ab, sg(f1(g['ver']) - f1(g['head'])), sg(f1(g['ver']) - f1(g['ctrl']))]
                row[a] = r
                blocks[a].append(r)
                data['t17'][a]['%s|%s' % (n, k)] = dict(k=g['band']['k_star'], hi=g['band']['band_hi_star'], share=g['res']['band_share'], crops=g['res']['band_candidates_test'] / nimg, acc_in_band=ab,
                                                       final_minus_first=f1(g['ver']) - f1(g['head']), reading_minus_ctrl=f1(g['ver']) - f1(g['ctrl']))
            a_, b_ = G['old'][n][k], G['unified'][n][k]
            blocks['delta'].append([row['old'][0], '%s → %s' % (row['old'][1], row['unified'][1]), sg(100 * (b_['res']['band_share'] - a_['res']['band_share']), 1) + ' pts', sg((b_['res']['band_candidates_test'] - a_['res']['band_candidates_test']) / a_['ver'].shape[0], 1), '',
                                    sg((f1(b_['ver']) - f1(b_['head'])) - (f1(a_['ver']) - f1(a_['head']))), sg((f1(b_['ver']) - f1(b_['ctrl'])) - (f1(a_['ver']) - f1(a_['ctrl'])))])
    out.append(('Table 17 (Doc; second stage of each recogniser). Delta: change (unified − old)', hdr, blocks))
    return out, data
