"""E31: TRAIN-only choice of the verification band factor k. CPU only, no TEST label or TEST output is read.

Nested cross-fit on the labelled TRAIN split: 5 outer image folds (the permutation of secondlook.second_stage.band_classifier.cross_fit). For each outer fold,
the stage-1 head and the band model are fitted on the other four folds exactly as in secondlook.second_stage.band_classifier (band training candidates defined by
a 5-fold cross-fitted stage-1 probability inside those folds) and applied to the held-out fold with cut c and band [0, min(k c, 1)]. The
pooled label-based TRAIN F1 is computed for every k of the grid; k* = the smallest k whose F1 is within TOL points of the best k
(pre-registered in the study log, E31). c is the TRAIN-only c_half of secondlook.selection.operating_point (E30).
Result class: approximate reproduction; TRAIN-only selection (in-sample spotter outputs for fine-tuned spotters, stated).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from secondlook.studies.ablation_controls import hgb
from secondlook.selection.operating_point import load
from secondlook.second_stage.band_classifier import cross_fit, verifier_features

KS = [1.2, 1.4, 1.6, 1.9, 2.2, 2.6, 3.0, 'all']
TOL = 0.10


def fit(x, y):
    m = y >= 0
    model = hgb()
    model.fit(x[m], y[m])
    return model


def main(a):
    W = Path(a.audit)
    c = json.load(open(W / ('op_audit_%s_%s.json' % (a.expert, a.task))))['c_half']
    work = Path(a.work) / ('%s_%s' % (a.expert, a.task))
    work.mkdir(parents=True, exist_ok=True)
    T = load(a.expert, a.task, work)
    ids = list(T.ids)
    if a.verifier == 'spotter':
        # E37: the other spotter as second reader (agreement features of secondlook.second_stage.band_classifier, same column order); default 'parseq' = unchanged E31 behaviour
        from secondlook.data.tasks import Task, TiclsTask
        from secondlook.studies.ablation_agreement import agreement
        (work / 'h').mkdir(exist_ok=True)
        H = (TiclsTask if a.expert == 'deepsolo' else Task)(a.task, work / 'h')
        agreement(T.tab, H.tab, ids)
        sp_order = [1, 2, 3, 0]
        vt = {s: T.tab[s]['ag'][:, sp_order] if len(T.tab[s]['ag']) else np.zeros((0, 4)) for s in ids}
    else:
        pq = json.load(open(Path(a.p3) / ('%s_%s_train_parseq.json' % (a.expert, a.task))))
        vt = verifier_features(T.tab, ids, pq)
    order = [ids[i] for i in np.random.RandomState(0).permutation(len(ids))]
    his = [(k, 1.0 if k == 'all' else min(k * c, 1.0)) for k in KS]
    keep = {str(k): {} for k, _ in his}
    keep['stage1'] = {}
    n_in = {str(k): 0 for k, _ in his}
    n_all = 0
    for f in range(5):
        held = order[f::5]
        hs = set(held)
        inner = [s for s in ids if s not in hs]
        head = fit(np.concatenate([T.tab[s]['xc'] for s in inner]), np.concatenate([T.lab[s] for s in inner]))
        p_cf = cross_fit(T, inner)
        p_out = {s: (head.predict_proba(T.tab[s]['xc'])[:, 1] if len(T.tab[s]['lines']) else np.zeros(0)) for s in held}
        for s in held:
            keep['stage1'][s] = p_out[s] >= c
            n_all += len(p_out[s])
            for k, hi in his:
                n_in[str(k)] += int((p_out[s] <= hi).sum())
        for k, hi in his:
            Xb, Yb = [], []
            for s in inner:
                sel = (p_cf[s] >= 0.0) & (p_cf[s] <= hi) & (T.lab[s] >= 0)
                if sel.any():
                    Xb.append(np.column_stack([T.tab[s]['xc'], p_cf[s], vt[s]])[sel])
                    Yb.append(T.lab[s][sel])
            band = fit(np.concatenate(Xb), np.concatenate(Yb))
            for s in held:
                p = p_out[s]
                inb = (p >= 0.0) & (p <= hi)
                if inb.any():
                    pb = band.predict_proba(np.column_stack([T.tab[s]['xc'], p, vt[s]]))[:, 1]
                    keep[str(k)][s] = np.where(inb, pb >= c, p >= c)
                else:
                    keep[str(k)][s] = p >= c
        print('fold', f, 'done', flush=True)
    f1 = {name: 100.0 * T.approx_f1(ids, kp) for name, kp in keep.items()}
    share = {str(k): n_in[str(k)] / max(n_all, 1) for k, _ in his}
    best = max(f1[str(k)] for k, _ in his)
    k_star = next(k for k, _ in his if f1[str(k)] >= best - TOL)
    hi_star = dict(his)[k_star]
    res = dict(expert=a.expert, task=a.task, c=c, tol=TOL, train_f1_stage1=f1['stage1'], train_f1_by_k={str(k): f1[str(k)] for k, _ in his},
               band_hi_by_k={str(k): hi for k, hi in his}, train_band_share_by_k=share, k_star=k_star, band_hi_star=round(hi_star, 4), train_gain_at_k_star=f1[str(k_star)] - f1['stage1'])
    (W / ('train_band_%s_%s.json' % (a.expert, a.task))).write_text(json.dumps(res, indent=1))
    print('BAND', a.expert, a.task, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k not in ('band_hi_by_k', 'train_band_share_by_k')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', required=True, choices=['totaltext', 'icdar2015', 'ctw1500'])
    p.add_argument('--audit', required=True, help='directory with op_audit_*.json (E30); outputs are written there')
    p.add_argument('--p3', required=True, help='directory with the cached PARSeq readings of the TRAIN crops')
    p.add_argument('--work', required=True)
    p.add_argument('--verifier', choices=['parseq', 'spotter'], default='parseq')
    main(p.parse_args())
