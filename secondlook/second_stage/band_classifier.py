"""P3 stage 3 (legacy env, CPU): does independent PARSeq-S verification of the uncertain band help? Pre-registered in the study log ('P3 go/no-go').
Band = ranker probability in [0.20, 0.60] (TRAIN: 5-fold image-grouped cross-fitted probability; TEST: ranker fitted on all TRAIN). Band model: HGB on TRAIN band
candidates with common11 + the cross-fitted ranker probability + verifier features (agreement, edit similarity, PARSeq sequence/min-token confidence, length difference).
Outside the band the E18 rule is kept (admit iff p >= 0.42). Exact pinned-evaluator TEST F1 vs own042, paired image bootstrap."""
import argparse
import json
from pathlib import Path

import numpy as np

from secondlook.data.cv_counts import counts_for, test_labels, SPEC
from secondlook.data.tasks import Task, TiclsTask
from secondlook.studies.ablation_agreement import lev
from secondlook.studies.ablation_controls import hgb
from secondlook.studies.ablation_learners import proba
from secondlook.evaluation.protocol import write_json

BAND = (0.20, 0.60)
CUT = 0.42
CLASSES = None


def norm(t):
    return ''.join(ch for ch in t.lstrip('#').lower() if ch.isalnum())


def own_text(line):
    toks = line.rstrip('\n').split(',')
    k = 0
    for t in toks:
        try:
            int(float(t))
            k += 1
        except ValueError:
            break
    k -= k % 2
    return norm(','.join(toks[k:]))


def verifier_features(tab, ids, pq):
    idx = {(o, c): r for o, c, r in zip(pq['owner'], pq['cidx'], pq['rows'])}
    feats = {}
    for s in ids:
        rows = []
        for i, line in enumerate(tab[s]['lines']):
            r = idx[(s, i)]
            a, b = own_text(line), norm(r['text'])
            sim = 1.0 - lev(a, b) / max(1, max(len(a), len(b)))
            rows.append([float(a == b), sim, r['seq'], r['mint'] if r['mint'] is not None else 0.0, r['meanlp'] if r['meanlp'] is not None else -10.0, float(abs(len(a) - len(b)))])
        feats[s] = np.asarray(rows, float).reshape(-1, 6)
    return feats


def cross_fit(T, ids, folds=5, seed=0):
    rng = np.random.RandomState(seed)
    order = [ids[i] for i in rng.permutation(len(ids))]
    p = {}
    for f in range(folds):
        te = order[f::folds]
        tr = [s for s in ids if s not in set(te)]
        x = np.concatenate([T.tab[s]['xc'] for s in tr]); y = np.concatenate([T.lab[s] for s in tr]); m = y >= 0
        mod = hgb(); mod.fit(x[m], y[m])
        for s in te:
            p[s] = mod.predict_proba(T.tab[s]['xc'])[:, 1] if len(T.tab[s]['xc']) else np.zeros(0)
    return p


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    m = y >= 0
    return float(roc_auc_score(y[m], p[m])) if m.sum() > 10 and len(set(y[m].tolist())) > 1 else float('nan')


def main(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    global BAND, CUT
    BAND, CUT = (a.band_lo, a.band_hi), a.cut
    if a.task == 'ctw1500':
        from secondlook.data.tasks_ctw1500 import CtwTask
        T = CtwTask(work, a.expert)
    elif a.expert == 'ppocr':
        from secondlook.data.tasks_ppocr import PpocrTask
        T = PpocrTask(a.task, work)
    else:
        T = (Task if a.expert == 'deepsolo' else TiclsTask)(a.task, work)
    R = Path(a.out) / 'p3'
    if a.verifier == 'spotter':
        from secondlook.studies.ablation_agreement import agreement
        (work / 'h').mkdir(exist_ok=True)
        H = (TiclsTask if a.expert == 'deepsolo' else Task)(a.task, work / 'h')
        agreement(T.tab, H.tab, T.ids)
        agreement(T.eval_tab, H.eval_tab, T.eval_ids)
        order = [1, 2, 3, 0]   # text_equal, edit similarity, other detector score, IoU (text_equal first: used as 'agreement alone')
        vt = {s: T.tab[s]['ag'][:, order] if len(T.tab[s]['ag']) else np.zeros((0, 4)) for s in T.ids}
        ve = {s: T.eval_tab[s]['ag'][:, order] if len(T.eval_tab[s]['ag']) else np.zeros((0, 4)) for s in T.eval_ids}
    else:
        pq_tr = json.load(open(R / ('%s_%s_train_parseq.json' % (a.expert, a.task))))
        pq_te = json.load(open(R / ('%s_%s_test_parseq.json' % (a.expert, a.task))))
        vt = verifier_features(T.tab, T.ids, pq_tr)
        ve = verifier_features(T.eval_tab, T.eval_ids, pq_te)
    nv = vt[T.ids[0]].shape[1]
    lab_te, gt_te = test_labels(T, work)
    p_cf = cross_fit(T, list(T.ids))
    x = np.concatenate([T.tab[s]['xc'] for s in T.ids]); y = np.concatenate([T.lab[s] for s in T.ids]); m = y >= 0
    rank = hgb(); rank.fit(x[m], y[m])
    p_te = proba(rank, T.eval_tab, T.eval_ids, SPEC)
    # TRAIN band candidates
    Xb, Yb = [], []
    for s in T.ids:
        sel = (p_cf[s] >= BAND[0]) & (p_cf[s] <= BAND[1]) & (T.lab[s] >= 0)
        if sel.any():
            Xb.append(np.column_stack([T.tab[s]['xc'], p_cf[s], vt[s]])[sel]); Yb.append(T.lab[s][sel])
    Xb, Yb = np.concatenate(Xb), np.concatenate(Yb)
    band_model = hgb(); band_model.fit(Xb, Yb)
    base_model = hgb(); base_model.fit(Xb[:, :12], Yb)   # same band rows without verifier features (control for re-fitting inside the band)
    keep_base, keep_ver, keep_ctrl = {}, {}, {}
    yb_list, pr_list, pb_list, pc_list, ag_list = [], [], [], [], []
    for s in T.eval_ids:
        p = p_te[s]
        inb = (p >= BAND[0]) & (p <= BAND[1])
        xe = np.column_stack([T.eval_tab[s]['xc'], p, ve[s]]) if len(p) else np.zeros((0, 12 + nv))
        pb = band_model.predict_proba(xe)[:, 1] if inb.any() else np.zeros(len(p))
        pc = base_model.predict_proba(xe[:, :12])[:, 1] if inb.any() else np.zeros(len(p))
        keep_base[s] = p >= CUT
        keep_ver[s] = np.where(inb, pb >= CUT, p >= CUT)
        keep_ctrl[s] = np.where(inb, pc >= CUT, p >= CUT)
        sel = inb & (lab_te[s] >= 0)
        yb_list.append(lab_te[s][sel]); pr_list.append(p[sel]); pb_list.append(pb[sel]); pc_list.append(pc[sel]); ag_list.append(ve[s][sel, 0])
    yb = np.concatenate(yb_list)
    res = dict(expert=a.expert, task=a.task, band_candidates_test=int(len(yb)), band_share=float(len(yb) / max(1, sum(len(p_te[s]) for s in T.eval_ids))), band_positive_rate=float(yb.mean()),
               auc_ranker=auc(yb, np.concatenate(pr_list)), auc_band_control_no_verifier=auc(yb, np.concatenate(pc_list)), auc_band_with_verifier=auc(yb, np.concatenate(pb_list)), auc_verifier_agree_alone=auc(yb, np.concatenate(ag_list)),
               train_band_candidates=int(len(Yb)))
    res['auc_gain_vs_ranker'] = res['auc_band_with_verifier'] - res['auc_ranker']
    res['auc_gain_vs_control'] = res['auc_band_with_verifier'] - res['auc_band_control_no_verifier']
    # TRAIN-side AUC of the verifier agreement in the band (cross-fit not needed: single feature)
    res['auc_verifier_agree_alone_train'] = auc(Yb, Xb[:, 12])
    c_base, c_ver, c_ctrl = (counts_for(T, work, k) for k in (keep_base, keep_ver, keep_ctrl))
    f = lambda c: 200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())
    n = len(c_base)
    M = np.random.RandomState(20261113).multinomial(n, [1.0 / n] * n, size=2000).astype(float)
    boot = lambda c: 200.0 * (M @ c[:, 0]) / (M @ (c[:, 1] + c[:, 2]))
    for name, c in (('verifier', c_ver), ('band_control', c_ctrl)):
        d = boot(c) - boot(c_base)
        res['f1_%s' % name] = float(f(c)); res['f1_%s_minus_own042' % name] = float(f(c) - f(c_base)); res['ci_%s' % name] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    d = boot(c_ver) - boot(c_ctrl)
    res['f1_verifier_minus_control'] = float(f(c_ver) - f(c_ctrl)); res['ci_verifier_minus_control'] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    res['f1_own042'] = float(f(c_base))
    # final method (head + verification) against the equal-information baseline (thr_train) and the author default, full-TEST paired bootstrap
    d = Path(a.out) / a.expert / a.task
    ex, gr = np.load(d / 'extra_arms.npz'), np.load(d / 'grid.npz')
    ref = {'thr_train': ex['thr_train'].astype(np.int64), 'author': gr['author'].astype(np.int64)}
    for name, c in ref.items():
        dd = boot(c_ver) - boot(c)
        res['final_minus_' + name] = float(f(c_ver) - f(c)); res['ci_final_minus_' + name] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
        res['f1_' + name] = float(f(c))
    np.savez_compressed(Path(a.out) / 'p3' / ('counts_%s_%s%s.npz' % (a.expert, a.task, a.tag)), head=c_base, verifier=c_ver, control=c_ctrl)
    write_json(Path(a.out) / 'p3' / ('result_%s_%s%s.json' % (a.expert, a.task, a.tag)), res, exclusive=False)
    print('P3', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--expert', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--work', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cut', type=float, default=0.42)
    p.add_argument('--band-lo', dest='band_lo', type=float, default=0.20)
    p.add_argument('--band-hi', dest='band_hi', type=float, default=0.60)
    p.add_argument('--tag', default='')
    p.add_argument('--verifier', choices=['parseq', 'spotter'], default='parseq')
    main(p.parse_args())
