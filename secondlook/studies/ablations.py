"""E39 (uniform methodology, 2026-11-07): Table 9 (signal groups), Table 10 (learners / calibration / votes / single signals), Section 6.3 (negative controls, hyper-parameters)
and Table 11 (fixed cuts) under the ONE pipeline of the study log ('Uniform pipeline'): pool = every labelled TRAIN image of the row, every arm is
fitted on the whole pool and its cut is c = round(F_cf*/2, 2) with F_cf* the maximum over the 0.01 grid of the label-based pooled F1 of the SAME arm cross-fitted
(5 image-grouped folds, the permutation of secondlook.second_stage.band_classifier.cross_fit); single-signal rules and thr_train are tuned on the whole pool. No selection split.
Exact pinned evaluator on the official TEST, evaluated once per arm. CPU only. Approximate reproduction. Counts are saved per arm and image (tp, kept, gt)."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from secondlook.data.cv_counts import counts_for
from secondlook.data.tasks import pick_thresholds, make_model
from secondlook.studies.ablation_controls import hgb, HP
from secondlook.studies.ablation_learners import build, best_signal_threshold, SIGNALS, CLASSIFIERS
from secondlook.selection.operating_point import FINE, load, pooled_f1

COLS = {
    'common11': list(range(11)), 'common8_det_rec_geo': list(range(8)), 'common5_det_rec': list(range(5)), 'no_detection': list(range(1, 11)), 'only_detection': [0],
    'no_recognition': [0, 5, 6, 7, 8, 9, 10], 'only_recognition': [1, 2, 3, 4], 'no_geometry': [0, 1, 2, 3, 4, 8, 9, 10], 'only_geometry': [5, 6, 7], 'only_transcript': [8, 9, 10],
}
FIXED = [0.30, 0.40, 0.50, 0.60]
SEED_PERM, SEED_NOISE, SEED_NOISE_TEST = 1000, 2000, 3000


def cf_predict(T, ids, X, Y, make, folds=5):
    """5-fold image-grouped cross-fitted probabilities (same permutation and fold structure as secondlook.second_stage.band_classifier.cross_fit). X, Y: per-image feature matrices / FIT labels."""
    order = [ids[i] for i in np.random.RandomState(0).permutation(len(ids))]
    p = {}
    for f in range(folds):
        te = order[f::folds]
        ts = set(te)
        tr = [s for s in ids if s not in ts]
        x = np.concatenate([X[s] for s in tr])
        y = np.concatenate([Y[s] for s in tr])
        m = y >= 0
        mod = make()
        mod.fit(x[m], y[m])
        for s in te:
            p[s] = mod.predict_proba(X[s])[:, 1] if len(X[s]) else np.zeros(0)
    return p


def cut_of(T, ids, p_cf):
    curve = [pooled_f1(T.lab, T.gt, p_cf, ids, c) for c in FINE]
    i = int(np.argmax(curve))
    return round(max(curve) / 2, 2), FINE[i], max(curve)


def fit_all(ids, X, Y, make):
    x = np.concatenate([X[s] for s in ids])
    y = np.concatenate([Y[s] for s in ids])
    m = y >= 0
    mod = make()
    mod.fit(x[m], y[m])
    return mod


def proba(mod, Xe, ids):
    return {s: (mod.predict_proba(Xe[s])[:, 1] if len(Xe[s]) else np.zeros(0)) for s in ids}


def f1c(c):
    return float(200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum()))


def job(a):
    out = Path(a.out) / ('%s_%s' % (a.expert, a.task))
    out.mkdir(parents=True, exist_ok=True)
    work = out / 'work'
    work.mkdir(exist_ok=True)
    T = load(a.expert, a.task, work)
    ids, eids = list(T.ids), list(T.eval_ids)
    cfg = getattr(T, 'cfg', None)
    counts, meta = {}, dict(expert=a.expert, task=a.task, train_images=len(ids), test_images=len(eids), arms={})

    def X_of(cols):
        return {s: T.tab[s]['xc'][:, cols] for s in ids}, {s: T.eval_tab[s]['xc'][:, cols] for s in eids}

    def record(name, keep, **kw):
        c = counts_for(T, work, {s: np.asarray(keep[s], bool) for s in eids})
        counts[name] = c
        meta['arms'][name] = dict(f1=f1c(c), **kw)
        print('E39', a.expert, a.task, name, '%.2f' % f1c(c), flush=True)

    def learner(name, cols, make, Y=None, X=None, Xe=None):
        if X is None:
            X, Xe = X_of(cols)
        Yf = Y if Y is not None else {s: T.lab[s] for s in ids}
        p_cf = cf_predict(T, ids, X, Yf, make)
        c, c_arg, fstar = cut_of(T, ids, p_cf)
        mod = fit_all(ids, X, Yf, make)
        pe = proba(mod, Xe, eids)
        record(name, {s: pe[s] >= c for s in eids}, cut=c, c_argmax=c_arg, f_cf_star=100 * fstar)
        if a.stage == 'table10_argmax':
            record(name + '@argmax', {s: pe[s] >= c_arg for s in eids}, cut=c_arg, c_argmax=c_arg, f_cf_star=100 * fstar)
        return p_cf, pe, c

    if a.stage in ('all', 'refs'):
        st, rt, f_in = pick_thresholds(T, ids)
        record('thr_train', {s: (T.eval_tab[s]['score'] >= st) & (T.eval_tab[s]['rec'] >= rt) for s in eids}, thr_score=float(st), thr_rec=float(rt), train_f1=100 * f_in)
    if a.stage in ('all', 'table9'):
        for name, cols in COLS.items():
            learner(name, cols, hgb)
    if a.stage in ('all', 'table10'):
        P = {}
        for kind in CLASSIFIERS:
            P[kind] = learner(kind, COLS['common11'], hgb if kind == 'hgb' else (lambda kind=kind: build(kind, cfg)))
        kinds = CLASSIFIERS
        pcf = {s: np.mean([P[k][0][s] for k in kinds], axis=0) for s in ids}
        pev = {s: np.mean([P[k][1][s] for k in kinds], axis=0) for s in eids}
        c, c_arg, fstar = cut_of(T, ids, pcf)
        record('vote_soft', {s: pev[s] >= c for s in eids}, cut=c, c_argmax=c_arg, f_cf_star=100 * fstar)
        record('vote_hard', {s: np.sum([P[k][1][s] >= P[k][2] for k in kinds], axis=0) >= 3 for s in eids}, cuts={k: P[k][2] for k in kinds})
        learner('lr5', COLS['common5_det_rec'], lambda: make_model('logistic', cfg))
        for name, (col, sign) in SIGNALS.items():
            thr = best_signal_threshold(T, ids, lambda t, col=col, sign=sign: sign * t['xc'][:, col])
            record(name, {s: sign * T.eval_tab[s]['xc'][:, col] >= thr for s in eids}, threshold=float(thr) if np.isfinite(thr) else None)
        prod = lambda t: (1.0 / (1.0 + np.exp(-t['xc'][:, 0]))) * t['xc'][:, 1]
        thr = best_signal_threshold(T, ids, prod)
        record('prod_det_geo', {s: prod(T.eval_tab[s]) >= thr for s in eids}, threshold=float(thr))
    if a.stage in ('all', 'controls'):
        X, Xe = X_of(COLS['common11'])
        valid = np.concatenate([T.lab[s] for s in ids])
        vals = valid[valid >= 0]
        perm = vals[np.random.RandomState(SEED_PERM).permutation(len(vals))]
        Yp, k = {}, 0
        for s in ids:
            y = T.lab[s].copy()
            m = y >= 0
            y[m] = perm[k:k + m.sum()]
            k += m.sum()
            Yp[s] = y
        learner('neg_shuffled_labels', COLS['common11'], hgb, Y=Yp, X=X, Xe=Xe)
        rn = np.random.RandomState(SEED_NOISE)
        Xn = {s: rn.randn(*X[s].shape) for s in ids}
        rt_ = np.random.RandomState(SEED_NOISE_TEST)
        Xne = {s: rt_.randn(*Xe[s].shape) for s in eids}
        learner('neg_noise_features', COLS['common11'], hgb, X=Xn, Xe=Xne)
        for tag, kw in HP.items():
            learner(tag, COLS['common11'], lambda kw=kw: hgb(**kw), X=X, Xe=Xe)
    if a.stage == 'table10_argmax':
        # post hoc sensitivity (E39 log): the same learners with the cut at the arg max of their own cross-fitted TRAIN F1 instead of F_cf*/2 (F/2 presumes calibrated probabilities)
        for kind in CLASSIFIERS:
            learner(kind, COLS['common11'], hgb if kind == 'hgb' else (lambda kind=kind: build(kind, cfg)))
        learner('lr5', COLS['common5_det_rec'], lambda: make_model('logistic', cfg))
    if a.stage in ('all', 'fixed'):
        X, Xe = X_of(COLS['common11'])
        mod = fit_all(ids, X, {s: T.lab[s] for s in ids}, hgb)
        pe = proba(mod, Xe, eids)
        for c in FIXED:
            record('fixed_%.2f' % c, {s: pe[s] >= c for s in eids}, cut=c)
    np.savez_compressed(out / ('counts_%s.npz' % a.stage), **counts)
    (out / ('meta_%s.json' % a.stage)).write_text(json.dumps(meta, indent=1))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', required=True, choices=['totaltext', 'icdar2015', 'ctw1500'])
    p.add_argument('--stage', default='all', choices=['all', 'refs', 'table9', 'table10', 'controls', 'fixed', 'table10_argmax'])
    p.add_argument('--out', required=True)
    job(p.parse_args())
