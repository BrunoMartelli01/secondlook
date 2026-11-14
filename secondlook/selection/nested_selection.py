"""E38 (2026-11-07): is a held-out part of TRAIN needed? Pre-registered in the study log (E38). CPU only, TRAIN only: no TEST label and no TEST output is read.

Per row, 10 random image-grouped 80/20 splits of the labelled TRAIN pool (split s: RandomState(20261106 + s) permutation, first 20% held out). On the 80% ('inner')
the whole E30/E31 pipeline runs (head, F_cf*, c_half = round(F_cf*/2, 2), nested 5-fold cross-fit for k* with the cached PARSeq-S readings, thr_train); on the held-out
20% the label-based pooled F1 of stage 1 at c_half, of stage 2 at k* and of thr_train are measured, and compared with the cross-fit estimates computed on the inner images.
The nested selection is a copy of secondlook.selection.verification_band.main for an arbitrary image list (checked: on the full pool it must give the stored E31 k*)."""
import argparse
import json
from pathlib import Path

import numpy as np

from secondlook.data.tasks import pick_thresholds
from secondlook.studies.ablation_controls import hgb
from secondlook.selection.operating_point import FINE, load, pooled_f1
from secondlook.second_stage.band_classifier import cross_fit, verifier_features
from secondlook.selection.verification_band import KS, TOL, fit
from secondlook.paths import RUNS

SEED0 = 20261106


def fit_head(T, ids):
    return fit(np.concatenate([T.tab[s]['xc'] for s in ids]), np.concatenate([T.lab[s] for s in ids]))


def pred(head, T, s):
    return head.predict_proba(T.tab[s]['xc'])[:, 1] if len(T.tab[s]['lines']) else np.zeros(0)


def band_fit(T, inner, p_cf, vt, hi):
    Xb, Yb = [], []
    for s in inner:
        sel = (p_cf[s] >= 0.0) & (p_cf[s] <= hi) & (T.lab[s] >= 0)
        if sel.any():
            Xb.append(np.column_stack([T.tab[s]['xc'], p_cf[s], vt[s]])[sel])
            Yb.append(T.lab[s][sel])
    return fit(np.concatenate(Xb), np.concatenate(Yb))


def band_apply(T, band, s, p, vt, hi, c):
    inb = (p >= 0.0) & (p <= hi)
    if inb.any():
        pb = band.predict_proba(np.column_stack([T.tab[s]['xc'], p, vt[s]]))[:, 1]
        return np.where(inb, pb >= c, p >= c)
    return p >= c


def nested_select(T, ids, c, vt):
    """Copy of secondlook.selection.verification_band.main (nested 5-fold cross-fit over `ids`); returns pooled label-based F1 (percent) per k and k*."""
    order = [ids[i] for i in np.random.RandomState(0).permutation(len(ids))]
    his = [(k, 1.0 if k == 'all' else min(k * c, 1.0)) for k in KS]
    keep = {str(k): {} for k, _ in his}
    keep['stage1'] = {}
    for f in range(5):
        held = order[f::5]
        hs = set(held)
        inner = [s for s in ids if s not in hs]
        head = fit_head(T, inner)
        p_cf = cross_fit(T, inner)
        p_out = {s: pred(head, T, s) for s in held}
        for s in held:
            keep['stage1'][s] = p_out[s] >= c
        for k, hi in his:
            band = band_fit(T, inner, p_cf, vt, hi)
            for s in held:
                keep[str(k)][s] = band_apply(T, band, s, p_out[s], vt, hi, c)
    f1 = {name: 100.0 * T.approx_f1(ids, kp) for name, kp in keep.items()}
    best = max(f1[str(k)] for k, _ in his)
    k_star = next(k for k, _ in his if f1[str(k)] >= best - TOL)
    return f1, k_star, dict(his)[k_star]


def thr_crossfit(T, ids):
    """5-fold cross-fitted two-threshold rule (same permutation as the head's cross-fit): thresholds tuned on four folds, applied to the fifth; pooled label-based F1 (percent)."""
    order = [ids[i] for i in np.random.RandomState(0).permutation(len(ids))]
    keep = {}
    for f in range(5):
        held = order[f::5]
        hs = set(held)
        tr = [s for s in ids if s not in hs]
        st, rt, _ = pick_thresholds(T, tr)
        for s in held:
            keep[s] = (T.tab[s]['score'] >= st) & (T.tab[s]['rec'] >= rt)
    return 100.0 * T.approx_f1(ids, keep)


def split_job(T, vt, ids, s):
    perm = [ids[i] for i in np.random.RandomState(SEED0 + s).permutation(len(ids))]
    nh = int(round(0.2 * len(ids)))
    held, inner = perm[:nh], perm[nh:]
    p_cf = cross_fit(T, inner)
    curve = [pooled_f1(T.lab, T.gt, p_cf, inner, c) for c in FINE]
    f_star = max(curve)
    c = round(f_star / 2, 2)
    cf_at_c = 100.0 * pooled_f1(T.lab, T.gt, p_cf, inner, c)
    f1n, k_star, hi = nested_select(T, inner, c, vt)
    st, rt, f_in = pick_thresholds(T, inner)
    thr_cf = thr_crossfit(T, inner)
    head = fit_head(T, inner)
    p_h = {x: pred(head, T, x) for x in held}
    band = band_fit(T, inner, p_cf, vt, hi)
    k1 = {x: p_h[x] >= c for x in held}
    k2 = {x: band_apply(T, band, x, p_h[x], vt, hi, c) for x in held}
    kt = {x: (T.tab[x]['score'] >= st) & (T.tab[x]['rec'] >= rt) for x in held}
    h1, h2, ht = (100.0 * T.approx_f1(held, k) for k in (k1, k2, kt))
    return dict(split=s, n_inner=len(inner), n_held=len(held), c_half=c, f_cf_star=100.0 * f_star, cf_f1_at_c=cf_at_c, nested_stage1=f1n['stage1'], nested_f1_by_k={k: f1n[k] for k in f1n},
                k_star=k_star, band_hi=hi, nested_gain_at_k=f1n[str(k_star)] - f1n['stage1'], thr_score=st, thr_rec=rt, thr_insample=100.0 * f_in, thr_crossfit=thr_cf,
                held_stage1=h1, held_stage2=h2, held_thr=ht, held_gain=h2 - h1, held_images=nh,
                D1=100.0 * f_star - h1, D1b=cf_at_c - h1, D2=(f1n[str(k_star)] - f1n['stage1']) - (h2 - h1), D3=(cf_at_c - thr_cf) - (h1 - ht), D3b=(cf_at_c - 100.0 * f_in) - (h1 - ht))


def main(a):
    out = Path(a.out) / ('%s_%s' % (a.expert, a.task))
    out.mkdir(parents=True, exist_ok=True)
    work = out / 'work'
    work.mkdir(exist_ok=True)
    T = load(a.expert, a.task, work)
    ids = list(T.ids)
    pq = json.load(open(Path(a.p3) / ('%s_%s_train_parseq.json' % (a.expert, a.task))))
    vt = verifier_features(T.tab, ids, pq)
    if a.check_full:
        c = json.load(open(Path(a.audit) / ('op_audit_%s_%s.json' % (a.expert, a.task))))['c_half']
        f1, k, hi = nested_select(T, ids, c, vt)
        ref = json.load(open(Path(a.audit) / ('train_band_%s_%s.json' % (a.expert, a.task))))
        print('CHECK', a.expert, a.task, 'k*', k, 'stored', ref['k_star'], 'stage1', round(f1['stage1'], 4), round(ref['train_f1_stage1'], 4), 'MATCH' if (k == ref['k_star'] and abs(f1['stage1'] - ref['train_f1_stage1']) < 1e-9) else 'MISMATCH', flush=True)
        return
    for s in a.splits:
        f = out / ('split%d.json' % s)
        if f.exists():
            continue
        r = split_job(T, vt, ids, s)
        r.update(expert=a.expert, task=a.task, pool=len(ids))
        f.write_text(json.dumps(r, indent=1))
        print('E38', a.expert, a.task, s, 'c', r['c_half'], 'k*', r['k_star'], 'D1 %+.2f D1b %+.2f D2 %+.2f D3 %+.2f' % (r['D1'], r['D1b'], r['D2'], r['D3']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', required=True, choices=['totaltext', 'icdar2015', 'ctw1500'])
    p.add_argument('--out', required=True)
    p.add_argument('--p3', default=str(RUNS / 'candidate-banks/p3'))
    p.add_argument('--audit', default=str(RUNS / 'operating-point'))
    p.add_argument('--splits', type=int, nargs='*', default=list(range(10)))
    p.add_argument('--check-full', action='store_true', help='sanity: nested selection on the FULL pool must reproduce the stored E31 k* (no output written)')
    main(p.parse_args())
