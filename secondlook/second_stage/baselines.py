"""Reviewer checks B and C on the final method. CPU only. Pre-declared in the study log ('Response to the reviewer verdict').
mode baselines: arms that use the SAME extra recogniser without a learned band model (agreement rules) + the final method, exact evaluator, full-TEST paired bootstrap.
mode partitions: the final configuration refitted on 11 random 80% subsets of the TRAIN images (head, band model and thresholds all refitted), TEST evaluated once per subset."""
import argparse
import json
from pathlib import Path

import numpy as np

from secondlook.data.cv_counts import counts_for, test_labels, SPEC
from secondlook.data.tasks import Task, TiclsTask, pick_thresholds
from secondlook.studies.ablation_controls import hgb
from secondlook.studies.ablation_learners import proba
from secondlook.evaluation.protocol import write_json
from secondlook.second_stage.band_classifier import verifier_features, cross_fit


def make_task(a, work):
    if a.task == 'ctw1500':
        from secondlook.data.tasks_ctw1500 import CtwTask
        return CtwTask(work, a.expert)   # E29: TiCLS / PP-OCR CTW1500 rows; DeepSolo unchanged
    if a.expert == 'ppocr':
        from secondlook.data.tasks_ppocr import PpocrTask
        return PpocrTask(a.task, work)
    return (Task if a.expert == 'deepsolo' else TiclsTask)(a.task, work)


def fit_stage1(T, ids):
    x = np.concatenate([T.tab[s]['xc'] for s in ids]); y = np.concatenate([T.lab[s] for s in ids]); m = y >= 0
    r = hgb(); r.fit(x[m], y[m])
    return r


def fit_band(T, ids, vt, lo, hi):
    p_cf = cross_fit(T, list(ids))
    Xb, Yb = [], []
    for s in ids:
        sel = (p_cf[s] >= lo) & (p_cf[s] <= hi) & (T.lab[s] >= 0)
        if sel.any():
            Xb.append(np.column_stack([T.tab[s]['xc'], p_cf[s], vt[s]])[sel]); Yb.append(T.lab[s][sel])
    m = hgb(); m.fit(np.concatenate(Xb), np.concatenate(Yb))
    return m


def final_keep(T, rank, band_model, ve, cut, lo, hi):
    p_te = proba(rank, T.eval_tab, T.eval_ids, SPEC)
    keep_head, keep_final = {}, {}
    for s in T.eval_ids:
        p = p_te[s]
        inb = (p >= lo) & (p <= hi)
        keep_head[s] = p >= cut
        if inb.any():
            xe = np.column_stack([T.eval_tab[s]['xc'], p, ve[s]])
            keep_final[s] = np.where(inb, band_model.predict_proba(xe)[:, 1] >= cut, p >= cut)
        else:
            keep_final[s] = p >= cut
    return p_te, keep_head, keep_final


def f1(c):
    return 200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())


def main(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    T = make_task(a, work)
    R = Path(a.out) / 'p3'
    pq_tr = json.load(open(R / ('%s_%s_train_parseq.json' % (a.expert, a.task))))
    pq_te = json.load(open(R / ('%s_%s_test_parseq.json' % (a.expert, a.task))))
    vt, ve = verifier_features(T.tab, T.ids, pq_tr), verifier_features(T.eval_tab, T.eval_ids, pq_te)
    lo, hi, cut = 0.0, a.band_hi, a.cut
    n = len(T.eval_ids)
    M = np.random.RandomState(20261114).multinomial(n, [1.0 / n] * n, size=2000).astype(float)
    boot = lambda c: 200.0 * (M @ c[:, 0]) / (M @ (c[:, 1] + c[:, 2]))
    if a.mode == 'baselines':
        rank = fit_stage1(T, T.ids)
        band = fit_band(T, T.ids, vt, lo, hi)
        p_te, keep_head, keep_final = final_keep(T, rank, band, ve, cut, lo, hi)
        st, rt, _ = pick_thresholds(T, T.ids)
        thr = {s: (T.eval_tab[s]['score'] >= st) & (T.eval_tab[s]['rec'] >= rt) for s in T.eval_ids}
        agree = {s: ve[s][:, 0] == 1 for s in T.eval_ids}
        inband = {s: (p_te[s] >= lo) & (p_te[s] <= hi) for s in T.eval_ids}
        arms = {'thr_train': thr, 'head_ruleR': keep_head, 'final_learned_band': keep_final,
                'thr_train+agree_in_band': {s: np.where(inband[s], agree[s], thr[s]) for s in T.eval_ids},
                'head+agree_in_band': {s: np.where(inband[s], agree[s], keep_head[s]) for s in T.eval_ids},
                'agree_all': agree,
                'head AND agree_all': {s: keep_head[s] & agree[s] for s in T.eval_ids},
                'thr_train AND agree_all': {s: thr[s] & agree[s] for s in T.eval_ids}}
        counts = {k: counts_for(T, work, v) for k, v in arms.items()}
        res = {}
        for k, c in counts.items():
            d = boot(c) - boot(counts['final_learned_band'])
            res[k] = dict(f1=float(f1(c)), minus_final=float(f1(c) - f1(counts['final_learned_band'])), ci_minus_final=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                          predictions=int(c[:, 1].sum()))
        write_json(R / ('baselines_%s_%s%s.json' % (a.expert, a.task, a.tag)), res, exclusive=False)
        for k, v in res.items():
            print('B %-9s %-9s %-26s F1 %.2f (vs final %+.2f [%+.2f,%+.2f]) pred %d' % (a.expert, a.task, k, v['f1'], v['minus_final'], *v['ci_minus_final'], v['predictions']), flush=True)
    else:
        lab_te, gt_te = None, None
        author = None
        out = []
        for seed in range(11):
            rng = np.random.RandomState(100 + seed)
            ids = [T.ids[i] for i in rng.permutation(len(T.ids))[:int(round(0.8 * len(T.ids)))]]
            rank = fit_stage1(T, ids)
            band = fit_band(T, ids, vt, lo, hi)
            p_te, keep_head, keep_final = final_keep(T, rank, band, ve, cut, lo, hi)
            st, rt, _ = pick_thresholds(T, ids)
            thr = {s: (T.eval_tab[s]['score'] >= st) & (T.eval_tab[s]['rec'] >= rt) for s in T.eval_ids}
            ch, cf, ct = (counts_for(T, work, k) for k in (keep_head, keep_final, thr))
            d = boot(cf) - boot(ct)
            out.append(dict(seed=seed, train_images=len(ids), f1_thr=float(f1(ct)), f1_head=float(f1(ch)), f1_final=float(f1(cf)), final_minus_thr=float(f1(cf) - f1(ct)), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]))
            print('C %s %s seed %d thr %.2f head %.2f final %.2f final-thr %+.2f [%+.2f,%+.2f]' % (a.expert, a.task, seed, f1(ct), f1(ch), f1(cf), f1(cf) - f1(ct), *out[-1]['ci']), flush=True)
        g = np.array([o['final_minus_thr'] for o in out])
        summ = dict(rows=out, mean=float(g.mean()), sd=float(g.std(ddof=1)), min=float(g.min()), max=float(g.max()), seeds_ci_above_zero=int(sum(o['ci'][0] > 0 for o in out)))
        write_json(R / ('partitions_%s_%s%s.json' % (a.expert, a.task, a.tag)), summ, exclusive=False)
        print('C SUMMARY %s %s final-thr mean %+.2f sd %.2f min %+.2f max %+.2f; CI>0 in %d/11 subsets' % (a.expert, a.task, summ['mean'], summ['sd'], summ['min'], summ['max'], summ['seeds_ci_above_zero']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['baselines', 'partitions'], required=True)
    p.add_argument('--expert', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--cut', type=float, required=True)
    p.add_argument('--band-hi', dest='band_hi', type=float, required=True)
    p.add_argument('--work', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tag', default='')   # E31: '_trainonly' outputs next to the original files
    main(p.parse_args())
