"""E39 (uniform methodology): stability with respect to the training sample (Section 6.9). Per row, 11 random 80% image subsets of the labelled TRAIN pool
(RandomState(100 + s), the subsets of the E31 check), and on EACH subset the WHOLE uniform pipeline is run again: head, c = round(F_cf*/2, 2) by cross-fit inside the
subset, k* by the nested cross-fit inside the subset (cached PARSeq-S readings), band model, thr_train; exact pinned evaluator on the official TEST, evaluated once per subset.
Paired image bootstrap 2000 draws, seed 20261106, ONE index matrix per row shared by all arms and subsets. A control with S = all TRAIN images must reproduce the stored
E31 `_trainonly` counts bit for bit. CPU only. Approximate reproduction."""
import argparse
import json
from pathlib import Path

import numpy as np

from secondlook.data.cv_counts import counts_for
from secondlook.data.tasks import pick_thresholds
from secondlook.selection.operating_point import FINE, load, pooled_f1
from secondlook.second_stage.band_classifier import cross_fit, verifier_features
from secondlook.selection.nested_selection import band_apply, band_fit, fit_head, nested_select, pred
from secondlook.paths import RUNS

BOOT, SEED = 2000, 20261106


def f1c(c):
    return 200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())


def pipeline(T, work, S, vt, ve):
    eids = list(T.eval_ids)
    p_cf = cross_fit(T, S)
    c = round(max(pooled_f1(T.lab, T.gt, p_cf, S, x) for x in FINE) / 2, 2)
    f1n, k_star, hi = nested_select(T, S, c, vt)
    head = fit_head(T, S)
    band = band_fit(T, S, p_cf, vt, hi)
    pe = {s: head.predict_proba(T.eval_tab[s]['xc'])[:, 1] if len(T.eval_tab[s]['lines']) else np.zeros(0) for s in eids}
    keep1 = {s: pe[s] >= c for s in eids}
    keep2 = {}
    for s in eids:
        inb = (pe[s] >= 0.0) & (pe[s] <= hi)
        if inb.any():
            xe = np.column_stack([T.eval_tab[s]['xc'], pe[s], ve[s]])
            keep2[s] = np.where(inb, band.predict_proba(xe)[:, 1] >= c, pe[s] >= c)
        else:
            keep2[s] = pe[s] >= c
    st, rt, _ = pick_thresholds(T, S)
    keept = {s: (T.eval_tab[s]['score'] >= st) & (T.eval_tab[s]['rec'] >= rt) for s in eids}
    cs = [counts_for(T, work, k) for k in (keep1, keep2, keept)]
    return dict(c=c, k_star=k_star, band_hi=hi), cs


def main(a):
    out = Path(a.out) / ('%s_%s' % (a.expert, a.task))
    out.mkdir(parents=True, exist_ok=True)
    work = out / 'work'
    work.mkdir(exist_ok=True)
    T = load(a.expert, a.task, work)
    ids = list(T.ids)
    R = Path(a.p3)
    vt = verifier_features(T.tab, ids, json.load(open(R / ('%s_%s_train_parseq.json' % (a.expert, a.task)))))
    ve = verifier_features(T.eval_tab, T.eval_ids, json.load(open(R / ('%s_%s_test_parseq.json' % (a.expert, a.task)))))
    n = len(T.eval_ids)
    M = np.random.default_rng(SEED).integers(0, n, (BOOT, n))
    boot = lambda c: 200.0 * c[M, 0].sum(1) / (c[M, 1].sum(1) + c[M, 2].sum(1))
    rows = []
    jobs = [('all', ids)] + [(s, [ids[i] for i in np.random.RandomState(100 + s).permutation(len(ids))[:int(round(0.8 * len(ids)))]]) for s in range(11)]
    for tag, S in jobs:
        meta, (c1, c2, ct) = pipeline(T, work, S, vt, ve)
        row = dict(subset=tag, images=len(S), **meta, f1_head=f1c(c1), f1_final=f1c(c2), f1_thr=f1c(ct))
        for name, ca, cb in (('final_minus_thr', c2, ct), ('head_minus_thr', c1, ct), ('final_minus_head', c2, c1)):
            d = boot(ca) - boot(cb)
            row[name] = f1c(ca) - f1c(cb)
            row['ci_' + name] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        if tag == 'all':
            ref = np.load(R / ('counts_%s_%s_%s.npz' % (a.expert, a.task, a.tag_main)))
            row['control_equals_E31'] = bool(np.array_equal(c1, ref['head']) and np.array_equal(c2, ref['verifier']))
            row['control_k_star'] = meta['k_star']
            ref_thr = np.load(R.parent / a.expert / a.task / 'extra_arms.npz')['thr_train'].astype(np.int64)
            row['control_thr_equals_E18'] = bool(np.array_equal(ct, ref_thr))
        rows.append(row)
        print('E39S', a.expert, a.task, tag, 'c %.2f k* %s final-thr %+.2f [%+.2f,%+.2f]%s' % (meta['c'], meta['k_star'], row['final_minus_thr'], *row['ci_final_minus_thr'], '' if tag != 'all' else ' control=%s thr=%s' % (row['control_equals_E31'], row['control_thr_equals_E18'])), flush=True)
    sub = [r for r in rows if r['subset'] != 'all']
    g = np.array([r['final_minus_thr'] for r in sub])
    h = np.array([r['head_minus_thr'] for r in sub])
    summ = dict(expert=a.expert, task=a.task, rows=rows, final_minus_thr_mean=float(g.mean()), final_minus_thr_sd=float(g.std(ddof=1)), final_minus_thr_min=float(g.min()), final_minus_thr_max=float(g.max()),
                head_minus_thr_mean=float(h.mean()), head_minus_thr_min=float(h.min()), head_minus_thr_max=float(h.max()), subsets_ci_above_zero=int(sum(r['ci_final_minus_thr'][0] > 0 for r in sub)),
                head_ci_above_zero=int(sum(r['ci_head_minus_thr'][0] > 0 for r in sub)), k_star_values=[r['k_star'] for r in sub], c_values=[r['c'] for r in sub])
    (out / 'stability.json').write_text(json.dumps(summ, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', required=True, choices=['totaltext', 'icdar2015', 'ctw1500'])
    p.add_argument('--out', required=True)
    p.add_argument('--p3', default=str(RUNS / 'candidate-banks/p3'))
    p.add_argument('--tag-main', default='trainonly', help='tag of the main counts file the S = all control is compared with (E36 clean run: clean)')
    main(p.parse_args())
