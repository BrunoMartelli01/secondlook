"""E44 (B): cross-dataset generalisation of the post-hoc layer, one spotter at a time (approximate reproduction; CPU only; the only things fitted are the HGB heads of pipeline U; no spotter touched).
The three TRAIN pools of the spotter (TT 1,251 / IC15 1,000 / CTW 1,000, 998 PP-OCR; the full E42 pools) are concatenated with prefixed image ids; the 11 shared signals `xc`, the care labels of each
benchmark's own native evaluator and the PARSeq-S* verifier features (readers/clean/p3) are used unchanged. Fit = pipeline U verbatim (secondlook.studies.stability.pipeline without thr_train):
5-fold cross-fit -> c = round(F_cf*/2, 2) on the POOLED label-based F1; nested 5-fold -> k*; head on the whole pool; band model. Because the pinned evaluators of different benchmarks live in
different upstream source trees (module name `adet`), the work is split in four stages, one benchmark per process where an evaluator is loaded:
  dump      --expert E --task T      load the E42 task, save the pool data (xc, labels, gt counts, verifier features) and the TEST-side xc / verifier features   (evaluator loaded by the task)
  fit       --expert E               pooled fits (no evaluator): gate fits on ONE benchmark's pool, B1 leave-one-benchmark-out, B2 pooled; keep sets of first stage and complete method on every TEST
  evaluate  --expert E --task T      exact pinned evaluator on the keep sets for the TEST of T (7 columns)
  assemble  --expert E               GATE: the one-benchmark fits reproduce the stored E42 c, k* and end-to-end TEST counts bit for bit (else nothing is written); write cross_E.{npz,json}
Run through secondlook.studies.fulltrain_launch for dump / evaluate."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from secondlook.evaluation.common import CLEAN, FIELDS, counts_full, f1e, stability_path

TASKS = ['totaltext', 'icdar2015', 'ctw1500']


class Pool:
    """Interface used by secondlook.second_stage.band_classifier.cross_fit / uniform_holdout_check: tab (xc, lines), lab, gt, approx_f1 (identical to Task.approx_f1)."""

    def __init__(self):
        self.ids, self.tab, self.lab, self.gt = [], {}, {}, {}

    def add(self, task, d):
        for s in d['ids']:
            k = '%s|%s' % (task, s)
            self.ids.append(k)
            self.tab[k] = dict(xc=d['xc'][s], lines=[None] * len(d['xc'][s]))
            self.lab[k] = d['lab'][s]
            self.gt[k] = d['gt'][s]

    def approx_f1(self, ids, keep):
        tp = p = 0
        for s in ids:
            k = np.asarray(keep[s], bool)
            lab = self.lab[s]
            tp += int(((lab == 1) & k).sum())
            p += int(((lab >= 0) & k).sum())
        g = sum(self.gt[s] for s in ids)
        return 2 * tp / (p + g) if p + g else 0.0


def dump(a):
    from secondlook.selection.operating_point import load
    from secondlook.second_stage.band_classifier import verifier_features
    e, t = a.expert, a.task
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / ('work_dump_%s_%s' % (e, t))
    work.mkdir(exist_ok=True)
    T = load(e, t, work)
    R = CLEAN / 'p3'
    ids, eids = list(T.ids), list(T.eval_ids)
    vt = verifier_features(T.tab, ids, json.load(open(R / ('%s_%s_train_parseq.json' % (e, t)))))
    ve = verifier_features(T.eval_tab, eids, json.load(open(R / ('%s_%s_test_parseq.json' % (e, t)))))
    d = dict(ids=ids, xc={s: np.asarray(T.tab[s]['xc'], np.float64) for s in ids}, lab={s: np.asarray(T.lab[s]) for s in ids}, gt={s: int(T.gt[s]) for s in ids}, vt=vt,
             eval_ids=eids, eval_xc={s: np.asarray(T.eval_tab[s]['xc'], np.float64) for s in eids}, ve=ve, n_eval_lines={s: len(T.eval_tab[s]['lines']) for s in eids})
    pickle.dump(d, open(out / ('pool_%s_%s.pkl' % (e, t)), 'wb'))
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    print('E44B dump', e, t, len(ids), 'train images', len(eids), 'test images', flush=True)


def fit_pool(P, S, vt):
    from secondlook.selection.operating_point import FINE, pooled_f1
    from secondlook.second_stage.band_classifier import cross_fit
    from secondlook.selection.nested_selection import band_fit, fit_head, nested_select
    p_cf = cross_fit(P, S)
    f_star = max(pooled_f1(P.lab, P.gt, p_cf, S, x) for x in FINE)
    c = round(f_star / 2, 2)
    f1n, k_star, hi = nested_select(P, S, c, vt)
    head = fit_head(P, S)
    band = band_fit(P, S, p_cf, vt, hi)
    return dict(c=c, k_star=k_star, band_hi=float(hi), f_cf_star=100.0 * f_star, images=len(S), candidates=int(sum(len(P.lab[s]) for s in S)),
                nested_f1_by_k={k: float(v) for k, v in f1n.items()}), head, band


def keeps_for(meta, head, band, d):
    c, hi = meta['c'], meta['band_hi']
    k1, k2 = {}, {}
    for s in d['eval_ids']:
        x = d['eval_xc'][s]
        pe = head.predict_proba(x)[:, 1] if d['n_eval_lines'][s] else np.zeros(0)
        k1[s] = pe >= c
        inb = (pe >= 0.0) & (pe <= hi)
        k2[s] = np.where(inb, band.predict_proba(np.column_stack([x, pe, d['ve'][s]]))[:, 1] >= c, pe >= c) if inb.any() else pe >= c
    return k1, k2


def fit_stage(a):
    out = Path(a.out)
    D = {t: pickle.load(open(out / ('pool_%s_%s.pkl' % (a.expert, t)), 'rb')) for t in TASKS}
    P, VT = Pool(), {}
    for t in TASKS:
        P.add(t, D[t])
        for s in D[t]['ids']:
            VT['%s|%s' % (t, s)] = D[t]['vt'][s]
    pool_ids = {t: ['%s|%s' % (t, s) for s in D[t]['ids']] for t in TASKS}
    jobs = [('gate_' + t, pool_ids[t], [t]) for t in TASKS] + [('B1_heldout_' + h, [k for t in TASKS if t != h for k in pool_ids[t]], [h]) for h in TASKS] + [('B2_pooled_all', [k for t in TASKS for k in pool_ids[t]], TASKS)]
    res = dict(expert=a.expert, pools={t: len(v) for t, v in pool_ids.items()}, fits={}, keeps={})
    for name, S, targets in jobs:
        meta, head, band = fit_pool(P, S, VT)
        res['fits'][name] = meta
        for t in targets:
            res['keeps']['%s__%s' % (name, t)] = dict(zip(('first', 'complete'), keeps_for(meta, head, band, D[t])))
        print('E44B fit', a.expert, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items() if k != 'nested_f1_by_k'}, flush=True)
    pickle.dump(res, open(out / ('fits_%s.pkl' % a.expert), 'wb'))


def evaluate_stage(a):
    from secondlook.selection.operating_point import load
    e, t = a.expert, a.task
    out = Path(a.out)
    fits = pickle.load(open(out / ('fits_%s.pkl' % e), 'rb'))
    work = out / ('work_eval_%s_%s' % (e, t))
    work.mkdir(exist_ok=True)
    T = load(e, t, work)
    arrays = {}
    for key, kp in fits['keeps'].items():
        name, target = key.rsplit('__', 1)
        if target != t:
            continue
        for stage in ('first', 'complete'):
            arrays['%s__%s__%s' % (name, t, stage)] = counts_full(T, work, {s: kp[stage][s] for s in T.eval_ids})
    np.savez_compressed(out / ('eval_%s_%s.npz' % (e, t)), ids=np.array([str(s) for s in T.eval_ids]), **arrays)
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    print('E44B evaluate', e, t, len(arrays), 'count arrays', flush=True)


def assemble(a):
    e, out = a.expert, Path(a.out)
    fits = pickle.load(open(out / ('fits_%s.pkl' % e), 'rb'))
    ev = {t: np.load(out / ('eval_%s_%s.npz' % (e, t))) for t in TASKS}
    meta = dict(expert=e, pools=fits['pools'], claim='approximate reproduction', fits={}, gates={})
    counts, ok_all = {}, True
    for t in TASKS:   # gate
        st = {str(r['subset']): r for r in json.load(open(stability_path(e, t)))['rows']}['all']
        ref = np.load(CLEAN / 'p3' / ('counts_%s_%s_clean.npz' % (e, t)))
        F = fits['fits']['gate_' + t]
        c1, c2 = ev[t]['gate_%s__%s__first' % (t, t)], ev[t]['gate_%s__%s__complete' % (t, t)]
        ok = bool(abs(F['c'] - st['c']) < 1e-12 and F['k_star'] == st['k_star'] and np.array_equal(c1[:, :3], ref['head']) and np.array_equal(c2[:, :3], ref['verifier']))
        meta['gates'][t] = dict(c=F['c'], k_star=F['k_star'], stored_c=st['c'], stored_k_star=st['k_star'], f1_first=f1e(c1), f1_complete=f1e(c2), ok=ok)
        ok_all &= ok
        print('E44B gate', e, t, meta['gates'][t], flush=True)
    if not ok_all:
        (out / ('cross_%s.json' % e)).write_text(json.dumps(dict(meta, status='GATE FAILED'), indent=1))
        raise SystemExit('GATE FAILED: the pooled code does not reproduce the stored E42 rows for %s' % e)
    for name in ['B1_heldout_' + t for t in TASKS] + ['B2_pooled_all']:
        meta['fits'][name] = dict(fits['fits'][name])
        for t in (TASKS if name.startswith('B2') else [name.split('_')[-1]]):
            for stage in ('first', 'complete'):
                counts['%s__%s__%s' % (name, t, stage)] = ev[t]['%s__%s__%s' % (name, t, stage)]
            meta['fits'][name]['f1_first_' + t] = f1e(counts['%s__%s__first' % (name, t)])
            meta['fits'][name]['f1_complete_' + t] = f1e(counts['%s__%s__complete' % (name, t)])
    np.savez_compressed(out / ('cross_%s.npz' % e), fields=np.array(FIELDS), **counts)
    meta['status'] = 'COMPLETE'
    (out / ('cross_%s.json' % e)).write_text(json.dumps(meta, indent=1))
    print('E44B assembled', e, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['dump', 'fit', 'evaluate', 'assemble'])
    p.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', choices=TASKS)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    {'dump': dump, 'fit': fit_stage, 'evaluate': evaluate_stage, 'assemble': assemble}[a.stage](a)
