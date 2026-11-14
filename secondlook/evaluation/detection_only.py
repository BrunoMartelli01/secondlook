"""E44 (A): detection-only metrics of the released setting, thr_train, the first stage and the complete method on one row (approximate reproduction; CPU only; nothing trained but the
post-hoc HGB heads, as in E42; no spotter inference). Run through secondlook.studies.fulltrain_launch so that the full-TRAIN-pool tasks of E42 are installed:
    python -m secondlook.studies.fulltrain_launch secondlook.evaluation.detection_only --expert E --task T --out DIR
Per row: pool S = all TRAIN ids and the eleven random 80% subsets (RandomState(100+s), the subsets of secondlook.studies.stability). For each S the stored (c, k*, band_hi) of the E42/E41
stability.json are re-used (the nested selection is not recomputed), the head / band model / thr_train are re-fitted by the code of secondlook.studies.stability.pipeline, and the three methods
are scored on the official TEST with all 7 evaluator columns. GATES: the end-to-end F1 of each method on each S equals the stored stability.json value (1e-9), the S = all end-to-end columns equal the
stored E42 counts bit for bit, and the released arm equals the stored grid.npz 'author' arm. A failed gate stops the row (no number written)."""
import argparse
import json
from pathlib import Path

import numpy as np

from secondlook.data.tasks import pick_thresholds
from secondlook.evaluation.common import (CLEAN, E18, FIELDS, RUNS, author_keep, counts_full, f1e, stability_path, thr_train_counts_path)
from secondlook.selection.operating_point import FINE, load, pooled_f1
from secondlook.second_stage.band_classifier import cross_fit, verifier_features
from secondlook.selection.nested_selection import band_fit, fit_head, pred


def fit_apply(T, work, S, vt, ve, c, hi):
    """uniform_stability.pipeline with c and the band edge given (taken from the stored stability.json); returns 7-column counts of first stage, complete method, thr_train."""
    eids = list(T.eval_ids)
    p_cf = cross_fit(T, S)
    c_re = round(max(pooled_f1(T.lab, T.gt, p_cf, S, x) for x in FINE) / 2, 2)
    if abs(c_re - c) > 1e-12:
        raise SystemExit('GATE FAILED: c recomputed %.2f != stored %.2f' % (c_re, c))
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
    return [counts_full(T, work, k) for k in (keep1, keep2, keept)], dict(thr_score=float(st), thr_rec=float(rt))


def main(a):
    e, t = a.expert, a.task
    out = Path(a.out) / ('%s_%s' % (e, t))
    out.mkdir(parents=True, exist_ok=True)
    work = out / 'work'
    work.mkdir(exist_ok=True)
    T = load(e, t, work)
    ids = list(T.ids)
    R = CLEAN / 'p3'
    vt = verifier_features(T.tab, ids, json.load(open(R / ('%s_%s_train_parseq.json' % (e, t)))))
    ve = verifier_features(T.eval_tab, T.eval_ids, json.load(open(R / ('%s_%s_test_parseq.json' % (e, t)))))
    stab = json.load(open(stability_path(e, t)))
    st_rows = {str(r['subset']): r for r in stab['rows']}
    # released arm + gate
    rel = counts_full(T, work, author_keep(T, e, t))
    g_author = np.load(E18 / e / t / 'grid.npz')['author'].astype(np.int64)
    if not np.array_equal(rel[:, :3], g_author):
        raise SystemExit('GATE FAILED: released arm differs from the stored grid.npz author arm for %s %s' % (e, t))
    jobs = [('all', ids)] + [(str(s), [ids[i] for i in np.random.RandomState(100 + s).permutation(len(ids))[:int(round(0.8 * len(ids)))]]) for s in range(11)]
    res, arrays = [], {'released': rel}
    for tag, S in jobs:
        sr = st_rows[tag]
        if sr['images'] != len(S):
            raise SystemExit('GATE FAILED: subset size %s != stored %s' % (len(S), sr['images']))
        (c1, c2, ct), thr = fit_apply(T, work, S, vt, ve, sr['c'], sr['band_hi'])
        for name, cc, key in (('first', c1, 'f1_head'), ('complete', c2, 'f1_final'), ('thr_train', ct, 'f1_thr')):
            if abs(f1e(cc) - sr[key]) > 1e-9:
                raise SystemExit('GATE FAILED %s %s subset %s: e2e F1 %.10f != stored %.10f' % (e, t, tag, f1e(cc), sr[key]))
        if tag == 'all':
            ref = np.load(CLEAN / 'p3' / ('counts_%s_%s_clean.npz' % (e, t)))
            bl = np.load(thr_train_counts_path(e, t))['thr_train'] if t != 'ctw1500' else np.load(E18 / e / t / 'extra_arms.npz')['thr_train']
            ok = np.array_equal(c1[:, :3], ref['head']) and np.array_equal(c2[:, :3], ref['verifier']) and np.array_equal(ct[:, :3], np.asarray(bl, np.int64))
            if not ok:
                raise SystemExit('GATE FAILED: S = all end-to-end columns differ from the stored E42 counts for %s %s' % (e, t))
            arrays.update(first=c1, complete=c2, thr_train=ct)
            res.append(dict(subset='all', gate_equal_stored_counts=True, **thr))
        else:
            arrays['first_s%s' % tag], arrays['complete_s%s' % tag], arrays['thr_train_s%s' % tag] = c1, c2, ct
            res.append(dict(subset=tag, **thr))
        print('E44A', e, t, tag, 'gates ok; e2e F1 first %.2f complete %.2f thr %.2f' % (f1e(c1), f1e(c2), f1e(ct)), flush=True)
    np.savez_compressed(out / 'det_counts.npz', fields=np.array(FIELDS), ids=np.array([str(s) for s in T.eval_ids]), **arrays)
    (out / 'det_meta.json').write_text(json.dumps(dict(expert=e, task=t, train_images=len(ids), test_images=len(T.eval_ids), subsets=res, gates='passed',
                                                       claim='approximate reproduction'), indent=1))
    import shutil
    shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', required=True, choices=['totaltext', 'icdar2015', 'ctw1500'])
    p.add_argument('--out', required=True)
    main(p.parse_args())
