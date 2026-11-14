"""E49 (approximate reproduction): the whole uniform pipeline (secondlook.studies.stability.pipeline, copied 1:1 with the 7-column counters of secondlook.evaluation.common.counts_full) on S = all TRAIN
images and on the eleven random 80% subsets (RandomState(100 + s)) of one row. Run through secondlook.reruns.fast.launch with the arm old (GATE) or unified:
    python -m secondlook.reruns.fast.launch {old|unified} secondlook.reruns.fast.stability --expert E --task T --out DIR
Per subset: c = round(F_cf*/2, 2) (cross-fit inside the subset), k* by the nested cross-fit inside the subset, band model, thr_train (pick_thresholds on the subset: FAST protocol, label-based F1 for
the classifier's c and k* and for the baseline thresholds alike); exact pinned evaluator on the full official TEST, all 7 columns (end-to-end and detection-only).
GATES (arm old): c, k*, band edge, F1 of first stage / complete / thr_train equal the stored stability.json (1e-9) on all 12 training sets; counts equal the stored E44 det_counts.npz arrays and the
released arm equals the stored grid.npz 'author' arm. (arm unified): S = all first/complete/thr_train end-to-end columns equal the E46 UNIFIED counts bit for bit."""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from secondlook.data.tasks import pick_thresholds
from secondlook.evaluation.common import CLEAN, E18, FIELDS, RUNS, author_keep, counts_full, f1e, stability_path
from secondlook.selection.operating_point import FINE, load, pooled_f1
from secondlook.second_stage.band_classifier import cross_fit, verifier_features
from secondlook.selection.nested_selection import band_fit, fit_head, nested_select

E44DET = RUNS / 'detection-crossdata/detection'
E46ROWS = RUNS / 'unified-features/rows'


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
    st, rt, fsel = pick_thresholds(T, S)
    keept = {s: (T.eval_tab[s]['score'] >= st) & (T.eval_tab[s]['rec'] >= rt) for s in eids}
    cs = [counts_full(T, work, k) for k in (keep1, keep2, keept)]
    return dict(c=c, k_star=k_star, band_hi=hi, thr_score=float(st), thr_rec=float(rt), approx_f1_selection=float(fsel)), cs


def main(a):
    arm = os.environ['SECONDLOOK_FEATURES']
    e, t = a.expert, a.task
    out = Path(a.out) / arm / 'stability' / ('%s_%s' % (e, t))
    out.mkdir(parents=True, exist_ok=True)
    work = out / 'work'
    work.mkdir(exist_ok=True)
    T = load(e, t, work)
    ids = list(T.ids)
    R = CLEAN / 'p3'
    vt = verifier_features(T.tab, ids, json.load(open(R / ('%s_%s_train_parseq.json' % (e, t)))))
    ve = verifier_features(T.eval_tab, T.eval_ids, json.load(open(R / ('%s_%s_test_parseq.json' % (e, t)))))
    rel = counts_full(T, work, author_keep(T, e, t))
    g_author = np.load(E18 / e / t / 'grid.npz')['author'].astype(np.int64)
    if not np.array_equal(rel[:, :3], g_author):
        raise SystemExit('GATE FAILED: released arm differs from the stored grid.npz author arm for %s %s' % (e, t))
    stab = {str(r['subset']): r for r in json.load(open(stability_path(e, t)))['rows']} if arm == 'old' else None
    ref_det = np.load(E44DET / ('%s_%s' % (e, t)) / 'det_counts.npz') if arm == 'old' else None
    jobs = [('all', ids)] + [(str(s), [ids[i] for i in np.random.RandomState(100 + s).permutation(len(ids))[:int(round(0.8 * len(ids)))]]) for s in range(11)]
    arrays, rows = {'released': rel}, []
    for tag, S in jobs:
        meta, (c1, c2, ct) = pipeline(T, work, S, vt, ve)
        row = dict(subset=tag, images=len(S), **meta, f1_head=f1e(c1), f1_final=f1e(c2), f1_thr=f1e(ct))
        if arm == 'old':
            sr = stab[tag]
            bad = [k for k, v in (('c', meta['c']), ('k_star', meta['k_star']), ('band_hi', meta['band_hi'])) if abs(float(v) - float(sr[k])) > 1e-12] if True else []
            bad += [k for k, v in (('f1_head', row['f1_head']), ('f1_final', row['f1_final']), ('f1_thr', row['f1_thr'])) if abs(v - sr[k]) > 1e-9]
            names = ('first', 'complete', 'thr_train') if tag == 'all' else ('first_s%s' % tag, 'complete_s%s' % tag, 'thr_train_s%s' % tag)
            for nm, cc in zip(names, (c1, c2, ct)):
                if not np.array_equal(cc, ref_det[nm]):
                    bad.append('counts:' + nm)
            if bad:
                raise SystemExit('GATE FAILED (old arm) %s %s subset %s: %s' % (e, t, tag, bad))
            row['gate_vs_stored'] = 'passed'
        elif tag == 'all':
            ref = np.load(E46ROWS / ('%s_%s' % (e, t)) / 'counts_UNIFIED.npz')
            ok = np.array_equal(c1[:, :3], ref['first']) and np.array_equal(c2[:, :3], ref['complete']) and np.array_equal(ct[:, :3], ref['thr_train'])
            if not ok:
                raise SystemExit('GATE FAILED (unified arm) %s %s: S = all differs from the E46 UNIFIED counts' % (e, t))
            row['gate_vs_E46'] = 'passed'
        names = ('first', 'complete', 'thr_train') if tag == 'all' else ('first_s%s' % tag, 'complete_s%s' % tag, 'thr_train_s%s' % tag)
        for nm, cc in zip(names, (c1, c2, ct)):
            arrays[nm] = cc
        rows.append(row)
        print('E49S', arm, e, t, tag, 'c %.2f k* %s first %.2f complete %.2f thr %.2f %s' % (meta['c'], meta['k_star'], row['f1_head'], row['f1_final'], row['f1_thr'], row.get('gate_vs_stored', row.get('gate_vs_E46', ''))), flush=True)
    np.savez_compressed(out / 'counts7.npz', fields=np.array(FIELDS), ids=np.array([str(s) for s in T.eval_ids]), **arrays)
    (out / 'stability.json').write_text(json.dumps(dict(expert=e, task=t, arm=arm, claim_label='approximate reproduction', selection_protocol='FAST (label-based F1 for c, k* and the baseline thresholds)',
                                                        train_images=len(ids), test_images=len(T.eval_ids), rows=rows, seeds=dict(subsets='RandomState(100+s), 80%, s = 0..10', cross_fit='RandomState(0)')), indent=1, default=float))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', required=True, choices=['totaltext', 'icdar2015', 'ctw1500'])
    p.add_argument('--out', required=True)
    main(p.parse_args())
