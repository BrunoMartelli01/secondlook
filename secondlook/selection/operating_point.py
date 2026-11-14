"""E30: audit of the label-free operating point c = F/2 of the stage-1 admission head. CPU only.

Reviewer objection addressed: rule R takes F from the spotter's PUBLISHED end-to-end F1, which is usually measured on the same TEST split,
so it carries aggregate TEST-label information; and the F/2 optimality result (Lipton et al., 2014) holds for calibrated probabilities and
the classifier's OWN optimal F, not for the released system's F1. This script, per row:

1. Calibration of the stage-1 head: 10-bin reliability table and ECE on (a) TRAIN candidates with 5-fold image-grouped cross-fitted
   probabilities (the same cross-fit as secondlook.second_stage.band_classifier) and (b) TEST candidates with the head fitted on all TRAIN (labels from the
   exact evaluator, used here only as a DIAGNOSTIC, never to choose c).
2. TRAIN-only operating points (no TEST information of any kind): label-based pooled F1 of the cross-fitted head on TRAIN on a 0.01 grid ->
   F_cf* (its maximum) and c_argmax (its arg max); c_half = F_cf*/2. Both are rounded to 0.01.
3. Diagnostic check of the F/2 relation on TEST: c_test* (arg max of the TEST F1 over the 0.01 grid) vs F_test*/2. Never a result.
4. Exact TEST counts (official evaluator) of stage 1 at c_half and c_argmax, saved for the bootstrap of the report.

Corpus-level F1 = 2 TP / (K + G) (K kept non-don't-care candidates, G ground-truth words, fixed). Adding a candidate with correctness
probability p raises E[F] iff p > TP / (K + G) = F / 2, so for calibrated, independent labels the optimal cut equals half the optimal F.
Labels are NOT independent of the keep set (duplicates compete for one ground-truth word), which is one reason the relation is checked.
Result class: approximate reproduction; TRAIN-only rules are a priori with respect to TEST; the TEST checks are diagnostics.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from secondlook.data.cv_counts import counts_for, test_labels
from secondlook.data.tasks import Task, TiclsTask
from secondlook.studies.ablation_controls import hgb
from secondlook.second_stage.band_classifier import cross_fit

FINE = [round(0.01 * i, 2) for i in range(100)]


def load(expert, task, work):
    if task == 'ctw1500':
        from secondlook.data.tasks_ctw1500 import CtwTask
        return CtwTask(work, expert)
    if expert == 'ppocr':
        from secondlook.data.tasks_ppocr import PpocrTask
        return PpocrTask(task, work)
    return (Task if expert == 'deepsolo' else TiclsTask)(task, work)


def pooled_f1(lab, gt, p, ids, c):
    tp = k = 0
    for s in ids:
        keep = p[s] >= c
        tp += int(((lab[s] == 1) & keep).sum())
        k += int(((lab[s] >= 0) & keep).sum())
    g = sum(gt[s] for s in ids)
    return 2.0 * tp / (k + g) if k + g else 0.0


def reliability(lab, p, ids, bins=10):
    y = np.concatenate([lab[s] for s in ids])
    q = np.concatenate([p[s] for s in ids])
    m = y >= 0
    y, q = y[m].astype(float), q[m]
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for b in range(bins):
        sel = (q >= edges[b]) & ((q < edges[b + 1]) if b < bins - 1 else (q <= 1.0))
        if not sel.any():
            rows.append(dict(lo=float(edges[b]), hi=float(edges[b + 1]), n=0, mean_p=None, pos_rate=None))
            continue
        mp, pr = float(q[sel].mean()), float(y[sel].mean())
        ece += sel.sum() / len(q) * abs(mp - pr)
        rows.append(dict(lo=float(edges[b]), hi=float(edges[b + 1]), n=int(sel.sum()), mean_p=mp, pos_rate=pr))
    return dict(ece=float(ece), mean_p=float(q.mean()), pos_rate=float(y.mean()), n=int(len(q)), brier=float(np.mean((q - y) ** 2)), bins=rows)


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(a.work) / ('%s_%s' % (a.expert, a.task))
    work.mkdir(parents=True, exist_ok=True)
    T = load(a.expert, a.task, work)
    ids = list(T.ids)
    p_cf = cross_fit(T, ids)
    x = np.concatenate([T.tab[s]['xc'] for s in ids])
    y = np.concatenate([T.lab[s] for s in ids])
    m = y >= 0
    head = hgb()
    head.fit(x[m], y[m])
    p_te = {s: (head.predict_proba(T.eval_tab[s]['xc'])[:, 1] if len(T.eval_tab[s]['lines']) else np.zeros(0)) for s in T.eval_ids}
    curve_tr = [pooled_f1(T.lab, T.gt, p_cf, ids, c) for c in FINE]
    i = int(np.argmax(curve_tr))
    f_star = curve_tr[i]
    res = dict(expert=a.expert, task=a.task, train_images=len(ids), test_images=len(T.eval_ids),
               train_cf_f1_curve=dict(zip(map(str, FINE), curve_tr)), train_cf_fstar=f_star, c_argmax=FINE[i], c_half=round(f_star / 2, 2),
               calibration_train_cf=reliability(T.lab, p_cf, ids))
    lab_te, gt_te = test_labels(T, work)
    curve_te = [pooled_f1(lab_te, gt_te, p_te, T.eval_ids, c) for c in FINE]
    j = int(np.argmax(curve_te))
    res.update(calibration_test=reliability(lab_te, p_te, T.eval_ids), diag_test_f1_curve_labels=dict(zip(map(str, FINE), curve_te)),
               diag_test_c_star=FINE[j], diag_test_fstar_half=curve_te[j] / 2)
    counts = {}
    for name in ('c_half', 'c_argmax'):
        c = res[name]
        counts[name] = counts_for(T, work, {s: p_te[s] >= c for s in T.eval_ids})
    for name in a.extra_cuts:
        counts['c_%.2f' % name] = counts_for(T, work, {s: p_te[s] >= name for s in T.eval_ids})
    f = lambda c: 200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())
    res['test_f1_exact'] = {k: float(f(v)) for k, v in counts.items()}
    np.savez_compressed(out / ('op_counts_%s_%s.npz' % (a.expert, a.task)), ids=np.array([str(s) for s in T.eval_ids]), **counts)
    (out / ('op_audit_%s_%s.json' % (a.expert, a.task))).write_text(json.dumps(res, indent=1))
    print('OP', a.expert, a.task, json.dumps({k: res[k] for k in ('train_cf_fstar', 'c_half', 'c_argmax', 'diag_test_c_star', 'diag_test_fstar_half', 'test_f1_exact')}),
          'ECE train %.3f test %.3f' % (res['calibration_train_cf']['ece'], res['calibration_test']['ece']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', required=True, choices=['totaltext', 'icdar2015', 'ctw1500'])
    p.add_argument('--work', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--extra-cuts', dest='extra_cuts', type=float, nargs='*', default=[])
    main(p.parse_args())
