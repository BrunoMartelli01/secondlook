"""E35: labelled-data learning curve (Table 13 / Figure 2) on ALL NINE rows of the draft, with the TRAIN-only operating point of E30/E31.

Per row (expert, task) and draw s (RandomState(1000+s) permutation of ALL labelled TRAIN images of the row; nested prefixes) and budget N in
{30, 60, 120, 240, 480, all}: the stage-1 head (HGB on the 11 shared signals, hyper-parameters of secondlook.studies.ablation_controls.hgb) is fitted on
the N images; its cut is c = round(F_cf*/2, 2), where F_cf* is the maximum over the 0.01 grid of the label-based pooled F1 of the 5-fold image-grouped
cross-fitted head INSIDE the N images (secondlook.second_stage.band_classifier.cross_fit, secondlook.selection.operating_point.pooled_f1: the E30/E31 rule, no selection split);
the comparison arm is the two-threshold rule (detector score, recognition score) tuned on the same N images (secondlook.data.tasks.
pick_thresholds, the `thr_train` rule at N = all). Both arms are evaluated with the exact pinned evaluator on the official TEST. 'all' = every labelled
TRAIN image of the row (deterministic: one value per row, computed on draw 0). CPU only, no neural call, no TEST label used for any choice."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from secondlook.data.cv_counts import counts_for
from secondlook.data.tasks import pick_thresholds
from secondlook.studies.ablation_controls import hgb
from secondlook.selection.operating_point import load, pooled_f1, FINE
from secondlook.second_stage.band_classifier import cross_fit

BUDGETS = [30, 60, 120, 240, 480]


def f1c(c):
    return float(200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum()))


def run(T, work, sub):
    ids = list(T.eval_ids)
    p_cf = cross_fit(T, sub)
    curve = [pooled_f1(T.lab, T.gt, p_cf, sub, c) for c in FINE]
    f_star = max(curve)
    cut = round(f_star / 2, 2)
    x = np.concatenate([T.tab[s]['xc'] for s in sub])
    y = np.concatenate([T.lab[s] for s in sub])
    m = y >= 0
    if len(set(y[m].tolist())) < 2:
        raise ValueError('Single-class labels in a budget subset of %d images' % len(sub))
    head = hgb()
    head.fit(x[m], y[m])
    p = {s: (head.predict_proba(T.eval_tab[s]['xc'])[:, 1] if len(T.eval_tab[s]['lines']) else np.zeros(0)) for s in ids}
    c_cls = counts_for(T, work, {s: p[s] >= cut for s in ids})
    st, rt, _ = pick_thresholds(T, sub)
    c_thr = counts_for(T, work, {s: (T.eval_tab[s]['score'] >= st) & (T.eval_tab[s]['rec'] >= rt) for s in ids})
    return dict(images=len(sub), f_star=float(f_star), cut=cut, thr_score=float(st), thr_rec=float(rt), f1_classifier=f1c(c_cls), f1_thresholds=f1c(c_thr),
                diff=f1c(c_cls) - f1c(c_thr), tp_cls=int(c_cls[:, 0].sum()), kept_cls=int(c_cls[:, 1].sum()), gt=int(c_cls[:, 2].sum()), tp_thr=int(c_thr[:, 0].sum()), kept_thr=int(c_thr[:, 1].sum()))


def job(a):
    out = Path(a.out) / a.expert / a.task
    out.mkdir(parents=True, exist_ok=True)
    f = out / ('seed%d.json' % a.seed)
    if f.exists():
        raise FileExistsError(f)
    work = out / ('work%d' % a.seed)
    work.mkdir(exist_ok=True)
    T = load(a.expert, a.task, work)
    ids = list(T.ids)
    perm = [ids[i] for i in np.random.RandomState(1000 + a.seed).permutation(len(ids))]
    res = dict(expert=a.expert, task=a.task, seed=a.seed, train_images=len(ids), test_images=len(T.eval_ids), budgets={})
    for n in BUDGETS:
        res['budgets'][str(n)] = run(T, work, perm[:n])
        print('E35', a.expert, a.task, a.seed, n, '%+.2f' % res['budgets'][str(n)]['diff'], flush=True)
    if a.seed == 0:
        res['budgets']['all'] = run(T, work, ids)
        print('E35', a.expert, a.task, 'all', '%+.2f' % res['budgets']['all']['diff'], flush=True)
    f.write_text(json.dumps(res, indent=1))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', required=True, choices=['totaltext', 'icdar2015', 'ctw1500'])
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    job(p.parse_args())
