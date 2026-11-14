"""E13: calibration baselines, classifier family and voting for the word-admission head.

CPU only, no neural call. Same tasks, partitions, 11 shared features ('common11'), candidate tables, cut grid
and cut-selection rule as E10/E12. Per-image evaluator counts are saved for every arm so a paired image bootstrap
can be computed. Arms (all evaluated with the exact pinned evaluator on the same evaluation images):

  author, tuned_all            references (author default; two-threshold rule tuned on ALL labelled images)
  hgb                          the E10/E12 classifier (HistGradientBoosting, common11)
  logistic, rf, extratrees, mlp   other classifier families on common11 (cut chosen on the selection images)
  vote_soft                    mean probability of the five classifiers, cut chosen on the selection images
  vote_hard                    majority (>=3 of 5) of the five classifiers, each with its own selection-chosen cut
  lr5                          logistic regression on the 5 confidence signals only (multivariate calibration)
  sig_logit, sig_geo, sig_minp, sig_entropy, sig_margin
                               ONE confidence signal, threshold tuned on ALL labelled images. Any monotone
                               re-calibration of a single signal (temperature scaling, Platt, isotonic) followed by an
                               F1-optimal cut is the same decision rule as thresholding that signal, so these arms
                               cover those calibrators (the Mor & Wolf margin is sig_margin).
  prod_det_geo                 detector probability x geometric-mean token probability, threshold tuned on ALL images

Approximate reproduction; TT/IC15 TEST was explored before.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from secondlook.data.tasks import (Task, TiclsTask, evaluate, make_model, pick_thresholds, stack, AUTHOR)
from secondlook.data.deepsolo_table import CUTS
from secondlook.evaluation.protocol import write_json, read_json

COMMON11 = ('xc', list(range(11)))
COMMON5 = ('xc', list(range(5)))
SIGNALS = {'sig_logit': (0, 1), 'sig_geo': (1, 1), 'sig_minp': (2, 1), 'sig_entropy': (3, -1), 'sig_margin': (4, 1)}
CLASSIFIERS = ['hgb', 'logistic', 'rf', 'extratrees', 'mlp']


def build(kind, cfg):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind in ('hgb', 'logistic'):
        return make_model(kind, cfg)
    if kind == 'rf':
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=1, random_state=11)
    if kind == 'extratrees':
        return ExtraTreesClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=1, random_state=11)
    if kind == 'mlp':
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=300, random_state=11))
    raise ValueError(kind)


def proba(model, tab, ids, spec):
    key, cols = spec
    return {s: (model.predict_proba(tab[s][key][:, cols])[:, 1] if len(tab[s]['lines']) else np.zeros(0)) for s in ids}


def cut_from(T, p, sel):
    best = max((T.approx_f1(sel, {s: p[s] >= c for s in sel}), -c, c) for c in CUTS)
    return best[2]


def best_signal_threshold(T, pool, fn):
    """Threshold maximizing the label-based pooled F1 on the pool images (candidates: 80 pooled quantiles + keep-all)."""
    vals = np.concatenate([fn(T.tab[s]) for s in pool])
    lab = np.concatenate([T.lab[s] for s in pool])
    g = sum(T.gt[s] for s in pool)
    cands = np.concatenate([[-np.inf], np.unique(np.quantile(vals, np.linspace(0.0, 0.97, 80)))])
    best = None
    for c in cands:
        k = vals >= c
        tp = int(((lab == 1) & k).sum())
        p = int(((lab >= 0) & k).sum())
        f = 2 * tp / (p + g) if p + g else 0.0
        if best is None or f > best[0]:
            best = (f, c)
    return best[1]


def job(args):
    out = Path(args.out) / args.task / ('seed%d' % args.seed)
    out.mkdir(parents=True, exist_ok=False)
    work = out / 'work'
    work.mkdir()
    T = (TiclsTask if args.model == 'ticls' else Task)(args.task, work)
    cfg = T.cfg
    fit, sel, aud = T.partition(args.seed)
    pool = list(fit) + list(sel)
    if T.holdout:
        tab, ids, names = T.tab, aud, T.names
        gt = work / 'aud_gt.zip'
        from secondlook.features.word_trust import subset_gt
        subset_gt(T.pool_gt, gt, T.metric.module, {s: T.names[s] for s in aud}, 'x')
    else:
        tab, ids, names, gt = T.eval_tab, T.eval_ids, T.eval_names, T.eval_gt
    counts, meta = {}, dict(model=args.model, task=args.task, seed=args.seed, eval_ids=[str(s) for s in ids], fit=len(fit), selection=len(sel), cuts={}, thresholds={})

    def exact(tag, keep):
        c = np.asarray(evaluate(None, work, tag, ids, tab, {s: np.asarray(keep[s], bool).tolist() for s in ids}, gt, names, T.metric)[0])
        for f in work.glob(tag + '.zip'):
            f.unlink()
        counts[tag] = c.astype(np.int64)

    exact('author', {s: [True] * len(tab[s]['lines']) for s in ids} if args.model == 'ticls' else {s: (tab[s]['score'] >= AUTHOR['score_threshold']) for s in ids})
    st, rt, _ = pick_thresholds(T, pool)
    exact('tuned_all', {s: (tab[s]['score'] >= st) & (tab[s]['rec'] >= rt) for s in ids})
    # single signals and the product heuristic (thresholds on ALL labelled images)
    for name, (col, sign) in SIGNALS.items():
        thr = best_signal_threshold(T, pool, lambda t, col=col, sign=sign: sign * t['xc'][:, col])
        meta['thresholds'][name] = float(thr) if np.isfinite(thr) else None
        exact(name, {s: sign * tab[s]['xc'][:, col] >= thr for s in ids})
    prod = lambda t: (1.0 / (1.0 + np.exp(-t['xc'][:, 0]))) * t['xc'][:, 1]
    thr = best_signal_threshold(T, pool, prod)
    meta['thresholds']['prod_det_geo'] = float(thr) if np.isfinite(thr) else None
    exact('prod_det_geo', {s: prod(tab[s]) >= thr for s in ids})
    # classifiers on common11, lr5 on the 5 confidence signals
    x, y = stack(T, fit, COMMON11)
    P_sel, P_eval, dec = {}, {}, {}
    for kind in CLASSIFIERS:
        m = build(kind, cfg)
        m.fit(x, y)
        P_sel[kind] = proba(m, T.tab, sel, COMMON11)
        P_eval[kind] = proba(m, tab, ids, COMMON11)
        cut = cut_from(T, P_sel[kind], sel)
        meta['cuts'][kind] = cut
        dec[kind] = {s: P_eval[kind][s] >= cut for s in ids}
        exact(kind, dec[kind])
    ens_sel = {s: np.mean([P_sel[k][s] for k in CLASSIFIERS], axis=0) for s in sel}
    ens_eval = {s: np.mean([P_eval[k][s] for k in CLASSIFIERS], axis=0) for s in ids}
    cut = cut_from(T, ens_sel, sel)
    meta['cuts']['vote_soft'] = cut
    exact('vote_soft', {s: ens_eval[s] >= cut for s in ids})
    exact('vote_hard', {s: np.sum([dec[k][s] for k in CLASSIFIERS], axis=0) >= 3 for s in ids})
    x5, y5 = stack(T, fit, COMMON5)
    m = build('logistic', cfg)
    m.fit(x5, y5)
    cut = cut_from(T, proba(m, T.tab, sel, COMMON5), sel)
    meta['cuts']['lr5'] = cut
    exact('lr5', {s: p >= cut for s, p in proba(m, tab, ids, COMMON5).items()})
    np.savez_compressed(out / 'counts.npz', **counts)
    write_json(out / 'meta.json', meta)
    shutil.rmtree(work, ignore_errors=True)
    f1 = {k: 200.0 * v[:, 0].sum() / (v[:, 1].sum() + v[:, 2].sum()) for k, v in counts.items()}
    print('E13', args.model, args.task, 'seed', args.seed, json.dumps({k: round(v, 2) for k, v in f1.items()}), flush=True)


def f1(c):
    return 200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())


def report(args):
    out = Path(args.out)
    rng = np.random.RandomState(20261106)
    summary = {}
    for task in args.tasks:
        runs = []
        for s in args.seeds:
            c = np.load(out / task / ('seed%d' % s) / 'counts.npz')
            runs.append(({k: c[k] for k in c.files}, read_json(out / task / ('seed%d' % s) / 'meta.json')['eval_ids']))
        arms = list(runs[0][0])
        same = all(r[1] == runs[0][1] for r in runs)
        t = dict(partitions=len(runs), eval_images=len(runs[0][1]), shared_eval_images=bool(same), arms={})
        draws = [rng.randint(0, len(runs[0][1]), len(runs[0][1])) for _ in range(args.draws)] if same else None
        for a in arms:
            mean = float(np.mean([f1(r[0][a]) for r in runs]))
            row = dict(mean_f1=mean)
            for b in ('hgb', 'tuned_all'):
                diff = np.array([f1(r[0][a]) - f1(r[0][b]) for r in runs])
                row['minus_' + b] = float(diff.mean())
                row['splits_above_' + b] = int((diff > 0).sum())
                if same and a != b:
                    d = np.array([np.mean([f1(r[0][a][i]) - f1(r[0][b][i]) for r in runs]) for i in draws])
                    row['ci_' + b] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            t['arms'][a] = row
        summary[task] = t
    write_json(out / ('family_%s.json' % args.model), summary, exclusive=False)
    for task, t in summary.items():
        print(args.model, task, 'imgs', t['eval_images'], 'CI only if shared images' if t['shared_eval_images'] else 'point estimates only')
        for a, r in t['arms'].items():
            ci = r.get('ci_tuned_all')
            print('  %-13s F1 %.2f | vs hgb %+.2f (%d/%d) | vs tuned_all %+.2f (%d/%d)%s' % (a, r['mean_f1'], r['minus_hgb'], r['splits_above_hgb'], t['partitions'], r['minus_tuned_all'], r['splits_above_tuned_all'], t['partitions'], (' CI [%+.2f,%+.2f]' % tuple(ci)) if ci else ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--stage', choices=['job', 'report'], required=True)
    p.add_argument('--model', choices=['deepsolo', 'ticls'], default='deepsolo')
    p.add_argument('--task', choices=['totaltext', 'icdar2015', 'cohort_totaltext', 'cohort_icdar2015'])
    p.add_argument('--tasks', nargs='+', default=['totaltext', 'icdar2015', 'cohort_totaltext', 'cohort_icdar2015'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(11)))
    p.add_argument('--draws', type=int, default=2000)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    job(a) if a.stage == 'job' else report(a)
