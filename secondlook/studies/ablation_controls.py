"""E14: negative controls, cut-rule ablation and hyper-parameter sensitivity for the word-admission head. CPU only, no neural call.

Same tasks / partitions / common11 interface / candidate tables / cut grid as E10-E13; exact pinned evaluator;
per-image counts saved. Arms:

  author, tuned_all            references
  hgb                          E10 classifier (60 iters, 7 leaves, L2 5, min leaf 20, lr 0.1), cut on selection images
  hgb_cut_fixed05, hgb_cut_fixed042
                               same model, NO selection images: fixed cut 0.5, and 0.42 (= F*/2 with F* ~ 0.84, the
                               F-measure-optimal threshold of a calibrated classifier, Lipton et al. 2014; value fixed in advance)
  neg_shuffled_labels          same model trained on the fit rows with PERMUTED labels, cut on selection images
  neg_noise_features           same model trained and applied on N(0,1) noise features (same shape), cut on selection images
  hp_*                         one-at-a-time hyper-parameter changes around the default (iters 30/120/240, leaves 3/15/31,
                               L2 1/20, min leaf 10/50, learning rate 0.05) and sklearn defaults
Interpretation rule fixed in advance: negative controls must NOT exceed tuned_all; hp arms are a robustness band.
Approximate reproduction; TT/IC15 TEST was explored before.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from secondlook.data.tasks import (Task, TiclsTask, evaluate, make_model, pick_thresholds, stack, AUTHOR)
from secondlook.studies.ablation_learners import proba, cut_from, f1
from secondlook.evaluation.protocol import write_json, read_json

COMMON11 = ('xc', list(range(11)))
HP = {
    'hp_iter30': dict(max_iter=30), 'hp_iter120': dict(max_iter=120), 'hp_iter240': dict(max_iter=240),
    'hp_leaves3': dict(max_leaf_nodes=3), 'hp_leaves15': dict(max_leaf_nodes=15), 'hp_leaves31': dict(max_leaf_nodes=31),
    'hp_l2_1': dict(l2_regularization=1.0), 'hp_l2_20': dict(l2_regularization=20.0),
    'hp_minleaf10': dict(min_samples_leaf=10), 'hp_minleaf50': dict(min_samples_leaf=50),
    'hp_lr005': dict(learning_rate=0.05),
    'hp_sklearn_default': dict(max_iter=100, max_leaf_nodes=31, l2_regularization=0.0, min_samples_leaf=20),
}


def hgb(**kw):
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = dict(max_iter=60, max_leaf_nodes=7, l2_regularization=5.0, min_samples_leaf=20, random_state=11, early_stopping=False)
    p.update(kw)
    return HistGradientBoostingClassifier(**p)


def job(args):
    out = Path(args.out) / args.task / ('seed%d' % args.seed)
    out.mkdir(parents=True, exist_ok=False)
    work = out / 'work'
    work.mkdir()
    T = (TiclsTask if args.model == 'ticls' else Task)(args.task, work)
    fit, sel, aud = T.partition(args.seed)
    pool = list(fit) + list(sel)
    if T.holdout:
        tab, ids, names = T.tab, aud, T.names
        gt = work / 'aud_gt.zip'
        from secondlook.features.word_trust import subset_gt
        subset_gt(T.pool_gt, gt, T.metric.module, {s: T.names[s] for s in aud}, 'x')
    else:
        tab, ids, names, gt = T.eval_tab, T.eval_ids, T.eval_names, T.eval_gt
    counts, meta = {}, dict(model=args.model, task=args.task, seed=args.seed, eval_ids=[str(s) for s in ids], cuts={})

    def exact(tag, keep):
        c = np.asarray(evaluate(None, work, tag, ids, tab, {s: np.asarray(keep[s], bool).tolist() for s in ids}, gt, names, T.metric)[0])
        for f in work.glob(tag + '.zip'):
            f.unlink()
        counts[tag] = c.astype(np.int64)

    exact('author', {s: [True] * len(tab[s]['lines']) for s in ids} if args.model == 'ticls' else {s: (tab[s]['score'] >= AUTHOR['score_threshold']) for s in ids})
    st, rt, _ = pick_thresholds(T, pool)
    exact('tuned_all', {s: (tab[s]['score'] >= st) & (tab[s]['rec'] >= rt) for s in ids})
    x, y = stack(T, fit, COMMON11)

    def run(tag, model, xfit, yfit, sel_p, eval_p, fixed=None):
        model.fit(xfit, yfit)
        ps, pe = sel_p(model), eval_p(model)
        cut = fixed if fixed is not None else cut_from(T, ps, sel)
        meta['cuts'][tag] = cut
        exact(tag, {s: pe[s] >= cut for s in ids})

    plain = (lambda m: proba(m, T.tab, sel, COMMON11), lambda m: proba(m, tab, ids, COMMON11))
    run('hgb', hgb(), x, y, *plain)
    m = hgb()
    m.fit(x, y)
    pe = proba(m, tab, ids, COMMON11)
    for tag, cut in (('hgb_cut_fixed05', 0.5), ('hgb_cut_fixed042', 0.42)):
        meta['cuts'][tag] = cut
        exact(tag, {s: pe[s] >= cut for s in ids})
    rng = np.random.RandomState(1000 + args.seed)
    run('neg_shuffled_labels', hgb(), x, y[rng.permutation(len(y))], *plain)
    noise = np.random.RandomState(2000 + args.seed)
    xn = noise.randn(*x.shape)

    def noise_proba(m, ids_, table):
        r = np.random.RandomState(3000 + args.seed)
        return {s: (m.predict_proba(r.randn(len(table[s]['lines']), x.shape[1]))[:, 1] if len(table[s]['lines']) else np.zeros(0)) for s in ids_}
    run('neg_noise_features', hgb(), xn, y,
        lambda m: noise_proba(m, sel, T.tab), lambda m: noise_proba(m, ids, tab))
    for tag, kw in HP.items():
        run(tag, hgb(**kw), x, y, *plain)
    np.savez_compressed(out / 'counts.npz', **counts)
    write_json(out / 'meta.json', meta)
    shutil.rmtree(work, ignore_errors=True)
    f = {k: 200.0 * v[:, 0].sum() / (v[:, 1].sum() + v[:, 2].sum()) for k, v in counts.items()}
    print('E14', args.model, args.task, 'seed', args.seed, json.dumps({k: round(v, 2) for k, v in f.items() if not k.startswith('hp_')}), flush=True)


def report(args):
    out = Path(args.out)
    rng = np.random.RandomState(20261107)
    summary = {}
    for task in args.tasks:
        runs = []
        for s in args.seeds:
            c = np.load(out / task / ('seed%d' % s) / 'counts.npz')
            runs.append(({k: c[k] for k in c.files}, read_json(out / task / ('seed%d' % s) / 'meta.json')['eval_ids']))
        arms = list(runs[0][0])
        same = all(r[1] == runs[0][1] for r in runs)
        draws = [rng.randint(0, len(runs[0][1]), len(runs[0][1])) for _ in range(args.draws)] if same else None
        t = dict(partitions=len(runs), eval_images=len(runs[0][1]), shared_eval_images=bool(same), arms={})
        for a in arms:
            row = dict(mean_f1=float(np.mean([f1(r[0][a]) for r in runs])))
            for b in ('tuned_all', 'hgb'):
                diff = np.array([f1(r[0][a]) - f1(r[0][b]) for r in runs])
                row['minus_' + b] = float(diff.mean())
                row['splits_above_' + b] = int((diff > 0).sum())
                if same and a != b:
                    d = np.array([np.mean([f1(r[0][a][i]) - f1(r[0][b][i]) for r in runs]) for i in draws])
                    row['ci_' + b] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            t['arms'][a] = row
        hp = [t['arms'][a]['minus_tuned_all'] for a in arms if a.startswith('hp_')]
        t['hp_band_minus_tuned_all'] = [float(min(hp)), float(max(hp))]
        summary[task] = t
    write_json(out / ('controls_%s.json' % args.model), summary, exclusive=False)
    for task, t in summary.items():
        print(args.model, task, 'imgs', t['eval_images'], 'hp band vs tuned_all [%+.2f, %+.2f]' % tuple(t['hp_band_minus_tuned_all']))
        for a, r in t['arms'].items():
            if a.startswith('hp_'):
                continue
            ci = r.get('ci_tuned_all')
            print('  %-20s F1 %.2f | vs tuned_all %+.2f (%d/%d)%s' % (a, r['mean_f1'], r['minus_tuned_all'], r['splits_above_tuned_all'], t['partitions'], (' CI [%+.2f,%+.2f]' % tuple(ci)) if ci else ''))


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
