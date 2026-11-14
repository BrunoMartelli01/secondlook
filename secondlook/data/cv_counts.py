"""E18: calibration on UNSEEN target images (cross-validation inside the official TEST set) for the word-admission head. CPU only, no neural call.

Why: E16 selected cuts/thresholds on TRAIN images that the experts were fine-tuned on (TiCLS: Total-Text and ICDAR2015 TRAIN)
and evaluated them on the official TEST, so (i) the calibration images are not exchangeable with the evaluation images and (ii)
risk guarantees (method B) could not hold. Here the TARGET pool is the official TEST set of the dataset: for each of R random
repeats, N labelled TEST images (nested prefixes of a random order, N in {30, 60, 120}) are the calibration set and the
evaluation set is the remaining images (always perm[120:], identical for every N and arm within a repeat). Heads are
trained on labelled TRAIN data (own dataset, or the other dataset as source) which is allowed because only the OPERATING
POINT / adaptation uses target labels. Exact per-image evaluator counts (additive over images) make every
selection and every evaluation exact. Results are partial-dataset results (subsets of TEST), approximate reproduction.

Stages: grid (per-image counts for cut grids of the own-TRAIN ranker and the other-dataset-source ranker, the 135-point
two-threshold grid, the author default); reps (per repeat: grid selections, risk-controlled rules, retrained arms: source+target
mix, own-TRAIN+target mix, target-only, label-free pseudo-label A); report (image-bootstrap CIs, FDR coverage).
Arms: author, thr_N (two-threshold rule tuned on the N calibration images), own_cut_N / src_cut_N (ranker + cut maximizing the exact F1
on the N images), own042 / src042 (no target label, cut 0.42), B rules (model_*/thr_* plug and ucb at alpha), dmix_N, ownmix_N,
scratch_N, A_Nu (no labels), as defined in the code. FDR = 1 - TP / predictions on the evaluation images.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from secondlook.data.deepsolo_table import CUTS, SCORE_GRID, REC_GRID
from secondlook.data.tasks import Task, TiclsTask, evaluate, AUTHOR
from secondlook.studies.ablation_agreement import agreement
from secondlook.studies.ablation_controls import hgb
from secondlook.studies.ablation_learners import proba
from secondlook.evaluation.protocol import write_json

SPEC = ('xc', list(range(11)))
GRID_CUTS = sorted(set(CUTS) | {0.2, 0.24, 0.28, 0.32, 0.33, 0.35, 0.37, 0.42, 0.45})   # fine grid around F/2 (E20/E21 grids; older grids keep their saved cuts)
NS = (30, 60, 120)
NMAX = 120
ALPHAS = (0.05, 0.10, 0.15)
MIX_NS = (30, 60)
A_NS = (60, 120)
THR = [(st, rt) for st in SCORE_GRID for rt in REC_GRID]


def load(task, expert, work):
    work = Path(work)
    for d in ('ds', 'ti', 'src', 'own'):
        (work / d).mkdir(parents=True, exist_ok=True)
    if expert == 'ppocr' and task != 'ctw1500':
        from secondlook.data.tasks_ppocr import PpocrTask
        if task not in ('totaltext', 'icdar2015'):
            raise ValueError('PP-OCR expert is wired for totaltext/icdar2015 only')
        T = PpocrTask(task, work / 'ds')
        return T, None, T, PpocrTask('icdar2015' if task == 'totaltext' else 'totaltext', work / 'src'), work / 'ds'
    if task == 'ctw1500':
        # Official CTW1500 DeepSolo R50 candidates of study E11 (read-only), see secondlook.data.tasks_ctw1500
        from secondlook.data.tasks_ctw1500 import CtwTask
        T = CtwTask(work / 'ds', expert)   # E29: TiCLS (own fine-tune bank) and PP-OCR (cached lines); DeepSolo path unchanged
        if expert == 'deepsolo':
            src = Task('totaltext', work / 'src')
        elif expert == 'ticls':
            src = TiclsTask('totaltext', work / 'src')
        else:
            from secondlook.data.tasks_ppocr import PpocrTask
            src = PpocrTask('totaltext', work / 'src')
        return T, None, T, src, work / 'ds'
    if task.startswith('cohort_'):
        # TextOCR cohort scored under the base dataset's rules (DeepSolo only, no TiCLS outputs): the cohort is the TARGET pool,
        # the ranker is trained on the base dataset's labelled TRAIN ('own') or on the other dataset's TRAIN ('src').
        if expert != 'deepsolo':
            raise ValueError('Cohort tasks exist for DeepSolo only')
        base = task[len('cohort_'):]
        T = Task(task, work / 'ds')
        T.eval_ids, T.eval_tab, T.eval_names, T.eval_gt = T.ids, T.tab, T.names, T.pool_gt
        return T, None, Task(base, work / 'own'), Task('icdar2015' if base == 'totaltext' else 'totaltext', work / 'src'), work / 'ds'
    D, L = Task(task, work / 'ds'), TiclsTask(task, work / 'ti')
    other = 'icdar2015' if task == 'totaltext' else 'totaltext'
    Tsrc = Task(other, work / 'src') if expert == 'deepsolo' else TiclsTask(other, work / 'src')
    T, H = (D, L) if expert == 'deepsolo' else (L, D)
    return T, H, T, Tsrc, work / ('ds' if expert == 'deepsolo' else 'ti')


class Shim:
    pass


def test_labels(T, work):
    if hasattr(T, 'test_labels'):
        return T.test_labels()
    sh = Shim()
    sh.tab, sh.ids, sh.names, sh.pool_gt, sh.metric, sh.work = T.eval_tab, T.eval_ids, T.eval_names, T.eval_gt, T.metric, Path(work)
    Task.label_pool(sh)
    return sh.lab, sh.gt


def counts_for(T, work, keep):
    if hasattr(T, 'counts'):
        return np.asarray(T.counts(keep), np.int64)[:, :3]
    c = evaluate(None, work, 'tmp', T.eval_ids, T.eval_tab, {s: np.asarray(keep[s], bool).tolist() for s in T.eval_ids}, T.eval_gt, T.eval_names, T.metric)[0]
    for f in Path(work).glob('tmp.zip'):
        f.unlink()
    return np.asarray(c, np.int64)[:, :3]


def train_hgb(rows, weights=None):
    x = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    w = None if weights is None else np.concatenate([np.full(len(r[1]), wt) for r, wt in zip(rows, weights)])
    m = y >= 0
    model = hgb()
    model.fit(x[m], y[m]) if w is None else model.fit(x[m], y[m], sample_weight=w[m])
    return model


def train_rows(T, ids):
    return [(T.tab[s]['xc'], T.lab[s]) for s in ids]


def grid_stage(args):
    out = Path(args.out) / args.expert / args.task
    if (out / 'grid.npz').exists():
        raise FileExistsError('grid.npz already exists in ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    T, H, Town, Tsrc, work = load(args.task, args.expert, out / 'work')
    ids = T.eval_ids
    own, src = train_hgb(train_rows(Town, Town.ids)), train_hgb(train_rows(Tsrc, Tsrc.ids))
    grids = {}
    for name, model in (('own', own), ('src', src)):
        p = proba(model, T.eval_tab, ids, SPEC)
        grids[name] = np.stack([counts_for(T, work, {s: p[s] >= c for s in ids}) for c in GRID_CUTS])
    grids['thr'] = np.stack([counts_for(T, work, {s: (T.eval_tab[s]['score'] >= st) & (T.eval_tab[s]['rec'] >= rt) for s in ids}) for st, rt in THR])
    grids['author'] = counts_for(T, work, {s: ([True] * len(T.eval_tab[s]['lines']) if args.expert in ('ticls', 'ppocr') else T.eval_tab[s]['score'] >= AUTHOR['score_threshold']) for s in ids})
    np.savez_compressed(out / 'grid.npz', cuts=np.array(GRID_CUTS), thr_params=np.array(THR), ids=np.array([str(s) for s in ids]), **grids)
    import shutil
    shutil.rmtree(out / 'work', ignore_errors=True)
    print('E18 grid', args.expert, args.task, 'author F1 %.2f own042 %.2f src042 %.2f' % tuple(f1(c) for c in (grids['author'], grids['own'][GRID_CUTS.index(0.42)], grids['src'][GRID_CUTS.index(0.42)])), flush=True)


def f1(c):
    return 200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())


def cal_f1(g, cal):
    return 200.0 * g[:, cal, 0].sum(1) / np.maximum((g[:, cal, 1] + g[:, cal, 2]).sum(1), 1)


def reps_stage(args):
    out = Path(args.out) / args.expert / args.task
    G = np.load(out / 'grid.npz')
    T, H, Town, Tsrc, work = load(args.task, args.expert, out / ('work_%d_%d' % (args.rep_from, args.rep_to)))
    ids = T.eval_ids
    n = len(ids)
    if H is not None:
        agreement(T.eval_tab, H.eval_tab, ids)
    lab, gt = test_labels(T, work)
    own = train_rows(Town, Town.ids)
    srcrows = train_rows(Tsrc, Tsrc.ids)
    gtc = G['author'][:, 2]
    abstain = np.stack([np.zeros(n, np.int64), np.zeros(n, np.int64), gtc], 1)
    cuts = list(G['cuts'])
    arms, perms = {}, []

    def put(name, r, c):
        arms.setdefault(name, {})[r] = np.asarray(c, np.int16)

    for r in range(args.rep_from, args.rep_to):
        perm = np.random.RandomState(5000 + r).permutation(n)
        perms.append(perm)
        put('author', r, G['author'])
        put('own042', r, G['own'][cuts.index(0.42)])
        put('src042', r, G['src'][cuts.index(0.42)])
        rng = np.random.RandomState(9000 + r)
        for N in NS:
            cal = perm[:N]
            put('thr_%d' % N, r, G['thr'][int(np.argmax(cal_f1(G['thr'], cal)))])
            put('own_cut_%d' % N, r, G['own'][int(np.argmax(cal_f1(G['own'], cal)))])
            put('src_cut_%d' % N, r, G['src'][int(np.argmax(cal_f1(G['src'], cal)))])
            idx = rng.randint(0, N, (300, N))
            for gname in ('own', 'thr'):
                g = G[gname]
                tp, pr = g[:, cal, 0], g[:, cal, 1]
                plug = 1 - tp.sum(1) / np.maximum(pr.sum(1), 1)
                ucb = np.percentile(1 - tp[:, idx].sum(2) / np.maximum(pr[:, idx].sum(2), 1), 95, axis=1)
                f1c = cal_f1(g, cal)
                for alpha in ALPHAS:
                    for rule, crit in (('plug', plug), ('ucb', ucb)):
                        ok = np.where((crit <= alpha) & (pr.sum(1) > 0))[0]
                        if gname == 'own':
                            pick = ok.min() if len(ok) else None
                        else:
                            pick = ok[np.argmax(f1c[ok])] if len(ok) else None
                        put('%s_%s_%d_%g' % ('model' if gname == 'own' else 'thr', rule, N, alpha), r, abstain if pick is None else g[pick])
        # retrained arms
        for N in MIX_NS:
            cal = [ids[i] for i in perm[:N]]
            rows = [(T.eval_tab[s]['xc'], lab[s]) for s in cal]
            for name, base in (('dmix', srcrows), ('ownmix', own)):
                m = train_hgb(base + rows, [1.0] * len(base) + [5.0] * len(rows))
                p = proba(m, T.eval_tab, ids, SPEC)
                put('%s_%d' % (name, N), r, counts_for(T, work, {s: p[s] >= 0.42 for s in ids}))
            m = train_hgb(rows)
            p = proba(m, T.eval_tab, ids, SPEC)
            put('scratch_%d' % N, r, counts_for(T, work, {s: p[s] >= 0.42 for s in ids}))
        for Nu in (A_NS if H is not None else ()):
            cal = [ids[i] for i in perm[:Nu]]
            x = np.concatenate([T.eval_tab[s]['xc'] for s in cal])
            y = np.concatenate([(T.eval_tab[s]['ag'][:, 1] == 1) for s in cal]).astype(int)
            m = hgb()
            m.fit(x, y)
            pu = np.concatenate([m.predict_proba(T.eval_tab[s]['xc'])[:, 1] if len(T.eval_tab[s]['xc']) else np.zeros(0) for s in cal])
            c_hat = float(np.clip(y[pu >= np.quantile(pu, 0.9)].mean(), 0.3, 1.0))
            p = proba(m, T.eval_tab, ids, SPEC)
            put('A_%d' % Nu, r, counts_for(T, work, {s: np.clip(p[s] / c_hat, 0, 1) >= 0.42 for s in ids}))
        print('E18 rep', args.expert, args.task, r, flush=True)
    reps = list(range(args.rep_from, args.rep_to))
    np.savez_compressed(out / ('reps_%d_%d.npz' % (args.rep_from, args.rep_to)), perms=np.array(perms), reps=np.array(reps),
                        **{a: np.stack([v[r] for r in reps]) for a, v in arms.items()})
    import shutil
    shutil.rmtree(out / ('work_%d_%d' % (args.rep_from, args.rep_to)), ignore_errors=True)


def merge(out):
    parts = sorted(out.glob('reps_*.npz'), key=lambda p: int(p.stem.split('_')[1]))
    data = [np.load(p) for p in parts]
    arms = {k: np.concatenate([d[k] for d in data]) for k in data[0].files if k not in ('perms', 'reps')}
    perms = np.concatenate([d['perms'] for d in data])
    extra = out / 'extra_arms.npz'
    if extra.exists():
        ex = np.load(extra)
        for k in ex.files:
            arms[k] = ex[k] if ex[k].ndim == 3 else np.broadcast_to(ex[k][None], (len(perms),) + ex[k].shape).copy()
    return arms, perms


def report(args):
    out = Path(args.out)
    rng = np.random.RandomState(20261110)
    summary = {}
    for task in args.tasks:
        for expert in ('deepsolo', 'ticls', 'ppocr'):
            if not (out / expert / task / 'grid.npz').exists():
                continue
            arms, perms = merge(out / expert / task)
            R, n = perms.shape
            e = np.zeros((R, n), bool)
            for r in range(R):
                e[r, perms[r, NMAX:]] = True
            gt = arms['author'][0, :, 2].astype(float)
            B = args.draws
            M = rng.multinomial(n, [1.0 / n] * n, size=B).astype(float)

            def boot(a):
                c = arms[a].astype(float)
                num = M @ (e * c[:, :, 0]).T
                den = M @ (e * (c[:, :, 1] + gt[None, :])).T
                return 200.0 * num / np.maximum(den, 1)
            def point(a):
                c = arms[a].astype(float)
                f = 200.0 * (e * c[:, :, 0]).sum(1) / np.maximum((e * (c[:, :, 1] + gt[None, :])).sum(1), 1)
                tp, pr = (e * c[:, :, 0]).sum(1), (e * c[:, :, 1]).sum(1)
                fdr = np.where(pr > 0, 100 * (1 - tp / np.maximum(pr, 1)), 0.0)
                return f, fdr
            res = {}
            for a in arms:
                f, fdr = point(a)
                row = dict(f1=float(f.mean()), fdr=float(fdr.mean()))
                refs = ['author']
                suf = a.split('_')[-1]
                if a.startswith('model_') and a.count('_') == 3:
                    refs.append('thr_%s' % a.split('_', 1)[1])
                    refs.append('thr_%s_%s' % (a.split('_')[2], 'x'))
                if suf.isdigit():
                    refs.append('thr_%s' % suf)
                if 'thr_train' in arms and a != 'thr_train' and a.startswith(('own', 'src', 'dmix', 'ownmix', 'A_', 'model_ucb')):
                    refs.append('thr_train')
                if a.startswith('A_') or a.startswith('dmix') or a.startswith('ownmix') or a.startswith('own_cut') or a.startswith('src_cut'):
                    refs += ['thr_%s' % suf]
                for ref in dict.fromkeys(refs):
                    if ref not in arms or ref == a:
                        continue
                    fr, _ = point(ref)
                    d = (boot(a) - boot(ref)).mean(1)
                    row['vs_' + ref] = dict(delta=float((f - fr).mean()), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], repeats_above=int(((f - fr) > 0).sum()))
                if a.startswith(('model_', 'thr_')) and a.count('_') == 3:
                    alpha = 100 * float(a.split('_')[-1])
                    row['coverage'] = float((fdr <= alpha).mean())
                    row['abstain'] = float((arms[a][:, :, 1].sum(1) == 0).mean())
                res[a] = row
            summary['%s/%s' % (expert, task)] = res
    write_json(out / 'summary.json', summary, exclusive=False)
    for key, res in summary.items():
        print('==', key)
        for a, row in res.items():
            ex = ''.join(' | %s %+.2f [%+.2f,%+.2f] %d/%d' % (k[3:], v['delta'], *v['ci'], v['repeats_above'], args.reps) for k, v in row.items() if k.startswith('vs_'))
            cov = (' | FDR<=alpha in %.0f%% (abstain %.0f%%)' % (100 * row['coverage'], 100 * row['abstain'])) if 'coverage' in row else ''
            print('  %-24s F1 %.2f FDR %.1f%s%s' % (a, row['f1'], row['fdr'], ex, cov))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--stage', choices=['grid', 'reps', 'report'], required=True)
    p.add_argument('--expert', choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', choices=['totaltext', 'icdar2015', 'cohort_totaltext', 'cohort_icdar2015', 'ctw1500'])
    p.add_argument('--tasks', nargs='+', default=['totaltext', 'icdar2015'])
    p.add_argument('--rep-from', dest='rep_from', type=int, default=0)
    p.add_argument('--rep-to', dest='rep_to', type=int, default=5)
    p.add_argument('--reps', type=int, default=20)
    p.add_argument('--draws', type=int, default=500)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    {'grid': grid_stage, 'reps': reps_stage, 'report': report}[a.stage](a)
