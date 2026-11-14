"""E10: signal ablation and labelled-data learning curve for the DeepSolo word-admission head.

Legacy Python 3.8 (sklearn 0.23.2). CPU only, cached DeepSolo outputs, no neural call.
Per job (task, seed): one partition as in E7/E8; cut / threshold selection with the label-based pooled
F1 approximation F1 = 2*TP / (P + G) on the selection images (validated against the exact evaluator on
seed 0); EVERY reported F1 on the evaluation target uses the exact pinned evaluator.
Tasks: totaltext / icdar2015 (evaluation = official TEST) and cohort_{totaltext,icdar2015,ctw1500_word}
(evaluation = held-out 128 of the TextOCR cohort split). Approximate reproduction.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from secondlook.evaluation.protocol import load_config, read_json, write_json, sha256, hmean_fraction
from secondlook.data.deepsolo_table import candidate_table, NAMES, CUTS, SCORE_GRID, REC_GRID, AUTHOR, QUALIFIED, DS, AUDIT, CFG
from secondlook.data.cohort import Ctx, evaluate, f1_of, COHORT, TASKS, drop_invalid_quads, subset_gt_zip
from secondlook.paths import RUNS

RUN = (RUNS / 'deepsolo-admission-base')
# Per-expert native feature groups (columns of the native 'x' matrix).
DS_GROUPS = {'score': [0], 'rec': [1], 'ctc': [2, 3, 4, 5, 6, 7, 8], 'geometry': [9, 10, 11], 'text': [12, 13, 14, 15]}
TI_GROUPS = {'score': [0], 'lm': [4, 5, 6, 7, 8, 9], 'geometry': [1, 2, 3], 'text': [10, 11, 12]}
# Common interface (same 11 definitions, same order, for both experts), matrix 'xc':
# 0 detector logit | 1 geometric-mean token prob | 2 min token prob | 3 mean token entropy | 4 min token margin |
# 5 log relative area | 6 log height | 7 log aspect | 8 log1p transcript length | 9 digit fraction | 10 letter fraction
COMMON_NAMES = ['detector_logit', 'geo_mean_token_prob', 'min_token_prob', 'mean_token_entropy', 'min_token_margin', 'log_rel_area', 'log_height', 'log_aspect', 'log_len', 'digit_frac', 'letter_frac']
COMMON = {'common5': list(range(5)), 'common8': list(range(8)), 'common11': list(range(11))}
BUDGETS = [30, 60, 120, 240, 480]


def make_variants(groups):
    allcols = sorted(c for cols in groups.values() for c in cols)
    v = {'full': ('x', allcols)}
    for g in groups:
        v['no_' + g] = ('x', [c for h, cols in groups.items() if h != g for c in cols])
        v['only_' + g] = ('x', list(groups[g]))
    if 'rec' in groups:
        v['score_rec'] = ('x', groups['score'] + groups['rec'])
        for g in ['ctc', 'geometry', 'text']:
            v['score_rec_' + g] = ('x', groups['score'] + groups['rec'] + groups[g])
    else:
        v['score_lm'] = ('x', groups['score'] + groups['lm'])
        for g in ['geometry', 'text']:
            v['score_lm_' + g] = ('x', groups['score'] + groups['lm'] + groups[g])
    for name, cols in COMMON.items():
        v[name] = ('xc', cols)
    return v


def add_common(tab, kind):
    """Build the shared-interface matrix 'xc' from each expert's native features."""
    for s, t in tab.items():
        x = t['x']
        if kind == 'deepsolo':
            sc = np.clip(x[:, 0], 1e-6, 1 - 1e-6)
            xc = np.stack([np.log(sc / (1 - sc)), np.exp(x[:, 3]), x[:, 2], x[:, 7], x[:, 4], x[:, 9], x[:, 10], x[:, 11], np.log1p(x[:, 12]), x[:, 13], x[:, 14]], 1)
        else:
            xc = x[:, [0, 5, 6, 7, 8, 1, 2, 3, 10, 11, 12]]
        t['xc'] = xc.reshape(-1, 11)
    return tab


class Task:
    """DeepSolo task (cached compact outputs)."""
    kind = 'deepsolo'

    def __init__(self, task, work):
        self.task = task
        self.work = Path(work)
        cfg = load_config(Path(CFG))
        self.cfg = cfg
        self.variants = make_variants(DS_GROUPS)
        self.aug = {}
        if task in ('totaltext', 'icdar2015'):
            ctx = Ctx(task)
            self.metric = ctx.metric
            train = read_json(DS / 'compact_deepsolo' / (task + '_train.json'))['samples']
            test = read_json(DS / 'compact_deepsolo' / (task + '_test.json'))['samples']
            self.ids = ctx.train_ids
            self.tab = candidate_table(train, self.ids, task)
            self.names = ctx.names
            self.pool_gt = ctx.base / 'native/source_gt.zip'
            self.official = (ctx.parts['fit_ids'], ctx.parts['selection_ids'], ctx.parts['calibration_ids'])
            self.sizes = (584, 80)
            self.eval_ids = sorted(test, key=lambda s: int(s.split(':')[-1]))
            self.eval_tab = add_common(candidate_table(test, self.eval_ids, task), 'deepsolo')
            self.eval_names = eval_names(cfg, task, self.eval_ids)
            self.eval_gt = AUDIT / ('gt_%s.zip' % task)
            self.holdout = False
        else:
            name = task[len('cohort_'):]
            cohort_name, exp_name = TASKS[name]
            dataset = 'icdar2015' if name == 'icdar2015' else 'totaltext'
            from secondlook.evaluation.declared_metric import DeclaredMetric
            self.metric = DeclaredMetric(cfg, cfg['protocol_calibration']['experiments'][exp_name]['metric'])
            records = read_json(COHORT / name / 'rgb.json')['records']
            self.ids = [r['sample_id'] for r in records]
            samples = read_json(RUN / 'cohort' / (cohort_name + '_cohort.json'))['samples']
            tab = candidate_table(samples, self.ids, dataset)
            self.tab = drop_invalid_quads(tab, self.ids) if dataset == 'icdar2015' else tab
            self.names = read_json(COHORT / name / 'id_map.json')
            self.pool_gt = COHORT / name / 'gt.zip'
            roles = read_json(COHORT / name / 'roles.json')
            self.official = (roles['fit'], roles['selection'], roles['audit'])
            self.sizes = (512, 128)
            self.holdout = True
        add_common(self.tab, 'deepsolo')
        self.label_pool()

    def label_pool(self):
        from secondlook.evaluation.declared_metric import counter_key, word_labels
        everything = {s: [True] * len(self.tab[s]['lines']) for s in self.ids}
        counts, native, counters = evaluate(None, self.work, 'pool_all', self.ids, self.tab, everything, self.pool_gt, self.names, self.metric)
        self.gt = {s: int(counts[i, 2]) for i, s in enumerate(self.ids)}
        self.lab = {}
        for s in self.ids:
            key = counter_key(self.names[s], self.metric.key_mode)
            sample = native['per_sample'][key]
            n = len(self.tab[s]['lines'])
            if n != len(sample['detTrans']):
                raise ValueError('Label identity mismatch')
            self.lab[s] = word_labels(list(range(n)), counters[key]['e2e_correct_indices'], sample['detDontCare'], n)

    def partition(self, seed):
        if seed == 0:
            return self.official
        order = [self.ids[i] for i in np.random.RandomState(seed).permutation(len(self.ids))]
        a, b = self.sizes
        return order[:a], order[a:a + b], order[a + b:]

    def approx_f1(self, ids, keep):
        tp = p = 0
        for s in ids:
            k = np.asarray(keep[s], bool)
            lab = self.lab[s]
            tp += int(((lab == 1) & k).sum())
            p += int(((lab >= 0) & k).sum())
        g = sum(self.gt[s] for s in ids)
        return 2 * tp / (p + g) if p + g else 0.0


def eval_names(cfg, task, ids):
    if task == 'totaltext':
        router = read_json(cfg['baseline_config'])
        mp = read_json(read_json(router['benchmark_audit_config'])['datasets']['totaltext']['spotting_id_map'])
        return {s: mp[s] for s in ids}
    return {s: 'res_img_' + s.split(':')[-1] + '.txt' for s in ids}


class TiclsTask(Task):
    """TiCLS task: source candidates from the prepared protocol-calibration native view, TEST candidates from the
    matched-benchmark journal (native_word action, author-default candidates, 13 real features)."""
    kind = 'ticls'

    def __init__(self, task, work):
        self.task = task
        self.work = Path(work)
        cfg = load_config(Path(CFG))
        self.cfg = cfg
        self.variants = make_variants(TI_GROUPS)
        ctx = Ctx(task)
        self.metric = ctx.metric
        base = ctx.base
        a = np.load(base / 'low/words.npz')
        bodies = read_json(base / 'low/bodies.json')
        ids = a['ids'].tolist()
        self.ids = ctx.train_ids
        self.tab, self.lab, self.gt = {}, {}, {}
        for i, s in enumerate(ids):
            w = bodies[s]['word_indices']
            x = a['features'][w].astype(np.float64)
            self.tab[s] = dict(lines=list(bodies[s]['lines']), x=x, score=a['detector_score'][w].astype(np.float64), rec=x[:, 6].copy())
            self.lab[s] = a['target'][w]
            self.gt[s] = int(a['counts'][i, 2])
        if set(self.tab) != set(self.ids):
            raise ValueError('TiCLS source population mismatch')
        aug = np.load(base / 'augmented/words.npz')
        self.aug = {}
        for j, s in enumerate(aug['ids'].tolist()):
            m = aug['image_index'] == j
            self.aug[s] = (aug['features'][m].astype(np.float64), aug['target'][m])
        add_common(self.tab, 'ticls')
        for s in self.aug:
            x, y = self.aug[s]
            xc = x[:, [0, 5, 6, 7, 8, 1, 2, 3, 10, 11, 12]]
            self.aug[s] = (x, xc, y)
        self.names = ctx.names
        self.pool_gt = base / 'native/source_gt.zip'
        self.official = (ctx.parts['fit_ids'], ctx.parts['selection_ids'], ctx.parts['calibration_ids'])
        self.sizes = (584, 80)
        self.holdout = False
        journal = (RUNS / 'ticls-source-banks/matched_benchmark_v2') / task / 'per_image.jsonl'
        rows = {}
        for line in journal.open():
            d = json.loads(line)
            if d['method'] == 'native_word':
                e = d['word_evidence'][0]
                x = np.asarray(e['features'], np.float64).reshape(-1, 13)
                rows[d['sample_id']] = dict(lines=list(e['lines']), x=x, score=np.asarray(e['detector_scores'], np.float64), rec=x[:, 6].copy())
        self.eval_ids = sorted(rows, key=lambda s: int(s.split(':')[-1]))
        self.eval_tab = add_common(rows, 'ticls')
        self.eval_names = eval_names(cfg, task, self.eval_ids)
        self.eval_gt = AUDIT / ('gt_%s.zip' % task)


def stack(T, ids, spec):
    key, cols = spec
    xs, ys = [], []
    for s in ids:
        x, y = T.tab[s][key][:, cols], T.lab[s]
        xs.append(x)
        ys.append(y)
        if s in T.aug:
            ax, axc, ay = T.aug[s]
            xs.append((ax if key == 'x' else axc)[:, cols])
            ys.append(ay)
    x, y = np.concatenate(xs), np.concatenate(ys)
    m = y >= 0
    return x[m], y[m]


def make_model(kind, cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from secondlook.features.word_model import new_word_model
    if kind == 'hgb':
        return new_word_model('hgb_x', cfg)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=11))


def prob(model, tab, s, spec):
    key, cols = spec
    return model.predict_proba(tab[s][key][:, cols])[:, 1] if len(tab[s]['lines']) else np.zeros(0)


def pick_cut(T, model, spec, sel_ids):
    p = {s: prob(model, T.tab, s, spec) for s in sel_ids}
    best = max((T.approx_f1(sel_ids, {s: p[s] >= c for s in sel_ids}), -c, c) for c in CUTS)
    return best[2], best[0]


def pick_thresholds(T, ids, rec_grid=None):
    rows = []
    for st in SCORE_GRID:
        for rt in (REC_GRID if rec_grid is None else rec_grid):
            keep = {s: (T.tab[s]['score'] >= st) & (T.tab[s]['rec'] >= rt) for s in ids}
            rows.append((T.approx_f1(ids, keep), st, rt))
    f, st, rt = max(rows)
    return st, rt, f


def job(args):
    from sklearn.metrics import roc_auc_score
    out = Path(args.out) / args.task / ('seed%d' % args.seed)
    out.mkdir(parents=True, exist_ok=False)
    work = out / 'work'
    work.mkdir()
    T = (TiclsTask if args.model == 'ticls' else Task)(args.task, work)
    cfg = T.cfg
    fit, sel, aud = T.partition(args.seed)
    if T.holdout:
        tab, ids, names = T.tab, aud, T.names
        gt = work / 'aud_gt.zip'
        from secondlook.features.word_trust import subset_gt
        subset_gt(T.pool_gt, gt, T.metric.module, {s: T.names[s] for s in aud}, 'x')
    else:
        tab, ids, names, gt = T.eval_tab, T.eval_ids, T.eval_names, T.eval_gt

    def exact(tag, keep):
        c = evaluate(None, work, tag, ids, tab, keep, gt, names, T.metric)[0]
        for f in work.glob(tag + '.zip'):
            f.unlink()
        return 100 * f1_of(c)

    def keep_model(model, spec, cut):
        return {s: (prob(model, tab, s, spec) >= cut).tolist() for s in ids}

    def keep_thr(st, rt):
        return {s: ((tab[s]['score'] >= st) & (tab[s]['rec'] >= rt)).tolist() for s in ids}
    full = T.variants['full']
    res = dict(model=args.model, task=args.task, seed=args.seed, fit=len(fit), selection=len(sel), eval_images=len(ids),
               expert_images='%d/%d (100%%)' % (len(ids), len(ids)), other_expert_images='0/%d (0%%)' % len(ids))
    res['author_default'] = exact('author', {s: [True] * len(tab[s]['lines']) for s in ids} if args.model == 'ticls' else {s: (tab[s]['score'] >= AUTHOR['score_threshold']).tolist() for s in ids})
    st, rt, fs = pick_thresholds(T, sel, [0.0])
    res['tuned_score_only_selection_only'] = dict(f1=exact('tuned1', keep_thr(st, rt)), score=st, selection_f1_approx=fs)
    st, rt, fs = pick_thresholds(T, sel)
    res['tuned_thresholds_selection_only'] = dict(f1=exact('tuned2', keep_thr(st, rt)), score=st, rec=rt, selection_f1_approx=fs)
    ab = {}
    for name, spec in T.variants.items():
        x, y = stack(T, fit, spec)
        model = make_model('hgb', cfg)
        model.fit(x, y)
        cut, fsel = pick_cut(T, model, spec, sel)
        ab[name] = dict(f1=exact('ab_' + name, keep_model(model, spec, cut)), cut=cut, selection_f1_approx=fsel)
        if name == 'full' and args.seed == 0:
            sel_gt = work / 'sel_gt.zip'
            from secondlook.features.word_trust import subset_gt
            subset_gt(T.pool_gt, sel_gt, T.metric.module, {s: T.names[s] for s in sel}, 'x')
            diffs = []
            for c in (0.3, 0.5, 0.7):
                keep = {s: (prob(model, T.tab, s, spec) >= c) for s in sel}
                ex = f1_of(evaluate(None, work, 'chk_%s' % c, sel, T.tab, {s: keep[s].tolist() for s in sel}, sel_gt, T.names, T.metric)[0])
                diffs.append(abs(ex - T.approx_f1(sel, keep)))
            res['approximation_max_abs_diff_selection'] = float(max(diffs))
    res['ablation'] = ab
    x, y = stack(T, fit, full)
    lg = make_model('logistic', cfg)
    lg.fit(x, y)
    cut, _ = pick_cut(T, lg, full, sel)
    res['logistic_full'] = dict(f1=exact('logistic', keep_model(lg, full, cut)), cut=cut)
    if args.seed == 0:
        names_ = NAMES if args.model == 'deepsolo' else ['detector_logit', 'log_relative_area', 'log_native_height', 'log_aspect', 'lm_log_length', 'lm_geometric_probability', 'lm_min_probability', 'lm_entropy', 'lm_min_margin', 'lm_finished', 'log_transcript_length', 'digit_fraction', 'letter_fraction']
        xa, ya = stack(T, fit, full)
        res['feature_auc'] = {names_[j]: float(roc_auc_score(ya, xa[:, j])) for j in range(len(names_))}
        res['fit_positive_rate'] = float(ya.mean())
        res['fit_words'] = int(len(ya))
    pool = list(fit) + list(sel)
    perm = [pool[i] for i in np.random.RandomState(1000 + args.seed).permutation(len(pool))]
    lc = {}
    for n in BUDGETS:
        sub = perm[:n]
        ns = max(10, int(round(0.12 * n)))
        sub_fit, sub_sel = sub[ns:], sub[:ns]
        x, y = stack(T, sub_fit, full)
        if len(y) < 20 or len(set(y.tolist())) < 2:
            continue
        h = make_model('hgb', cfg)
        h.fit(x, y)
        ch, _ = pick_cut(T, h, full, sub_sel)
        l = make_model('logistic', cfg)
        l.fit(x, y)
        cl, _ = pick_cut(T, l, full, sub_sel)
        st, rt, _ = pick_thresholds(T, sub)
        lc[str(n)] = dict(hgb=exact('lc_h', keep_model(h, full, ch)), logistic=exact('lc_l', keep_model(l, full, cl)), tuned_thresholds_all_n=exact('lc_t', keep_thr(st, rt)), fit_images=len(sub_fit), selection_images=len(sub_sel))
    st, rt, _ = pick_thresholds(T, pool)
    lc['all'] = dict(hgb=ab['full']['f1'], logistic=res['logistic_full']['f1'], tuned_thresholds_all_n=exact('lc_tall', keep_thr(st, rt)), fit_images=len(fit), selection_images=len(sel))
    res['learning_curve'] = lc
    write_json(out / 'results.json', res)
    shutil.rmtree(work, ignore_errors=True)
    print('E10', args.model, args.task, 'seed', args.seed, 'full %.2f common11 %.2f tuned2 %.2f author %.2f' % (ab['full']['f1'], ab['common11']['f1'], res['tuned_thresholds_selection_only']['f1'], res['author_default']), flush=True)


def report(args):
    out = Path(args.out)
    summary = {}
    for task in args.tasks:
        R = {s: read_json(out / task / ('seed%d' % s) / 'results.json') for s in args.seeds}
        arr = lambda f: np.array([f(R[s]) for s in args.seeds])
        t = {}
        full = arr(lambda r: r['ablation']['full']['f1'])
        tuned = arr(lambda r: r['tuned_thresholds_selection_only']['f1'])
        tuned1 = arr(lambda r: r['tuned_score_only_selection_only']['f1'])
        auth = arr(lambda r: r['author_default'])
        t['references'] = dict(author_default=float(auth.mean()), tuned_score_only=float(tuned1.mean()), tuned_thresholds=float(tuned.mean()), logistic_full=float(arr(lambda r: r['logistic_full']['f1']).mean()))
        t['ablation'] = {}
        for v in R[args.seeds[0]]['ablation']:
            f = arr(lambda r: r['ablation'][v]['f1'])
            t['ablation'][v] = dict(mean_f1=float(f.mean()), minus_full=float((f - full).mean()), minus_tuned=float((f - tuned).mean()), seeds_above_tuned=int((f > tuned).sum()), seeds_below_full=int((f < full).sum()))
        t['feature_auc_seed0'] = R[args.seeds[0]].get('feature_auc')
        t['approx_check_seed0'] = R[args.seeds[0]].get('approximation_max_abs_diff_selection')
        t['learning_curve'] = {}
        for n in [str(b) for b in BUDGETS] + ['all']:
            rows = [R[s]['learning_curve'][n] for s in args.seeds if n in R[s]['learning_curve']]
            if not rows:
                continue
            h = np.array([r['hgb'] for r in rows])
            l = np.array([r['logistic'] for r in rows])
            b = np.array([r['tuned_thresholds_all_n'] for r in rows])
            t['learning_curve'][n] = dict(runs=len(rows), hgb=float(h.mean()), logistic=float(l.mean()), tuned_thresholds=float(b.mean()), hgb_minus_tuned=float((h - b).mean()), runs_hgb_above_tuned=int((h > b).sum()), logistic_minus_tuned=float((l - b).mean()))
        summary[task] = t
    write_json(out / ('summary_%s.json' % args.model), summary, exclusive=False)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--stage', choices=['job', 'report'], required=True)
    p.add_argument('--model', choices=['deepsolo', 'ticls'], default='deepsolo')
    p.add_argument('--task', choices=['totaltext', 'icdar2015', 'cohort_totaltext', 'cohort_icdar2015', 'cohort_ctw1500_word'])
    p.add_argument('--tasks', nargs='+', default=['totaltext', 'icdar2015', 'cohort_totaltext', 'cohort_icdar2015', 'cohort_ctw1500_word'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(11)))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    job(a) if a.stage == 'job' else report(a)
