"""E46 (approximate reproduction): does the label-based F1 approximation 2TP/(P+G) change the (st, rt) pair of the re-tuned baseline `thr_train` that the EXACT pinned evaluator
would choose on the TRAIN pool? And what does one exact TRAIN evaluation cost? CPU only, nothing trained.
    python -m secondlook.studies.fulltrain_launch secondlook.selection.exact_selection --expert E --task T --out DIR [--top K | --full]
For each recognition statistic of the baseline (old `rec` of the row, and rec = geometric mean = column 1 of the shared interface) and over the 135-pair grid of
secondlook.data.deepsolo_table (SCORE_GRID x REC_GRID): approximate F1 of every pair on the whole TRAIN pool (`Task.approx_f1`, as in `pick_thresholds`); the exact F1 (2 correct /
(predictions + ground truth) from the pinned evaluator, TRAIN GT) of the top-K pairs by approximate F1 (--full: all 135); the arg max of each; and the exact TEST F1 of the pair chosen
by the approximation and of the pair chosen by the exact evaluator. Wall-clock of every call is stored."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from secondlook.selection import timing
from secondlook.data.deepsolo_table import REC_GRID, SCORE_GRID
from secondlook.evaluation.common import FIELDS
from secondlook.selection.operating_point import load


def exact_train_counts(T, work, keep):
    """[correct, predictions, ground truth] summed over the TRAIN pool with the pinned evaluator (the same call that produced the TRAIN labels)."""
    if hasattr(T, 'off_tr'):   # CTW1500
        from secondlook.spotters.deepsolo_ctw1500 import entries_of
        _, counters, _ = T.off_tr.run(entries_of(T.tab, T.ids, {i: list(map(bool, keep[i])) for i in T.ids}), None)
        c = np.asarray([[counters['%07d' % int(i)][f] for f in FIELDS[:3]] for i in T.ids], np.int64)
    else:
        from secondlook.data.tasks import evaluate
        c = np.asarray(evaluate(None, work, 'e46_train', T.ids, T.tab, {s: np.asarray(keep[s], bool).tolist() for s in T.ids}, T.pool_gt, T.names, T.metric)[0], np.int64)[:, :3]
        for f in Path(work).glob('e46_train.zip'):
            f.unlink()
    return c


def f1(c):
    return 200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())


def main(a):
    e, t = a.expert, a.task
    out = Path(a.out) / ('%s_%s' % (e, t))
    out.mkdir(parents=True, exist_ok=True)
    work = out / 'work'
    work.mkdir(exist_ok=True)
    tm = timing.Timer()
    counts_for = timing.install_exact_timer(tm, e, t)
    T = load(e, t, work)
    ids = list(T.ids)
    res = dict(expert=e, task=t, claim_label='approximate reproduction', train_images=len(ids), grid=dict(score=SCORE_GRID, rec=REC_GRID))
    for rec_name in ('old_rec', 'geometric_mean'):
        rec = {s: (T.tab[s]['rec'] if rec_name == 'old_rec' else T.tab[s]['xc'][:, 1]) for s in ids}
        rec_te = {s: (T.eval_tab[s]['rec'] if rec_name == 'old_rec' else T.eval_tab[s]['xc'][:, 1]) for s in T.eval_ids}
        rows = []
        with tm('approx_f1_all_135_pairs_%s' % rec_name):
            for st in SCORE_GRID:
                for rt in REC_GRID:
                    keep = {s: (T.tab[s]['score'] >= st) & (rec[s] >= rt) for s in ids}
                    rows.append(dict(st=st, rt=rt, approx_f1=100.0 * T.approx_f1(ids, keep)))
        order = sorted(range(len(rows)), key=lambda i: (-rows[i]['approx_f1'], -rows[i]['st'], -rows[i]['rt']))   # ties as pick_thresholds: max over (f, st, rt)
        chosen_approx = rows[order[0]]
        todo = order if a.full else order[:a.top]
        for i in todo:
            r = rows[i]
            keep = {s: (T.tab[s]['score'] >= r['st']) & (rec[s] >= r['rt']) for s in ids}
            with tm('exact_train_evaluation_%s' % rec_name, n_images=len(ids)):
                r['exact_train_f1'] = f1(exact_train_counts(T, work, keep))
        evald = [rows[i] for i in todo]
        chosen_exact = max(evald, key=lambda r: (r['exact_train_f1'], r['st'], r['rt']))

        def test_f1(r):
            keep = {s: (T.eval_tab[s]['score'] >= r['st']) & (rec_te[s] >= r['rt']) for s in T.eval_ids}
            return f1(counts_for(T, work, keep))
        res[rec_name] = dict(pairs=rows, evaluated_exactly=len(evald), chosen_by_approximation=dict(chosen_approx, test_f1=test_f1(chosen_approx)),
                             chosen_by_exact_among_evaluated=dict(chosen_exact, test_f1=test_f1(chosen_exact)),
                             argmax_changes=(chosen_approx['st'], chosen_approx['rt']) != (chosen_exact['st'], chosen_exact['rt']),
                             max_abs_gap_approx_vs_exact_train_f1=max(abs(r['approx_f1'] - r['exact_train_f1']) for r in evald))
        print(e, t, rec_name, 'approx pair', chosen_approx['st'], chosen_approx['rt'], 'exact pair', chosen_exact['st'], chosen_exact['rt'],
              'changes', res[rec_name]['argmax_changes'], 'max gap %.3f' % res[rec_name]['max_abs_gap_approx_vs_exact_train_f1'], flush=True)
    res['timing'] = tm.summary()
    (out / ('exact_selection_%s.json' % ('full' if a.full else 'top%d' % a.top))).write_text(json.dumps(res, indent=1, default=float))
    print('done', e, t, json.dumps({k: v for k, v in res['timing'].items() if 'exact' in k or 'approx' in k}, default=float)[:600], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', required=True, choices=['totaltext', 'icdar2015', 'ctw1500'])
    p.add_argument('--out', required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--top', type=int, default=10)
    g.add_argument('--full', action='store_true')
    main(p.parse_args())
