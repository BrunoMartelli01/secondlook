"""E44 shared helpers (approximate reproduction). Detection-only counters from the SAME pinned upstream evaluator that scores the end-to-end F1 of every row.

`counts_full(T, work, keep)` is secondlook.data.cv_counts.counts_for without the truncation to the first three (end-to-end) columns: it returns the 7 columns
FIELDS = e2e_correct, e2e_predictions, e2e_gt_words, e2e_localized, correct, predictions, gt_words (columns 4-6 are the evaluator's det-only branch: IoU 0.5 between
polygons, transcription ignored, don't-care regions handled by the evaluator; the det-only aggregate is checked against the evaluator's own printed hmean to 1e-12 by
secondlook.evaluation.detection_counters). Nothing in the evaluator is modified.
"""
import json
from pathlib import Path

import numpy as np
from secondlook.paths import RUNS

FIELDS = ['e2e_correct', 'e2e_predictions', 'e2e_gt_words', 'e2e_localized', 'correct', 'predictions', 'gt_words']
RUNS = RUNS
E42 = RUNS / 'fulltrain'
CLEAN = E42 / 'readers/clean'
E18 = RUNS / 'candidate-banks'
OUT = RUNS / 'detection-crossdata'
ROWS9 = [('deepsolo', 'totaltext'), ('deepsolo', 'icdar2015'), ('ticls', 'totaltext'), ('ticls', 'icdar2015'), ('ppocr', 'totaltext'), ('ppocr', 'icdar2015'),
         ('deepsolo', 'ctw1500'), ('ticls', 'ctw1500'), ('ppocr', 'ctw1500')]
BOOT, SEED = 2000, 20261106
TEST_N = {'totaltext': 300, 'icdar2015': 500, 'ctw1500': 500}


def stability_path(e, t):
    k = '%s_%s' % (e, t)
    if t == 'ctw1500':   # CTW rows are unchanged by E42; stability with the clean readings (E41 C8)
        return RUNS / 'reader-harness/parseq-s-clean/tables/stability' / k / k / 'stability.json'
    return E42 / 'stability' / k / 'stability.json'


def counts_full(T, work, keep):
    if hasattr(T, 'off_te'):   # CTW1500: the unmodified Official path, all counters
        from secondlook.spotters.deepsolo_ctw1500 import entries_of
        _, counters, _ = T.off_te.run(entries_of(T.eval_tab, T.eval_ids, {i: list(map(bool, keep[i])) for i in T.eval_ids}), None)
        return np.asarray([[counters['%07d' % int(i)][f] for f in FIELDS] for i in T.eval_ids], np.int64)
    from secondlook.data.tasks import evaluate
    c = evaluate(None, work, 'tmp', T.eval_ids, T.eval_tab, {s: np.asarray(keep[s], bool).tolist() for s in T.eval_ids}, T.eval_gt, T.eval_names, T.metric)[0]
    for f in Path(work).glob('tmp.zip'):
        f.unlink()
    c = np.asarray(c, np.int64)
    if c.shape != (len(T.eval_ids), 7):
        raise ValueError('Unexpected counter shape %s' % (c.shape,))
    return c


def f1e(c):    # end-to-end F1 (percent), columns 0-2
    return 200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())


def det_rates(c):   # detection (percent): precision, recall, F
    k, p, g = float(c[:, 4].sum()), float(c[:, 5].sum()), float(c[:, 6].sum())
    return dict(precision=100.0 * k / p if p else 0.0, recall=100.0 * k / g if g else 0.0, f=200.0 * k / (p + g) if p + g else 0.0)


def boot_rates(c, M):
    """Per-draw detection precision / recall / F (percent) for the index matrix M (draws x images)."""
    k, p, g = (c[M, i].sum(1).astype(float) for i in (4, 5, 6))
    return dict(precision=100.0 * k / p, recall=100.0 * k / g, f=200.0 * k / (p + g))


def index_matrix(n):
    return np.random.default_rng(SEED).integers(0, n, (BOOT, n))


def ci(d):
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def author_keep(T, expert, task):
    """The released setting, exactly the expression of secondlook.data.cv_counts.grid_stage / secondlook.data.tasks_ctw1500.full_stage."""
    if expert in ('ticls', 'ppocr'):
        return {s: [True] * len(T.eval_tab[s]['lines']) for s in T.eval_ids}
    return {s: T.eval_tab[s]['score'] >= 0.4 for s in T.eval_ids}


def thr_train_counts_path(e, t):
    return CLEAN / 'tables/baselines' / ('counts_%s_%s.npz' % (e, t))
