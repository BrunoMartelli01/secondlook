"""E42: Task objects whose TRAIN pool is the FULL official TRAIN of Total-Text (1,251 native-valid of 1,253 non-duplicate) and ICDAR2015 (1,000).

`load_full(expert, task, work)` returns the same object as secondlook.selection.operating_point.load for CTW1500 (unchanged), and for totaltext/icdar2015 the old task with
the TRAIN pool extended by the extra images:
  - DeepSolo : compact samples of the extra ids from the existing raw responses (research/fulltrain_extract.py), same candidate_table / add_common.
  - TiCLS    : expanded_word_evidence_v2/<task>/native (13 features, labels, per-image counts; bit-identical to the old tables on the 728 old ids).
  - PP-OCR   : cached old rows + extra rows of research/fulltrain_collect.py (same CompactWorker/format_rows), same word_tables.
  - labels   : DeepSolo/PP-OCR through the pinned native evaluator on the expanded source_gt.zip (label_pool); TiCLS from its word table.
The TEST side (eval_ids, eval_tab, eval_names, eval_gt) is untouched. `restrict_old(T)` returns the old 728-id view (gate T).
"""
import json
from pathlib import Path

import numpy as np

from secondlook.selection.operating_point import load
from secondlook.data.tasks import Task, TiclsTask, candidate_table, add_common
from secondlook.evaluation.protocol import read_json
from secondlook.paths import RUNS

COLLECT = (RUNS / 'fulltrain/collect')
EVID = (RUNS / 'ticls-source-banks/expanded_word_evidence_v2')
PPOCR_RUN = (RUNS / 'ppocr-predictions')
SRC = Path(__file__).resolve()


def _tail(s):
    return int(s.split(':')[-1])


def load_full(expert, task, work):
    return extend(load(expert, task, work), expert, task)


def extend(T, expert, task):
    import os
    if task not in ('totaltext', 'icdar2015') or os.environ.get('FT_OLD') == '1':   # FT_OLD=1: end-to-end gate, the unchanged 728-image pool through the same launcher
        return T
    old_ids = list(T.ids)
    bodies = read_json(EVID / task / 'native/bodies.json')
    names = {s: v['filename'] for s, v in bodies.items()}
    for s in old_ids:
        if names[s] != T.names[s]:
            raise ValueError('Filename mapping differs on an old id: ' + s)
    pool_gt = EVID / task / 'native/source_gt.zip'
    if expert == 'deepsolo':
        extra = read_json(COLLECT / ('%s_train_extra.json' % task))['samples']
        new_ids = sorted(extra, key=_tail)
        T.tab.update(add_common(candidate_table(extra, new_ids, task), 'deepsolo'))
        if os.environ.get('FT_DS_ALL_SAMPLES') and task == 'icdar2015':   # E42 SENSITIVITY ARM (1440): every pool image from the TEST-resolution inference; labelled as such by the caller
            allp = read_json(os.environ['FT_DS_ALL_SAMPLES'])['samples']
            T.tab = add_common(candidate_table(allp, old_ids + new_ids, task), 'deepsolo')
    elif expert == 'ppocr':
        from secondlook.data.tasks_ppocr import word_tables
        from secondlook.spotters.ppocr_collect import valid_quad
        extra = read_json(COLLECT / ('ppocr_%s_extra.json' % task))['rows']
        new_ids = sorted(extra, key=_tail)
        # The submission body already omits polygons rejected by the native rule (format_rows); word_tables expects words and body lines aligned, so the
        # rejected words (same valid_quad test) are removed from the word list first. On the 728 old rows nothing is removed (checked by gate T).
        rows, removed = {}, 0
        for s in new_ids:
            r = dict(extra[s])
            keep = [w for w in r['words'] if valid_quad([list(q) for q in w['points']])]
            removed += len(r['words']) - len(keep)
            r['words'] = keep
            rows[s] = r
        T.ppocr_rejected_polygons_new = removed
        T.tab.update(word_tables(rows))
    elif expert == 'ticls':
        a = np.load(EVID / task / 'native/words.npz')
        ids = a['ids'].tolist()
        new_ids = sorted(set(ids) - set(old_ids), key=_tail)
        tab, lab, gt = {}, {}, {}
        for i, s in enumerate(ids):
            w = bodies[s]['word_indices']
            x = a['features'][w].astype(np.float64)
            tab[s] = dict(lines=list(bodies[s]['lines']), x=x, score=a['detector_score'][w].astype(np.float64), rec=x[:, 6].copy())
            lab[s] = a['target'][w]
            gt[s] = int(a['counts'][i, 2])
        add_common(tab, 'ticls')
        if set(old_ids) - set(ids):
            raise ValueError('Expanded TiCLS table lacks old ids')
        T.tab, T.lab, T.gt = tab, lab, gt
        T.aug = {}
    else:
        raise ValueError(expert)
    T.ids = old_ids + new_ids
    T.names = names
    T.pool_gt = pool_gt
    if expert != 'ticls':
        T.label_pool()
    T.old_ids = old_ids
    T.new_ids = new_ids
    T.official = None
    return T


def assert_old_equal(Tf, To):
    """Gate T: on the old ids the full-pool task carries exactly the old task's lines, features, labels and ground-truth counts."""
    bad = []
    for s in To.ids:
        a, b = Tf.tab[s], To.tab[s]
        ok = (a['lines'] == b['lines'] and np.array_equal(a['x'], b['x']) and np.array_equal(a['xc'], b['xc']) and np.array_equal(a['score'], b['score'])
              and np.array_equal(a['rec'], b['rec']) and np.array_equal(Tf.lab[s], To.lab[s]) and Tf.gt[s] == To.gt[s])
        if not ok:
            bad.append(s)
    return bad


def install():
    """Make EVERY construction of the DeepSolo / TiCLS / PP-OCR Total-Text and ICDAR2015 tasks (in this process) return the full-TRAIN-pool task. Used by
    research/fulltrain_launch.py so that the unchanged pipeline scripts (operating_point_audit, train_only_band, p3_analyze, uniform_*, ...) run on the full pool."""
    import secondlook.data.tasks as dsa
    import secondlook.data.tasks_ppocr as pp
    for cls, expert in ((dsa.Task, 'deepsolo'), (dsa.TiclsTask, 'ticls'), (pp.PpocrTask, 'ppocr')):
        if cls.__dict__.get('_fulltrain_installed', False):
            continue
        orig = cls.__init__

        def wrapped(self, task, work, _o=orig, _e=expert):
            _o(self, task, work)
            if task in ('totaltext', 'icdar2015'):
                extend(self, _e, task)
        cls.__init__ = wrapped
        cls._fulltrain_installed = True
