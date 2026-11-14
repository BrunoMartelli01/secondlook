"""E49 launcher (approximate reproduction): run an UNCHANGED pipeline module of the repository with the full-TRAIN-pool tasks of E42 installed (as secondlook.studies.fulltrain_launch does) and,
for features=unified, with the E46 unified feature definitions substituted for `xc` and `rec` of every task object built in the process.

    python -m secondlook.reruns.fast.launch {old|unified} MODULE [module args...]

MODULE is any importable module (repository modules such as secondlook.studies.stability, or the E49 modules of this directory).
features=old is the GATE arm: it adds nothing to the task objects, so the module sees exactly what it saw when the manuscript numbers were produced.
features=unified wraps `Task`, `TiclsTask`, `PpocrTask` and `CtwTask` constructors: after the (E42-extended) constructor has built the tables, `build_unified` computes the unified
`xc` (columns 3-7 for the spotters where they differ, see the study log) and `rec` (= geometric mean for every spotter) and replaces them in `T.tab` and `T.eval_tab`.
Nothing else of the task (candidates, labels, ground truth, evaluator, readings) changes. The geometry gates of secondlook.studies.unified_pipeline are enforced; a failed gate stops the process.
The PP-OCR word-level entropy/margin come from the E46 re-run files (path in SECONDLOOK_PPOCR_STATS)."""
import json
import os
import runpy
import sys
from pathlib import Path
from secondlook.paths import REPO_ROOT, RUNS

REPO = REPO_ROOT
HERE = Path(__file__).resolve().parent

PPOCR_STATS = Path(os.environ.get('SECONDLOOK_PPOCR_STATS', str(RUNS / 'unified-features/ppocr_stats')))
_STATS = {}


def ppocr_stats(task):
    if task not in _STATS:
        _STATS[task] = {key: json.load(open(PPOCR_STATS / ('ppocr_stats_%s_%s.json' % (task, split))))['rows'] for key, split in (('train', 'train'), ('eval', 'test'))}
    return _STATS[task]


def geometry_gate(e, t, gate):
    def need(cond, text):
        if not cond:
            raise SystemExit('GEOMETRY GATE FAILED %s %s: %s %s' % (e, t, text, json.dumps(gate)))
    if e == 'ticls':
        need(gate['old_col5']['median'] <= 0.02 and gate['old_col5']['p99'] <= 0.2 and gate['old_col7']['median'] <= 0.02, 'TiCLS polygon source')
    elif e == 'deepsolo' and t != 'ctw1500':
        need(gate['old_col5']['max_abs'] <= 1e-9, 'DeepSolo area')
        if t == 'totaltext':
            need(gate['old_col6']['median'] <= 1e-9 and gate['old_col7']['median'] <= 1e-9, 'DeepSolo height/aspect')
    else:
        need(all(gate['old_col%d' % c]['max_abs'] <= 1e-9 for c in (5, 6, 7)), 'strict geometry')


def apply_unified(T, e, t):
    from secondlook.features.unified_features import build_unified
    stats = ppocr_stats(t) if e == 'ppocr' else None
    new_xc, new_rec, rep = build_unified(T, e, t, stats)
    geometry_gate(e, t, rep['geometry_gate'])
    T.tab = {s: dict(r, xc=new_xc['train'][s], rec=new_rec['train'][s]) if s in new_xc['train'] else r for s, r in T.tab.items()}
    T.eval_tab = {s: dict(r, xc=new_xc['eval'][s], rec=new_rec['eval'][s]) if s in new_xc['eval'] else r for s, r in T.eval_tab.items()}
    T.layer_features = 'unified'
    T.layer_report = rep


def wrap_all(features):
    from secondlook.data.tasks_fulltrain import install
    install()
    if features == 'old':
        return
    import secondlook.data.tasks_ctw1500 as ctw
    import secondlook.data.tasks_ppocr as pp
    import secondlook.data.tasks as dsa
    for cls, expert in ((dsa.Task, 'deepsolo'), (dsa.TiclsTask, 'ticls'), (pp.PpocrTask, 'ppocr')):
        if cls.__dict__.get('_layer_wrapped', False):
            continue
        orig = cls.__init__

        def wrapped(self, task, work, _o=orig, _c=cls, _e=expert):
            _o(self, task, work)
            if type(self) is _c and not getattr(self, 'layer_features', None):
                apply_unified(self, _e, task)
        cls.__init__ = wrapped
        cls._layer_wrapped = True
    if not ctw.CtwTask.__dict__.get('_layer_wrapped', False):
        orig_ctw = ctw.CtwTask.__init__

        def wrapped_ctw(self, work, expert='deepsolo', *a, **k):
            orig_ctw(self, work, expert, *a, **k)
            if not getattr(self, 'layer_features', None):
                apply_unified(self, expert, 'ctw1500')
        ctw.CtwTask.__init__ = wrapped_ctw
        ctw.CtwTask._layer_wrapped = True


if __name__ == '__main__':
    features, module = sys.argv[1], sys.argv[2]
    if features not in ('old', 'unified'):
        raise SystemExit('features must be old or unified')
    os.environ['SECONDLOOK_FEATURES'] = features
    wrap_all(features)
    sys.argv = [module] + sys.argv[3:]
    runpy.run_module(module, run_name='__main__', alter_sys=True)
