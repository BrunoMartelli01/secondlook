"""E50-exact launcher (approximate reproduction). Runs an UNCHANGED repository / E49 module with the UNIFIED features of E46/E49 (secondlook.features.unified_features through results/unified_features_full_rerun/secondlook.reruns.fast.launch)
and, for the arm `exact`, with every selection made by the EXACT pinned evaluator instead of the label-based F1 2TP/(P+G):

    python -m secondlook.reruns.exact.launch {fast|exact} MODULE [module args...]

arm fast  : nothing is patched; identical to the E49 `unified` arm (GATE: the E50 outputs of this arm equal the E49 files bit for bit).
arm exact : the three places where the repository selects a parameter on the TRAIN pool are replaced by exact counterparts (same arguments, same return value, same tie rules):
    secondlook.selection.operating_point.pooled_f1(lab, gt, p, ids, c)        -> c (100-point out-of-fold curve; the 0..99 grid is evaluated as one batch)
    Task/TiclsTask/PpocrTask/CtwTask.approx_f1(ids, keep)               -> k* (nested cross-fit), pick_cut, every direct use
    secondlook.data.tasks.pick_thresholds(T, ids, grid)  -> the baseline pair (all 135 pairs, ties to the larger score threshold then the larger rec threshold)
The exact value is the pooled F1 2*correct/(predictions+gt) of the pinned evaluator on the TRAIN GT, summed over the images of the fit (secondlook.selection.exact_selection / e46_exact_pipeline rule),
obtained through secondlook.reruns.exact.exact_service.Store (per-image cache of the unchanged evaluator, gate G1). Calls on data that is not a registered TRAIN pool (TEST diagnostics, other label sets) fall through to the
original function and are LISTED in exact_cache/fallbacks_<pid>.json; nothing falls back silently.
Environment: SECONDLOOK_SELECTION = fast|exact; SECONDLOOK_FEATURES = the arm name (used by the E49 scripts as the output sub-directory)."""
import atexit
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from secondlook.paths import REPO_ROOT, RUNS

REPO = REPO_ROOT
HERE = Path(__file__).resolve().parent

CACHE_ROOT = Path(os.environ.get('SECONDLOOK_EXACT_CACHE', str(RUNS / 'rerun-exact/exact_cache')))
FALLBACKS = []
USED = dict(approx_f1=0, pooled_f1_curves=0, pooled_f1_single=0, pick_thresholds=0, best_signal_threshold=0)   # exact calls actually served by the patches (proof that they were active)
STORES = {}        # id(lab dict) -> Store ; id(task) -> Store
_CURVES = {}       # content hash of (ids, p) -> {c: exact F1}


def _store_of(T):
    return STORES.get(id(T))


def register(T, expert, task, work):
    from secondlook.reruns.exact.exact_service import Store
    row = '%s_%s' % (expert, task)
    st = Store(T, Path(work), row)
    STORES[id(T)] = st
    STORES[id(T.lab)] = st
    T._exact_store = st
    T._e50_row = row


def install_exact():
    import secondlook.data.tasks_ctw1500 as ctw
    import secondlook.data.tasks_ppocr as pp
    import secondlook.data.tasks as dsa
    import secondlook.selection.operating_point as opa
    for cls, expert in ((dsa.Task, 'deepsolo'), (dsa.TiclsTask, 'ticls'), (pp.PpocrTask, 'ppocr')):
        orig = cls.__init__

        def wrapped(self, task, work, _o=orig, _c=cls, _e=expert):
            _o(self, task, work)
            if type(self) is _c and not hasattr(self, '_exact_store'):
                register(self, _e, task, work)
        cls.__init__ = wrapped
    orig_ctw = ctw.CtwTask.__init__

    def wrapped_ctw(self, work, expert='deepsolo', *a, **k):
        orig_ctw(self, work, expert, *a, **k)
        if not hasattr(self, '_exact_store'):
            register(self, expert, 'ctw1500', work)
    ctw.CtwTask.__init__ = wrapped_ctw

    # ---------------- approx_f1 -> exact pooled F1 of the TRAIN pool (k*, pick_cut, ...)
    def make_f1(orig):
        def exact_f1(self, ids, keep, *a, **k):
            st = _store_of(self)
            if st is None or a or k or not all(s in st.pool for s in ids):
                FALLBACKS.append(dict(where='approx_f1', cls=type(self).__name__, images=len(list(ids)), extra_args=bool(a or k), registered=st is not None))
                return orig(self, ids, keep, *a, **k)
            USED['approx_f1'] += 1
            return st.f1(list(ids), keep)
        return exact_f1
    for cls in (dsa.Task, dsa.TiclsTask, pp.PpocrTask, ctw.CtwTask):
        if 'approx_f1' in cls.__dict__:
            cls.approx_f1 = make_f1(cls.approx_f1)

    # ---------------- pooled_f1 -> exact (the whole FINE grid as one batch)
    orig_pooled = opa.pooled_f1

    def exact_pooled_f1(lab, gt, p, ids, c):
        st = STORES.get(id(lab))
        ids = list(ids)
        if st is None or not all(s in st.pool for s in ids):
            FALLBACKS.append(dict(where='pooled_f1', images=len(ids), registered=st is not None))
            return orig_pooled(lab, gt, p, ids, c)
        h = hashlib.sha256()
        for s in ids:
            h.update(str(s).encode() + b'|' + p[s].astype('float64').tobytes() + b';')
        key = (id(st), h.hexdigest())
        if key not in _CURVES:
            if len(_CURVES) > 64:
                _CURVES.clear()
            grid = opa.FINE
            f = st.f1_many(ids, [{s: p[s] >= x for s in ids} for x in grid])
            _CURVES[key] = dict(zip(grid, f))
            USED['pooled_f1_curves'] += 1
        if c in _CURVES[key]:
            return _CURVES[key][c]
        USED['pooled_f1_single'] += 1
        return st.f1(ids, {s: p[s] >= c for s in ids})
    opa.pooled_f1 = exact_pooled_f1
    for name, mod in list(sys.modules.items()):
        if mod is not None and getattr(mod, 'pooled_f1', None) is orig_pooled:
            mod.pooled_f1 = exact_pooled_f1

    # ---------------- pick_thresholds -> exact argmax over the 135 pairs
    orig_pick = dsa.pick_thresholds

    def exact_pick(T, ids, rec_grid=None):
        st = _store_of(T)
        ids = list(ids)
        if st is None or not all(s in st.pool for s in ids):
            FALLBACKS.append(dict(where='pick_thresholds', images=len(ids), registered=st is not None))
            return orig_pick(T, ids, rec_grid)
        pairs = [(a, b) for a in dsa.SCORE_GRID for b in (dsa.REC_GRID if rec_grid is None else rec_grid)]
        keeps = [{s: (T.tab[s]['score'] >= a) & (T.tab[s]['rec'] >= b) for s in ids} for a, b in pairs]
        f = st.f1_many(ids, keeps)
        rows = [(fi, a, b) for fi, (a, b) in zip(f, pairs)]
        fbest, a, b = max(rows)
        USED['pick_thresholds'] += 1
        return a, b, fbest
    dsa.pick_thresholds = exact_pick
    for name, mod in list(sys.modules.items()):
        if mod is not None and getattr(mod, 'pick_thresholds', None) is orig_pick:
            mod.pick_thresholds = exact_pick

    # ---------------- best_signal_threshold (single-signal rules of the learner table) -> exact; same candidates (80 pooled quantiles + keep-all), first maximum wins
    import numpy as np
    import secondlook.studies.ablation_learners as fam
    orig_sig = fam.best_signal_threshold

    def exact_signal(T, pool, fn):
        st = _store_of(T)
        pool = list(pool)
        if st is None or not all(s in st.pool for s in pool):
            FALLBACKS.append(dict(where='best_signal_threshold', images=len(pool), registered=st is not None))
            return orig_sig(T, pool, fn)
        vals = np.concatenate([fn(T.tab[s]) for s in pool])
        cands = np.concatenate([[-np.inf], np.unique(np.quantile(vals, np.linspace(0.0, 0.97, 80)))])
        per = {s: fn(T.tab[s]) for s in pool}
        f = st.f1_many(pool, [{s: per[s] >= c for s in pool} for c in cands])
        best = None
        for fi, c in zip(f, cands):
            if best is None or fi > best[0]:
                best = (fi, c)
        USED['best_signal_threshold'] += 1
        return best[1]
    fam.best_signal_threshold = exact_signal
    for name, mod in list(sys.modules.items()):
        if mod is not None and getattr(mod, 'best_signal_threshold', None) is orig_sig:
            mod.best_signal_threshold = exact_signal


def dump_fallbacks():
    try:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        (CACHE_ROOT / ('fallbacks_%s_%d.json' % (os.uname().nodename, os.getpid()))).write_text(json.dumps(dict(argv=sys.argv, n=len(FALLBACKS), entries=FALLBACKS[:200], exact_calls_served=USED, arm=os.environ.get('SECONDLOOK_SELECTION')), indent=1))
    except Exception:
        pass


if __name__ == '__main__':
    arm, module = sys.argv[1], sys.argv[2]
    if arm not in ('fast', 'exact'):
        raise SystemExit('arm must be fast or exact')
    sys.modules.setdefault('secondlook.reruns.exact.launch', sys.modules['__main__'])   # modules run below that `import secondlook.reruns.exact.launch` must see THIS instance (STORES registry)
    os.environ['SECONDLOOK_SELECTION'] = arm
    os.environ['SECONDLOOK_FEATURES'] = arm
    import secondlook.reruns.fast.launch as fast_launch
    fast_launch.wrap_all('unified')
    if arm == 'exact':
        install_exact()
    atexit.register(dump_fallbacks)
    sys.argv = [module] + sys.argv[3:]
    if arm == 'exact' and module == 'secondlook.selection.operating_point':
        # runpy would execute the module source again as __main__ and re-define its own label-based pooled_f1 (the patch would not be seen):
        # import it (patched) and call main with the module's own argument parser
        import argparse
        import secondlook.selection.operating_point as m
        pa = argparse.ArgumentParser(m.__doc__)
        pa.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
        pa.add_argument('--task', required=True, choices=['totaltext', 'icdar2015', 'ctw1500'])
        pa.add_argument('--work', required=True)
        pa.add_argument('--out', required=True)
        pa.add_argument('--extra-cuts', dest='extra_cuts', type=float, nargs='*', default=[])
        m.main(pa.parse_args(sys.argv[1:]))
        sys.exit(0)
    runpy.run_module(module, run_name='__main__', alter_sys=True)
