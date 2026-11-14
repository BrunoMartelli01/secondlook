"""E50-exact: secondlook.studies.cross_dataset (leave-one-benchmark-out and pooled fits) with EXACT selection. The module is the unchanged E44 (B) code; this wrapper
  (1) points its GATE to the arm's own S = all chain and stability files (RUN50/<arm>/main, RUN50/<arm>/stability), as secondlook.reruns.fast.cross_dataset did for the unified arm;
  (2) arm exact: replaces the label-based F1 of the pooled TRAIN pool by the exact one. The pooled pool mixes three benchmarks whose pinned evaluators live in different upstream source trees
      (module name `adet`), so the `fit` stage never loads an evaluator. It reads the per-image exact cache of each benchmark (exact_cache/<expert>_<benchmark>/part_*.pkl, written by secondlook.reruns.exact.exact_service.Store)
      and, for every (image, mask) it does not find, records the request, continues with the LABEL-BASED count of that image as a PROVISIONAL value and, at the end, writes
      `misses_<expert>.pkl` and exits with code 75 WITHOUT writing any fit. The `serve` stage (one process per benchmark, evaluator loaded) evaluates the requested pairs exactly; the driver
      (e50_cross_chain.sh) repeats fit -> serve until a fit finishes with no miss, i.e. until every selection of the stored fits was made from exact values only.
    python -m secondlook.reruns.exact.launch {fast|exact} secondlook.reruns.exact.cross_dataset {dump|fit|serve|evaluate|assemble} --expert E [--task T] --out DIR"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from secondlook.paths import RUNS

RUN = (RUNS / 'rerun-exact')
TASKS = ['totaltext', 'icdar2015', 'ctw1500']


class PoolStore:
    """Read-only exact lookup for the concatenated pool (image ids 'benchmark|id'); duck-types secondlook.reruns.exact.exact_service.Store for the patched pooled_f1 / approx_f1."""

    def __init__(self, expert, P):
        from secondlook.reruns.exact.exact_service import CACHE_ROOT, _key
        self._key = _key
        self.P, self.pool, self.row = P, set(P.ids), '%s_pooled' % expert
        self.cache = {b: {} for b in TASKS}
        for b in TASKS:
            for f in sorted((CACHE_ROOT / ('%s_%s' % (expert, b))).glob('part_*.pkl')):
                try:
                    d = pickle.load(open(f, 'rb'))
                except Exception:
                    continue
                for s, masks in d.items():
                    self.cache[b].setdefault(s, {}).update(masks)
        self.misses = {}
        self.stats = dict(lookups=0, misses=0)

    def counts_many(self, ids, keeps):
        out = []
        for kp in keeps:
            arr = np.zeros((len(ids), 3), np.int64)
            for j, k in enumerate(ids):
                b, s = k.split('|', 1)
                m = self._key(kp[k])
                v = self.cache[b].get(s, {}).get(m)
                self.stats['lookups'] += 1
                if v is None:
                    self.misses.setdefault(b, {}).setdefault(s, set()).add(m)
                    self.stats['misses'] += 1
                    lab, kk = self.P.lab[k], np.asarray(kp[k], bool)
                    v = (int(((lab == 1) & kk).sum()), int(((lab >= 0) & kk).sum()), int(self.P.gt[k]))
                arr[j] = v
            out.append(arr)
        return out

    def f1_many(self, ids, keeps):
        res = []
        for c in self.counts_many(list(ids), keeps):
            d = c[:, 1].sum() + c[:, 2].sum()
            res.append(2.0 * c[:, 0].sum() / d if d else 0.0)
        return res

    def f1(self, ids, keep):
        return self.f1_many(ids, [keep])[0]


def serve(a):
    """Evaluate the requested (image, mask) pairs of one benchmark with the pinned evaluator and add them to the benchmark's exact cache."""
    from secondlook.selection.operating_point import load
    out = Path(a.out)
    path = out / ('misses_%s.pkl' % a.expert)
    miss = pickle.load(open(path, 'rb')).get(a.task, {})
    n = sum(len(v) for v in miss.values())
    if not n:
        print('E50X serve', a.expert, a.task, 'nothing to evaluate', flush=True)
        return
    work = out / ('work_serve_%s_%s' % (a.expert, a.task))
    work.mkdir(exist_ok=True)
    T = load(a.expert, a.task, work)
    st = T._exact_store
    st._fill({s: list(v) for s, v in miss.items()})
    st.save()
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    print('E50X serve', a.expert, a.task, 'pairs', n, 'rounds', st.stats['rounds'], 'evaluator seconds %.0f' % st.stats['evaluator_seconds'], flush=True)


if __name__ == '__main__':
    import secondlook.evaluation.common as common
    arm = os.environ['SECONDLOOK_SELECTION']
    # dump / fit / serve / evaluate read only the (feature-independent) PARSeq-S* readings of the E42 clean reader; the GATE of `assemble` reads the arm's own S = all chain (counts_<row>_clean.npz)
    common.CLEAN = (RUN / arm / 'main') if (len(sys.argv) > 1 and sys.argv[1] == 'assemble') else (RUN.parent / 'fulltrain' / 'readers' / 'clean')
    common.stability_path = lambda e, t: RUN / arm / 'stability' / ('%s_%s' % (e, t)) / 'stability.json'
    import secondlook.studies.cross_dataset as x
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['dump', 'fit', 'serve', 'evaluate', 'assemble'])
    p.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', choices=x.TASKS)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if a.stage == 'serve':
        serve(a)
    elif a.stage == 'fit' and arm == 'exact':
        import secondlook.reruns.exact.launch as launch
        holder = {}

        class ExactPool(x.Pool):
            def __init__(self):
                super().__init__()
                holder['pool'] = self

            def approx_f1(self, ids, keep):
                return holder['store'].f1(list(ids), keep)
        x.Pool = ExactPool
        real_dump = pickle.dump

        def guarded_dump(obj, f, *args, **kw):
            st = holder.get('store')
            if st is not None and st.misses:
                (Path(a.out) / ('misses_%s.pkl' % a.expert)).write_bytes(pickle.dumps(st.misses))
                f.close()
                Path(f.name).unlink()   # the module opened the output file before calling dump: remove the empty file
                print('E50X fit: %d lookups, %d pairs missing in the exact cache (provisional values used) -> no fit written, exit 75' % (st.stats['lookups'], sum(len(v) for b in st.misses.values() for v in b.values())), flush=True)
                raise SystemExit(75)
            return real_dump(obj, f, *args, **kw)
        x.pickle = type('PickleShim', (), dict(load=staticmethod(pickle.load), dump=staticmethod(guarded_dump)))

        orig_cross_fit = None
        # the store needs the pool after its concatenation: register lazily at the first use of the pool in a fit
        orig_fit_pool = x.fit_pool

        def fit_pool(P, S, vt):
            if 'store' not in holder:
                holder['store'] = PoolStore(a.expert, P)
                launch.STORES[id(P.lab)] = holder['store']
                launch.STORES[id(P)] = holder['store']
            return orig_fit_pool(P, S, vt)
        x.fit_pool = fit_pool
        x.fit_stage(a)
        st = holder['store']
        print('E50X fit complete: %d exact lookups, 0 misses' % st.stats['lookups'], flush=True)
    else:
        {'dump': x.dump, 'fit': x.fit_stage, 'evaluate': x.evaluate_stage, 'assemble': x.assemble}[a.stage](a)
