"""E50-exact (approximate reproduction): the EXACT pinned evaluator on the TRAIN pool as a per-image, cached service.

Every selection of pipeline U (c, k*, the baseline pair) needs the end-to-end counts [correct, predictions, ground truth] that the pinned evaluator gives for a kept set on a (sub)set
of TRAIN images. The evaluators (Total-Text / ICDAR2015 DetEval-type scripts, CTW1500 upstream script) score each image independently; therefore the counts of one image depend ONLY on
(the image, the mask of kept candidates). This module evaluates every distinct (image, mask) once with the unchanged evaluator call, caches it (memory + shared files), and sums the
cached rows over any set of images: the result is the exact pooled count, identical to evaluating the whole pool with that kept set (verified by `verify`, gate E50-G1).
Missing pairs are evaluated in rounds; round r submits to the evaluator only the images that still need their r-th new mask, with a ground-truth archive restricted to those images
(Total-Text / ICDAR2015: secondlook.features.word_trust.subset_gt; CTW1500: the same selection of ground-truth files). Nothing is approximated and nothing is skipped.
"""
import atexit
import hashlib
import json
import os
import pickle
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from secondlook.paths import RUNS

CACHE_ROOT = Path(os.environ.get('SECONDLOOK_EXACT_CACHE', str(RUNS / 'rerun-exact/exact_cache')))


def _key(m):
    return np.asarray(m, bool).tobytes()


class Store:
    def __init__(self, T, work, row):
        self.T, self.row = T, row
        self.work = Path(tempfile.mkdtemp(prefix='e50x_', dir=str(work)))
        self.ctw = hasattr(T, 'off_tr')
        self.pool = set(T.ids)
        self.cache = {}                      # image id -> {mask bytes: (correct, predictions, gt)}  (everything known)
        self.own = {}                        # the entries this process computed (the only ones it writes)
        self.last_reload = 0.0
        self.dir = CACHE_ROOT / row
        self.dir.mkdir(parents=True, exist_ok=True)
        self.part = self.dir / ('part_%s_%d.pkl' % (os.uname().nodename, os.getpid()))
        self.seen_mtime = {}
        self.dirty = 0
        self.last_save = time.time()
        self.stats = dict(lookups=0, hits=0, evaluator_calls=0, image_evaluations=0, evaluator_seconds=0.0, rounds=0, requests=0, row=row)
        self.stat_path = self.dir / ('stats_%s_%d.json' % (os.uname().nodename, os.getpid()))
        if self.ctw:
            from secondlook.spotters.deepsolo_ctw1500 import Official
            from secondlook.data.tasks_ctw1500 import ASSETS, SRC
            self.official = Official(SRC, ASSETS, self.work / 'scratch', T.off_tr.gt)
            self.module = self.official.module
            self.src_gt = Path(T.off_tr.gt)
        else:
            self.module = T.metric.module
            self.src_gt = Path(T.pool_gt)
        self.pattern = re.compile(self.module.default_evaluation_params()['GT_SAMPLE_NAME_2_ID'])
        self._reload(force=True)
        atexit.register(self.close)

    # ------------------------------------------------------------------ persistence
    def _reload(self, force=False):
        if not force and time.time() - self.last_reload < 30:
            return
        self.last_reload = time.time()
        for f in sorted(self.dir.glob('part_*.pkl')):
            if f == self.part:
                continue
            m = f.stat().st_mtime
            if self.seen_mtime.get(f) == m:
                continue
            try:
                d = pickle.load(open(f, 'rb'))
            except Exception:       # a file being replaced; it is re-read at the next miss
                continue
            self.seen_mtime[f] = m
            for s, masks in d.items():
                self.cache.setdefault(s, {}).update(masks)

    def save(self):
        if not self.dirty:
            return
        tmp = self.part.with_suffix('.tmp')
        pickle.dump(self.own, open(tmp, 'wb'), protocol=4)
        os.replace(tmp, self.part)
        self.dirty, self.last_save = 0, time.time()

    def close(self):
        try:
            self.save()
            self.stat_path.write_text(json.dumps(self.stats, indent=1))
            shutil.rmtree(self.work, ignore_errors=True)
        except Exception:
            pass

    # ------------------------------------------------------------------ evaluator
    def _subset_gt(self, ids, dest):
        keys = {int(x) if self.ctw else None for x in ids}
        names = {Path(self.T.names[s]).stem.replace('res_img_', '') for s in ids} if not self.ctw else None
        found = set()
        with zipfile.ZipFile(self.src_gt) as src, zipfile.ZipFile(dest, 'x', compression=zipfile.ZIP_DEFLATED) as tgt:
            for name in src.namelist():
                m = self.pattern.search(name)
                if not m:
                    continue
                k = int(m.group(1)) if self.ctw else m.group(1)
                if (self.ctw and k in keys) or (not self.ctw and k in names):
                    if k in found:
                        raise ValueError('Duplicate GT sample')
                    found.add(k)
                    tgt.writestr(name, src.read(name))
        if found != (keys if self.ctw else names):
            raise ValueError('Incomplete GT population for the subset (%d of %d)' % (len(found), len(ids)))

    def _evaluate(self, ids, keep):
        """Exact counts [correct, predictions, gt] of the images `ids` with the kept masks `keep` (unchanged evaluator, GT restricted to `ids`)."""
        t0 = time.perf_counter()
        gt = self.work / 'sub_gt.zip'
        if gt.exists():
            gt.unlink()
        self._subset_gt(ids, gt)
        if self.ctw:
            from secondlook.spotters.deepsolo_ctw1500 import entries_of
            self.official.gt = gt
            _, counters, _ = self.official.run(entries_of(self.T.tab, ids, {i: list(map(bool, keep[i])) for i in ids}), None)
            c = np.asarray([[counters['%07d' % int(i)][f] for f in ('e2e_correct', 'e2e_predictions', 'e2e_gt_words')] for i in ids], np.int64)
        else:
            from secondlook.data.tasks import evaluate
            c = np.asarray(evaluate(None, self.work, 'e50', ids, self.T.tab, {s: np.asarray(keep[s], bool).tolist() for s in ids}, gt, self.T.names, self.T.metric)[0], np.int64)[:, :3]
            for f in self.work.glob('e50.zip'):
                f.unlink()
        gt.unlink()
        self.stats['evaluator_calls'] += 1
        self.stats['image_evaluations'] += len(ids)
        self.stats['evaluator_seconds'] += time.perf_counter() - t0
        return c

    def _fill(self, pending):
        """pending: {image id: [mask bytes, ...] (distinct, not cached)}; evaluates them in rounds."""
        self._reload()
        pending = {s: [m for m in ms if m not in self.cache.get(s, {})] for s, ms in pending.items()}
        pending = {s: ms for s, ms in pending.items() if ms}
        r = 0
        while pending:
            ids = sorted(pending, key=lambda x: int(x.split(':')[-1]) if not self.ctw else int(x))
            keep = {s: np.frombuffer(pending[s][0], dtype=bool) for s in ids}
            c = self._evaluate(ids, keep)
            for j, s in enumerate(ids):
                v = (int(c[j, 0]), int(c[j, 1]), int(c[j, 2]))
                self.cache.setdefault(s, {})[pending[s][0]] = v
                self.own.setdefault(s, {})[pending[s][0]] = v
                pending[s] = pending[s][1:]
            pending = {s: ms for s, ms in pending.items() if ms}
            self.dirty += len(ids)
            self.stats['rounds'] += 1
            r += 1
        if self.dirty and (time.time() - self.last_save > 120):
            self.save()
        return r

    # ------------------------------------------------------------------ public API
    def counts_many(self, ids, keeps):
        """Exact per-image counts [correct, predictions, gt] (array len(ids) x 3) for each kept set of `keeps` (dict image -> bool mask over its candidates)."""
        ids = list(ids)
        bad = [s for s in ids if s not in self.pool]
        if bad:
            raise KeyError('images outside the TRAIN pool of %s: %s' % (self.row, bad[:3]))
        pending = {}
        n_lookup = 0
        for kp in keeps:
            for s in ids:
                m = _key(kp[s])
                n_lookup += 1
                if m not in self.cache.get(s, {}):
                    pending.setdefault(s, {})[m] = None
        self.stats['lookups'] += n_lookup
        self.stats['hits'] += n_lookup - sum(len(v) for v in pending.values())
        self.stats['requests'] += len(keeps)
        if pending:
            self._fill({s: list(v) for s, v in pending.items()})
        return [np.asarray([self.cache[s][_key(kp[s])] for s in ids], np.int64) for kp in keeps]

    def f1_many(self, ids, keeps):
        out = []
        for c in self.counts_many(ids, keeps):
            d = c[:, 1].sum() + c[:, 2].sum()
            out.append(2.0 * c[:, 0].sum() / d if d else 0.0)
        return out

    def f1(self, ids, keep):
        return self.f1_many(ids, [keep])[0]

    # ------------------------------------------------------------------ gate E50-G1
    def verify(self, n_sets=4, seed=0):
        """Per-image counts of the cached/subset path == the WHOLE-POOL evaluator (the call of secondlook.selection.exact_selection.exact_train_counts) for kept sets of different kinds."""
        from secondlook.selection.exact_selection import exact_train_counts
        rng = np.random.RandomState(seed)
        ids = list(self.T.ids)
        sets = [{s: np.ones(len(self.T.tab[s]['lines']), bool) for s in ids}, {s: np.zeros(len(self.T.tab[s]['lines']), bool) for s in ids}]
        for _ in range(n_sets):
            q = rng.uniform(0.2, 0.9)
            sets.append({s: rng.uniform(size=len(self.T.tab[s]['lines'])) < q for s in ids})
        t0 = time.perf_counter()
        sub = self.counts_many(ids, sets)
        t_sub = time.perf_counter() - t0
        t0 = time.perf_counter()
        whole = [exact_train_counts(self.T, self.work, kp)[:, :3] for kp in sets]
        t_whole = time.perf_counter() - t0
        ok = all(np.array_equal(a, b) for a, b in zip(sub, whole))
        return dict(row=self.row, kept_sets=len(sets), images=len(ids), identical=bool(ok), per_image_equal=[bool(np.array_equal(a, b)) for a, b in zip(sub, whole)],
                    seconds_subset_cached_path=t_sub, seconds_whole_pool_path=t_whole, rounds=self.stats['rounds'])
