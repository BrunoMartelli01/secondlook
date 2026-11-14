"""E46: wall-clock accounting. `Timer` records named stages; `install_exact_timer` returns a drop-in `counts_for(T, work, keep)` that times every call of the EXACT pinned
evaluator (and, where the code path allows, its sub-steps: submission zip writing and the upstream metric itself), so that the label-based F1 approximation used for threshold
selection can later be replaced by the exact evaluator at a known cost."""
import contextlib
import time

import numpy as np


class Timer:
    def __init__(self):
        self.records = []

    @contextlib.contextmanager
    def __call__(self, name, **extra):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.records.append(dict(name=name, seconds=time.perf_counter() - t0, **extra))

    def summary(self):
        out = {}
        for r in self.records:
            d = out.setdefault(r['name'], dict(calls=0, seconds=0.0, max_seconds=0.0))
            d['calls'] += 1
            d['seconds'] += r['seconds']
            d['max_seconds'] = max(d['max_seconds'], r['seconds'])
        for d in out.values():
            d['mean_seconds'] = d['seconds'] / d['calls']
        if 'exact_evaluator' in out:
            n = [r.get('n_images', 0) for r in self.records if r['name'] == 'exact_evaluator']
            out['exact_evaluator']['images_per_call'] = int(np.median(n)) if n else 0
            out['exact_evaluator']['seconds_per_image'] = out['exact_evaluator']['mean_seconds'] / max(out['exact_evaluator']['images_per_call'], 1)
        return out


def install_exact_timer(tm, expert, task):
    import secondlook.data.cv_counts as ac
    import secondlook.data.cohort as v2

    import secondlook.evaluation.official_benchmarks as ob
    original_zip = ob.write_zip              # v2.evaluate imports write_zip from here at call time
    if not getattr(original_zip, '_timed', False):
        def timed_zip(*args, **kwargs):
            with tm('exact_evaluator_write_submission_zip'):
                return original_zip(*args, **kwargs)
        timed_zip._timed = True
        ob.write_zip = timed_zip
    if task == 'ctw1500':
        import secondlook.spotters.deepsolo_ctw1500 as co
        original_run = co.Official.run

        def timed_run(self, *args, **kwargs):
            with tm('exact_evaluator_upstream_metric'):
                return original_run(self, *args, **kwargs)
        co.Official.run = timed_run

    def counts_for(T, work, keep):
        if task != 'ctw1500' and not getattr(T.metric, '_timed', False):
            original = T.metric.evaluate

            def timed_evaluate(*args, **kwargs):
                with tm('exact_evaluator_upstream_metric'):
                    return original(*args, **kwargs)
            try:
                T.metric.evaluate = timed_evaluate
                T.metric._timed = True
            except AttributeError:
                pass
        with tm('exact_evaluator', n_images=len(T.eval_ids)):
            return ac.counts_for(T, work, keep)
    return counts_for
