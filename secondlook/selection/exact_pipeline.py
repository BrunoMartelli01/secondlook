"""E46b (approximate reproduction): pipeline U of E38/E39/E42 with EVERY selection made by the exact pinned evaluator (c, k*, thr_train); see the pre-registration in the study log.
    python -m secondlook.studies.fulltrain_launch secondlook.selection.exact_pipeline --expert E --task T --runs DIR --exact-sel DIR --exact-base DIR --out DIR
Arms OLD_EXACT (old features) and UNIFIED_EXACT (secondlook.features.unified_features). Head, band model, cross-fit seeds, readings and the TEST evaluation are those of secondlook.studies.unified_pipeline / secondlook.studies.stability.
Every exact evaluation is cached by the hash of its kept set (identical keep sets are evaluated once) and timed."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from secondlook.selection import timing
from secondlook.evaluation.common import CLEAN
from secondlook.selection.exact_selection import exact_train_counts
from secondlook.features.unified_features import build_unified
from secondlook.studies.unified_pipeline import with_features
from secondlook.selection.operating_point import FINE, load, pooled_f1
from secondlook.second_stage.band_classifier import cross_fit, verifier_features
from secondlook.selection.verification_band import KS, TOL
from secondlook.selection.nested_selection import band_apply, band_fit, fit_head, pred
from secondlook.paths import RUNS


class ExactF1:
    """Exact pooled F1 (fraction) of a keep set over the TRAIN pool T.ids, cached by content."""

    def __init__(self, T, work, tm):
        self.T, self.work, self.tm, self.cache, self.calls = T, work, tm, {}, 0

    def __call__(self, keep):
        ids = list(self.T.ids)
        h = hashlib.sha256()
        for s in ids:
            h.update(np.asarray(keep[s], bool).tobytes() + b'|')
        key = h.hexdigest()
        if key not in self.cache:
            with self.tm('exact_train_evaluation', n_images=len(ids)):
                c = exact_train_counts(self.T, self.work, {s: np.asarray(keep[s], bool) for s in ids})
            self.cache[key] = 2.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())
            self.calls += 1
        return self.cache[key]


def pipeline_exact(T, S, vt, ve, ex, counts_for, work, tm):
    eids = list(T.eval_ids)
    p_cf = cross_fit(T, S)
    approx_curve = [pooled_f1(T.lab, T.gt, p_cf, S, x) for x in FINE]
    with tm('c_curve_exact_100_thresholds'):
        exact_curve = [ex({s: p_cf[s] >= x for s in S}) for x in FINE]
    c_approx, c = round(max(approx_curve) / 2, 2), round(max(exact_curve) / 2, 2)
    order = [S[i] for i in np.random.RandomState(0).permutation(len(S))]
    his = [(k, 1.0 if k == 'all' else min(k * c, 1.0)) for k in KS]
    keep = {str(k): {} for k, _ in his}
    keep['stage1'] = {}
    for f in range(5):
        held = order[f::5]
        hs = set(held)
        inner = [s for s in S if s not in hs]
        head_f = fit_head(T, inner)
        p_in = cross_fit(T, inner)
        p_out = {s: pred(head_f, T, s) for s in held}
        for s in held:
            keep['stage1'][s] = p_out[s] >= c
        for k, hi in his:
            band_f = band_fit(T, inner, p_in, vt, hi)
            for s in held:
                keep[str(k)][s] = band_apply(T, band_f, s, p_out[s], vt, hi, c)
    with tm('k_selection_exact'):
        f1 = {name: 100.0 * ex(kp) for name, kp in keep.items()}
    best = max(f1[str(k)] for k, _ in his)
    k_star = next(k for k, _ in his if f1[str(k)] >= best - TOL)
    hi = dict(his)[k_star]
    head = fit_head(T, S)
    band = band_fit(T, S, p_cf, vt, hi)
    pe = {s: head.predict_proba(T.eval_tab[s]['xc'])[:, 1] if len(T.eval_tab[s]['lines']) else np.zeros(0) for s in eids}
    keep1 = {s: pe[s] >= c for s in eids}
    keep2 = {}
    for s in eids:
        inb = (pe[s] >= 0.0) & (pe[s] <= hi)
        if inb.any():
            xe = np.column_stack([T.eval_tab[s]['xc'], pe[s], ve[s]])
            keep2[s] = np.where(inb, band.predict_proba(xe)[:, 1] >= c, pe[s] >= c)
        else:
            keep2[s] = pe[s] >= c
    cs = [counts_for(T, work, k) for k in (keep1, keep2)]
    return dict(c_exact=c, c_approx=c_approx, F_exact_star=float(max(exact_curve)), F_approx_star=float(max(approx_curve)), k_star_exact=k_star, band_hi=hi,
                k_f1_exact={k: float(v) for k, v in f1.items()}, max_abs_gap_c_curve=float(max(abs(a - b) for a, b in zip(approx_curve, exact_curve))),
                distinct_exact_evaluations=ex.calls), cs


def main(a):
    e, t = a.expert, a.task
    row = '%s_%s' % (e, t)
    out = Path(a.out) / row
    out.mkdir(parents=True, exist_ok=True)
    work = out / 'work'
    work.mkdir(exist_ok=True)
    tm = timing.Timer()
    counts_for = timing.install_exact_timer(tm, e, t)
    T = load(e, t, work)
    ids = list(T.ids)
    vt = verifier_features(T.tab, ids, json.load(open(CLEAN / 'p3' / ('%s_%s_train_parseq.json' % (e, t)))))
    ve = verifier_features(T.eval_tab, T.eval_ids, json.load(open(CLEAN / 'p3' / ('%s_%s_test_parseq.json' % (e, t)))))
    ex = ExactF1(T, work, tm)
    # G1: the exact evaluator with every candidate kept reproduces the label-based counts
    every = {s: np.ones(len(T.tab[s]['lines']), bool) for s in ids}
    tp = sum(int((T.lab[s] == 1).sum()) for s in ids)
    k = sum(int((T.lab[s] >= 0).sum()) for s in ids)
    g = sum(T.gt[s] for s in ids)
    f_label, f_exact = 2.0 * tp / (k + g), ex(every)
    if abs(f_label - f_exact) > 1e-12:
        raise SystemExit('GATE G1 FAILED %s: exact F1 with every candidate kept %.12f != label-based %.12f' % (row, f_exact, f_label))
    res = dict(expert=e, task=t, claim_label='approximate reproduction', gate_G1=dict(label_based=f_label, exact=f_exact, passed=True), arms={})
    stats = None
    if e == 'ppocr':
        stats = {sp: json.load(open(Path(a.ppocr_stats) / ('ppocr_stats_%s_%s.json' % (t, spn))))['rows'] for sp, spn in (('train', 'train'), ('eval', 'test'))}
    new_xc, new_rec, _ = build_unified(T, e, t, stats)
    arms = {'OLD_EXACT': T, 'UNIFIED_EXACT': with_features(T, (new_xc, new_rec))}
    for name, U in [(n, arms[n]) for n in a.arms]:   # default: UNIFIED_EXACT only (the old features are not recomputed under the exact protocol)
        ex.T = U
        meta, cs = pipeline_exact(U, ids, vt, ve, ex, counts_for, work, tm)
        res['arms'][name] = meta
        np.savez(out / ('counts_%s.npz' % name), first=cs[0], complete=cs[1])
        f1c = lambda c: 200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())
        meta['f1_first'], meta['f1_complete'] = f1c(cs[0]), f1c(cs[1])
        print(row, name, 'c %.2f (approx %.2f) k* %s first %.2f complete %.2f' % (meta['c_exact'], meta['c_approx'], meta['k_star_exact'], meta['f1_first'], meta['f1_complete']), flush=True)
    res['timing'] = tm.summary()
    (out / 'exact_pipeline.json').write_text(json.dumps(res, indent=1, default=float))
    print('done', row, 'exact train evaluations', ex.calls, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--expert', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--arms', nargs='+', default=['UNIFIED_EXACT'], choices=['OLD_EXACT', 'UNIFIED_EXACT'])
    p.add_argument('--ppocr-stats', default=str(RUNS / 'unified-features/ppocr_stats'))
    main(p.parse_args())
