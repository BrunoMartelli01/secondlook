"""E46 (approximate reproduction): one row (expert x task) of the feature-unification experiment, with a wall-clock account of every stage.

Run through the E42 launcher so the full-TRAIN-pool tasks are installed:
    python -m secondlook.studies.fulltrain_launch secondlook.studies.unified_pipeline --expert E --task T --out DIR [--ppocr-stats DIR]
Arms (all use pipeline U of E38/E39/E42 on S = all TRAIN images, exact pinned evaluator on the official TEST; code lines of secondlook.studies.stability.pipeline copied 1:1 and timed):
  OLD        old xc, old rec                       -> gate: equals the stored stability.json row 'all' (c, f1_head, f1_final, f1_thr) and grid.npz 'author'.
  UNIFIED    unified xc, rec = geometric mean      -> the feature-unified pipeline.
  OLDFEAT_G  old xc, rec = geometric mean          -> isolates the baseline-statistic change.
  UNIFIED_OLDREC  unified xc, old rec              -> isolates the feature change against the old baseline.
Extra baselines (no classifier): detection-score-only re-tuning (rec grid {0}) with old and with unified rec; DeepSolo ICDAR2015 released at the official 0.3.
TIMING: every call of the exact evaluator, of the approximate F1 selection and of the learners is timed (secondlook.selection.timing); the timings are written next to the counts so that the exact
evaluator can later replace the label-based approximation.
Outputs under --out/<expert>_<task>/: counts_<arm>.npz (per-image [correct, predictions, gt] for first/complete/thr), arms.json, timing.json, unified_report.json."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from secondlook.selection import timing
from secondlook.data.tasks import pick_thresholds
from secondlook.evaluation.common import CLEAN, E18, author_keep, stability_path
from secondlook.features.unified_features import build_unified
from secondlook.selection.operating_point import FINE, load, pooled_f1
from secondlook.second_stage.band_classifier import cross_fit, verifier_features
from secondlook.selection.nested_selection import band_fit, fit_head, nested_select
from secondlook.paths import RUNS


def f1c(c):
    return 200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())


def with_features(T, xc_rec):
    """Shallow copy of the task whose tables carry other `xc` / `rec` (candidates, labels, evaluator, work directory shared)."""
    new_xc, new_rec = xc_rec
    U = copy.copy(T)
    U.tab = {s: dict(r, xc=new_xc['train'][s], rec=new_rec['train'][s]) if s in new_xc['train'] else r for s, r in T.tab.items()}
    U.eval_tab = {s: dict(r, xc=new_xc['eval'][s], rec=new_rec['eval'][s]) if s in new_xc['eval'] else r for s, r in T.eval_tab.items()}
    return U


def pipeline(T, counts_for, work, S, vt, ve, tm):
    """secondlook.studies.stability.pipeline, identical logic, every stage timed. Returns meta and the three 3-column count arrays (first stage, complete, thr_train)."""
    eids = list(T.eval_ids)
    with tm('cross_fit'):
        p_cf = cross_fit(T, S)
    with tm('c_selection_label_f1'):
        c = round(max(pooled_f1(T.lab, T.gt, p_cf, S, x) for x in FINE) / 2, 2)
    with tm('nested_select_k'):
        f1n, k_star, hi = nested_select(T, S, c, vt)
    with tm('fit_head'):
        head = fit_head(T, S)
    with tm('fit_band'):
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
    with tm('pick_thresholds_approx_f1'):
        st, rt, fsel = pick_thresholds(T, S)
    with tm('pick_thresholds_detection_only_approx_f1'):
        st1, _, fsel1 = pick_thresholds(T, S, rec_grid=[0.0])
    keept = {s: (T.eval_tab[s]['score'] >= st) & (T.eval_tab[s]['rec'] >= rt) for s in eids}
    keep1t = {s: T.eval_tab[s]['score'] >= st1 for s in eids}
    cs = [counts_for(T, work, k) for k in (keep1, keep2, keept, keep1t)]
    return dict(c=c, k_star=k_star, band_hi=hi, thr_score=float(st), thr_rec=float(rt), thr_score_only=float(st1), approx_f1_selection=float(fsel), approx_f1_selection_score_only=float(fsel1)), cs


def main(a):
    e, t = a.expert, a.task
    out = Path(a.out) / ('%s_%s' % (e, t))
    out.mkdir(parents=True, exist_ok=True)
    work = out / 'work'
    work.mkdir(exist_ok=True)
    tm = timing.Timer()
    counts_for = timing.install_exact_timer(tm, e, t)
    with tm('load_task_and_labels'):
        T = load(e, t, work)
    ids = list(T.ids)
    R = CLEAN / 'p3'
    with tm('verifier_features'):
        vt = verifier_features(T.tab, ids, json.load(open(R / ('%s_%s_train_parseq.json' % (e, t)))))
        ve = verifier_features(T.eval_tab, T.eval_ids, json.load(open(R / ('%s_%s_test_parseq.json' % (e, t)))))
    stab = {str(r['subset']): r for r in json.load(open(stability_path(e, t)))['rows']}['all']
    report = dict(expert=e, task=t, claim_label='approximate reproduction', train_images=len(ids), test_images=len(T.eval_ids))
    arms = {}
    # ---- released arm + gate ----
    rel = counts_for(T, work, author_keep(T, e, t))
    g_author = np.load(E18 / e / t / 'grid.npz')['author'].astype(np.int64)
    if not np.array_equal(rel, g_author):
        raise SystemExit('GATE FAILED: released arm differs from the stored grid.npz author arm for %s %s' % (e, t))
    arms['released'] = dict(f1=f1c(rel))
    np.savez(out / 'counts_released.npz', released=rel)
    if e == 'deepsolo' and t == 'icdar2015':   # official operating point of the ICDAR2015 DeepSolo config
        k03 = counts_for(T, work, {s: T.eval_tab[s]['score'] >= 0.3 for s in T.eval_ids})
        arms['released_official_0.3'] = dict(f1=f1c(k03))
        np.savez(out / 'counts_released_official_0.3.npz', released=k03)
    # ---- OLD pipeline + gate ----
    meta, cs = pipeline(T, counts_for, work, ids, vt, ve, tm)
    for name, cc, key in (('first', cs[0], 'f1_head'), ('complete', cs[1], 'f1_final'), ('thr_train', cs[2], 'f1_thr')):
        if abs(f1c(cc) - stab[key]) > 1e-9:
            raise SystemExit('GATE FAILED %s %s: %s F1 %.10f != stored %.10f' % (e, t, name, f1c(cc), stab[key]))
    if abs(meta['c'] - stab['c']) > 1e-12:
        raise SystemExit('GATE FAILED c %s != stored %s' % (meta['c'], stab['c']))
    arms['OLD'] = dict(meta=meta, f1_first=f1c(cs[0]), f1_complete=f1c(cs[1]), f1_thr_train=f1c(cs[2]), f1_thr_score_only=f1c(cs[3]), gate='passed (c, f1_head, f1_final, f1_thr equal stability.json row all)')
    np.savez(out / 'counts_OLD.npz', first=cs[0], complete=cs[1], thr_train=cs[2], thr_score_only=cs[3])
    print('GATE PASSED', e, t, 'OLD first %.2f complete %.2f thr %.2f' % (f1c(cs[0]), f1c(cs[1]), f1c(cs[2])), flush=True)
    # ---- unified features ----
    stats = None
    if e == 'ppocr':
        stats = {}
        for split, key in (('train', 'train'), ('test', 'eval')):
            stats[key] = json.load(open(Path(a.ppocr_stats) / ('ppocr_stats_%s_%s.json' % (t, split))))['rows']
    with tm('build_unified_features'):
        new_xc, new_rec, urep = build_unified(T, e, t, stats)
    report['unified_report'] = urep
    gate = urep['geometry_gate']
    # Geometry gate (polygon + image-size sources are the ones the old tables used). TiCLS: the submitted polygon is the integer rounding of the float polygon of the old
    # features, so the comparison is a distribution (median/p99); DeepSolo: area strict, height/aspect came from the detector BOUNDARY bbox (not recomputable from the
    # submitted quad on ICDAR2015, strict-median on Total-Text); PP-OCR and CTW1500 experts: strict.
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
    changed = {}
    for c in range(11):
        diff = [float(np.max(np.abs(new_xc['train'][s][:, c] - T.tab[s]['xc'][:, c]))) for s in ids if len(T.tab[s]['xc'])]
        diff_e = [float(np.max(np.abs(new_xc['eval'][s][:, c] - T.eval_tab[s]['xc'][:, c]))) for s in T.eval_ids if len(T.eval_tab[s]['xc'])]
        changed[c] = dict(max_abs_change_train=max(diff) if diff else 0.0, max_abs_change_test=max(diff_e) if diff_e else 0.0)
    report['column_changes'] = changed
    g_only = ({'train': {s: T.tab[s]['xc'] for s in ids}, 'eval': {s: T.eval_tab[s]['xc'] for s in T.eval_ids}},
              {'train': {s: T.tab[s]['xc'][:, 1].copy() for s in ids}, 'eval': {s: T.eval_tab[s]['xc'][:, 1].copy() for s in T.eval_ids}})
    old_rec = ({'train': new_xc['train'], 'eval': new_xc['eval']}, {'train': {s: T.tab[s]['rec'] for s in ids}, 'eval': {s: T.eval_tab[s]['rec'] for s in T.eval_ids}})
    for name, spec in (('UNIFIED', (new_xc, new_rec)), ('OLDFEAT_G', g_only), ('UNIFIED_OLDREC', old_rec)):
        U = with_features(T, spec)
        meta, cs = pipeline(U, counts_for, work, ids, vt, ve, tm)
        arms[name] = dict(meta=meta, f1_first=f1c(cs[0]), f1_complete=f1c(cs[1]), f1_thr_train=f1c(cs[2]), f1_thr_score_only=f1c(cs[3]))
        np.savez(out / ('counts_%s.npz' % name), first=cs[0], complete=cs[1], thr_train=cs[2], thr_score_only=cs[3])
        print(name, e, t, 'first %.2f complete %.2f thr %.2f thr_score_only %.2f' % (f1c(cs[0]), f1c(cs[1]), f1c(cs[2]), f1c(cs[3])), flush=True)
    report['arms'] = arms
    report['timing'] = tm.summary()
    (out / 'arms.json').write_text(json.dumps(report, indent=1, default=float))
    (out / 'timing.json').write_text(json.dumps(tm.records, indent=1, default=float))
    print('done', e, t, 'exact evaluator calls', report['timing'].get('exact_evaluator', {}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', required=True, choices=['totaltext', 'icdar2015', 'ctw1500'])
    p.add_argument('--out', required=True)
    p.add_argument('--ppocr-stats', default=str(RUNS / 'unified-features/ppocr_stats'))
    main(p.parse_args())
