"""E39 (uniform methodology): Tables 3, 4, 5, 6, 12 and 14 of the draft on ONE bootstrap convention (paired image draws, 2000, seed 20261106, ONE index matrix per row shared by all arms),
from the saved per-image counts of the unchanged E18/E30/E31/E32 chains (the uniform pipeline does not change these computations: pool = all TRAIN images, c = c_half, k* by nested cross-fit).
Stage `baselines` re-runs the same-reader rules of Table 6 (secondlook.second_stage.baselines `baselines` mode, unchanged logic) and SAVES their per-image counts in the new directory (the old run saved
only the summary), so Table 6 can use the shared index matrix. Stage `tables` assembles everything and checks the validation gate (point estimates equal the old ones). CPU only."""
import argparse
import json
from pathlib import Path

import numpy as np
from secondlook.paths import RUNS

R = (RUNS / 'candidate-banks')
W = (RUNS / 'operating-point')
ROWS = [('deepsolo', 'totaltext'), ('deepsolo', 'icdar2015'), ('ticls', 'totaltext'), ('ticls', 'icdar2015'), ('ppocr', 'totaltext'), ('ppocr', 'icdar2015'),
        ('deepsolo', 'ctw1500'), ('ticls', 'ctw1500'), ('ppocr', 'ctw1500')]
SWEEP_ROWS = ROWS[:4]
SWEEP_K = ['1.2', '1.4', '1.6', '1.9', '2.2', '2.6', '3.0', 'all']
BOOT, SEED = 2000, 20261106
# optional redirection (defaults keep the E39 behaviour): readings/counts directory, band/audit directory, tags of the counts files
P3, BAND = R / 'p3', W
TAG_MAIN, TAG_TARGMAX, TAG_FINAL, TAG_SW = 'trainonly', 'targmax', 'final', 'sw'
CHECK_GATE5 = True
GATE3 = [0.64, 0.60, 0.62, 1.68, 2.92, 2.34, 0.15, 0.50, 0.79]
GATE5 = [1.81, 2.50, 0.96, 2.45, 3.71, 2.92, 1.23, 0.72, 1.14]


def f1(c):
    return float(200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum()))


def prf(c):
    tp, pr, gt = c.sum(0)[:3]
    return float(100 * tp / pr), float(100 * tp / gt), float(200 * tp / (pr + gt)), int(pr), int(gt)


def baselines(a):
    from secondlook.data.cv_counts import counts_for
    from secondlook.data.tasks import pick_thresholds
    from secondlook.second_stage.band_classifier import verifier_features
    from secondlook.second_stage.baselines import fit_band, fit_stage1, final_keep, make_task
    out = Path(a.out) / 'baselines'
    out.mkdir(parents=True, exist_ok=True)
    for e, t in ROWS:
        if a.only and (e, t) != tuple(a.only):
            continue
        if (out / ('counts_%s_%s.npz' % (e, t))).exists():
            continue
        work = out / ('work_%s_%s' % (e, t))
        work.mkdir(exist_ok=True)
        ns = argparse.Namespace(task=t, expert=e)
        T = make_task(ns, work)
        pq_tr = json.load(open(P3 / ('%s_%s_train_parseq.json' % (e, t))))
        pq_te = json.load(open(P3 / ('%s_%s_test_parseq.json' % (e, t))))
        vt, ve = verifier_features(T.tab, T.ids, pq_tr), verifier_features(T.eval_tab, T.eval_ids, pq_te)
        band_ = json.load(open(BAND / ('train_band_%s_%s.json' % (e, t))))
        cut, hi = band_['c'], band_['band_hi_star']
        rank = fit_stage1(T, T.ids)
        band = fit_band(T, T.ids, vt, 0.0, hi)
        p_te, keep_head, keep_final = final_keep(T, rank, band, ve, cut, 0.0, hi)
        st, rt, _ = pick_thresholds(T, T.ids)
        thr = {s: (T.eval_tab[s]['score'] >= st) & (T.eval_tab[s]['rec'] >= rt) for s in T.eval_ids}
        agree = {s: ve[s][:, 0] == 1 for s in T.eval_ids}
        inb = {s: (p_te[s] >= 0.0) & (p_te[s] <= hi) for s in T.eval_ids}
        arms = {'thr_train': thr, 'head': keep_head, 'final': keep_final,
                'thr_train+agree_in_band': {s: np.where(inb[s], agree[s], thr[s]) for s in T.eval_ids},
                'head+agree_in_band': {s: np.where(inb[s], agree[s], keep_head[s]) for s in T.eval_ids},
                'agree_all': agree, 'head AND agree_all': {s: keep_head[s] & agree[s] for s in T.eval_ids}, 'thr_train AND agree_all': {s: thr[s] & agree[s] for s in T.eval_ids}}
        counts = {k: counts_for(T, work, v) for k, v in arms.items()}
        ref = np.load(P3 / ('counts_%s_%s_%s.npz' % (e, t, TAG_MAIN)))
        ok = bool(np.array_equal(counts['final'], ref['verifier']) and np.array_equal(counts['head'], ref['head']))
        print('BASE', e, t, 'final/head equal to E31 counts:', ok, flush=True)
        if not ok:
            raise SystemExit('gate failed: baselines run does not reproduce the E31 counts for %s %s' % (e, t))
        np.savez_compressed(out / ('counts_%s_%s.npz' % (e, t)), **{k.replace(' ', '_'): v for k, v in counts.items()})
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def tables(a):
    out = Path(a.out)
    res = {}
    gate_fail = []
    for ri, (e, t) in enumerate(ROWS):
        g, x = np.load(R / e / t / 'grid.npz'), np.load(R / e / t / 'extra_arms.npz')
        T, A = (np.load(P3 / ('counts_%s_%s_%s.npz' % (e, t, s))) for s in (TAG_MAIN, TAG_TARGMAX))
        F = np.load(P3 / ('counts_%s_%s_%s.npz' % (e, t, TAG_FINAL))) if (P3 / ('counts_%s_%s_%s.npz' % (e, t, TAG_FINAL))).exists() else None
        arms = dict(author=g['author'], thr=x['thr_train'], s1=T['head'], s2=T['verifier'], ctrl=T['control'], s1A=A['head'], s2A=A['verifier'])
        if F is not None:
            arms.update(s2R=F['verifier'], s1R=F['head'])
        arms = {k: v.astype(np.int64) for k, v in arms.items()}
        n = arms['author'].shape[0]
        idx = np.random.default_rng(SEED).integers(0, n, (BOOT, n))

        def d(ca, cb):
            bd = 200.0 * ca[idx, 0].sum(1) / (ca[idx, 1].sum(1) + ca[idx, 2].sum(1)) - 200.0 * cb[idx, 0].sum(1) / (cb[idx, 1].sum(1) + cb[idx, 2].sum(1))
            return [f1(ca) - f1(cb), float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
        band = json.load(open(BAND / ('train_band_%s_%s.json' % (e, t))))
        aud = json.load(open(BAND / ('op_audit_%s_%s.json' % (e, t))))
        r1 = json.load(open(P3 / ('result_%s_%s_%s.json' % (e, t, TAG_MAIN))))
        r = dict(name='%s_%s' % (e, t), n=n, c=band['c'], k=band['k_star'], hi=band['band_hi_star'], c_argmax=aud['c_argmax'], prf={k: prf(v) for k, v in arms.items()}, f1={k: f1(v) for k, v in arms.items()},
                 thr_m_author=d(arms['thr'], arms['author']), s1_m_author=d(arms['s1'], arms['author']), s1_m_thr=d(arms['s1'], arms['thr']), s2_m_s1=d(arms['s2'], arms['s1']), s2_m_thr=d(arms['s2'], arms['thr']),
                 s2_m_author=d(arms['s2'], arms['author']), ctrl_m_s1=d(arms['ctrl'], arms['s1']), s2_m_ctrl=d(arms['s2'], arms['ctrl']), s2_m_s2R=d(arms['s2'], arms['s2R']) if F is not None else None, s2A_m_s2=d(arms['s2A'], arms['s2']),
                 band_share=r1['band_share'], crops_per_image=r1['band_candidates_test'] / n, auc=[r1['auc_ranker'], r1['auc_band_with_verifier'], r1['auc_band_control_no_verifier']])
        # gate (validation): point estimates equal the numbers of Tables 3 and 5 of the previous version
        if abs(round(r['s1_m_thr'][0], 2) - GATE3[ri]) > 0.005 or (CHECK_GATE5 and abs(round(r['s2_m_thr'][0], 2) - GATE5[ri]) > 0.005):
            gate_fail.append((e, t, round(r['s1_m_thr'][0], 3), GATE3[ri], round(r['s2_m_thr'][0], 3), GATE5[ri]))
        bf = out / 'baselines' / ('counts_%s_%s.npz' % (e, t))
        if bf.exists():
            B = np.load(bf)
            r['table6'] = {k: dict(f1=f1(B[k]), minus_final=d(B[k], B['final'])) for k in B.files if k not in ('final', 'head', 'thr_train')}
            r['table6']['_final_equals_E31'] = bool(np.array_equal(B['final'], arms['s2']))
        if (e, t) in SWEEP_ROWS:
            sw = {}
            for k in SWEEP_K:
                S = np.load(P3 / ('counts_%s_%s_%s%s.npz' % (e, t, TAG_SW, k)))
                rj = json.load(open(P3 / ('result_%s_%s_%s%s.json' % (e, t, TAG_SW, k))))
                sw[k] = dict(share=rj['band_share'], crops=rj['band_candidates_test'] / n, gain=d(S['verifier'], S['head']), final_m_thr=d(S['verifier'], arms['thr']), f1=f1(S['verifier']),
                             head_equals_s1=bool(np.array_equal(S['head'], arms['s1'])))
            ks = str(band['k_star'])
            sw['_k_star_equals_main'] = bool(np.array_equal(np.load(P3 / ('counts_%s_%s_%s%s.npz' % (e, t, TAG_SW, ks)))['verifier'], arms['s2'])) if ks in SWEEP_K else None
            r['sweep'] = sw
        res['%s_%s' % (e, t)] = r
        print('TAB', e, t, 'stage1-thr %+.2f [%+.2f,%+.2f]  final-thr %+.2f [%+.2f,%+.2f]' % (*r['s1_m_thr'], *r['s2_m_thr']), flush=True)
    res['_gate'] = dict(passed=not gate_fail, failures=gate_fail)
    (out / 'tables.json').write_text(json.dumps(res, indent=1))
    print('GATE (Table 3 stage1-thr, Table 5 final-thr point estimates equal the old numbers):', 'PASS' if not gate_fail else 'FAIL %s' % gate_fail)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--stage', choices=['baselines', 'tables'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--only', nargs=2, help='expert task: one row per process (the DeepSolo and TiCLS evaluators both import `adet` and cannot share an interpreter)')
    p.add_argument('--p3', help='directory with the readings (<e>_<t>_{train,test}_parseq.json) and the counts/result files; default: candidate-banks/p3')
    p.add_argument('--band', help='directory with train_band_*.json and op_audit_*.json; default: operating-point')
    p.add_argument('--tag-main', default='trainonly'); p.add_argument('--tag-targmax', default='targmax'); p.add_argument('--tag-final', default='final'); p.add_argument('--tag-sweep', default='sw')
    p.add_argument('--no-gate5', action='store_true', help='do not compare the final minus thr_train point estimates with the released-reader numbers (use for another reader)')
    a = p.parse_args()
    if a.p3:
        P3 = Path(a.p3)
    if a.band:
        BAND = Path(a.band)
    TAG_MAIN, TAG_TARGMAX, TAG_FINAL, TAG_SW = a.tag_main, a.tag_targmax, a.tag_final, a.tag_sweep
    CHECK_GATE5 = not a.no_gate5
    baselines(a) if a.stage == 'baselines' else tables(a)
