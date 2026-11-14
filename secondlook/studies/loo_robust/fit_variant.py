"""Diagnostic (approximate reproduction, CPU): leave-one-dataset-out with a restricted feature set, the REST of pipeline U verbatim (secondlook.studies.cross_dataset.fit_pool / keeps_for).
 stages: fit EXPERT TARGET COLSET OUT    |    evaluate EXPERT TARGET COLSET OUT (run through secondlook.reruns.fast.launch unified, exact pinned evaluator, TEST scored once per frozen fit)
COLSETS: noText = columns 0-7 (drops log_len, digit_frac, letter_frac); detrec = columns 0-4 (detector + recognition statistics only)."""
import pickle, sys, json, numpy as np
from pathlib import Path
from secondlook.paths import RUNS
COLS = {'noText': list(range(8)), 'detrec': list(range(5)), 'full': list(range(11)), 'Ra': list(range(8)) + [11], 'Rb': list(range(8)) + [11, 12]}
AUG = (RUNS / 'loo-diagnosis')
R = (RUNS / 'rerun-fast/unified/crossdata')
def slim(d, cols):
    d = dict(d); d['xc'] = {s: v[:, cols] for s, v in d['xc'].items()}; d['eval_xc'] = {s: v[:, cols] for s, v in d['eval_xc'].items()}; return d
if __name__ == '__main__':
    stage, e, target, cs, mode, out = sys.argv[1:7]   # mode: loo | own
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if stage == 'fit':
        from secondlook.studies.cross_dataset import Pool, TASKS, fit_pool, keeps_for
        D = {t: slim(pickle.load(open(AUG / ('aug_pool_%s_%s.pkl' % (e, t)), 'rb')), COLS[cs]) for t in TASKS}
        P, VT = Pool(), {}
        for t in TASKS:
            if (mode == 'loo' and t == target) or (mode == 'own' and t != target): continue
            P.add(t, D[t])
            for s in D[t]['ids']: VT['%s|%s' % (t, s)] = D[t]['vt'][s]
        S = list(P.ids)
        meta, head, band = fit_pool(P, S, VT)
        k1, k2 = keeps_for(meta, head, band, D[target])
        pickle.dump(dict(meta=meta, keeps=dict(first=k1, complete=k2)), open(out / ('fit_%s_%s_%s_%s.pkl' % (e, target, cs, mode)), 'wb'))
        print('FIT', e, target, cs, mode, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items() if k != 'nested_f1_by_k'}, flush=True)
    else:
        from secondlook.selection.operating_point import load
        from secondlook.evaluation.common import counts_full
        f = pickle.load(open(out / ('fit_%s_%s_%s_%s.pkl' % (e, target, cs, mode)), 'rb'))
        work = out / ('work_%s_%s_%s_%s' % (e, target, cs, mode)); work.mkdir(exist_ok=True)
        T = load(e, target, work)
        res = {}
        for st in ('first', 'complete'):
            c = counts_full(T, work, {s: f['keeps'][st][s] for s in T.eval_ids})
            tp, pr, gt = (float(x) for x in c[:, :3].sum(0)); res[st] = dict(P=100 * tp / pr, R=100 * tp / gt, F=200 * tp / (pr + gt), n=int(pr), counts=c.tolist())
            print('EVAL %s %s %s %s %s P=%.2f R=%.2f F=%.2f npred=%d' % (e, target, cs, mode, st, res[st]['P'], res[st]['R'], res[st]['F'], int(pr)), flush=True)
        json.dump(dict(meta=f['meta'], res=res), open(out / ('eval_%s_%s_%s_%s.json' % (e, target, cs, mode)), 'w'))
