"""E7: DeepSolo word admission - repeated source partitions and sealed TextOCR cohorts.

Stages (legacy Python 3.8, sklearn 0.23.2):
  seeds  : --dataset D --seed S. Refit HGB / logistic / two-threshold grid on a random (S>0) or the
           E6 (S=0) 584/80/64 partition of the 728 TRAIN ids, evaluate ONCE on TEST, and freeze the policy
           (joblib models + cuts + thresholds + sha256) under <out>/<D>/seed<S>/.
  infer  : DeepSolo (existing qualified profile) on the TextOCR cohort images -> compact samples. GPU.
  sealed : apply every frozen policy to the cohort (TT task, IC15 task, VimTS word task) with the pinned
           evaluators and unchanged cohort GT. No fitting, no cut choice.
  report : aggregate seeds and sealed results (paired bootstrap, seed 20261104, 10000 draws).
Approximate reproduction. TEST is exploratory; cohort images are 'sealed' only w.r.t. our own fitting."""

from pathlib import Path

import numpy as np

from secondlook.evaluation.protocol import load_config, read_json, hmean_fraction

from secondlook.data.deepsolo_table import CAL, CFG
from secondlook.paths import RUNS

COHORT = (RUNS / 'ticls-source-banks/textocr_cohort_v1')

TASKS = {'totaltext': ('totaltext', 'totaltext_native'), 'icdar2015': ('icdar2015', 'icdar2015_native'), 'ctw1500_word': ('totaltext', 'ctw_source_native')}  # VimTS word-evaluator task on the TT cohort images

class Ctx:
    def __init__(self, dataset):
        self.cfg = load_config(Path(CFG))
        self.dataset = dataset
        self.exp = self.cfg['protocol_calibration']['experiments'][dataset + '_native']
        from secondlook.evaluation.declared_metric import DeclaredMetric
        self.metric = DeclaredMetric(self.cfg, self.exp['metric'])
        self.base = CAL / (dataset + '_native') / dataset
        self.parts = read_json(self.base / 'partitions.json')
        self.train_ids = self.parts['fit_ids'] + self.parts['selection_ids'] + self.parts['calibration_ids']
        self.names = {sid: v['filename'] for sid, v in read_json(self.base / 'native/bodies.json').items()}

def evaluate(ctx, work, tag, ids, tab, keep, gt, filenames, metric=None):
    from secondlook.evaluation.official_benchmarks import write_zip
    z = Path(work) / (tag + '.zip')
    if z.exists():
        z.unlink()
    write_zip(z, {filenames[s]: ''.join(l for l, k in zip(tab[s]['lines'], keep[s]) if k) for s in ids})
    return (metric or ctx.metric).evaluate(gt, z, ids, {s: dict(filename=filenames[s]) for s in ids})

def f1_of(c):
    return float(hmean_fraction(c))

def subset_gt_zip(ctx, ids, dest):
    from secondlook.features.word_trust import subset_gt
    dest = Path(dest)
    if dest.exists():
        dest.unlink()
    subset_gt(ctx.base / 'native/source_gt.zip', dest, ctx.metric.module, {s: ctx.names[s] for s in ids}, ctx.dataset)
    return dest

def drop_invalid_quads(tab, ids):
    """IC15 evaluator rejects non-clockwise quadrilaterals (hard error). Remove such candidates for EVERY rule
    (ASSUMPTION: an unscorable submission line cannot be part of any policy's output); counts are saved."""
    removed = 0
    for s in ids:
        keep = []
        for line in tab[s]['lines']:
            v = [int(float(x)) for x in line.split(',')[:8]]
            area = (v[2] - v[0]) * (v[3] + v[1]) + (v[4] - v[2]) * (v[5] + v[3]) + (v[6] - v[4]) * (v[7] + v[5]) + (v[0] - v[6]) * (v[1] + v[7])
            keep.append(area <= 0)
        removed += len(keep) - sum(keep)
        k = np.asarray(keep, bool)
        tab[s] = dict(lines=[l for l, ok in zip(tab[s]['lines'], keep) if ok], x=tab[s]['x'][k], score=tab[s]['score'][k], rec=tab[s]['rec'][k])
    print('dropped invalid IC15 quadrilaterals:', removed, flush=True)
    return tab
