"""E49: secondlook.studies.cross_dataset (leave-one-benchmark-out and pooled fits) re-run through the E49 launcher. The module is the unchanged E44 (B) code; the only difference is where its GATE
looks for the reference of a single-benchmark fit:
  arm old     : the stored E42/E44 files (fulltrain, as in E44);
  arm unified : the E49 unified S = all chain (RUN/unified/main: counts_<row>_clean.npz) and the E49 unified stability.json (row 'all'), which themselves passed the E46 gate.
The patch of `secondlook.evaluation.common.CLEAN` / `stability_path` is made BEFORE secondlook.studies.cross_dataset is imported, so its `from secondlook.evaluation.common import ...` binds the E49 locations.
    python -m secondlook.reruns.fast.launch {old|unified} secondlook.reruns.fast.cross_dataset {dump|fit|evaluate|assemble} --expert E [--task T] --out DIR"""
import argparse
import os
import sys
from pathlib import Path
from secondlook.paths import RUNS

RUN = (RUNS / 'rerun-fast')

if __name__ == '__main__':
    import secondlook.evaluation.common as common
    arm = os.environ['SECONDLOOK_FEATURES']
    if arm == 'unified':
        common.CLEAN = RUN / 'unified' / 'main'      # CLEAN / 'p3' holds the readings (symlinks) and counts_<row>_clean.npz of the unified chain
        common.stability_path = lambda e, t: RUN / 'unified' / 'stability' / ('%s_%s' % (e, t)) / 'stability.json'
    import secondlook.studies.cross_dataset as x
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['dump', 'fit', 'evaluate', 'assemble'])
    p.add_argument('--expert', required=True, choices=['deepsolo', 'ticls', 'ppocr'])
    p.add_argument('--task', choices=x.TASKS)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    {'dump': x.dump, 'fit': x.fit_stage, 'evaluate': x.evaluate_stage, 'assemble': x.assemble}[a.stage](a)
