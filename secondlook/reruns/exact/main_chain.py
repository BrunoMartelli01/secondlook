"""E50-exact copy of the E49 orchestrator (arms fast|exact through secondlook.reruns.exact.launch) for the S = all chain of the manuscript (Tables 4-7, 13-15 of the Doc): the E42 `clean` reader chain of secondlook.studies.fulltrain_chain, re-run row by row with an arm
(old = GATE, unified) through the E49 launcher. Stages per row (each one an unchanged repository module in its own process, as the DeepSolo and TiCLS evaluators cannot share an interpreter):
  op        secondlook.selection.operating_point      -> audit/op_audit_<row>.json (ECE, c_half, c_argmax), op_counts
  band      secondlook.selection.verification_band             -> audit/train_band_<row>.json (k*, band edge)
  test      secondlook.second_stage.band_classifier  at (c_half, [0, band edge])            tag _clean
  targmax   secondlook.second_stage.band_classifier  at (c_argmax, [0, min(k* c_argmax, 1)]) tag _targmax
  sweep     secondlook.second_stage.band_classifier  at c_half for every k of the grid      tags _sw<k>
  baselines secondlook.studies.tables_fulltrain --stage baselines                  -> tables/baselines/counts_<row>.npz (thr_train, head, final and the same-reading rules)
Readings: the PARSeq-S* files of fulltrain/readers/clean/p3 (read-only symlinks; they read the crops of the submitted polygons and do not depend on the classifier features).
    python -m secondlook.reruns.fast.main_chain --arm unified --out DIR [--rows deepsolo:totaltext ...] [--jobs 4] [--threads 2]"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from secondlook.paths import REPO_ROOT, RUNS

REPO = REPO_ROOT
RUNS = RUNS
E18 = RUNS / 'candidate-banks'
E42 = RUNS / 'fulltrain'
PY = os.environ.get('SECONDLOOK_PYTHON', sys.executable)
LAUNCH = 'secondlook.reruns.exact.launch'
ROWS9 = [('deepsolo', 'totaltext'), ('deepsolo', 'icdar2015'), ('ticls', 'totaltext'), ('ticls', 'icdar2015'), ('ppocr', 'totaltext'), ('ppocr', 'icdar2015'),
         ('deepsolo', 'ctw1500'), ('ticls', 'ctw1500'), ('ppocr', 'ctw1500')]
SWK = ['1.2', '1.4', '1.6', '1.9', '2.2', '2.6', '3.0', 'all']
TAG = '_clean'


def run(arm, module, args, log, threads):
    cmd = ['nice', '-n', '10', str(PY), '-m', LAUNCH, arm, module] + [str(a) for a in args]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), PYTHONPATH=str(REPO))
    with open(log, 'a') as f:
        f.write('$ %s\n' % ' '.join(cmd))
        f.flush()
        r = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit('FAILED (%d): %s  see %s' % (r.returncode, ' '.join(cmd[4:10]), log))


def hi_of(k, c):
    return 1.0 if str(k) == 'all' else round(min(float(k) * c, 1.0), 4)


def row_chain(a, e, t):
    k = '%s_%s' % (e, t)
    d = Path(a.out) / a.arm / 'main'
    for sub in ('p3', 'audit', 'logs', 'work', 'tables'):
        (d / sub).mkdir(parents=True, exist_ok=True)
    for ex in ('deepsolo', 'ticls', 'ppocr'):
        try:
            if not (d / ex).exists():
                (d / ex).symlink_to(E18 / ex)
        except FileExistsError:   # another row thread created it (race)
            pass
    for sp in ('train', 'test'):
        dst = d / 'p3' / ('%s_%s_parseq.json' % (k, sp))
        if not dst.exists():
            dst.symlink_to((E42 / 'readers/clean/p3' / ('%s_%s_parseq.json' % (k, sp))).resolve())
    log = d / 'logs' / ('%s.log' % k)
    t0 = time.time()
    if not (d / 'audit' / ('op_audit_%s.json' % k)).exists():
        (d / 'work' / ('op_%s' % k)).mkdir(exist_ok=True)
        run(a.arm, 'secondlook.selection.operating_point', ['--expert', e, '--task', t, '--work', d / 'work' / 'op', '--out', d / 'audit'], log, a.threads)
    if not (d / 'audit' / ('train_band_%s.json' % k)).exists():
        (d / 'work' / 'band').mkdir(exist_ok=True)
        run(a.arm, 'secondlook.selection.verification_band', ['--expert', e, '--task', t, '--audit', d / 'audit', '--p3', d / 'p3', '--work', d / 'work' / 'band'], log, a.threads)
    band = json.load(open(d / 'audit' / ('train_band_%s.json' % k)))
    aud = json.load(open(d / 'audit' / ('op_audit_%s.json' % k)))
    c, ks = band['c'], band['k_star']

    def p3(cut, hi, tag):
        if (d / 'p3' / ('result_%s%s.json' % (k, tag))).exists():
            return
        w = d / 'work' / ('x%s_%s' % (tag, k))
        w.mkdir(parents=True, exist_ok=True)
        run(a.arm, 'secondlook.second_stage.band_classifier', ['--expert', e, '--task', t, '--work', w, '--out', d, '--cut', cut, '--band-lo', 0, '--band-hi', hi, '--tag', tag], log, a.threads)
    p3(c, hi_of(ks, c), TAG)
    p3(aud['c_argmax'], hi_of(ks, aud['c_argmax']), '_targmax')
    for kk in SWK:
        p3(c, hi_of(kk, c), '_sw' + kk)
    if not (d / 'tables' / 'baselines' / ('counts_%s.npz' % k)).exists():
        run(a.arm, 'secondlook.studies.tables_fulltrain', ['--stage', 'baselines', '--only', e, t, '--out', d / 'tables', '--p3', d / 'p3', '--band', d / 'audit', '--tag-main', 'clean', '--no-gate5'], log, a.threads)
    print('row done', a.arm, k, '%.0fs' % (time.time() - t0), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', required=True, choices=['fast', 'exact'])
    p.add_argument('--out', required=True)
    p.add_argument('--rows', nargs='*', help='expert:task ...')
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    rows = [tuple(r.split(':')) for r in a.rows] if a.rows else ROWS9
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda r: row_chain(a, *r), rows))
    import shutil
    shutil.rmtree(Path(a.out) / a.arm / 'main' / 'work', ignore_errors=True)


if __name__ == '__main__':
    main()
