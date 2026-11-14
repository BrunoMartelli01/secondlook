#!/usr/bin/env python3
"""E41/E33-PROMOTION driver for the reader-dependent tables of one reader directory (default: parseq-s-clean). CPU only, nice 10, at most 8 threads (jobs x threads-per-job).
Stages: targmax (complete method at c_argmax, band [0, min(k* c_argmax, 1)], tag _targmax), sweep (four rows, k in the grid, cut c_half, tag _sw<k>), baselines + tables (secondlook.studies.tables with --p3/--band redirection).
Writes only inside the reader directory (new tags); no existing result file is overwritten. Approximate reproduction."""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from secondlook.paths import REPO_ROOT, RUNS
REPO = REPO_ROOT; H = (RUNS / 'reader-harness')
E30 = (RUNS / 'operating-point')
PY = Path(os.environ.get('SECONDLOOK_PYTHON', sys.executable))
ROWS = [('deepsolo', 'totaltext'), ('deepsolo', 'icdar2015'), ('ticls', 'totaltext'), ('ticls', 'icdar2015'), ('ppocr', 'totaltext'), ('ppocr', 'icdar2015'),
        ('deepsolo', 'ctw1500'), ('ticls', 'ctw1500'), ('ppocr', 'ctw1500')]
SWK = ['1.2', '1.4', '1.6', '1.9', '2.2', '2.6', '3.0', 'all']
key = lambda e, t: '%s_%s' % (e, t)
def run(cmd, log, threads):
    cmd = ['nice', '-n', '10'] + [str(c) for c in cmd]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), PYTHONPATH=str(REPO))
    with open(log, 'a') as f:
        f.write('$ %s\n' % ' '.join(cmd)); f.flush()
        r = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit('FAILED (%d): %s  see %s' % (r.returncode, ' '.join(cmd[:8]), log))
def band(out, e, t):
    b = json.load(open(out / 'audit' / ('train_band_%s.json' % key(e, t))))
    return b['c'], b['k_star']
def hi_of(k, c):
    return 1.0 if k == 'all' else round(min(float(k) * c, 1.0), 4)
def p3a(out, e, t, cut, hi, tag, threads, force):
    if (out / 'p3' / ('result_%s%s.json' % (key(e, t), tag))).exists() and not force:
        return
    w = out / 'work' / ('x%s_%s' % (tag, key(e, t))); w.mkdir(parents=True, exist_ok=True)
    import shutil
    for d in w.glob('scratch_*'):
        shutil.rmtree(d)
    run([PY, '-m', 'secondlook.second_stage.band_classifier', '--expert', e, '--task', t, '--work', w, '--out', out, '--cut', cut, '--band-lo', 0, '--band-hi', hi, '--tag', tag], out / 'logs' / ('tables_%s%s.log' % (key(e, t), tag)), threads)
def main():
    p = argparse.ArgumentParser(); p.add_argument('--reader-dir', default=str(H / 'parseq-s-clean')); p.add_argument('--main-tag', default='clean')
    p.add_argument('--stages', default='targmax,sweep,baselines,tables'); p.add_argument('--jobs', type=int, default=3); p.add_argument('--threads-per-job', type=int, default=2)
    p.add_argument('--only', nargs=2); p.add_argument('--force', action='store_true'); p.add_argument('--tables-out', default=None); p.add_argument('--no-gate5', action='store_true')
    a = p.parse_args(); out = Path(a.reader_dir); (out / 'logs').mkdir(exist_ok=True)
    if a.jobs * a.threads_per_job > 8:
        sys.exit('refusing more than 8 threads in all')
    st = a.stages.split(','); tout = Path(a.tables_out) if a.tables_out else out / 'tables'; tout.mkdir(parents=True, exist_ok=True)
    rows = [tuple(a.only)] if a.only else ROWS
    for e in ('deepsolo', 'ticls', 'ppocr'):
        if not (out / e).exists():
            (out / e).symlink_to(str(RUNS / 'candidate-banks/') + e)
    jobs = []
    if 'targmax' in st:
        for e, t in rows:
            c, k = band(out, e, t); ca = json.load(open(out / 'audit' / ('op_audit_%s.json' % key(e, t))))['c_argmax']
            jobs.append((e, t, ca, hi_of(k, ca), '_targmax'))
    if 'sweep' in st:
        for e, t in [r for r in rows if r in ROWS[:4]]:
            c, k = band(out, e, t)
            for kk in SWK:
                jobs.append((e, t, c, hi_of(kk, c), '_sw' + kk))
    t0 = time.time()
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda j: p3a(out, j[0], j[1], j[2], j[3], j[4], a.threads_per_job, a.force), jobs))
    print('p3_analyze jobs: %d in %.0fs' % (len(jobs), time.time() - t0), flush=True)
    base = [PY, '-m', 'secondlook.studies.tables', '--out', tout, '--p3', out / 'p3', '--band', out / 'audit', '--tag-main', a.main_tag] + (['--no-gate5'] if a.no_gate5 else [])
    if 'baselines' in st:                                                  # one row per process (adet clash)
        def one(r):
            run(base + ['--stage', 'baselines', '--only', r[0], r[1]], out / 'logs' / ('tables_baselines_%s.log' % key(*r)), a.threads_per_job)
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(one, rows))
    if 'tables' in st:
        run(base + ['--stage', 'tables'], out / 'logs' / 'tables_tables.log', a.threads_per_job)
if __name__ == '__main__':
    main()
