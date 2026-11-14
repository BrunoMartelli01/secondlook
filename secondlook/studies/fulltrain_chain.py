#!/usr/bin/env python3
"""E42 driver (adapted from reader-harness/run_reader.py and run_tables.py, which are NOT modified): the reader-dependent chain on the FULL TRAIN pool.
One directory per reader: OUT/readers/<name>/{p3,audit,logs,work,<expert> links,tables}. Stages:
  merge     p3/<expert>_<task>_{train,test}_parseq.json for the nine rows: the six Total-Text/ICDAR2015 TRAIN files = old readings + readings of the extra images
            (readings/<reader>/<expert>_<task>_train_extra_parseq.json, written by the reader harness); TEST files and CTW1500 files are links to the old readings; audit/ links.
  band      secondlook.selection.verification_band (k* by nested 5-fold on the full pool with THIS reader's readings)     [through secondlook.studies.fulltrain_launch]
  test      secondlook.second_stage.band_classifier at (c_half, band [0, min(k* c_half, 1)]), tag _<tag>
  targmax   p3_analyze at c_argmax with the same k*, tag _targmax                              (main reader)
  sweep     p3_analyze at c_half for every k of the grid, tags _sw<k>                          (main reader, four rows)
  baselines secondlook.studies.tables_fulltrain --stage baselines (same-reader rules of Table 6 + thr_train re-tuned on the full pool)
  tables    secondlook.studies.tables_fulltrain --stage tables
CTW1500 rows are unchanged: their band/audit/counts/result/baselines files are links to the old reader directory.
CPU only, nice 10; jobs x threads-per-job <= 4. Approximate reproduction. No spotter is trained."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from secondlook.paths import REPO_ROOT, RUNS

REPO = REPO_ROOT
RUNS = RUNS
E18, E30 = RUNS / 'candidate-banks', RUNS / 'operating-point'
H = RUNS / 'reader-harness' / 'parseq-s-clean'
OUT = RUNS / 'fulltrain'
PY = Path(os.environ.get('SECONDLOOK_PYTHON', sys.executable))
ROWS9 = [('deepsolo', 'totaltext'), ('deepsolo', 'icdar2015'), ('ticls', 'totaltext'), ('ticls', 'icdar2015'), ('ppocr', 'totaltext'), ('ppocr', 'icdar2015'),
         ('deepsolo', 'ctw1500'), ('ticls', 'ctw1500'), ('ppocr', 'ctw1500')]
ROWS4 = ROWS9[:4]
CHANGED = ROWS9[:6]
NEW5 = [ROWS9[4], ROWS9[5], ROWS9[6], ROWS9[7], ROWS9[8]]   # E43: rows without ViTSTR-S / ABINet / shuffled results
SWK = ['1.2', '1.4', '1.6', '1.9', '2.2', '2.6', '3.0', 'all']
# name -> (old readings dir, old reader dir with audit/p3/tables (for CTW links), tag used by p3_analyze here, rows)
READERS = {
    'clean': dict(old_p3=H / 'p3', old_dir=H, tag='clean', rows=ROWS9, rdir='parseq-s-clean'),
    'released': dict(old_p3=E18 / 'p3', old_dir=None, tag='released', rows=CHANGED, rdir='parseq-s-released'),
    'vitstr': dict(old_p3=E18 / 'vitstr_run/p3', old_dir=None, tag='vitstr', rows=ROWS4, rdir='vitstr'),
    'shuffled': dict(old_p3=OUT / 'shuffled_src', old_dir=None, tag='shuffled', rows=ROWS4, rdir=None, complete=True),
    'abinet': dict(old_p3=E18 / 'abinet_run/p3', old_dir=None, tag='abinet', rows=ROWS4, rdir='abinet'),
}
key = lambda e, t: '%s_%s' % (e, t)


def run(cmd, log, threads, env_extra=None):
    cmd = ['nice', '-n', '10'] + [str(c) for c in cmd]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), PYTHONPATH=str(REPO), **(env_extra or {}))
    with open(log, 'a') as f:
        f.write('$ %s\n' % ' '.join(cmd))
        f.flush()
        r = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit('FAILED (%d): %s  see %s' % (r.returncode, ' '.join(cmd[:9]), log))


def clean_scratch(w):
    for d in Path(w).glob('scratch_*'):
        shutil.rmtree(d, ignore_errors=True)


def link(dst, src):
    dst = Path(dst)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(Path(src).resolve())


def stage_merge(a, d, spec):
    (d / 'p3').mkdir(exist_ok=True)
    (d / 'audit').mkdir(exist_ok=True)
    for e in ('deepsolo', 'ticls', 'ppocr'):
        if not (d / e).exists():
            (d / e).symlink_to(E18 / e)
    ex = OUT / 'readings' / (spec['rdir'] or 'none')
    for e, t in spec['rows']:
        k = key(e, t)
        old_tr = json.load(open(spec['old_p3'] / ('%s_train_parseq.json' % k)))
        if (e, t) in CHANGED and spec.get('complete'):
            link(d / 'p3' / ('%s_train_parseq.json' % k), spec['old_p3'] / ('%s_train_parseq.json' % k))
        elif (e, t) in CHANGED:
            new = json.load(open(ex / ('%s_train_extra_parseq.json' % k)))
            crops = np_load_owner(OUT / 'crops' / ('%s_train_extra.npz' % k))
            if new['owner'] != crops['owner'] or new['cidx'] != crops['cidx'] or len(new['rows']) != len(crops['owner']):
                raise SystemExit('extra readings are not aligned with the extra crops: ' + k)
            if set(new['owner']) & set(old_tr['owner']):
                raise SystemExit('extra readings overlap the old TRAIN readings: ' + k)
            merged = dict(owner=old_tr['owner'] + new['owner'], cidx=old_tr['cidx'] + new['cidx'], rows=old_tr['rows'] + new['rows'])
            tgt = d / 'p3' / ('%s_train_parseq.json' % k)
            if tgt.is_symlink() or tgt.exists():
                tgt.unlink()
            tgt.write_text(json.dumps(merged))
        else:
            link(d / 'p3' / ('%s_train_parseq.json' % k), spec['old_p3'] / ('%s_train_parseq.json' % k))
        link(d / 'p3' / ('%s_test_parseq.json' % k), spec['old_p3'] / ('%s_test_parseq.json' % k))
        # audit: op_audit of the changed rows from this run, CTW from E30; train_band of CTW rows from the old reader directory
        link(d / 'audit' / ('op_audit_%s.json' % k), (OUT / 'op' if (e, t) in CHANGED else E30) / ('op_audit_%s.json' % k))
        if (e, t) not in CHANGED:
            if spec['old_dir'] is not None:
                link(d / 'audit' / ('train_band_%s.json' % k), spec['old_dir'] / 'audit' / ('train_band_%s.json' % k))
                for f in (spec['old_dir'] / 'p3').glob('*_%s_*' % k):
                    if f.name.startswith(('counts_', 'result_')):
                        link(d / 'p3' / f.name, f)
                bl = spec['old_dir'] / 'tables' / 'baselines' / ('counts_%s.npz' % k)
                if bl.exists():
                    (d / 'tables' / 'baselines').mkdir(parents=True, exist_ok=True)
                    link(d / 'tables' / 'baselines' / bl.name, bl)
    print('merge done', a.reader)


def stage_merge_e43(a, d, spec):
    """E43: readings of the five missing rows. ppocr TT/IC15: readings_e43 (old 728 TRAIN images; TEST) + the E42 extra-image readings; CTW1500: readings_e43 as they are. Alignment with the cached crop files is checked."""
    (d / 'p3').mkdir(exist_ok=True)
    (d / 'audit').mkdir(exist_ok=True)
    for e in ('deepsolo', 'ticls', 'ppocr'):
        if not (d / e).exists():
            (d / e).symlink_to(E18 / e)
    src = OUT / ('shuffled_src_e43' if spec.get('complete') else 'readings_e43/%s' % spec['rdir'])
    ex = OUT / 'readings' / (spec['rdir'] or 'none')
    for e, t in NEW5:
        k = key(e, t)
        for sp in ('train', 'test'):
            tgt = d / 'p3' / ('%s_%s_parseq.json' % (k, sp))
            ref = np_load_owner(E18 / 'p3' / ('%s_%s.npz' % (k, sp)))
            part = json.load(open(src / ('%s_%s_parseq.json' % (k, sp))))
            if sp == 'train' and (e, t) in CHANGED and not spec.get('complete'):
                if part['owner'] != ref['owner'] or part['cidx'] != ref['cidx']:
                    raise SystemExit('old-pool readings not aligned with the cached crops: ' + k)
                new = json.load(open(ex / ('%s_train_extra_parseq.json' % k)))
                crops = np_load_owner(OUT / 'crops' / ('%s_train_extra.npz' % k))
                if new['owner'] != crops['owner'] or new['cidx'] != crops['cidx']:
                    raise SystemExit('extra readings not aligned: ' + k)
                merged = dict(owner=part['owner'] + new['owner'], cidx=part['cidx'] + new['cidx'], rows=part['rows'] + new['rows'])
                if tgt.is_symlink() or tgt.exists():
                    tgt.unlink()
                tgt.write_text(json.dumps(merged))
            else:
                if sp == 'test' or (e, t) not in CHANGED:
                    if part['owner'] != ref['owner'] or part['cidx'] != ref['cidx']:
                        raise SystemExit('readings not aligned with the cached crops: %s %s' % (k, sp))
                link(tgt, src / ('%s_%s_parseq.json' % (k, sp)))
        link(d / 'audit' / ('op_audit_%s.json' % k), (OUT / 'op' if (e, t) in CHANGED else E30) / ('op_audit_%s.json' % k))
    print('merge e43 done', a.reader)


def np_load_owner(p):
    import numpy as np
    z = np.load(p)
    return dict(owner=z['owner'].tolist(), cidx=z['cidx'].tolist())


def band_of(d, e, t):
    b = json.load(open(d / 'audit' / ('train_band_%s.json' % key(e, t))))
    return b['c'], b['k_star']


def hi_of(k, c):
    return 1.0 if str(k) == 'all' else round(min(float(k) * c, 1.0), 4)


def launch(mod):
    return [PY, '-m', 'secondlook.studies.fulltrain_launch', mod]


def parallel(fn, items, jobs):
    with ThreadPoolExecutor(jobs) as ex:
        return list(ex.map(fn, items))


def p3a(d, spec, e, t, cut, hi, tag, threads, force, env_extra):
    if (d / 'p3' / ('result_%s%s.json' % (key(e, t), tag))).exists() and not force:
        return
    w = d / 'work' / ('x%s_%s' % (tag, key(e, t)))
    w.mkdir(parents=True, exist_ok=True)
    clean_scratch(w)
    run(launch('secondlook.second_stage.band_classifier') + ['--expert', e, '--task', t, '--work', w, '--out', d, '--cut', cut, '--band-lo', 0, '--band-hi', hi, '--tag', tag], d / 'logs' / ('test_%s%s.log' % (key(e, t), tag)), threads, env_extra)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reader', required=True, choices=list(READERS))
    p.add_argument('--stages', default='merge,band,test')
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--threads-per-job', type=int, default=2)
    p.add_argument('--only', nargs=2)
    p.add_argument('--force', action='store_true')
    p.add_argument('--e43', action='store_true', help='E43: process the five rows NEW5 (ViTSTR-S / ABINet / shuffled readings from readings_e43)')
    p.add_argument('--old-pool', action='store_true', help='END-TO-END GATE: FT_OLD=1, the unchanged 728-image pool through the same launcher (output dir suffix _oldpool)')
    a = p.parse_args()
    if a.jobs * a.threads_per_job > 4:
        sys.exit('refusing: more than 4 CPU threads')
    spec = READERS[a.reader]
    d = OUT / 'readers' / (a.reader + ('_oldpool' if a.old_pool else ''))
    for sub in ('p3', 'audit', 'logs', 'work', 'tables'):
        (d / sub).mkdir(parents=True, exist_ok=True)
    env_extra = {'FT_OLD': '1'} if a.old_pool else {}
    rows = [tuple(a.only)] if a.only else (NEW5 if a.e43 else spec['rows'])
    changed = list(rows) if a.e43 else [r for r in rows if r in CHANGED]
    st = a.stages.split(',')
    tag = '_' + spec['tag']
    t0 = time.time()
    if 'merge' in st:
        if a.old_pool:
            sys.exit('old-pool gate: build the directory with the old-pool script')
        (stage_merge_e43 if a.e43 else stage_merge)(a, d, spec)
    if 'band' in st:
        def one(r):
            e, t = r
            if (d / 'audit' / ('train_band_%s.json' % key(e, t))).exists() and not a.force:
                return
            w = d / 'work' / 'band'
            w.mkdir(exist_ok=True)
            clean_scratch(w / key(e, t))
            run(launch('secondlook.selection.verification_band') + ['--expert', e, '--task', t, '--audit', d / 'audit', '--p3', d / 'p3', '--work', w], d / 'logs' / ('band_%s.log' % key(e, t)), a.threads_per_job, env_extra)
        parallel(one, changed, a.jobs)
    if 'test' in st:
        def one(r):
            e, t = r
            c, k = band_of(d, e, t)
            p3a(d, spec, e, t, c, hi_of(k, c), tag, a.threads_per_job, a.force, env_extra)
        parallel(one, changed, a.jobs)
    if 'targmax' in st:
        def one(r):
            e, t = r
            c, k = band_of(d, e, t)
            ca = json.load(open(d / 'audit' / ('op_audit_%s.json' % key(e, t))))['c_argmax']
            p3a(d, spec, e, t, ca, hi_of(k, ca), '_targmax', a.threads_per_job, a.force, env_extra)
        parallel(one, changed, a.jobs)
    if 'sweep' in st:
        jobs = []
        for e, t in [r for r in changed if r in ROWS4]:
            c, k = band_of(d, e, t)
            jobs += [(e, t, c, hi_of(kk, c), '_sw' + kk) for kk in SWK]
        parallel(lambda j: p3a(d, spec, j[0], j[1], j[2], j[3], j[4], a.threads_per_job, a.force, env_extra), jobs, a.jobs)
    base = launch('secondlook.studies.tables_fulltrain') + ['--out', d / 'tables', '--p3', d / 'p3', '--band', d / 'audit', '--tag-main', spec['tag'], '--no-gate5']
    if 'baselines' in st:
        parallel(lambda r: run(base + ['--stage', 'baselines', '--only', r[0], r[1]], d / 'logs' / ('baselines_%s.log' % key(*r)), a.threads_per_job, env_extra), changed, a.jobs)
    if 'tables' in st:
        run(base + ['--stage', 'tables'], d / 'logs' / 'tables.log', a.threads_per_job, env_extra)
    print('stages %s done for %s in %.0fs' % (a.stages, a.reader, time.time() - t0))


if __name__ == '__main__':
    main()
