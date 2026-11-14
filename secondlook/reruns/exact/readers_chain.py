"""E50-exact copy of the E49 orchestrator (arms fast|exact through secondlook.reruns.exact.launch) for the second-reader comparison (Doc Tables 16-17), four rows (DeepSolo and TiCLS on Total-Text and ICDAR2015) x readers {vitstr, abinet, shuffled}, one arm (old = gate, unified).
It repeats, with the E42 reader chain's stage definitions (secondlook.studies.fulltrain_chain: band, test), the second stage for each reader's OWN readings: k* re-selected on TRAIN by the nested cross-fit
(secondlook.selection.verification_band --p3 <reader readings>), then secondlook.second_stage.band_classifier at (c_half, [0, band edge]) with the reader's readings and the tag of the reader. c_half comes from the arm's own
S = all chain (RUN/<arm>/main/audit/op_audit_<row>.json, symlinked into each reader's audit directory). The readings are the E42 merged reading files (read-only symlinks); they read crops of the
submitted polygons, so they do not depend on the classifier features (the unified geometry changes no polygon). Word accuracy (overall and in the reader's own band) is recomputed with
secondlook.second_stage.reader_accuracy for every reader, the main PARSeq-S* included (reader `clean` = RUN/<arm>/main).
    python -m secondlook.reruns.fast.readers_chain --arm unified --out DIR [--jobs 4]"""
import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from secondlook.reruns.exact.main_chain import E18, E42, LAUNCH, PY, REPO, hi_of

ROWS4 = [('deepsolo', 'totaltext'), ('deepsolo', 'icdar2015'), ('ticls', 'totaltext'), ('ticls', 'icdar2015')]
READERS = {'vitstr': 'vitstr', 'abinet': 'abinet', 'shuffled': 'shuffled'}   # name -> p3 tag (secondlook.studies.fulltrain_chain.READERS)


def run(arm, module, args, log, threads):
    cmd = ['nice', '-n', '10', str(PY), '-m', LAUNCH, arm, module] + [str(a) for a in args]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), PYTHONPATH=str(REPO))
    with open(log, 'a') as f:
        f.write('$ %s\n' % ' '.join(cmd))
        f.flush()
        r = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit('FAILED (%d): %s  see %s' % (r.returncode, ' '.join(cmd[4:10]), log))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def one(a, name, e, t):
    k = '%s_%s' % (e, t)
    root = Path(a.out) / a.arm
    d = root / 'readers' / name
    for sub in ('p3', 'audit', 'logs', 'work'):
        (d / sub).mkdir(parents=True, exist_ok=True)
    def link(dst, src):
        try:
            if not dst.exists():
                dst.symlink_to(src)
        except FileExistsError:   # another thread of the same reader created it
            pass
    for ex in ('deepsolo', 'ticls', 'ppocr'):
        link(d / ex, E18 / ex)
    for sp in ('train', 'test'):
        link(d / 'p3' / ('%s_%s_parseq.json' % (k, sp)), (E42 / 'readers' / name / 'p3' / ('%s_%s_parseq.json' % (k, sp))).resolve())
    link(d / 'audit' / ('op_audit_%s.json' % k), (root / 'main' / 'audit' / ('op_audit_%s.json' % k)).resolve())
    log = d / 'logs' / ('%s.log' % k)
    if not (d / 'audit' / ('train_band_%s.json' % k)).exists():
        (d / 'work' / 'band').mkdir(exist_ok=True)
        run(a.arm, 'secondlook.selection.verification_band', ['--expert', e, '--task', t, '--audit', d / 'audit', '--p3', d / 'p3', '--work', d / 'work' / 'band'], log, a.threads)
    band = json.load(open(d / 'audit' / ('train_band_%s.json' % k)))
    tag = '_' + READERS[name]
    if not (d / 'p3' / ('result_%s%s.json' % (k, tag))).exists():
        w = d / 'work' / ('x%s_%s' % (tag, k))
        w.mkdir(parents=True, exist_ok=True)
        run(a.arm, 'secondlook.second_stage.band_classifier', ['--expert', e, '--task', t, '--work', w, '--out', d, '--cut', band['c'], '--band-lo', 0, '--band-hi', hi_of(band['k_star'], band['c']), '--tag', tag], log, a.threads)
    print('reader row done', a.arm, name, k, flush=True)


def accuracy(a, e, t):
    root = Path(a.out) / a.arm
    out = root / 'accuracy'
    out.mkdir(parents=True, exist_ok=True)
    if (out / ('acc.%s_%s.json' % (e, t))).exists():
        return
    args = ['--reader', 'clean=%s' % (root / 'main')] + sum((['--reader', '%s=%s' % (n, root / 'readers' / n)] for n in READERS), []) + ['--work', out / 'work', '--out', out / 'acc', '--only', e, t]
    run(a.arm, 'secondlook.second_stage.reader_accuracy', args, out / ('%s_%s.log' % (e, t)), a.threads)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', required=True, choices=['fast', 'exact'])
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    for e, t in ROWS4:   # the main chain of the arm must be complete
        if not (Path(a.out) / a.arm / 'main' / 'p3' / ('result_%s_%s_clean.json' % (e, t))).exists():
            raise SystemExit('run secondlook.reruns.exact.main_chain first for %s %s' % (e, t))
    jobs = [(n, e, t) for n in READERS for e, t in ROWS4]
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda j: one(a, *j), jobs))
        list(ex.map(lambda r: accuracy(a, *r), ROWS4))
    meta = {n: {'%s_%s_%s' % (e, t, sp): sha(E42 / 'readers' / n / 'p3' / ('%s_%s_%s_parseq.json' % (e, t, sp))) for e, t in ROWS4 for sp in ('train', 'test')} for n in list(READERS) + ['clean']}
    (Path(a.out) / a.arm / 'readers' / 'reading_files_sha256.json').write_text(json.dumps(meta, indent=1))
    import shutil
    for n in READERS:
        shutil.rmtree(Path(a.out) / a.arm / 'readers' / n / 'work', ignore_errors=True)
    shutil.rmtree(Path(a.out) / a.arm / 'accuracy' / 'work', ignore_errors=True)


if __name__ == '__main__':
    main()
