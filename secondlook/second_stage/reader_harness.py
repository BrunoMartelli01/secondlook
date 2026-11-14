#!/usr/bin/env python3
"""E36 driver: evaluate ONE second reader on the nine rows of the paper draft (approximate reproduction, CPU only, TRAIN-only choices).

Stages (each skips work whose output exists unless --force):
  read    cached crops (TRAIN+TEST, 18 npz) -> OUT/p3/<expert>_<task>_<split>_parseq.json with secondlook.second_stage.read_parseq_star (needs --checkpoint),
          or, for a dry run, symlinks to existing readings (--readings-from DIR; no new reading is made)
  gate    gate A (accuracy on matched TRAIN candidates, 6 rows that have candidate pickles) + label-based proxy on all 9 rows -> OUT/gate.json
  band    secondlook.selection.verification_band per row (k* re-selected on TRAIN with THIS reader's TRAIN readings) -> OUT/audit/train_band_*.json
  test    secondlook.second_stage.band_classifier per row at (c_half, k*), tag _clean -> OUT/p3/{result,counts}_<expert>_<task>_clean.*
  summary per-row machine-readable summary OUT/summary_<NAME>.json (+ .md): stage-1 F1, both-stage F1, gain over thr_train, gain of the reading features over
          the band control, paired image bootstrap (2000, seed 20261106), band share, crops per image, routing counts (stored-output decisions, not online)
Nothing under candidate-banks, operating-point or parseq-retrain is written or read except the existing inputs of E30/E31 (read-only).
Run with the legacy analysis python: python (the read stage launches the reader environment itself)."""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from secondlook.paths import ARTIFACTS, REPO_ROOT, RUNS

REPO = REPO_ROOT
RUNS = RUNS
E18 = RUNS / 'candidate-banks'
E30 = RUNS / 'operating-point'
HERE = Path(__file__).resolve().parent
ANALYSIS_PY = Path(os.environ.get('SECONDLOOK_PYTHON', sys.executable))
READ_PY = Path(os.environ.get('SECONDLOOK_READER_PYTHON', sys.executable))
ROWS = [('deepsolo', 'totaltext'), ('deepsolo', 'icdar2015'), ('ticls', 'totaltext'), ('ticls', 'icdar2015'), ('ppocr', 'totaltext'), ('ppocr', 'icdar2015'),
        ('deepsolo', 'ctw1500'), ('ticls', 'ctw1500'), ('ppocr', 'ctw1500')]
GATE_ROWS = ROWS[:6]                      # rows with candidate pickles (candidate-banks/quick/<expert>_<task>.pkl)
GATE_A_ROWS = ROWS[:4]                    # the four rows of the pre-registered gate (E33)
GATE_A_MARGIN = 3.0
BOOT, SEED = 2000, 20261106
FORBIDDEN = str(ARTIFACTS / 'parseq-retrain')


def env(threads):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), PYTHONPATH=str(REPO))
    return e


def run(cmd, log, threads, cwd=REPO):
    cmd = ['nice', '-n', '10'] + [str(c) for c in cmd]
    t = time.time()
    with open(log, 'a') as f:
        f.write('$ %s\n' % ' '.join(cmd)); f.flush()
        r = subprocess.run(cmd, cwd=cwd, env=env(threads), stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t
    if r.returncode != 0:
        raise SystemExit('FAILED (exit %d, %.0fs): %s\n  see %s' % (r.returncode, dt, ' '.join(cmd[:6]), log))
    return dt


def clean_scratch(w):
    """The CTW1500 evaluator fails if a scratch_* directory of an interrupted run is left behind; remove only those (inside this reader's own out dir)."""
    import shutil
    for d in Path(w).glob('scratch_*'):
        shutil.rmtree(d)


def key(e, t):
    return '%s_%s' % (e, t)


def parallel(fn, items, jobs):
    with ThreadPoolExecutor(jobs) as ex:
        return list(ex.map(fn, items))


def stage_read(a, out):
    p3 = out / 'p3'
    p3.mkdir(exist_ok=True)
    files = [(e, t, s) for e, t in ROWS for s in ('train', 'test')]
    target = lambda e, t, s: p3 / ('%s_%s_%s_parseq.json' % (e, t, s))
    if a.readings_from:
        src = Path(a.readings_from)
        for e, t, s in files:
            tg, sr = target(e, t, s), src / ('%s_%s_%s_parseq.json' % (e, t, s))
            if not sr.exists():
                raise SystemExit('missing reading %s' % sr)
            if tg.is_symlink() or tg.exists():
                tg.unlink()
            tg.symlink_to(sr)
        (p3 / 'provenance.json').write_text(json.dumps(dict(status='approximate reproduction', dry_run=True, readings_from=str(src), note='no model was run; the released PARSeq-S readings are linked as the reader'), indent=1))
        print('read: linked %d existing reading files from %s' % (len(files), src))
        return 0.0
    ck = Path(a.checkpoint)
    if not ck.is_file():
        raise SystemExit('checkpoint not found: %s' % ck)
    todo = [E18 / 'p3' / ('%s_%s_%s.npz' % f) for f in files if a.force or not target(*f).exists()]
    if not todo:
        print('read: all readings exist'); return 0.0
    dt = run([READ_PY, '-m', 'secondlook.second_stage.read_parseq_star', '--checkpoint', ck, '--out-dir', p3, '--files', *todo, '--threads', a.threads, '--batch', 64], out / 'logs/read.log', a.threads)
    for e, t, sp in files:                                    # every reading file must be aligned with the cached crops (same owner/cidx order as the released readings)
        new, ref = json.load(open(target(e, t, sp))), json.load(open(E18 / 'p3' / ('%s_%s_%s_parseq.json' % (e, t, sp))))
        if new['owner'] != ref['owner'] or new['cidx'] != ref['cidx'] or len(new['rows']) != len(ref['rows']):
            raise SystemExit('reading file %s is not aligned with the cached crops' % target(e, t, sp))
    return dt


def stage_gate(a, out):
    res = out / 'gate.json'
    if res.exists() and not a.force:
        print('gate: exists'); return 0.0
    rd = ['--reader', 'released=%s' % (E18 / 'p3'), '--reader', '%s=%s' % (a.name, out / 'p3')]
    t = run([ANALYSIS_PY, HERE / 'gate_rows.py', '--mode', 'gate', *rd, '--out', out / 'gate_raw.json'], out / 'logs/gate.log', min(a.threads, 2))
    (out / 'work/gate').mkdir(parents=True, exist_ok=True)
    def one(r):
        k_ = key(*r)
        if (out / ('proxy_%s.json' % k_)).exists() and not a.force:
            return 0.0
        clean_scratch(out / 'work/gate' / k_)
        return run([ANALYSIS_PY, HERE / 'gate_rows.py', '--mode', 'proxy', *rd, '--row', k_, '--work', out / 'work/gate', '--out', out / ('proxy_%s.json' % k_)], out / ('logs/proxy_%s.log' % k_), a.threads_per_job)
    t += sum(parallel(one, ROWS, a.jobs))
    raw = json.load(open(out / 'gate_raw.json'))
    proxy = {key(*r): json.load(open(out / ('proxy_%s.json' % key(*r)))) for r in ROWS}
    gate = {}
    for e, tk in GATE_ROWS:
        r = raw['rows'][key(e, tk)]
        gate[key(e, tk)] = dict(n_matched=r['n_matched'], acc_released_parseq_s=r['acc']['released'], acc_reader=r['acc'][a.name], diff=r['acc'][a.name] - r['acc']['released'],
                                spotter_own_acc=r['acc']['spotter'], in_gate_a=(e, tk) in GATE_A_ROWS,
                                passed=bool(r['acc'][a.name] >= r['acc']['released'] - GATE_A_MARGIN))
    gate_a = all(v['passed'] for k, v in gate.items() if v['in_gate_a'])
    out_json = dict(reader=a.name, status='approximate reproduction', margin_points=GATE_A_MARGIN, gate_A_rows=[key(*r) for r in GATE_A_ROWS], gate_A_pass=gate_a, rows=gate,
                    rows_without_candidate_pickles=[key(*r) for r in ROWS if r not in GATE_ROWS],
                    proxy_agreement_on_spotter_correct_train_candidates=proxy,
                    note='gate A = accuracy on TRAIN candidates matched to a GT word (IoU>=0.5, labelled), exactly research/ideas_reader_gate.py; only rows with a candidate pickle can be scored (6 of 9). The proxy covers all nine rows but is NOT gate A.')
    res.write_text(json.dumps(out_json, indent=1))
    print('gate A (4 rows, margin %.1f): %s' % (GATE_A_MARGIN, 'PASS' if gate_a else 'FAIL'))
    for k, v in gate.items():
        print('  %-20s released %.1f  %s %.1f  (%+.1f)%s' % (k, v['acc_released_parseq_s'], a.name, v['acc_reader'], v['diff'], '' if v['in_gate_a'] else '  [extra row, not in gate A]'))
    return t


def stage_band(a, out):
    audit = out / 'audit'
    audit.mkdir(exist_ok=True)
    for e, t in ROWS:                                       # read-only inputs, linked; train_only_band writes train_band_*.json INTO this directory
        ln = audit / ('op_audit_%s.json' % key(e, t))
        if not ln.exists():
            ln.symlink_to(E30 / ('op_audit_%s.json' % key(e, t)))
    def one(r):
        e, t = r
        if (audit / ('train_band_%s.json' % key(e, t))).exists() and not a.force:
            return 0.0
        clean_scratch(out / 'work/band' / key(e, t))
        return run([ANALYSIS_PY, '-m', 'secondlook.selection.verification_band', '--expert', e, '--task', t, '--audit', audit, '--p3', out / 'p3', '--work', out / 'work/band'], out / ('logs/band_%s.log' % key(e, t)), a.threads_per_job)
    (out / 'work/band').mkdir(parents=True, exist_ok=True)
    return sum(parallel(one, ROWS, a.jobs))


def cut_and_hi(out, e, t):
    b = json.load(open(out / 'audit' / ('train_band_%s.json' % key(e, t))))
    k, c = b['k_star'], b['c']
    return c, (1.0 if k == 'all' else round(min(k * c, 1.0), 4)), k


def stage_test(a, out):
    p3 = out / 'p3'
    for e in ('deepsolo', 'ticls', 'ppocr'):                  # p3_analyze reads <out>/<expert>/<task>/{extra_arms,grid}.npz (stage-1 reference arms), read-only links
        ln = out / e
        if not ln.exists():
            ln.symlink_to(E18 / e)
    def one(r):
        e, t = r
        if (p3 / ('result_%s_clean.json' % key(e, t))).exists() and not a.force:
            return 0.0
        c, hi, k = cut_and_hi(out, e, t)
        w = out / 'work' / ('t_%s' % key(e, t))
        w.mkdir(parents=True, exist_ok=True)
        clean_scratch(w)
        return run([ANALYSIS_PY, '-m', 'secondlook.second_stage.band_classifier', '--expert', e, '--task', t, '--work', w, '--out', out, '--cut', c, '--band-lo', 0, '--band-hi', hi, '--tag', '_clean'],
                   out / ('logs/test_%s.log' % key(e, t)), a.threads_per_job)
    return sum(parallel(one, ROWS, a.jobs))


def f1(c):
    s = c.sum(-2)
    return 200.0 * s[..., 0] / (s[..., 1] + s[..., 2])


def stage_summary(a, out):
    rows = {}
    for e, t in ROWS:
        k_ = key(e, t)
        res = json.load(open(out / 'p3' / ('result_%s_clean.json' % k_)))
        cnt = np.load(out / 'p3' / ('counts_%s_clean.npz' % k_))
        thr = np.load(E18 / e / t / 'extra_arms.npz')['thr_train'].astype(np.int64)
        arms = dict(thr_train=thr, stage1=cnt['head'].astype(np.int64), both_stage=cnt['verifier'].astype(np.int64), band_control=cnt['control'].astype(np.int64))
        n = thr.shape[0]
        idx = np.random.default_rng(SEED).integers(0, n, (BOOT, n))       # one index matrix per row: all contrasts are paired on the same resamples
        base = {k: f1(v) for k, v in arms.items()}
        boot = {k: f1(v[idx]) for k, v in arms.items()}
        def d(x, y):
            return dict(diff=float(base[x] - base[y]), ci95=[float(v) for v in np.percentile(boot[x] - boot[y], [2.5, 97.5])])
        c, hi, k = cut_and_hi(out, e, t)
        sent = n if e == 'deepsolo' else 0
        b = json.load(open(out / 'audit' / ('train_band_%s.json' % k_)))
        rows[k_] = dict(
            expert=e, task=t, images=int(n), c_half=c, k_star=k, band_hi=hi,
            f1_thr_train=float(base['thr_train']), f1_stage1=float(base['stage1']), f1_both_stage=float(base['both_stage']), f1_band_control=float(base['band_control']),
            gain_both_stage_over_thr_train=d('both_stage', 'thr_train'),
            gain_both_stage_over_stage1=d('both_stage', 'stage1'),
            gain_stage1_over_thr_train=d('stage1', 'thr_train'),
            gain_reading_features_over_band_control=d('both_stage', 'band_control'),
            band_share_of_test_candidates=res['band_share'], band_candidates_test=res['band_candidates_test'], crops_per_image=res['band_candidates_test'] / n,
            auc_ranker=res['auc_ranker'], auc_band_with_reader=res['auc_band_with_verifier'], auc_band_control=res['auc_band_control_no_verifier'],
            train_f1_by_k=b['train_f1_by_k'], train_f1_stage1=b['train_f1_stage1'],
            routing=dict(images_sent_to_deepsolo=sent, total_images=int(n), percent=round(100.0 * sent / n, 2), text='%d/%d, %.2f%%' % (sent, n, 100.0 * sent / n),
                         kind='logical decision on stored outputs (one spotter per row); not online inference and not measured latency'))
    summ = dict(reader=a.name, status='approximate reproduction', created=time.strftime('%F %T'), bootstrap=dict(draws=BOOT, seed=SEED, unit='image', note='one index matrix per row, contrasts paired'),
                config='p3_analyze --cut c_half --band-lo 0 --band-hi min(k* c_half, 1) --tag _clean; k* re-selected on TRAIN with this reader (train_only_band)', rows=rows)
    gp = out / 'gate.json'
    if gp.exists():
        summ['gate'] = json.load(open(gp))
    (out / ('summary_%s.json' % a.name)).write_text(json.dumps(summ, indent=1))
    md = ['# Reader `%s` on the nine rows (approximate reproduction)\n' % a.name,
          '| row | images | c | k* (hi) | thr_train F1 | stage-1 F1 | both-stage F1 | both-stage - thr_train [95% CI] | reading - band control [95% CI] | band share | crops/img | to DeepSolo |', '|' + '---|' * 12]
    fm = lambda v: '%+.2f [%+.2f, %+.2f]' % (v['diff'], *v['ci95'])
    for k_, r in rows.items():
        md.append('| %s | %d | %.2f | %s (%.2f) | %.2f | %.2f | %.2f | %s | %s | %.1f%% | %.2f | %s |' % (
            k_, r['images'], r['c_half'], r['k_star'], r['band_hi'], r['f1_thr_train'], r['f1_stage1'], r['f1_both_stage'], fm(r['gain_both_stage_over_thr_train']),
            fm(r['gain_reading_features_over_band_control']), 100 * r['band_share_of_test_candidates'], r['crops_per_image'], r['routing']['text']))
    (out / ('summary_%s.md' % a.name)).write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    return 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', required=True, help='reader name, used in file names and the summary (e.g. parseq-s-clean, dryrun-released)')
    p.add_argument('--out', required=True, help='output directory (new; never an existing result directory)')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint', help='PARSeq-S-CLEAN Lightning checkpoint (final_swa.ckpt), read by secondlook.second_stage.read_parseq_star with strict=True')
    g.add_argument('--readings-from', help='DRY RUN: directory with existing <expert>_<task>_{train,test}_parseq.json to link instead of reading crops')
    p.add_argument('--stages', default='read,gate,band,test,summary')
    p.add_argument('--threads', type=int, default=4, help='threads of the read stage (max 4)')
    p.add_argument('--jobs', type=int, default=2, help='rows in parallel for band/test (jobs x threads-per-job <= 4)')
    p.add_argument('--threads-per-job', dest='threads_per_job', type=int, default=2)
    p.add_argument('--only-rows', help='comma list of <expert>_<task> (partial test of the chain; the summary is then incomplete)')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    out = Path(a.out).resolve()
    if a.threads > 4 or a.jobs * a.threads_per_job > 4:
        raise SystemExit('refusing: more than 4 CPU threads')
    for bad in (FORBIDDEN, str(E18), str(E30)):
        if str(out) == bad or str(out).startswith(bad + '/'):
            raise SystemExit('refusing to write under %s' % bad)
    if a.checkpoint and str(Path(a.checkpoint).resolve()).startswith(FORBIDDEN) and not a.checkpoint.endswith('final_swa.ckpt'):
        raise SystemExit('only final_swa.ckpt may be read from the training directory (E33 checkpoint rule: no selection among checkpoints)')
    if a.only_rows:
        global ROWS, GATE_ROWS
        ROWS = [r for r in ROWS if key(*r) in a.only_rows.split(',')]
        GATE_ROWS = [r for r in GATE_ROWS if r in ROWS]
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    (out / 'run_args.json').write_text(json.dumps(dict(vars(a), argv=sys.argv, started=time.strftime('%F %T')), indent=1))
    timing = {}
    for s in a.stages.split(','):
        t0 = time.time()
        print('=== stage %s ===' % s, flush=True)
        {'read': stage_read, 'gate': stage_gate, 'band': stage_band, 'test': stage_test, 'summary': stage_summary}[s](a, out)
        timing[s] = round(time.time() - t0, 1)
        print('stage %s: %.0fs wall' % (s, timing[s]), flush=True)
    (out / ('timing_%s.json' % a.name)).write_text(json.dumps(timing, indent=1))


if __name__ == '__main__':
    main()
