"""E46 (approximate reproduction): re-run the pinned PP-OCRv5 Mobile worker on the TRAIN/TEST images of one benchmark, expose per-character entropy and margin, and GATE
every original field against the cached rows. GPU (the worker requires CUDAExecutionProvider; CPU fallback is forbidden by the worker); nothing trained, no spotter changed.
    python -m secondlook.spotters.ppocr_stats --task totaltext --split train --out DIR [--limit N]
Sources (read only): records = the `train_inputs` / `test_inputs` lists of configs/approximate/ppocr_cascade_v3.json plus, for the Total-Text / ICDAR2015 extra TRAIN images,
secondlook.data.collect_extra_train.extra_records; cached rows = ppocr-predictions/predictions/<split>/<task>.json and fulltrain/collect/ppocr_<task>_extra.json.
GATE (per image): every field of the cached row except {status, request_id, worker_seconds, body, submission_rejections} must be JSON-identical in the new row; the only extra fields are
`char_entropy` and `char_margin` of each line row. A failed gate stops the run and writes nothing."""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from secondlook.evaluation.protocol import load_config, read_json, sha256, write_json
from secondlook.paths import RUNS

CONFIG = Path('configs/approximate/ppocr_cascade_v3.json')
CASCADE = (RUNS / 'ppocr-predictions')
COLLECT = (RUNS / 'fulltrain/collect')
SKIP = {'status', 'request_id', 'worker_seconds', 'body', 'submission_rejections'}
NEW = {'char_entropy', 'char_margin'}


def records(task, split, cfg):
    spec = cfg['datasets'][task]
    raw = read_json(spec['train_inputs' if split == 'train' else 'test_inputs'])
    recs = list(raw['records'] if isinstance(raw, dict) else raw)
    cached = dict(read_json(CASCADE / 'predictions' / split / (task + '.json'))['rows'])
    if split == 'train' and task in ('totaltext', 'icdar2015'):
        from secondlook.data.collect_extra_train import extra_records
        extra, _ = extra_records(task)
        extra_rows = read_json(COLLECT / ('ppocr_%s_extra.json' % task))['rows']
        by_id = {r['sample_id']: r for r in recs}
        for r in extra:
            if r['sample_id'] in extra_rows:
                by_id[r['sample_id']] = r
                cached[r['sample_id']] = extra_rows[r['sample_id']]
        recs = list(by_id.values())
    recs = [r for r in recs if r['sample_id'] in cached]
    return recs, cached


def strip_new(value):
    if isinstance(value, dict):
        return {k: strip_new(v) for k, v in value.items() if k not in NEW}
    if isinstance(value, list):
        return [strip_new(v) for v in value]
    return value


def main(a):
    cfg = load_config(CONFIG)
    recs, cached = records(a.task, a.split, cfg)
    if a.limit:
        recs = recs[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    paths = cfg['compact_runtime']['cuda_library_paths']
    env['LD_LIBRARY_PATH'] = ':'.join(paths) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    log = (out / ('worker_%s_%s.log' % (a.task, a.split))).open('w')
    proc = subprocess.Popen([cfg['compact_runtime']['python'], '-m', 'secondlook.spotters.ppocr_stats_worker', '--config', str(CONFIG)], env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)

    def receive(request_id=None):
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError('E46 worker exited; see ' + log.name)
        value = json.loads(line)
        if value.get('status') != 'OK':
            raise RuntimeError('E46 worker failed: ' + json.dumps(value))
        if request_id is not None and value.get('request_id') != request_id:
            raise ValueError('Out-of-order response')
        return value

    ready = receive()
    rows, tick, bad = {}, time.perf_counter(), []
    try:
        for i, r in enumerate(recs):
            if sha256(r['image_path']) != r['image_sha256']:
                raise ValueError('Changed raster ' + r['sample_id'])
            req = dict(r, request_id=uuid.uuid4().hex)
            proc.stdin.write(json.dumps(req) + '\n')
            proc.stdin.flush()
            new = receive(req['request_id'])
            old = cached[r['sample_id']]
            diff = [k for k in old if k not in SKIP and json.dumps(strip_new(new.get(k)), sort_keys=True) != json.dumps(old[k], sort_keys=True)]
            if diff:
                bad.append(dict(sample_id=r['sample_id'], differing_fields=diff))
                if len(bad) >= 5:
                    break
            for line_row in new['rows']:
                if len(line_row['char_entropy']) != len(line_row['char_probs']) or len(line_row['char_margin']) != len(line_row['char_probs']):
                    raise ValueError('Per-character statistics not aligned with char_probs for ' + r['sample_id'])
            rows[r['sample_id']] = {k: v for k, v in new.items() if k not in ('status', 'request_id', 'worker_seconds', 'features', 'rejected_detection_scores')}
            if i % 100 == 0:
                print('ppocr', a.task, a.split, i + 1, '/', len(recs), '%.0fs' % (time.perf_counter() - tick), flush=True)
    finally:
        if proc.poll() is None:
            proc.stdin.write('{"command":"stop"}\n')
            proc.stdin.flush()
            proc.wait(timeout=60)
        log.close()
    if bad:
        write_json(out / ('ppocr_%s_%s_GATE_FAILED.json' % (a.task, a.split)), dict(bad=bad), exclusive=False)
        raise SystemExit('E46 PP-OCR gate FAILED: %s' % bad[:2])
    tag = '%s_%s%s' % (a.task, a.split, '_smoke' if a.limit else '')
    write_json(out / ('ppocr_stats_%s.json' % tag), dict(rows=rows, images=len(rows), task=a.task, split=a.split, ready=ready, gate_pass=True, wall_seconds=time.perf_counter() - tick,
                                                          source_sha256=sha256(__file__), claim_label='approximate reproduction'), exclusive=False)
    print('done', tag, len(rows), 'images, gate pass, %.0fs' % (time.perf_counter() - tick), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=['totaltext', 'icdar2015', 'ctw1500'])
    p.add_argument('--split', required=True, choices=['train', 'test'])
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=0)
    main(p.parse_args())
