"""E42: spotter outputs for the extra official TRAIN images of Total-Text (523) and ICDAR2015 (272). Inference only; no model is trained or changed.

Stages (one per process; DeepSolo needs the legacy host env, PP-OCR runs through its own worker venv):
  --stage deepsolo : same call path as research/selfview_infer.py at scale 1.0 (== deepsolo_admission_v2.infer_stage), same checkpoint/profile checks,
                     0.05 floor, FP32, batch 1, extract_compact_deepsolo_sample. Extra images + GATE images (first --gate old TRAIN images, re-inferred and
                     compared to the cached compact samples).
  --stage ppocr    : secondlook.spotters.ppocr_collect.CompactWorker + format_rows with configs/approximate/ppocr_cascade_v3.json (read only). Extra images +
                     GATE images compared with the cached rows.
Output only under --out (own directory). Approximate reproduction. No ground truth is read.
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

from secondlook.evaluation.protocol import load_config, read_json, write_json, sha256
from secondlook.paths import RUNS

COMPACT = (RUNS / 'deepsolo-compact-outputs/compact_deepsolo')
PPOCR_RUN = (RUNS / 'ppocr-predictions')
EXPANDED = (RUNS / 'ticls-source-banks/expanded_source_v3')
EVID = (RUNS / 'ticls-source-banks/expanded_word_evidence_v2')


def extra_records(task):
    name = {'totaltext': 'totaltext', 'icdar2015': 'icdar2015'}[task]
    recs = read_json(EXPANDED / ('%s_rgb_only.json' % name))['records']
    pool = read_json(EVID / task / 'native/bodies.json')       # sample ids retained by the native evaluator
    keep = [r for r in recs if r['sample_id'] in pool]
    dropped = [r['sample_id'] for r in recs if r['sample_id'] not in pool]
    return keep, dropped


def gate_records(task, n):
    prim = read_json(COMPACT / (task + '_train.json'))
    ids = set(prim['samples'])
    old = [r for r in read_json(prim['rgb_manifest_path'])['records'] if r['sample_id'] in ids]
    rng = np.random.RandomState(20261107)
    pick = sorted(rng.choice(len(old), n, replace=False).tolist())
    return [old[i] for i in pick]


def run_deepsolo(a):
    import cv2
    import torch
    from PIL import Image
    cfg = load_config(Path('configs/approximate/optimized_online_v1.json'))
    spec = cfg['online']['datasets'][a.task]
    profile = read_json(spec['deepsolo_profile'])
    from secondlook.spotters.deepsolo_candidates import verify_host_source_identity, resolve_runtime_candidate
    runtime = verify_host_source_identity(Path(spec['deepsolo_runtime']), Path(spec['deepsolo_source']))
    resolved = resolve_runtime_candidate(runtime, profile['candidate'])
    if resolved['checkpoint_sha256'] != profile['checkpoint_sha256']:
        raise ValueError('Wrong DeepSolo weights')
    sys.path.insert(0, spec['deepsolo_source'])
    from adet.config import get_cfg
    from secondlook.spotters.deepsolo_worker import _load_model_and_validate, _predict, _serialize_instances
    from secondlook.spotters.deepsolo_resize import configure_model_resize
    from secondlook.spotters.deepsolo_outputs import extract_compact_deepsolo_sample
    torch.set_num_threads(profile['cpu_threads'])
    torch.manual_seed(profile['seed'])
    np.random.seed(profile['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    native_cfg = get_cfg()
    native_cfg.merge_from_file(resolved['config_path'])
    native_cfg.MODEL.WEIGHTS = resolved['checkpoint_path']
    native_cfg.MODEL.DEVICE = profile['device']
    native_cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = profile['inference_floor']
    native_cfg.INPUT.MIN_SIZE_TEST, native_cfg.INPUT.MAX_SIZE_TEST = profile['resolutions']['native']
    native_cfg.freeze()
    model, compat = _load_model_and_validate(native_cfg, resolved['checkpoint_path'])
    configure_model_resize(model, native_cfg)
    extra, dropped = extra_records(a.task)
    gate = gate_records(a.task, a.gate)
    if a.all:   # E42 sensitivity arm: ALL pool images (old 728 + extras) at the profile resolution of the TEST outputs; no gate against the 1024 cache
        prim = read_json(COMPACT / (a.task + '_train.json'))
        old = [r for r in read_json(prim['rgb_manifest_path'])['records'] if r['sample_id'] in set(prim['samples'])]
        extra, gate = old + extra, []
    if a.limit:
        extra = extra[:a.limit]
    todo = [('gate', r) for r in gate] + [('extra', r) for r in extra]
    samples, tick = {}, time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for i, (kind, r) in enumerate(todo):
        if sha256(r['image_path']) != r['image_sha256']:
            raise ValueError('Changed raster ' + r['sample_id'])
        image = cv2.imread(r['image_path'], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Unreadable raster')
        t0 = time.perf_counter()
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            torch.cuda.synchronize()
            prediction = _predict(model, native_cfg, image, True)
            torch.cuda.synchronize()
        detections = _serialize_instances(prediction['instances'], native_cfg)
        with Image.open(r['image_path']) as im:
            orientation = int(im.getexif().get(274, 1) or 1)
        raw = dict(ground_truth_used=False, contract=dict(image_sha256=r['image_sha256']), seconds=max(time.perf_counter() - t0, 1e-12),
                   response=dict(success=True, detections=detections))
        samples[r['sample_id']] = extract_compact_deepsolo_sample(raw, a.task, r['sample_id'], r['image_wh'], orientation, r['image_sha256'])
        if i % 25 == 0:
            print('infer', a.task, kind, i + 1, len(todo), flush=True)
    # gate: gate images vs the cached compact samples
    cached = read_json(COMPACT / (a.task + '_train.json'))['samples']
    gate_report, ok = [], True
    for r in gate:
        new, old = samples[r['sample_id']], cached[r['sample_id']]
        dn, do = new['detections'], old['detections']
        rep = dict(sample_id=r['sample_id'], n_new=len(dn), n_old=len(do))
        if len(dn) == len(do):
            rep['text_equal'] = all(x['text'] == y['text'] for x, y in zip(dn, do))
            rep['max_score_diff'] = max([abs(x['score'] - y['score']) for x, y in zip(dn, do)] or [0.0])
            rep['max_rec_conf_diff'] = max([abs(x['recognition_confidence'] - y['recognition_confidence']) for x, y in zip(dn, do)] or [0.0])
            rep['points_equal'] = all(x['int_points'] == y['int_points'] for x, y in zip(dn, do))
            good = rep['text_equal'] and rep['max_score_diff'] < 1e-4 and rep['max_rec_conf_diff'] < 1e-4 and rep['points_equal']
        else:
            good = False
        rep['pass'] = bool(good)
        ok = ok and good
        gate_report.append(rep)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = '%s%s' % (a.task, '_smoke' if a.limit else '')
    extra_samples = {k: v for k, v in samples.items() if k in {r['sample_id'] for r in extra}}
    if a.all:
        tag += '_all_testres'
    write_json(out / ('deepsolo_%s_extra.json' % tag), dict(
        samples=extra_samples, dataset=a.task, split='train', images=len(extra), dropped_native_invalid=dropped, wall_seconds=time.perf_counter() - tick,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(), profile=profile['candidate'],
        compatibility=compat, no_gt_used=True, source_sha256=sha256(__file__), claim_label='approximate reproduction', gate_pass=ok), exclusive=False)
    write_json(out / ('deepsolo_%s_gate.json' % tag), dict(gate=gate_report, all_pass=ok), exclusive=False)
    print('done', tag, len(extra), 'gate', ok, '%.1fs' % (time.perf_counter() - tick), 'peak alloc %.2f GB reserved %.2f GB' % (
        torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9), flush=True)
    if not ok:
        raise SystemExit('DeepSolo gate FAILED: see ' + str(out / ('deepsolo_%s_gate.json' % tag)))


def run_ppocr(a):
    from secondlook.spotters.ppocr_collect import CompactWorker, format_rows
    config_path = Path('configs/approximate/ppocr_cascade_v3.json')
    cfg = load_config(config_path)
    spec = cfg['datasets'][a.task]
    extra, dropped = extra_records(a.task)
    gate = gate_records(a.task, a.gate)
    if a.limit:
        extra = extra[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    worker = CompactWorker(cfg, config_path, out / ('worker_%s_%d' % (a.task, int(time.time()))))
    rows, tick = {}, time.perf_counter()
    try:
        for r in worker_warm(gate):
            worker.request(r)
        for i, (kind, r) in enumerate([('gate', g) for g in gate] + [('extra', e) for e in extra]):
            if sha256(r['image_path']) != r['image_sha256']:
                raise ValueError('Changed raster ' + r['sample_id'])
            row = worker.request(r)
            row['body'], row['submission_rejections'] = format_rows(row['words'], spec['evaluation_dataset'])
            rows[r['sample_id']] = row
            if i % 25 == 0:
                print('ppocr', a.task, kind, i + 1, flush=True)
        ready = worker.ready
    finally:
        worker.close()
    cached = read_json(PPOCR_RUN / 'predictions/train' / (a.task + '.json'))['rows']
    skip = {'status', 'request_id', 'worker_seconds'}
    rep, ok = [], True
    for r in gate:
        new, old = rows[r['sample_id']], cached[r['sample_id']]
        bad = [k for k in old if k not in skip and json.dumps(new.get(k), sort_keys=True) != json.dumps(old[k], sort_keys=True)]
        rep.append(dict(sample_id=r['sample_id'], differing_fields=bad, **{'pass': not bad}))
        ok = ok and not bad
    gate_ids = {r['sample_id'] for r in gate}
    tag = '%s%s' % (a.task, '_smoke' if a.limit else '')
    write_json(out / ('ppocr_%s_extra.json' % tag), dict(rows={k: v for k, v in rows.items() if k not in gate_ids}, dataset=a.task, split='train', dropped_native_invalid=dropped,
                                                         wall_seconds=time.perf_counter() - tick, initialization=ready, gate_pass=ok, source_sha256=sha256(__file__),
                                                         claim_label='approximate reproduction'), exclusive=False)
    write_json(out / ('ppocr_%s_gate.json' % tag), dict(gate=rep, all_pass=ok), exclusive=False)
    print('done', tag, len(extra), 'gate', ok, '%.1fs' % (time.perf_counter() - tick), flush=True)
    if not ok:
        raise SystemExit('PP-OCR gate FAILED: see ' + str(out / ('ppocr_%s_gate.json' % tag)))


def worker_warm(gate):
    return gate[:2]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['deepsolo', 'ppocr'])
    p.add_argument('--task', required=True, choices=['totaltext', 'icdar2015'])
    p.add_argument('--out', required=True)
    p.add_argument('--gate', type=int, default=8)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--all', action='store_true', help='E42 sensitivity: infer ALL pool images at the profile (TEST) resolution; output suffix _all_testres')
    a = p.parse_args()
    (run_deepsolo if a.stage == 'deepsolo' else run_ppocr)(a)
