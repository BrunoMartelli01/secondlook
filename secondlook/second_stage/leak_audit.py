"""Verifier-leakage audit: PARSeq-S exact accuracy on GROUND-TRUTH word crops, same crop recipe as stage 2 of the method, TRAIN vs TEST of
Total-Text, CTW1500 (line crops <= 25 normalised characters) and ICDAR2015. Stage 'crops' runs in the legacy env, stage 'score' in the main env. Defence analysis only:
PARSeq-S real training data include ArT (which contains Total-Text and CTW1500 images, test splits included); ICDAR2015 is an evaluation set. Caveat stated in advance: TT/CTW TRAIN
crops are in ArT too, so a TRAIN-vs-TEST comparison alone cannot prove or exclude leakage; the comparison with the unseen ICDAR2015 gap and with the absolute level is what is informative."""
import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from secondlook.paths import ARTIFACTS, DATASETS, RUNS

OUT = (RUNS / 'candidate-banks/p3_audit')
B = (DATASETS / 'scene-segmentation-v1/datasets')
AUDIT = (ARTIFACTS / 'official-benchmark-audit-20261031-v1')
E11 = (RUNS / 'ctw1500-deepsolo')


def norm(t):
    return ''.join(ch for ch in t.lstrip('#').lower() if ch.isalnum())


def parse_gt(line):
    toks = line.rstrip('\r\n').split(',')
    k = 0
    for t in toks:
        try:
            int(float(t))
            k += 1
        except ValueError:
            break
    k -= k % 2
    pts = np.asarray([int(float(t)) for t in toks[:k]], np.float32).reshape(-1, 2)
    return pts, ','.join(toks[k:])


def crops_stage(a):
    from secondlook.second_stage.crops import crop
    from secondlook.evaluation.protocol import read_json
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    if a.dataset in ('totaltext', 'icdar2015'):
        from secondlook.data.cohort import Ctx
        from secondlook.data.tasks import eval_names
        ctx = Ctx(a.dataset)
        cfg = ctx.cfg
        d = 'total_text' if a.dataset == 'totaltext' else 'icdar2015'
        cfgp = read_json('configs/approximate/protocol_calibration_v1.json')
        pool = read_json(read_json(cfgp['baseline_config'])['expert_pool_config'])
        rec = {r['sample_id']: r['image_path'] for r in read_json(pool['evaluation']['rgb_only_inputs'][a.dataset]['path'])['records']}
        tz = zipfile.ZipFile(ctx.base / 'native/source_gt.zip')
        jobs.append(('train', [(s, B / d / 'train' / 'images' / ('%s.jpg' % s.split(':')[-1]), (ctx.names[s].replace('res_img_', 'gt_img_') if a.dataset == 'icdar2015' else ctx.names[s])) for s in ctx.train_ids], tz))
        ez = zipfile.ZipFile(AUDIT / ('gt_%s.zip' % a.dataset))
        if a.dataset == 'totaltext':
            names = eval_names(cfg, a.dataset, list(rec))
        else:
            names = {s: 'gt_img_%s.txt' % s.split(':')[-1] for s in rec}
        jobs.append(('test', [(s, Path(rec[s]), names[s]) for s in rec], ez))
    else:
        root = B / 'ctw1500'
        ovr = E11 / 'train_override'
        for split, zp in (('train', E11 / 'work/gt_ctw1500_train.zip'), ('test', E11 / 'assets/evaluation/gt_ctw1500.zip')):
            cj = read_json(E11 / ('cands_%s_floor0.05.json' % split))['images']
            items = []
            for s, im in cj.items():
                fn = im['file']
                p = root / split / 'images' / fn
                if not p.exists():
                    p = root / split / 'images' / fn.lstrip('0')
                if not p.exists():
                    p = [q for q in (root / split / 'images').iterdir() if q.stem == Path(fn).stem.lstrip('0')][0]
                if split == 'train' and (ovr / fn.lstrip('0')).exists():
                    p = ovr / fn.lstrip('0')
                items.append((s, p, '%07d.txt' % int(s)))
            jobs.append((split, items, zipfile.ZipFile(zp)))
    for split, items, z in jobs:
        crops, texts = [], []
        for s, path, gtname in items:
            img = np.asarray(Image.open(path).convert('RGB'))
            for line in z.read(gtname).decode('utf-8', 'replace').splitlines():
                if not line.strip():
                    continue
                pts, text = parse_gt(line)
                t = norm(text)
                if not t or text.strip() in ('###', '#') or len(pts) < 3:
                    continue
                if a.dataset == 'ctw1500' and len(t) > 25:
                    continue
                crops.append(crop(img, pts))
                texts.append(t)
        np.savez_compressed(OUT / ('%s_%s.npz' % (a.dataset, split)), crops=np.stack(crops), owner=np.arange(len(crops)).astype(str), cidx=np.zeros(len(crops), int), texts=np.array(texts))
        print('gt crops', a.dataset, split, len(crops), flush=True)


def score_stage(a):
    res = {}
    for ds in ('totaltext', 'ctw1500', 'icdar2015'):
        for split in ('train', 'test'):
            d = np.load(OUT / ('%s_%s.npz' % (ds, split)))
            rows = json.load(open(OUT / ('%s_%s_parseq.json' % (ds, split))))['rows']
            texts = d['texts'].tolist()
            ok = [norm(r['text']) == t for r, t in zip(rows, texts)]
            res['%s/%s' % (ds, split)] = dict(n=len(ok), accuracy=float(np.mean(ok)))
            print('%-18s n %5d PARSeq-S exact (normalised) accuracy %.4f' % ('%s/%s' % (ds, split), len(ok), np.mean(ok)), flush=True)
    for ds in ('totaltext', 'ctw1500', 'icdar2015'):
        print('%-10s TEST - TRAIN = %+.4f' % (ds, res[ds + '/test']['accuracy'] - res[ds + '/train']['accuracy']))
    (OUT / 'audit_result.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['crops', 'score'], required=True)
    p.add_argument('--dataset')
    a = p.parse_args()
    crops_stage(a) if a.stage == 'crops' else score_stage(a)
