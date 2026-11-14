"""V-ABI stage 1 (legacy env, CPU): official ABINet (vision + bidirectional language model, iter_size 3; MJSynth + SynthText only) reads every TRAIN/TEST candidate crop of
Total-Text / ICDAR2015. Writes JSON in the same format as research/p3_parseq.py so that research/p3_analyze.py runs unchanged. Pre-registered in the study log ('V-ABI').

The model code is the unmodified official source (FangShancheng/ABINet, commit pinned below, licence: non-commercial) kept OUTSIDE this repository in --src; the only deviation is a
three-line stand-in for `fastai.vision` (its star-import is used only for `nn` and `ifnone`). Evaluation path = the official one (main.py + dataset.py + demo.py): RGB, cv2.resize to 128x32 (default
linear interpolation), ToTensor in [0, 1], ImageNet mean/std normalisation.
Charset: 36 lower-case alphanumeric characters plus the end token (case-insensitive model); max length 25 + end token."""
import argparse
import hashlib
import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from secondlook.data.tasks import Task, TiclsTask
from secondlook.evaluation.protocol import read_json
from secondlook.second_stage.read_vitstr import native_patch
from secondlook.second_stage.crops import B, DIRS, polygon
from secondlook.paths import ARTIFACTS

ART = (ARTIFACTS / 'abinet')
SRC = ART / 'src'
WEIGHTS = ART / 'best-train-abinet.pth'
SOURCE_COMMIT = '789743c56eacbf114c2d63dfaeada533098dd19a'
SHA256 = ''   # filled from the artefact manifest in main() (ART / 'sha256.txt'); a mismatch is an error


def install_fastai_stand_in():
    """`from fastai.vision import *` in the official model files only needs `nn` and `ifnone`."""
    mod = types.ModuleType('fastai.vision')
    mod.nn = nn
    mod.ifnone = lambda a, b: b if a is None else a
    pkg = types.ModuleType('fastai')
    pkg.vision = mod
    sys.modules['fastai'] = pkg
    sys.modules['fastai.vision'] = mod


@contextmanager
def chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build(src=SRC, weights=WEIGHTS, expected_sha=None):
    if expected_sha is not None and sha256_of(weights) != expected_sha:
        raise ValueError('ABINet checkpoint hash mismatch')
    install_fastai_stand_in()
    sys.path.insert(0, str(src))
    with chdir(src):
        from utils import Config, CharsetMapper
        config = Config('configs/train_abinet.yaml')
        config.model_vision_checkpoint, config.model_language_checkpoint = None, None   # the full checkpoint is loaded below
        config.global_phase = 'test'
        from modules.model_abinet_iter import ABINetIterModel
        model = ABINetIterModel(config)
        charset = CharsetMapper(filename=config.dataset_charset_path, max_length=config.dataset_max_length + 1)
    state = torch.load(weights, map_location='cpu')
    if 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=True)    # strict: any missing / unexpected key is an error
    return model.eval(), charset, config


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def to_tensor(patch):
    """Official eval path (dataset.py `_process_test` + ToTensor, then main.py line 88 `.normalize(imagenet_stats)`, identical to demo.py): cv2.resize(img, (128, 32)),
    [0, 1], ImageNet mean/std. `patch` is an RGB uint8 array. (A first version omitted the normalisation: 6/8 of the official demo images read correctly instead of 8/8.)"""
    x = cv2.resize(patch, (128, 32))
    return (torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float() / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


@torch.inference_mode()
def read_batch(model, charset, tensors):
    a_res, _, _ = model(torch.stack(tensors))     # eval mode: (alignment result, last language result, vision result)
    prob = a_res['logits'].softmax(-1)
    conf, idx = prob.max(-1)
    out = []
    for b in range(len(tensors)):
        ids = idx[b].tolist()
        end = ids.index(charset.null_label) if charset.null_label in ids else len(ids)
        text = ''.join(charset.label_to_char[i] for i in ids[:end])
        c = conf[b][:min(end + 1, len(ids))]      # characters + the end token (as the official postprocess keeps)
        out.append(dict(text=text, seq=float(torch.prod(c)), mint=float(c.min()), meanlp=float(torch.log(c).mean()), n=int(end)))
    return out


def main(a):
    torch.set_num_threads(a.threads)
    sha = (ART / 'sha256.txt').read_text().split()[0]
    model, charset, _ = build(Path(a.src), Path(a.weights), sha)
    print('params', sum(p.numel() for p in model.parameters()), 'classes', charset.num_classes, flush=True)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    T = (Task if a.expert == 'deepsolo' else TiclsTask)(a.task, work)
    cfg = read_json('configs/approximate/protocol_calibration_v1.json')
    pool = read_json(read_json(cfg['baseline_config'])['expert_pool_config'])
    rec = {r['sample_id']: r['image_path'] for r in read_json(pool['evaluation']['rgb_only_inputs'][a.task]['path'])['records']}
    for split, ids, tab in (('train', T.ids, T.tab), ('test', T.eval_ids, T.eval_tab)):
        if a.limit:
            ids = ids[:a.limit]
        owner, cidx, rows, buf = [], [], [], []

        def flush():
            for r in read_batch(model, charset, buf):
                rows.append(r)
            buf.clear()
        for k, s in enumerate(ids):
            if not len(tab[s]['lines']):
                continue
            path = rec[s] if split == 'test' else str(B / DIRS[a.task] / 'train' / 'images' / ('%s.jpg' % s.split(':')[-1]))
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            for i, line in enumerate(tab[s]['lines']):
                buf.append(to_tensor(native_patch(img, polygon(line))))
                owner.append(s); cidx.append(i)
                if len(buf) >= a.batch:
                    flush()
            if k % 50 == 0:
                print(a.expert, a.task, split, k, len(ids), len(rows), flush=True)
        flush()
        Path(a.out).mkdir(parents=True, exist_ok=True)
        Path(a.out, '%s_%s_%s_parseq.json' % (a.expert, a.task, split)).write_text(json.dumps(dict(owner=owner, cidx=cidx, rows=rows)))
        print('done', a.expert, a.task, split, len(rows), flush=True)
        if a.limit:
            for s, i, r in list(zip(owner, cidx, rows))[:12]:
                print('  ', repr(tab[s]['lines'][i].rstrip().split(',')[-1][:30]), '->', repr(r['text']), round(r['seq'], 3))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--expert', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--work', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--src', default=str(SRC))
    p.add_argument('--weights', default=str(WEIGHTS))
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--limit', type=int, default=0)
    main(p.parse_args())
