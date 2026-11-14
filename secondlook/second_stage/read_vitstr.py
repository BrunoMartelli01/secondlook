"""V-CLEAN stage 1 (legacy env, CPU): ViTSTR-small (MJSynth+SynthText only; strict weight load) reads every TRAIN/TEST candidate crop of Total-Text / ICDAR2015.
Writes JSON in the same format as research/p3_parseq.py so that research/p3_analyze.py runs unchanged. Pre-registered in the study log ('V-CLEAN').
Upstream test path for --Transformer without --PAD: grayscale, bicubic resize to 224x224, ToTensor in [0,1] (no normalisation); 94-character sensitive charset."""
import argparse
import json
import string
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from timm.models.vision_transformer import VisionTransformer

from secondlook.data.tasks import Task, TiclsTask
from secondlook.evaluation.protocol import read_json
from secondlook.second_stage.crops import B, DIRS, polygon
from secondlook.paths import ARTIFACTS

WEIGHTS = (ARTIFACTS / 'vitstr/vitstr_small_patch16_224_aug.pth')
SHA256 = '3ff5c0490dabcb99b8540e66feb61d60e57821b55dabaacecbf31920824e42aa'
LABELS = ['[GO]', '[s]'] + list(string.printable[:-6])   # 2 special tokens + 94 characters = 96 classes


class ViTSTR(VisionTransformer):
    """Upstream ViTSTR (Atienza 2021): a ViT whose first seqlen output tokens are classified independently."""

    def forward_features(self, x):
        """All tokens (timm 0.5.4 returns only the pooled CLS token): patch embedding, prepended CLS token, position embedding, blocks, final norm."""
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return self.norm(self.blocks(self.pos_drop(x + self.pos_embed)))

    def forward(self, x, seqlen: int = 25):
        x = self.forward_features(x)
        x = x[:, :seqlen]
        b, s, e = x.size()
        return self.head(x.reshape(b * s, e)).view(b, s, self.num_classes)


def build():
    import hashlib
    h = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    if h != SHA256:
        raise ValueError('ViTSTR checkpoint hash mismatch: ' + h)
    sd = torch.load(WEIGHTS, map_location='cpu')
    sd = {k[len('module.vitstr.'):]: v for k, v in sd.items()}
    m = ViTSTR(img_size=224, patch_size=16, in_chans=1, num_classes=len(LABELS), embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True)
    m.load_state_dict(sd, strict=True)
    return m.eval()


def native_patch(img, pts):
    """Same warp and rotation rule as research/p3_crops.crop, without the final 128x32 resize."""
    if len(pts) < 3:
        x0, y0, x1, y1 = [int(v) for v in (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max() + 1, pts[:, 1].max() + 1)] if len(pts) else (0, 0, 2, 2)
        patch = img[max(y0, 0):max(y1, y0 + 1), max(x0, 0):max(x1, x0 + 1)]
    else:
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.float32)
        s, d = box.sum(1), np.diff(box, axis=1).ravel()
        tl, br, tr, bl = box[np.argmin(s)], box[np.argmax(s)], box[np.argmin(d)], box[np.argmax(d)]
        src = np.stack([tl, tr, br, bl]).astype(np.float32)
        w = max(float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - bl)), 2.0)
        h = max(float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr)), 2.0)
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        patch = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (int(round(w)), int(round(h))))
        if patch.shape[0] > 1.5 * patch.shape[1]:
            patch = np.rot90(patch)
    if patch.size == 0:
        patch = np.zeros((32, 128, 3), np.uint8)
    return np.ascontiguousarray(patch)


def to_tensor(patch):
    im = Image.fromarray(patch).convert('L').resize((224, 224), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0)[None]


@torch.inference_mode()
def read_batch(model, tensors):
    prob = model(torch.stack(tensors), seqlen=27).softmax(-1)
    conf, idx = prob.max(-1)
    out = []
    for b in range(len(tensors)):
        ids = idx[b, 1:].tolist()                 # drop the CLS / [GO] slot: 26 positions
        end = ids.index(1) if 1 in ids else len(ids)
        text = ''.join(LABELS[i] for i in ids[:end] if i >= 2)
        c = conf[b, 1:][:min(end + 1, len(ids))]  # characters + the end token
        out.append(dict(text=text, seq=float(torch.prod(c)), mint=float(c.min()), meanlp=float(torch.log(c).mean()), n=int(end)))
    return out


def main(a):
    torch.set_num_threads(a.threads)
    model = build()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    T = (Task if a.expert == 'deepsolo' else TiclsTask)(a.task, work)
    cfg = read_json('configs/approximate/protocol_calibration_v1.json')
    pool = read_json(read_json(cfg['baseline_config'])['expert_pool_config'])
    rec = {r['sample_id']: r['image_path'] for r in read_json(pool['evaluation']['rgb_only_inputs'][a.task]['path'])['records']}
    for split, ids, tab in (('train', T.ids, T.tab), ('test', T.eval_ids, T.eval_tab)):
        if a.limit:
            ids = ids[:a.limit]
        owner, cidx, rows, buf, meta = [], [], [], [], []

        def flush():
            for r in read_batch(model, buf):
                rows.append(r)
            buf.clear()
        for k, s in enumerate(ids):
            if not len(tab[s]['lines']):
                continue
            path = rec[s] if split == 'test' else str(B / DIRS[a.task] / 'train' / 'images' / ('%s.jpg' % s.split(':')[-1]))
            img = np.asarray(Image.open(path).convert('RGB'))
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
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--limit', type=int, default=0)
    main(p.parse_args())
