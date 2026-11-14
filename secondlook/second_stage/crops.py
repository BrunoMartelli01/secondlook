"""P3 stage 1 (legacy env, CPU): perspective crops of every TiCLS/DeepSolo candidate polygon (EXIF ignored), resized to 32x128 RGB uint8.
File interface to the PARSeq stage (main env). Pre-registered in the study log ('P3 go/no-go')."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from secondlook.data.tasks import Task, TiclsTask
from secondlook.paths import DATASETS

B = (DATASETS / 'scene-segmentation-v1/datasets')
DIRS = {'totaltext': 'total_text', 'icdar2015': 'icdar2015'}


def polygon(line):
    nums = []
    for tok in line.rstrip('\n').split(','):
        try:
            nums.append(int(float(tok)))
        except ValueError:
            break
    n = len(nums) - (len(nums) % 2)
    return np.asarray(nums[:n], np.float32).reshape(-1, 2)


def crop(img, pts):
    if len(pts) < 3:
        x0, y0, x1, y1 = [int(v) for v in (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max() + 1, pts[:, 1].max() + 1)] if len(pts) else (0, 0, 2, 2)
        patch = img[max(y0, 0):max(y1, y0 + 1), max(x0, 0):max(x1, x0 + 1)]
    else:
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.float32)
        # order: top-left, top-right, bottom-right, bottom-left
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
    return cv2.resize(np.ascontiguousarray(patch), (128, 32), interpolation=cv2.INTER_CUBIC)


def main_ctw(a):
    """CTW1500 official (E11 candidates): polygons from the cached candidates, original images (override for the six EXIF-rotated TRAIN rasters)."""
    from secondlook.data.tasks_ctw1500 import CtwTask, E11
    from secondlook.evaluation.protocol import read_json
    T = CtwTask(Path(a.work), a.expert)   # E29: same images and E11 file/size lookup for every CTW1500 expert
    root = (DATASETS / 'scene-segmentation-v1/datasets/ctw1500')
    ovr = E11 / 'train_override'
    for split, ids, tab in (('train', T.ids, T.tab), ('test', T.eval_ids, T.eval_tab)):
        cj = read_json(E11 / ('cands_%s_floor0.05.json' % split))['images']
        crops, owner, cidx = [], [], []
        for s in ids:
            if not len(tab[s]['cands']):
                continue
            fn = cj[s]['file']
            path = root / split / 'images' / fn
            if not path.exists():
                path = root / split / 'images' / fn.lstrip('0')
            if not path.exists():
                cand = [q for q in (root / split / 'images').iterdir() if q.stem == Path(fn).stem.lstrip('0')]
                path = cand[0]
            if split == 'train' and (ovr / fn.lstrip('0')).exists():
                path = ovr / fn.lstrip('0')
            img = np.asarray(Image.open(path).convert('RGB'))
            if img.shape[:2] != (cj[s]['height'], cj[s]['width']):
                raise ValueError('Raster size differs from the annotation for ' + str(path))
            for i, c in enumerate(tab[s]['cands']):
                crops.append(crop(img, np.asarray(c['polys'], np.float32)))
                owner.append(s)
                cidx.append(i)
        np.savez_compressed(Path(a.out) / ('%s_%s_%s.npz' % (a.expert, a.task, split)), crops=np.stack(crops), owner=np.array(owner), cidx=np.array(cidx))
        print('crops', a.expert, a.task, split, len(crops), flush=True)


def main(a):
    if a.task == 'ctw1500':
        return main_ctw(a)
    from secondlook.evaluation.protocol import read_json
    if a.expert == 'ppocr':
        from secondlook.data.tasks_ppocr import PpocrTask
        T = PpocrTask(a.task, Path(a.work))
    else:
        T = (Task if a.expert == 'deepsolo' else TiclsTask)(a.task, Path(a.work))
    cfg = read_json('configs/approximate/protocol_calibration_v1.json')
    pool = read_json(read_json(cfg['baseline_config'])['expert_pool_config'])
    rec = {r['sample_id']: r['image_path'] for r in read_json(pool['evaluation']['rgb_only_inputs'][a.task]['path'])['records']}
    for split, ids, tab in (('train', T.ids, T.tab), ('test', T.eval_ids, T.eval_tab)):
        crops, owner, cidx = [], [], []
        for s in ids:
            if not len(tab[s]['lines']):
                continue
            path = rec[s] if split == 'test' else str(B / DIRS[a.task] / 'train' / 'images' / ('%s.jpg' % s.split(':')[-1]))
            img = np.asarray(Image.open(path).convert('RGB'))
            for i, line in enumerate(tab[s]['lines']):
                crops.append(crop(img, polygon(line)))
                owner.append(s)
                cidx.append(i)
        np.savez_compressed(Path(a.out) / ('%s_%s_%s.npz' % (a.expert, a.task, split)), crops=np.stack(crops), owner=np.array(owner), cidx=np.array(cidx))
        print('crops', a.expert, a.task, split, len(crops), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--expert', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--work', required=True)
    p.add_argument('--out', required=True)
    main(p.parse_args())
