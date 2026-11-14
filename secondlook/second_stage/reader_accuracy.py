"""E42: word accuracy of each reader on FULL-POOL TRAIN candidates matched to a GT word (IoU >= 0.5 of the candidate bounding box with a GT word box, candidate labelled), overall and
inside the reader's own band [0, band_hi*] of the 5-fold cross-fitted stage-1 probability (same logic as research/reader_retune_accuracy.py, with the full-pool task, the expanded GT zip and
candidate boxes taken from the candidate polygons exactly as research/quick_prep.py). Also the spotter's own accuracy. TRAIN only; no TEST label. Run through secondlook.studies.fulltrain_launch.
Usage: --reader name=<reader dir with p3/ and audit/> ... (rows: the six Total-Text / ICDAR2015 rows present in every reader dir)."""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from secondlook.second_stage.band_classifier import cross_fit, own_text
from secondlook.second_stage.leak_audit import parse_gt, norm
from secondlook.second_stage.crops import polygon
from secondlook.second_stage.geometry_helpers import iou_cross

ROWS = [('deepsolo', 'totaltext'), ('deepsolo', 'icdar2015'), ('ticls', 'totaltext'), ('ticls', 'icdar2015'), ('ppocr', 'totaltext'), ('ppocr', 'icdar2015')]


def gt_words_full(task, T):
    z = zipfile.ZipFile(T.pool_gt)
    out = {}
    for s in T.ids:
        name = T.names[s].replace('res_img_', 'gt_img_') if task == 'icdar2015' else T.names[s]
        boxes, texts = [], []
        for line in z.read(name).decode('utf-8', 'replace').splitlines():
            if not line.strip():
                continue
            pts, text = parse_gt(line)
            t = norm(text)
            if len(pts) < 3 or not t or text.strip() in ('###', '#'):
                continue
            boxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
            texts.append(t)
        out[s] = (np.asarray(boxes, float).reshape(-1, 4), texts)
    return out


def matched(T, task):
    G = gt_words_full(task, T)
    m = {}
    for s in T.ids:
        lines = T.tab[s]['lines']
        if not lines:
            continue
        boxes = []
        for l in lines:
            p = polygon(l)
            boxes.append([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] if len(p) else [0, 0, 1, 1])
        gb, gtx = G[s]
        iou = iou_cross(np.asarray(boxes, float).reshape(-1, 4), gb)
        for j in range(len(lines)):
            g = int(np.argmax(iou[j])) if iou.shape[1] and iou[j].max() >= 0.5 else -1
            if g >= 0 and T.lab[s][j] >= 0:
                m[(s, j)] = (gtx[g], own_text(lines[j]))
    return m


def main(a):
    from secondlook.selection.operating_point import load
    readers = dict(r.split('=', 1) for r in a.reader)
    res = {}
    for e, task in ROWS:
        if a.only and (e, task) != tuple(a.only):
            continue
        work = Path(a.work) / ('%s_%s' % (e, task))
        work.mkdir(parents=True, exist_ok=True)
        T = load(e, task, work)
        p_cf = cross_fit(T, list(T.ids))
        m = matched(T, task)
        spot = np.array([norm(t) == g for g, t in m.values()])
        row = dict(n_matched=len(m), spotter_own_acc=100.0 * float(spot.mean()), readers={})
        for name, d in readers.items():
            j = json.load(open(Path(d) / 'p3' / ('%s_%s_train_parseq.json' % (e, task))))
            mp = {(o, c): x for o, c, x in zip(j['owner'], j['cidx'], j['rows'])}
            hi = json.load(open(Path(d) / 'audit' / ('train_band_%s_%s.json' % (e, task))))['band_hi_star']
            ok = np.array([norm(mp[k]['text']) == g for k, (g, _) in m.items()])
            inb = np.array([p_cf[k[0]][k[1]] <= hi for k in m])
            row['readers'][name] = dict(acc_matched=100.0 * float(ok.mean()), band_hi=hi, n_in_band=int(inb.sum()), acc_in_band=100.0 * float(ok[inb].mean()) if inb.any() else None,
                                        spotter_acc_in_band=100.0 * float(spot[inb].mean()) if inb.any() else None)
        res['%s_%s' % (e, task)] = row
        print('ACC', e, task, row['n_matched'], round(row['spotter_own_acc'], 1), {r: (round(v['acc_matched'], 1), v['acc_in_band'] and round(v['acc_in_band'], 1)) for r, v in row['readers'].items()}, flush=True)
        Path(a.out).with_suffix('.%s_%s.json' % (e, task)).write_text(json.dumps(row, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reader', action='append', required=True)
    p.add_argument('--work', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--only', nargs=2)
    main(p.parse_args())
