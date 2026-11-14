"""Third, lightweight expert for the E18 protocol: PP-OCRv5 Mobile detector + English recognizer (12.7 MB ONNX), cached outputs of the
compact-OCR cascade study. CPU only, no neural call. Candidate = one word prediction of the cached
`predictions/{train,test}/<dataset>.json` (recognition admission .5 already applied by that study; the head can only REJECT).
Evaluator lines are the cached submission `body` lines (identical to the study's own scoring path).

Common-interface features (same order as the other experts): 0 detector logit (line det_score) | 1 geometric-mean character probability
of the word's characters | 2 min character probability of the word | 3 line mean entropy (line-level; ASSUMPTION) | 4 line min margin (line-level;
ASSUMPTION) | 5 log polygon area in PIXELS (not image-normalized: image size is not stored; ASSUMPTION) | 6 log box height | 7 log aspect | 8 log1p length |
9 digit fraction | 10 letter fraction. 'score' = detector score, 'rec' = word recognition score (thresholds baseline).
"""
import json
import math
from pathlib import Path

import numpy as np

from secondlook.data.tasks import Task, eval_names, CFG, AUDIT
from secondlook.data.cohort import Ctx
from secondlook.spotters.ppocr_collect import valid_quad
from secondlook.evaluation.protocol import load_config
from secondlook.paths import RUNS

RUN = (RUNS / 'ppocr-predictions')


def word_tables(rows):
    tab = {}
    for sid, r in rows.items():
        body = [l for l in r['body'].split('\n') if l]
        words = r['words']
        if len(words) != len(body):
            raise ValueError('Word/body alignment unavailable for %s (%d vs %d)' % (sid, len(words), len(body)))
        xs, sc, rec, lines = [], [], [], []
        for w, line in zip(words, body):
            if not valid_quad([list(q) for q in w['points']]):
                continue   # native submission rule of the cascade study: invalid predicted polygons are excluded (image stays in the denominator)
            row = r['rows'][w['line_id']]
            text, probs = row['text'], row['char_probs']
            pos = text.find(w['text'])
            p = probs[pos:pos + len(w['text'])] if (pos >= 0 and len(probs) == len(text)) else probs
            p = np.clip(np.asarray(p if len(p) else [row['score']], float), 1e-6, 1.0)
            pts = np.asarray(w['points'], float)
            wd, ht = max(np.ptp(pts[:, 0]), 1.0), max(np.ptp(pts[:, 1]), 1.0)
            area = 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1)))
            t = w['text']
            n = max(len(t), 1)
            ds = float(np.clip(row['det_score'], 1e-6, 1 - 1e-6))
            xs.append([math.log(ds / (1 - ds)), float(np.exp(np.log(p).mean())), float(p.min()), row['entropy'], row['margin'], math.log(max(area, 1.0)), math.log(ht),
                       math.log(wd / ht), math.log1p(len(t)), sum(c.isdigit() for c in t) / n, sum(c.isalpha() for c in t) / n])
            sc.append(row['det_score'])
            rec.append(w['score'])
            lines.append(line + '\n')
        xc = np.asarray(xs, float).reshape(-1, 11)
        tab[sid] = dict(lines=lines, xc=xc, x=xc, score=np.asarray(sc, float), rec=np.asarray(rec, float))
    return tab


class PpocrTask(Task):
    kind = 'ppocr'

    def __init__(self, task, work):
        self.task, self.work = task, Path(work)
        self.cfg = load_config(Path(CFG))
        self.variants, self.aug, self.holdout = {}, {}, False
        ctx = Ctx(task)
        self.metric, self.ids, self.names = ctx.metric, ctx.train_ids, ctx.names
        self.pool_gt = ctx.base / 'native/source_gt.zip'
        self.official = (ctx.parts['fit_ids'], ctx.parts['selection_ids'], ctx.parts['calibration_ids'])
        self.sizes = (584, 80)
        rows = json.load(open(RUN / 'predictions' / 'train' / ('%s.json' % task)))['rows']
        if set(rows) != set(self.ids):
            raise ValueError('PP-OCR TRAIN population differs from the source population')
        self.tab = word_tables(rows)
        trows = json.load(open(RUN / 'predictions' / 'test' / ('%s.json' % task)))['rows']
        self.eval_ids = sorted(trows, key=lambda s: int(s.split(':')[-1]))
        self.eval_tab = word_tables(trows)
        self.eval_names = eval_names(self.cfg, task, self.eval_ids)
        self.eval_gt = AUDIT / ('gt_%s.zip' % task)
        self.label_pool()
