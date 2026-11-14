"""E20: the E18 protocol on CTW1500 with the OFFICIAL DeepSolo R50 CTW1500 candidates. CPU only.

Read-only inputs from study E11 (agreed with it, conditions 1-5): cands_{train,test}_floor0.05.json, assets/, work/gt_ctw1500_train.zip.
Own work dirs only; the unmodified upstream evaluator path (secondlook.spotters.deepsolo_ctw1500.Official) is instantiated with MY work dir;
candidate tables use the same table(cj, 0.2) pre-filter so line indices align with the evaluator. CTW TEST was already evaluated once per
frozen policy in E11, so everything here is EXPLORATORY: 'approximate reproduction, TEST-internal cross-validation'.
`CtwTask` exposes the interface secondlook.data.cv_counts expects (ids/tab/lab/gt for TRAIN, eval_ids/eval_tab for TEST, counts(), test_labels(),
approx_f1()). Stage `full` (no target label): lexicon-aware vs lexicon-free calibration under the official None and Full (weak) lexicon.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from secondlook.spotters.deepsolo_ctw1500 import Official, table, entries_of, counts_array, read_json, CAND_FLOOR, OFFICIAL_SCORE
from secondlook.data.tasks import add_common
from secondlook.evaluation.declared_metric import word_labels
from secondlook.paths import ARTIFACTS, RUNS

SRC = str(ARTIFACTS / 'deepsolo-host/DeepSolo-upstream/DeepSolo')
E11 = (RUNS / 'ctw1500-deepsolo')
ASSETS = E11 / 'assets'
TRAIN_GT = E11 / 'work' / 'gt_ctw1500_train.zip'


TICLS_BANK = (RUNS / 'ctw1500-ticls-finetune/bank')
PPOCR_RUN = (RUNS / 'ppocr-predictions/predictions')
EXIF_TRAIN = E11 / 'train_override'


def _valid(polys):
    from shapely.geometry import Polygon
    try:
        return Polygon([(float(p[0]), float(p[1])) for p in polys]).is_valid
    except Exception:
        return False


def ticls_table(split):
    """E29: own TiCLS CTW1500 fine-tune bank (E27; pool = author CTW threshold 0.3). Same official validity pre-filter as `table`
    (score <= 0.1 dropped by to_eval_format, invalid polygons dropped by sort_detection) so line indices align with the evaluator;
    'x' = the 13 deployment features (same order as TiCLS Total-Text/ICDAR2015), 'rec' = x[:, 6] as in TiclsTask."""
    cj = read_json(TICLS_BANK / split / ('cands_%s_ticls.json' % split))
    if cj.get('status') != 'COMPLETE':
        raise ValueError('TiCLS CTW1500 bank incomplete: ' + split)
    tab = {}
    for iid, im in cj['images'].items():
        rows = [c for c in im['cands'] if c['score'] > 0.1 and _valid(c['polys'])]
        x = np.asarray([c['x13'] for c in rows], np.float64).reshape(-1, 13)
        tab[iid] = dict(cands=rows, x=x, score=np.asarray([c['score'] for c in rows], np.float64).reshape(-1), rec=x[:, 6].copy(),
                        width=im['width'], height=im['height'], file=im['file'])
    return add_common(tab, 'ticls')


def ppocr_table(split, sizes):
    """E29: PP-OCRv5 Mobile cached LINE outputs (det_score >= 0.6, rec >= 0.5 applied upstream). One candidate per text line, official entry
    format (image_id, polys = the detector quad, rec = line text, score = detector score). 11 common signals as in secondlook.data.tasks_ppocr
    (E21) but computed on the whole line; log area relative to the image size (as for DeepSolo/TiCLS on CTW1500); 'rec' = line recognition score."""
    import math
    from secondlook.spotters.ppocr_collect import valid_quad
    rows_all = read_json(PPOCR_RUN / split / 'ctw1500.json')['rows']
    tab, geom_mismatch = {}, []
    for sid, r in rows_all.items():
        iid = str(int(sid.split(':')[-1]))
        wd_img, ht_img = sizes[iid]
        cands, xs, sc, rec = [], [], [], []
        for row in r['rows']:
            pts = [list(map(float, q)) for q in row['points']]
            if not valid_quad([list(q) for q in row['points']]) or not _valid(pts):
                continue
            t = row['text']
            p = np.clip(np.asarray(row['char_probs'] if len(row['char_probs']) else [row['score']], float), 1e-6, 1.0)
            a = np.asarray(pts)
            w, h = max(np.ptp(a[:, 0]), 1.0), max(np.ptp(a[:, 1]), 1.0)
            area = 0.5 * abs(np.dot(a[:, 0], np.roll(a[:, 1], -1)) - np.dot(a[:, 1], np.roll(a[:, 0], -1)))
            n = max(len(t), 1)
            ds = float(np.clip(row['det_score'], 1e-6, 1 - 1e-6))
            xs.append([math.log(ds / (1 - ds)), float(np.exp(np.log(p).mean())), float(p.min()), row['entropy'], row['margin'],
                       math.log(max(area, 1.0) / (wd_img * ht_img)), math.log(h), math.log(w / h), math.log1p(len(t)),
                       sum(c.isdigit() for c in t) / n, sum(c.isalpha() for c in t) / n])
            cands.append(dict(image_id=int(iid), category_id=1, polys=pts, rec=t, score=float(row['det_score'])))
            sc.append(row['det_score'])
            rec.append(row['score'])
            if a[:, 0].max() > wd_img + 2 or a[:, 1].max() > ht_img + 2:
                geom_mismatch.append(iid)
        xc = np.asarray(xs, float).reshape(-1, 11)
        tab[iid] = dict(cands=cands, x=xc, xc=xc, score=np.asarray(sc, float), rec=np.asarray(rec, float))
    return tab, sorted(set(geom_mismatch), key=int)


class CtwTask:
    kind = 'deepsolo'

    def __init__(self, work, expert='deepsolo'):
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        self.kind = expert
        self.excluded_train = []
        if expert == 'deepsolo':
            tr, _ = table(read_json(E11 / 'cands_train_floor0.05.json'), CAND_FLOOR)
            te, _ = table(read_json(E11 / 'cands_test_floor0.05.json'), CAND_FLOOR)
        elif expert == 'ticls':
            tr, te = ticls_table('train'), ticls_table('test')
        elif expert == 'ppocr':
            sz = {}
            for split in ('train', 'test'):
                for iid, im in read_json(E11 / ('cands_%s_floor0.05.json' % split))['images'].items():
                    sz[iid] = (im['width'], im['height'])
            tr, mm_tr = ppocr_table('train', sz)
            te, mm_te = ppocr_table('test', sz)
            if mm_te:
                raise ValueError('PP-OCR TEST geometry exceeds the annotated image size for images %s' % mm_te)
            exif = {f.stem.lstrip('0') for f in EXIF_TRAIN.iterdir()}
            bad = set(mm_tr)
            if bad - exif:
                raise ValueError('PP-OCR TRAIN geometry mismatch outside the known EXIF list: %s' % sorted(bad - exif, key=int))
            self.excluded_train = sorted(bad, key=int)   # declared in E29: EXIF-rotated rasters read without the override
            for iid in self.excluded_train:
                del tr[iid]
        else:
            raise ValueError('Unknown CTW1500 expert ' + expert)
        self.ids, self.eval_ids = sorted(tr, key=int), sorted(te, key=int)
        for tab in (tr, te):
            for t in tab.values():
                t['lines'] = [c['rec'] for c in t['cands']]
        if expert == 'deepsolo':
            self.tab, self.eval_tab = add_common(tr, 'deepsolo'), add_common(te, 'deepsolo')
        else:
            self.tab, self.eval_tab = tr, te
        # excluded TRAIN images keep their GT file in the zip; only the counters of self.ids are read, so they do not enter labels or F1
        self.off_tr = Official(SRC, ASSETS, work / 'scratch_train', TRAIN_GT)
        self.off_te = Official(SRC, ASSETS, work / 'scratch_test', ASSETS / 'evaluation' / 'gt_ctw1500.zip')
        self.lab, self.gt = self._labels(self.off_tr, self.tab, self.ids, None)

    @staticmethod
    def _labels(off, tab, ids, lex):
        native, counters, _ = off.run(entries_of(tab, ids, {i: [True] * len(tab[i]['cands']) for i in ids}), lex)
        lab, gt = {}, {}
        for i in ids:
            key = '%07d' % int(i)
            n = len(tab[i]['cands'])
            s = native['per_sample'][key]
            if n != len(s['detTrans']):
                raise ValueError('Label identity mismatch for image ' + i)
            lab[i] = word_labels(list(range(n)), counters[key]['e2e_correct_indices'], s['detDontCare'], n)
            gt[i] = counters[key]['e2e_gt_words']
        return lab, gt

    def train_labels(self, lex):
        return self._labels(self.off_tr, self.tab, self.ids, lex)

    def counts(self, keep, lexicon=None):
        _, c, _ = self.off_te.run(entries_of(self.eval_tab, self.eval_ids, {i: list(map(bool, keep[i])) for i in self.eval_ids}), lexicon)
        return counts_array(c, self.eval_ids)

    def test_labels(self):
        return self._labels(self.off_te, self.eval_tab, self.eval_ids, None)

    def approx_f1(self, ids, keep, lab=None, gt=None):
        lab, gt = lab or self.lab, gt or self.gt
        tp = pr = g = 0
        for i in ids:
            k = np.asarray(keep[i], bool)
            tp += int(((lab[i] == 1) & k).sum())
            pr += int(((lab[i] >= 0) & k).sum())
            g += gt[i]
        return 2 * tp / (pr + g) if pr + g else 0.0


def f1(c):
    return 200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())


def full_stage(args):
    """Fixed decisions (no target label) evaluated on ALL 500 TEST images under None and Full lexicon."""
    from secondlook.data.cv_counts import SPEC, train_hgb
    from secondlook.studies.ablation_learners import proba
    out = Path(args.out) / 'deepsolo' / 'ctw1500'
    T = CtwTask(out / 'work_full')
    lab_none, gt_none = T.lab, T.gt
    lab_full, gt_full = T.train_labels(1)
    ids = T.eval_ids
    rows = lambda lab: [(T.tab[s]['xc'], lab[s]) for s in T.ids]
    rankers = {'none': train_hgb(rows(lab_none)), 'full': train_hgb(rows(lab_full))}
    keeps = {'author': {s: T.eval_tab[s]['score'] >= OFFICIAL_SCORE for s in ids}}
    for name, m in rankers.items():
        p = proba(m, T.eval_tab, ids, SPEC)
        keeps['own042_%strained' % name] = {s: p[s] >= 0.42 for s in ids}
    from secondlook.data.tasks import pick_thresholds
    for name, (lab, gt) in {'none': (lab_none, gt_none), 'full': (lab_full, gt_full)}.items():
        T.lab, T.gt = lab, gt
        st, rt, _ = pick_thresholds(T, T.ids)
        keeps['thr_train_%s' % name] = {s: (T.eval_tab[s]['score'] >= st) & (T.eval_tab[s]['rec'] >= rt) for s in ids}
    arms = {}
    for tag, keep in keeps.items():
        for lexname, lex in (('none', None), ('full', 1)):
            arms['%s__%s' % (tag, lexname)] = T.counts(keep, lex)
    rng = np.random.RandomState(20261111)
    n = len(ids)
    M = rng.multinomial(n, [1.0 / n] * n, size=2000).astype(float)

    def boot(c):
        c = c.astype(float)
        return 200.0 * (M @ c[:, 0]) / (M @ (c[:, 1] + c[:, 2]))
    res = {}
    for lexname in ('none', 'full'):
        ref = arms['author__%s' % lexname]
        for tag in keeps:
            c = arms['%s__%s' % (tag, lexname)]
            d = boot(c) - boot(ref)
            res['%s [%s lexicon]' % (tag, lexname)] = dict(f1=float(f1(c)), minus_author=float(f1(c) - f1(ref)), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                                          fdr=float(100 * (1 - c[:, 0].sum() / max(c[:, 1].sum(), 1))), predictions=int(c[:, 1].sum()))
    np.savez_compressed(out / 'full_counts.npz', ids=np.array(ids), **arms)
    (out / 'full_summary.json').write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print('%-36s F1 %.2f (%+.2f [%+.2f,%+.2f]) FDR %.1f pred %d' % (k, v['f1'], v['minus_author'], *v['ci'], v['fdr'], v['predictions']), flush=True)
    import shutil
    shutil.rmtree(out / 'work_full', ignore_errors=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--stage', choices=['full'], required=True)
    p.add_argument('--out', type=Path, required=True)
    full_stage(p.parse_args())
