"""E46 (approximate reproduction): the UNIFIED feature specification of the filtering layer and its builder for the nine rows.

Specification (one rule for DeepSolo, TiCLS and PP-OCR; the 11 columns of `xc` keep their order and names):
  0 detector_logit      logit(clip(score, 1e-6, 1-1e-6)). Formula unified; the score itself is spotter specific (query classification probability vs DB box probability).
  1 geo_mean_token_prob exp(mean log p) over the EMITTED symbols of the candidate.      } emitted symbol = CTC: argmax non-blank, repeats merged; BART: token before the first
  2 min_token_prob      min p over the emitted symbols.                                   } stop token; PP-OCR word level: the characters of the word in the line's CTC path.
  3 mean_token_entropy  mean over emitted symbols of H / log(V)  (V = output vocabulary; a per-feature monotone scale is irrelevant for trees).
  4 min_token_margin    min over emitted symbols of (top1 - top2).
  5 log_rel_area        log(max(|shoelace(P)|, 1) / (W*H))      P = the polygon submitted to the evaluator (integer pixel coordinates of the ORIGINAL image).
  6 log_height          log(max(extent_y(P), 1))                 pixels of the original image, axis-aligned extent, no +1 convention.
  7 log_aspect          log(max(extent_x(P), 1) / max(extent_y(P), 1)).
  8 log_len, 9 digit_frac, 10 letter_frac  of the submitted transcription (unchanged: already identical for the three spotters).
Baseline `rec` (re-tuned thresholds) = column 1 (the geometric mean) for every spotter.
What changes with respect to the pre-unification tables (audit: the study log): columns 5-7 for DeepSolo (+1 px convention) / TiCLS (box area -> polygon area,
input-resolution height -> original pixels) / PP-OCR on Total-Text and ICDAR2015 (area was in pixels, not divided by W*H); columns 3-4 for PP-OCR (line-level max entropy and line-level
min margin -> word-level mean entropy and word-level min margin; on CTW1500 the candidate is the line, so line = candidate); `rec` for TiCLS (min) and PP-OCR (line arithmetic mean).
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from secondlook.paths import RUNS

RUNS = RUNS
CONFIG = Path('configs/approximate/ppocr_cascade_v3.json')
UNIFIED_COLUMNS = ['detector_logit', 'geo_mean_token_prob', 'min_token_prob', 'mean_token_entropy', 'min_token_margin', 'log_rel_area', 'log_height', 'log_aspect', 'log_len',
                   'digit_frac', 'letter_frac']


def shoelace(P):
    return 0.5 * abs(float(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1))))


def unified_geometry(P, wh):
    """Columns 5-7 of the unified specification for the submitted polygon P (n x 2) of an image of size wh = (W, H)."""
    P = np.asarray(P, float)
    w, h = max(float(P[:, 0].max() - P[:, 0].min()), 1.0), max(float(P[:, 1].max() - P[:, 1].min()), 1.0)
    return [math.log(max(shoelace(P), 1.0) / (float(wh[0]) * float(wh[1]))), math.log(h), math.log(w / h)]


def parse_line_polygon(line, task):
    """Polygon of one submission line. Total-Text: leading integers up to the '####text' field; ICDAR2015: exactly the first 8 integers (4 corners)."""
    parts = line.rstrip('\n').split(',')
    if task == 'icdar2015':
        nums = [int(v) for v in parts[:8]]
    else:
        nums = []
        for v in parts:
            if re.fullmatch(r'-?\d+', v):
                nums.append(int(v))
            else:
                break
    if len(nums) < 6 or len(nums) % 2:
        raise ValueError('Cannot parse a polygon from submission line %r' % line[:80])
    return np.asarray(nums, float).reshape(-1, 2)


def image_sizes(task):
    """sample_id -> (W, H) of the original raster for Total-Text / ICDAR2015 TRAIN and TEST (records of the cascade config and of the E42 extra images)."""
    from secondlook.evaluation.protocol import load_config, read_json
    cfg = load_config(CONFIG)
    spec = cfg['datasets'][task]
    out = {}
    for key in ('train_inputs', 'test_inputs'):
        raw = read_json(spec[key])
        for r in (raw['records'] if isinstance(raw, dict) else raw):
            out[r['sample_id']] = tuple(r['image_wh'])
    from secondlook.data.collect_extra_train import extra_records
    for r in extra_records(task)[0]:
        out[r['sample_id']] = tuple(r['image_wh'])
    return out


def ctw_sizes():
    from secondlook.data.tasks_ctw1500 import E11
    from secondlook.evaluation.protocol import read_json
    out = {}
    for split in ('train', 'test'):
        for iid, im in read_json(E11 / ('cands_%s_floor0.05.json' % split))['images'].items():
            out[iid] = (im['width'], im['height'])
    return out


def ctw_ticls_sizes():
    from secondlook.data.tasks_ctw1500 import TICLS_BANK
    from secondlook.evaluation.protocol import read_json
    out = {}
    for split in ('train', 'test'):
        for iid, im in read_json(TICLS_BANK / split / ('cands_%s_ticls.json' % split))['images'].items():
            out[iid] = (im['width'], im['height'])
    return out


# ---- old geometry rules, used ONLY to prove that the polygon + size sources are the ones the old tables used ----
def old_geometry(e, t, P, wh):
    W, H = wh
    x0, x1, y0, y1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    if e == 'deepsolo' and t != 'ctw1500':   # poly_bbox = [min, max + 1]
        w, h = max(x1 + 1 - x0, 1.0), max(y1 + 1 - y0, 1.0)
        return [math.log(max(shoelace(P), 1.0) / (W * H)), math.log(h), math.log(w / h)]
    if e == 'ticls':                          # axis-aligned box area; height at the network input scale (not reproducible without input_h)
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        return [math.log(w * h / (W * H)), None, math.log(w / h)]
    if e == 'ppocr' and t != 'ctw1500':       # pixels, not divided by W*H
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        return [math.log(max(shoelace(P), 1.0)), math.log(h), math.log(w / h)]
    return unified_geometry(P, wh)            # DeepSolo CTW1500 and PP-OCR CTW1500 already follow the unified rule


def ppocr_word_candidates(row, dataset_is_line):
    """Yield (word_text, polygon, entropy_array, margin_array, probs_array) in the order of secondlook.data.tasks_ppocr.word_tables (Total-Text / ICDAR2015)."""
    from secondlook.spotters.ppocr_collect import valid_quad
    out = []
    for w in row['words']:
        if not valid_quad([list(q) for q in w['points']]):
            continue
        line = row['rows'][w['line_id']]
        text, probs = line['text'], line['char_probs']
        pos = text.find(w['text'])
        ok = pos >= 0 and len(probs) == len(text)
        sl = slice(pos, pos + len(w['text'])) if ok else slice(None)
        ent = np.asarray(line['char_entropy'][sl] if len(line['char_entropy']) == len(probs) else line['char_entropy'], float)
        mar = np.asarray(line['char_margin'][sl] if len(line['char_margin']) == len(probs) else line['char_margin'], float)
        out.append((w['text'], np.asarray(w['points'], float), ent if len(ent) else np.asarray(line['char_entropy'], float),
                    mar if len(mar) else np.asarray(line['char_margin'], float), line))
    return out


def build_unified(T, e, t, ppocr_stats=None):
    """Return ({'train': {sid: xc_new}, 'eval': {sid: xc_new}}, rec_new dicts, report). Old tables are NOT modified; the caller installs the result."""
    report = dict(expert=e, task=t, geometry_gate={}, changed_columns={})
    if e == 'ppocr' and t == 'ctw1500':   # CTW1500 image ids are the integer tail of the sample id (as in admission_cv_ctw.ppocr_table)
        ppocr_stats = {k: {str(int(sid.split(':')[-1])): v for sid, v in d.items()} for k, d in ppocr_stats.items()}
    new_tr, new_te, rec_tr, rec_te = {}, {}, {}, {}
    if t == 'ctw1500':
        sizes = ctw_ticls_sizes() if e == 'ticls' else ctw_sizes()
    else:
        sizes = image_sizes(t)
    errs = {c: [] for c in (5, 6, 7)}
    height_consts = {}
    for split, tab, ids, dst, rdst in (('train', T.tab, T.ids, new_tr, rec_tr), ('eval', T.eval_tab, T.eval_ids, new_te, rec_te)):
        for s in ids:
            xo = np.asarray(tab[s]['xc'], float)
            n = len(xo)
            if t == 'ctw1500':
                polys = [np.asarray(c['polys'], float).reshape(-1, 2) for c in tab[s]['cands']]
            else:
                polys = [parse_line_polygon(l, t) for l in tab[s]['lines']]
            if len(polys) != n:
                raise ValueError('Candidate/polygon count mismatch for %s %s' % (e, s))
            xn = xo.copy()
            for i, P in enumerate(polys):
                wh = sizes[s if t != 'ctw1500' else s]
                old = old_geometry(e, t, P, wh)
                for c, ov in zip((5, 6, 7), old):
                    if ov is not None:
                        errs[c].append(abs(ov - xo[i, c]))
                if e == 'ticls':
                    height_consts.setdefault((split, s), []).append(math.log(max(P[:, 1].max() - P[:, 1].min(), 1.0)) - xo[i, 6])
                xn[i, 5:8] = unified_geometry(P, wh)
            if e == 'ppocr':
                if ppocr_stats is None:
                    raise ValueError('PP-OCR statistics (E46 re-run) required')
                row = ppocr_stats[split][s if t != 'ctw1500' else s]
                if t != 'ctw1500':
                    cands = ppocr_word_candidates(row, False)
                    if len(cands) != n:
                        raise ValueError('PP-OCR candidate count mismatch for %s: %d vs %d' % (s, len(cands), n))
                    for i, (_, P, ent, mar, line) in enumerate(cands):
                        # gate: the old (line-level) columns are reproduced from the re-run
                        if abs(float(np.max(line['char_entropy'])) - xo[i, 3]) > 1e-9 or abs(float(np.min(line['char_margin'])) - xo[i, 4]) > 1e-9:
                            raise ValueError('PP-OCR old line-level entropy/margin not reproduced for %s' % s)
                        xn[i, 3] = float(ent.mean())
                        xn[i, 4] = float(mar.min())
                else:
                    from secondlook.spotters.ppocr_collect import valid_quad
                    from secondlook.data.tasks_ctw1500 import _valid
                    kept = [r for r in row['rows'] if valid_quad([list(q) for q in r['points']]) and _valid([list(map(float, q)) for q in r['points']])]
                    if len(kept) != n:
                        raise ValueError('PP-OCR CTW candidate count mismatch for %s' % s)
                    for i, r in enumerate(kept):
                        if abs(float(np.max(r['char_entropy'])) - xo[i, 3]) > 1e-9 or abs(float(np.min(r['char_margin'])) - xo[i, 4]) > 1e-9:
                            raise ValueError('PP-OCR CTW old line-level entropy/margin not reproduced for %s' % s)
                        xn[i, 3] = float(np.mean(r['char_entropy']))
                        xn[i, 4] = float(np.min(r['char_margin']))
            dst[s] = xn
            rdst[s] = xn[:, 1].copy()          # baseline rec = geometric mean for every spotter
    report['geometry_gate'] = {('old_col%d' % c): (dict(n=len(v), max_abs=float(np.max(v)), p99=float(np.percentile(v, 99)), median=float(np.median(v))) if v else None) for c, v in errs.items()}
    if height_consts:   # TiCLS: log(extent_y) - old log_height must be constant within an image (= log(H / input_h))
        spread = [max(v) - min(v) for v in height_consts.values() if len(v) > 1]
        report['ticls_height_constant_within_image'] = dict(images=len(spread), max_spread=float(max(spread)) if spread else 0.0, median_spread=float(np.median(spread)) if spread else 0.0,
                                                              note='spread > 0 only from rounding of the submitted polygon to integers')
        ratios = [float(np.mean(v)) for v in height_consts.values()]
        report['ticls_log_H_over_input_h'] = dict(mean=float(np.mean(ratios)), min=float(np.min(ratios)), max=float(np.max(ratios)))
    return dict(train=new_tr, eval=new_te), dict(train=rec_tr, eval=rec_te), report
