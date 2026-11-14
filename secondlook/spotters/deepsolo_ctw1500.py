"""CTW1500 with the OFFICIAL DeepSolo checkpoint/config/evaluator.

Run with the legacy DeepSolo Python (torch 1.9 + detectron2 0.6 + sklearn 0.23.2) from the
repository root: ``PYTHONPATH=. python -m secondlook.spotters.deepsolo_ctw1500``.

Stages
  infer   GPU. Official R50 CTW1500 checkpoint + official config (short side 1000, max 1200),
          candidate floor 0.05, FP32, batch 1. Saves GT-free candidates (official polygon
          construction and CTC decoding via the unmodified upstream TextEvaluator methods)
          plus 97-class CTC statistics.
  gate    CPU. Fidelity gate: official score threshold 0.4 on all TEST candidates, scored with
          the UNMODIFIED upstream ``to_eval_format`` / ``sort_detection`` / ``text_eval_script``
          (lexicon None and the official "Full" weak lexicon) against the paper/README 64.2 / 81.4.
  fit     CPU. Calibrate the word-admission head ONLY on CTW1500 TRAIN (11 partitions), then ONE
          exact official evaluation on the 500 TEST images per partition and arm.

Approximate reproduction. TEST is evaluated once per frozen policy; nothing is re-chosen on it.
Classification: official checkpoint/config/evaluator = PAPER/REPOSITORY_SPECIFIED; head, floor 0.2,
97-class feature adaptation, label-based selection F1 = ASSUMPTION.
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

FLOOR_INFER = 0.05
CAND_FLOOR = 0.2          # ASSUMPTION (same as E6/E7 on TT/IC15)
OFFICIAL_SCORE = 0.4      # configs/R_50/Base_det.yaml INFERENCE_TH_TEST
OFFICIAL_EVAL_MIN = 0.1   # to_eval_format drops score <= 0.1
NAMES = ['score', 'rec_conf', 'min_char_prob', 'mean_char_logp', 'min_char_margin', 'mean_char_margin', 'max_entropy', 'mean_entropy', 'blank_fraction',
         'log_rel_area', 'log_height', 'log_aspect', 'text_len', 'digit_frac', 'alpha_frac', 'upper_frac']
CUTS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
SCORE_GRID = [round(0.2 + 0.05 * i, 2) for i in range(15)]
REC_GRID = [0.0, 0.5, 0.75, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99]
CTLABELS = [chr(i) for i in range(32, 127)]   # identical to the 95-character list in the upstream evaluator (ASCII 32..126)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def ctc_stats(probs):
    """Same definitions as secondlook.spotters.deepsolo_outputs.ctc_token_statistics, for K classes (blank = K-1)."""
    p = np.asarray(probs, np.float64)
    k = p.shape[1]
    log_den = math.log(k)
    emitted, margins, ents, blank, prev = [], [], [], 0, None
    for row in p:
        tok = int(row.argmax())
        if tok >= k - 1:
            blank += 1
            prev = None
            continue
        if tok != prev:
            second = np.max(np.delete(row, tok))
            nz = row[row > 1e-15]
            ent = float(-(nz * np.log(nz)).sum() / log_den)
            emitted.append(float(row[tok]))
            margins.append(float(max(0.0, row[tok] - second)))
            ents.append(min(max(ent, 0.0), 1.0))
        prev = tok
    bf = blank / len(p)
    if not emitted:
        return dict(rec_conf=0.0, min_char_prob=0.0, mean_char_logp=-12.0, min_char_margin=0.0, mean_char_margin=0.0, max_entropy=1.0, mean_entropy=1.0, blank_fraction=bf)
    logs = [math.log(max(v, 1e-12)) for v in emitted]
    return dict(rec_conf=float(math.exp(sum(logs) / len(logs))), min_char_prob=min(emitted), mean_char_logp=sum(logs) / len(logs), min_char_margin=min(margins),
                mean_char_margin=sum(margins) / len(margins), max_entropy=max(ents), mean_entropy=sum(ents) / len(ents), blank_fraction=bf)


def official_stub(source, lexicon_type=None):
    """Instance of the unmodified upstream TextEvaluator (attributes set without a detectron2 dataset catalogue)."""
    sys.path.insert(0, str(source))
    from adet.evaluation.text_evaluation_all import TextEvaluator
    ev = object.__new__(TextEvaluator)
    ev.dataset_name = 'ctw1500'
    ev.voc_size = 96
    ev.CTLABELS = CTLABELS
    ev.lexicon_type = lexicon_type
    return ev


# ----------------------------------------------------------------------------- inference

def infer(args):
    import cv2
    import torch
    source = Path(args.source)
    sys.path.insert(0, str(source))
    ev = official_stub(source)
    from adet.config import get_cfg
    from secondlook.spotters.deepsolo_worker import _load_model_and_validate, _predict
    from secondlook.spotters.deepsolo_resize import configure_model_resize
    assets = Path(args.assets)
    ckpt = Path(args.checkpoint)
    cfg = get_cfg()
    cfg_path = source / 'configs/R_50/CTW1500/finetune_96voc_50maxlen.yaml'
    cfg.merge_from_file(str(cfg_path))
    cfg.MODEL.WEIGHTS = str(ckpt)
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = args.floor
    cfg.freeze()
    torch.manual_seed(20261105)
    np.random.seed(20261105)
    torch.backends.cudnn.benchmark = False
    model, compat = _load_model_and_validate(cfg, str(ckpt))
    configure_model_resize(model, cfg)
    ann = read_json(assets / 'ctw1500' / ('test.json' if args.split == 'test' else 'train_96voc.json'))
    images = ann['images'][:args.limit] if args.limit else ann['images']
    root = Path(args.images) / args.split / 'images'
    out = {}
    dump = None
    if getattr(args, 'dump_posteriors', None):   # optional, does not change any cached output (E22 prerequisite, not yet run on real data)
        from secondlook.spotters.deepsolo_posteriors import Dump
        dump = Dump(floor=FLOOR_INFER)
    tick = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for i, im in enumerate(images):
        path = root / im['file_name']
        if not path.exists():
            path = root / (im['file_name'].lstrip('0'))
        if not path.exists():
            cand = [p for p in root.iterdir() if p.stem == Path(im['file_name']).stem.lstrip('0')]
            if len(cand) != 1:
                raise FileNotFoundError('Cannot resolve image ' + im['file_name'])
            path = cand[0]
        if args.override and (Path(args.override) / (im['file_name'].lstrip('0'))).exists():
            # six TRAIN rasters exist locally only as EXIF-rotated PNG copies whose size differs from the annotations; use the original JPEG
            path = Path(args.override) / (im['file_name'].lstrip('0'))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None and image.shape[:2] != (im['height'], im['width']):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None or image.shape[:2] != (im['height'], im['width']):
            raise ValueError('Unreadable or resized raster ' + str(path))
        t0 = time.perf_counter()
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            torch.cuda.synchronize()
            pred = _predict(model, cfg, image, False)
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        inst = pred['instances'].to('cpu')
        coco = ev.instances_to_coco_json(inst, dict(image_id=im['id'], width=im['width'], height=im['height'])) if len(inst) else []
        cands = []
        for j, c in enumerate(coco):
            st = ctc_stats(inst.rec_scores[j].numpy())
            c = dict(c)
            c.update(st)
            cands.append(c)
        if dump is not None and len(inst):
            bd = inst.bd.numpy() if inst.has('bd') and not isinstance(inst.bd, list) else None
            dump.add_image(im['id'], inst.scores.numpy(), inst.rec_scores.numpy(), inst.recs.numpy(), inst.ctrl_points.numpy(), [c['rec'] for c in coco], boundary=bd)
        out[str(im['id'])] = dict(width=im['width'], height=im['height'], file=im['file_name'], seconds=sec, cands=cands)
        if i % 50 == 0:
            print('infer', args.split, i + 1, len(images), flush=True)
    write_json(Path(args.out) / ('cands_%s_floor%s.json' % (args.split, args.floor)), dict(
        images=out, split=args.split, floor=args.floor, checkpoint=str(ckpt), checkpoint_sha256=sha256(ckpt), config_sha256=sha256(cfg_path),
        compatibility=compat, wall_seconds=time.perf_counter() - tick, peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        no_gt_used=True, source_sha256=sha256(__file__), cfg_min_max=[cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST]))
    if dump is not None:
        dump.write(args.dump_posteriors, args.split, CTLABELS, dict(checkpoint_sha256=sha256(ckpt), config_sha256=sha256(cfg_path), split=args.split, images=len(images), source_sha256=sha256(__file__),
                                                              candidates_json='cands_%s_floor%s.json' % (args.split, args.floor)))
    print('infer done', args.split, len(images), time.perf_counter() - tick, flush=True)


# ----------------------------------------------------------------------------- evaluation

class Official:
    """Unmodified upstream ``to_eval_format``/``sort_detection``/``text_eval_script`` in a scratch directory."""

    def __init__(self, source, assets, work, gt_zip):
        self.source, self.work, self.gt = Path(source), Path(work), Path(gt_zip)
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / 'datasets/ctw1500').mkdir(parents=True, exist_ok=True)
        for n in ['weak_voc_new.txt', 'weak_voc_pair_list.txt']:
            shutil.copy(Path(assets) / 'ctw1500' / n, self.work / 'datasets/ctw1500' / n)
        sys.path.insert(0, str(self.source))
        import adet.evaluation.text_eval_script as module
        from secondlook.evaluation.detection_counters import evaluate_with_detection_counts
        self.module, self.counts_fn = module, evaluate_with_detection_counts
        self.tag = 0

    def run(self, entries, lexicon_type=None):
        """entries: official coco-json dicts (image_id, polys, rec, score). Returns (counts rows by image, native result, counters)."""
        self.tag += 1
        scratch = self.work / ('run%d' % self.tag)
        scratch.mkdir()
        os.symlink(self.work / 'datasets', scratch / 'datasets')
        cwd = os.getcwd()
        ev = official_stub(self.source, lexicon_type)
        os.chdir(scratch)
        try:
            write_json('text_results.json', entries)
            with contextlib.redirect_stdout(io.StringIO()):
                ev.to_eval_format('text_results.json', 'temp_det_results/')
                det, det_full = ev.sort_detection('temp_det_results/')
            det = os.path.abspath(det_full if lexicon_type is not None else det)
        finally:
            os.chdir(cwd)
        self.module.WORD_SPOTTING = False   # DeepSolo TextEvaluator: ctw1500 -> _word_spotting False
        params = self.module.default_evaluation_params()
        self.module.validate_data(str(self.gt), det, params)
        native, counters = self.counts_fn(self.module, str(self.gt), det, params)
        official = self.module.text_eval_main(det, str(self.gt), False)     # unmodified headline function
        if official['e2e_method'] != native['e2e_method']:
            raise ValueError('Counter wrapper changed the official E2E string')
        shutil.rmtree(scratch)
        return native, counters, official


def counts_array(counters, image_ids):
    rows = []
    for iid in image_ids:
        key = '%07d' % int(iid)    # evaluator key = capturing group of '([0-9]+).txt'
        c = counters[key]
        rows.append([c['e2e_correct'], c['e2e_predictions'], c['e2e_gt_words']])
    return np.asarray(rows, np.int64)


def f1_of(c):
    d = c[:, 1].sum() + c[:, 2].sum()
    return float(2 * c[:, 0].sum() / d) if d else 0.0


def parse(s):
    import re
    m = re.search(r'precision: ([\d.eE+-]+), recall: ([\d.eE+-]+), hmean: ([\d.eE+-]+)', s)
    return dict(precision=float(m.group(1)), recall=float(m.group(2)), hmean=float(m.group(3)))


def features(c, wh):
    xs = [p[0] for p in c['polys']]
    ys = [p[1] for p in c['polys']]
    w, h = max(max(xs) - min(xs), 1.0), max(max(ys) - min(ys), 1.0)
    t = c['rec']
    n = max(len(t), 1)
    from shapely.geometry import Polygon
    area = Polygon([(p[0], p[1]) for p in c['polys']]).area
    return [c['score'], c['rec_conf'], c['min_char_prob'], c['mean_char_logp'], c['min_char_margin'], c['mean_char_margin'], c['max_entropy'], c['mean_entropy'],
            c['blank_fraction'], math.log(max(area, 1.0) / (wh[0] * wh[1])), math.log(h), math.log(w / h), len(t), sum(ch.isdigit() for ch in t) / n,
            sum(ch.isalpha() for ch in t) / n, sum(ch.isupper() for ch in t) / n]


def table(cand_json, floor):
    """Official validity/threshold pre-filtering (same tests as upstream, applied once so line indices align)."""
    from shapely.geometry import Polygon
    tab, dropped = {}, 0
    for iid, im in cand_json['images'].items():
        rows = []
        for c in im['cands']:
            if c['score'] <= OFFICIAL_EVAL_MIN or c['score'] < floor:
                continue
            try:
                pg = Polygon([(float(p[0]), float(p[1])) for p in c['polys']])
                valid = pg.is_valid
            except Exception:
                valid = False
            if not valid:
                dropped += 1
                continue
            rows.append(c)
        tab[iid] = dict(cands=rows, x=np.asarray([features(c, (im['width'], im['height'])) for c in rows], np.float64).reshape(-1, len(NAMES)),
                        score=np.asarray([c['score'] for c in rows]), rec=np.asarray([c['rec_conf'] for c in rows]))
    return tab, dropped


def entries_of(tab, ids, keep):
    out = []
    for iid in ids:
        out.extend(c for c, k in zip(tab[iid]['cands'], keep[iid]) if k)
    return out


def gate(args):
    assets = Path(args.assets)
    cj = read_json(Path(args.out) / ('cands_test_floor%s.json' % FLOOR_INFER))
    ids = sorted(cj['images'], key=int)
    off = Official(args.source, assets, Path(args.out) / 'scratch_gate', assets / 'evaluation' / 'gt_ctw1500.zip')
    # (a) the unmodified native path: all candidates >= 0.4 exactly as the upstream evaluator would receive them
    native_entries = [c for iid in ids for c in cj['images'][iid]['cands'] if c['score'] >= OFFICIAL_SCORE]
    res = {}
    for name, lex in [('None', None), ('Full', 1)]:
        native, counters, official = off.run(native_entries, lex)
        res[name] = dict(e2e=parse(native['e2e_method']), det_only=parse(native['det_only_method']), official_string=official['e2e_method'],
                         counts=counts_array(counters, ids).tolist())
        print('official', name, res[name]['e2e'], flush=True)
    # (b) our pre-filtered table must reproduce the same counts (alignment check used by every later arm)
    tab, dropped = table(cj, OFFICIAL_SCORE)
    keep = {iid: [True] * len(tab[iid]['cands']) for iid in ids}
    native, counters, _ = off.run(entries_of(tab, ids, keep), None)
    same = counts_array(counters, ids).tolist() == res['None']['counts']
    print('prefiltered table reproduces official counts:', same, 'dropped invalid:', dropped, flush=True)
    write_json(Path(args.out) / 'gate.json', dict(
        status='COMPLETE', mode='approximate', claim_label='approximate reproduction (official CTW1500 checkpoint/config/evaluator, fidelity gate)', images=len(ids),
        reference_paper=dict(README_None=64.2, README_Full=81.4), result=res, prefiltered_equals_official=same, dropped_invalid=dropped,
        deepsolo_images='%d/%d (100%%)' % (len(ids), len(ids)), ticls_images='0/%d (0%%)' % len(ids), source_sha256=sha256(__file__),
        evaluator_sha256={p.name: sha256(p) for p in [Path(args.source) / 'adet/evaluation/text_eval_script.py', Path(args.source) / 'adet/evaluation/text_evaluation_all.py', Path(args.source) / 'adet/evaluation/rrc_evaluation_funcs.py']},
        gt_sha256=sha256(assets / 'evaluation' / 'gt_ctw1500.zip')))


def partitions(ids, seed):
    rng = np.random.default_rng(20261105 + seed)
    perm = [ids[i] for i in rng.permutation(len(ids))]
    return perm[:800], perm[800:900], perm[900:]


def fit(args):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from secondlook.features.word_model import new_word_model
    from secondlook.evaluation.protocol import load_config
    from secondlook.evaluation.declared_metric import word_labels
    assets, out = Path(args.assets), Path(args.out)
    cfg = load_config(Path(args.config))
    train = read_json(out / ('cands_train_floor%s.json' % FLOOR_INFER))
    test = read_json(out / ('cands_test_floor%s.json' % FLOOR_INFER))
    tr, _ = table(train, CAND_FLOOR)
    te, _ = table(test, CAND_FLOOR)
    tr_ids = sorted(tr, key=int)
    te_ids = sorted(te, key=int)
    off_tr = Official(args.source, assets, out / 'scratch_train', Path(args.train_gt))
    off_te = Official(args.source, assets, out / 'scratch_test', assets / 'evaluation' / 'gt_ctw1500.zip')
    # labels for every TRAIN candidate (exact evaluator on the full candidate set)
    native, counters, _ = off_tr.run(entries_of(tr, tr_ids, {i: [True] * len(tr[i]['cands']) for i in tr_ids}), None)
    labels, gtw = {}, {}
    for i in tr_ids:
        key = '%07d' % int(i)
        n = len(tr[i]['cands'])
        s = native['per_sample'][key]
        if n != len(s['detTrans']):
            raise ValueError('Label identity mismatch for image ' + i)
        labels[i] = word_labels(list(range(n)), counters[key]['e2e_correct_indices'], s['detDontCare'], n)
        gtw[i] = counters[key]['e2e_gt_words']
    print('train candidates', sum(len(v) for v in labels.values()), 'positives', int(sum((v == 1).sum() for v in labels.values())), flush=True)
    seeds = list(range(args.seeds))
    results = {}
    default_keep = {i: (te[i]['score'] >= OFFICIAL_SCORE).tolist() for i in te_ids}
    base_counts = None
    base_full = None
    for seed in seeds:
        fit_ids, sel_ids, aud_ids = partitions(tr_ids, seed)
        x = np.concatenate([tr[i]['x'] for i in fit_ids if len(tr[i]['cands'])])
        y = np.concatenate([labels[i] for i in fit_ids if len(tr[i]['cands'])])
        m = y >= 0
        models = dict(hgb=new_word_model('hgb_x', cfg), logistic=make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=11)))
        for mod in models.values():
            mod.fit(x[m], y[m])

        def approx_f1(ids, keep):   # label-based pooled F1 on the selection images (ASSUMPTION; TEST numbers use the exact evaluator)
            tp = pr = g = 0
            for i in ids:
                k = np.asarray(keep[i], bool)
                lab = labels[i]
                tp += int(((lab == 1) & k).sum())
                pr += int(((lab >= 0) & k).sum())
                g += gtw[i]
            return 2 * tp / (pr + g) if pr + g else 0.0

        sel = {}
        for name, mod in models.items():
            p = {i: (mod.predict_proba(tr[i]['x'])[:, 1] if len(tr[i]['cands']) else np.zeros(0)) for i in sel_ids}
            rows = [dict(cut=cut, f1=approx_f1(sel_ids, {i: p[i] >= cut for i in sel_ids})) for cut in CUTS]
            sel[name] = max(rows, key=lambda r: (r['f1'], -r['cut']))
        rows = [dict(score=a, rec=b, f1=approx_f1(sel_ids, {i: (tr[i]['score'] >= a) & (tr[i]['rec'] >= b) for i in sel_ids})) for a in SCORE_GRID for b in REC_GRID]
        sel['grid'] = max(rows, key=lambda r: (r['f1'], r['score'], r['rec']))
        sel['author_default_selection_f1'] = approx_f1(sel_ids, {i: tr[i]['score'] >= OFFICIAL_SCORE for i in sel_ids})
        pt = {n: {i: (mod.predict_proba(te[i]['x'])[:, 1] if len(te[i]['cands']) else np.zeros(0)) for i in te_ids} for n, mod in models.items()}
        keeps = {'author_default': default_keep,
                 'selection_tuned_thresholds': {i: ((te[i]['score'] >= sel['grid']['score']) & (te[i]['rec'] >= sel['grid']['rec'])).tolist() for i in te_ids},
                 'logistic_head': {i: (pt['logistic'][i] >= sel['logistic']['cut']).tolist() for i in te_ids},
                 'hgb_head': {i: (pt['hgb'][i] >= sel['hgb']['cut']).tolist() for i in te_ids}}
        res = dict(selection=sel, fit_words=int(m.sum()), positive_rate=float(y[m].mean()))
        arr = {}
        for tag, keep in keeps.items():
            if tag == 'author_default' and base_counts is not None:
                arr[tag] = base_counts
            else:
                _, c, _ = off_te.run(entries_of(te, te_ids, keep), None)
                arr[tag] = counts_array(c, te_ids)
                if tag == 'author_default':
                    base_counts = arr[tag]
            full = None
            if args.full:
                if tag == 'author_default' and base_full is not None:
                    full = base_full
                else:
                    _, cf, _ = off_te.run(entries_of(te, te_ids, keep), 1)
                    full = counts_array(cf, te_ids)
                    if tag == 'author_default':
                        base_full = full
                arr['full_' + tag] = full
            res[tag] = dict(full_lexicon_f1_pct=(100 * f1_of(full) if full is not None else None), f1_pct=100 * f1_of(arr[tag]), correct=int(arr[tag][:, 0].sum()), predictions=int(arr[tag][:, 1].sum()), gt=int(arr[tag][:, 2].sum()),
                            deepsolo_images='%d/%d (100%%)' % (len(te_ids), len(te_ids)), ticls_images='0/%d (0%%)' % len(te_ids))
        rng = np.random.default_rng(20261104)
        draws = rng.integers(0, len(te_ids), size=(10000, len(te_ids)))

        def pooled(c):
            return 2 * c[..., 0].sum(-1) / (c[..., 1].sum(-1) + c[..., 2].sum(-1))
        for ref in ['author_default', 'selection_tuned_thresholds']:
            for tag in ['hgb_head', 'logistic_head']:
                d = pooled(arr[tag][draws]) - pooled(arr[ref][draws])
                res[tag]['vs_' + ref] = dict(delta_pp=float(100 * (pooled(arr[tag]) - pooled(arr[ref]))), ci95_pp=[float(100 * np.percentile(d, 2.5)), float(100 * np.percentile(d, 97.5))])
        np.savez(out / (args.tag + 'test_counts_seed%d.npz' % seed), ids=np.asarray(te_ids), **arr)
        results[seed] = res
        print('seed', seed, json.dumps({k: (v['f1_pct'] if isinstance(v, dict) and 'f1_pct' in v else None) for k, v in res.items()}), flush=True)
        write_json(out / (args.tag + 'fit_results.json'), dict(status='PARTIAL' if seed != seeds[-1] else 'COMPLETE', mode='approximate', claim_label='approximate reproduction (CTW1500 TRAIN calibration; one exact official TEST evaluation per frozen policy)',
            partitions='800/100/100 of 1000 TRAIN images, seeds 20261105+k', feature_names=NAMES, candidate_floor=CAND_FLOOR, results={str(k): v for k, v in results.items()},
            test_images=len(te_ids), source_sha256=sha256(__file__)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--stage', choices=['infer', 'gate', 'fit'], required=True)
    p.add_argument('--source', required=True, help='DeepSolo upstream source root (commit dbadae99...)')
    p.add_argument('--assets', type=Path, required=True, help='extracted official assets directory')
    p.add_argument('--checkpoint', help='official R50 CTW1500 checkpoint')
    p.add_argument('--images', help='CTW1500 image root containing train/images and test/images')
    p.add_argument('--split', choices=['train', 'test'], default='test')
    p.add_argument('--floor', type=float, default=FLOOR_INFER)
    p.add_argument('--dump-posteriors', help='infer stage: also write per-point CTC posteriors (score >= %s, schema deepsolo-posteriors/2) to this directory' % FLOOR_INFER)
    p.add_argument('--limit', type=int)
    p.add_argument('--override', help='directory with original JPEGs for rasters whose local copy was rotated')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train-gt', help='converted TRAIN GT zip (research/ctw1500_gt.py)')
    p.add_argument('--config', type=Path, default=Path('configs/approximate/protocol_calibration_v1.json'))
    p.add_argument('--seeds', type=int, default=11)
    p.add_argument('--full', action='store_true', help='also score with the official weak (Full) lexicon')
    p.add_argument('--tag', default='')
    a = p.parse_args()
    {'infer': infer, 'gate': gate, 'fit': fit}[a.stage](a)
