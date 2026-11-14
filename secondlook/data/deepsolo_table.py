"""E6: learned word admission on cached DeepSolo outputs. CPU only.

Run with the legacy Python (sklearn 0.23.2, same family as the TiCLS heads).
Question: does the same idea as the TiCLS word-admission head (per-candidate quality
classifier + F1-aware cut chosen on source data) help a second, CTC-based expert,
DeepSolo, beyond (a) the author default, (b) the repo's two-threshold
``qualified_calibrated`` rule and (c) a two-threshold grid re-tuned on the SAME
source selection set (the score-only / standard-calibration control)?

Everything is read from cached DeepSolo compact outputs; no neural call. Source =
TRAIN 584 FIT / 80 selection / 64 audit ids of ``protocol_calibration_v1``; TEST
300 (Total-Text) / 500 (ICDAR2015) scored once with the pinned native evaluator.
Primary comparison (declared before TEST): HGB head vs the selection-tuned
two-threshold control. Candidate floor 0.2 is an ASSUMPTION (same for all rows).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from secondlook.evaluation.protocol import load_config, read_json, write_json, sha256, hmean_fraction
from secondlook.paths import ARTIFACTS, RUNS

CAL = (RUNS / 'ticls-source-banks/protocol_calibration_v1')
DS = (RUNS / 'deepsolo-compact-outputs')
AUDIT = (ARTIFACTS / 'official-benchmark-audit-20261031-v1')
CFG = 'configs/approximate/protocol_calibration_v1.json'
FLOOR = dict(score_threshold=0.2, recognition_threshold=0.0, invalid_polygon_policy='drop')
AUTHOR = dict(score_threshold=0.4, recognition_threshold=0.0, invalid_polygon_policy='drop')
QUALIFIED = dict(score_threshold=0.35, recognition_threshold=0.925, invalid_polygon_policy='drop')
CUTS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
SCORE_GRID = [round(0.2 + 0.05 * i, 2) for i in range(15)]
REC_GRID = [0.0, 0.5, 0.75, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99]
NAMES = ['score', 'rec_conf', 'min_char_prob', 'mean_char_logp', 'min_char_margin', 'mean_char_margin', 'max_entropy', 'mean_entropy',
         'blank_fraction', 'log_rel_area', 'log_height', 'log_aspect', 'text_len', 'digit_frac', 'alpha_frac', 'upper_frac']


def features(row, wh):
    x0, y0, x1, y1 = row['bbox']
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    t = row['submission_text']
    n = max(len(t), 1)
    return [row['score'], row['recognition_confidence'], row['min_char_prob'], row['mean_char_log_prob'], row['min_char_margin'], row['mean_char_margin'],
            row['max_char_entropy'], row['mean_char_entropy'], row['blank_fraction'], math.log(max(row['polygon_area'], 1.0) / (wh[0] * wh[1])),
            math.log(h), math.log(w / h), len(t), sum(c.isdigit() for c in t) / n, sum(c.isalpha() for c in t) / n, sum(c.isupper() for c in t) / n]


def candidate_table(samples, ids, dataset):
    from secondlook.spotters.deepsolo_outputs import format_deepsolo_variant
    table = {}
    for sid in ids:
        s = samples[sid]
        fmt = format_deepsolo_variant(s, dataset, FLOOR, 'None')
        lines = [l + '\n' for l in fmt['body'].split('\n')[:-1]]
        rows = fmt['admitted_rows']
        if len(lines) != len(rows):
            raise ValueError('Formatter lines/rows misaligned ' + sid)
        table[sid] = dict(lines=lines, x=np.asarray([features(r, s['image_wh']) for r in rows], np.float64).reshape(-1, len(NAMES)),
                          score=np.asarray([r['score'] for r in rows]), rec=np.asarray([r['recognition_confidence'] for r in rows]))
    return table


def main(args):
    import sklearn
    from secondlook.evaluation.official_benchmarks import write_zip
    from secondlook.evaluation.declared_metric import DeclaredMetric, counter_key, word_labels
    from secondlook.features.word_model import new_word_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    dataset = args.dataset
    cfg = load_config(Path(CFG))
    exp = cfg['protocol_calibration']['experiments'][dataset + '_native']
    metric = DeclaredMetric(cfg, exp['metric'])
    base = CAL / (dataset + '_native') / dataset
    parts = read_json(base / 'partitions.json')
    fit_ids, sel_ids, aud_ids = parts['fit_ids'], parts['selection_ids'], parts['calibration_ids']
    train_ids = fit_ids + sel_ids + aud_ids
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    work = out / 'work'
    work.mkdir()
    train = read_json(DS / 'compact_deepsolo' / (dataset + '_train.json'))['samples']
    test = read_json(DS / 'compact_deepsolo' / (dataset + '_test.json'))['samples']
    names = {sid: v['filename'] for sid, v in read_json(base / 'native/bodies.json').items()}
    test_ids = sorted(test, key=lambda s: int(s.split(':')[-1]))
    router = read_json(cfg['baseline_config'])
    if dataset == 'totaltext':
        mp = read_json(read_json(router['benchmark_audit_config'])['datasets']['totaltext']['spotting_id_map'])
        test_names = {sid: mp[sid] for sid in test_ids}
    else:
        test_names = {sid: 'res_img_' + sid.split(':')[-1] + '.txt' for sid in test_ids}
    tt = candidate_table(train, train_ids, dataset)
    te = candidate_table(test, test_ids, dataset)

    def evaluate(tag, ids, tab, keep, gt, filenames):
        z = work / (tag + '.zip')
        write_zip(z, {filenames[s]: ''.join(l for l, k in zip(tab[s]['lines'], keep[s]) if k) for s in ids})
        return metric.evaluate(gt, z, ids, {s: dict(filename=filenames[s]) for s in ids})

    everything = {s: [True] * len(tt[s]['lines']) for s in train_ids}
    _, native, counters = evaluate('train_all', train_ids, tt, everything, base / 'native/source_gt.zip', names)
    labels = {}
    for s in train_ids:
        key = counter_key(names[s], metric.key_mode)
        sample = native['per_sample'][key]
        n = len(tt[s]['lines'])
        if n != len(sample['detTrans']):
            raise ValueError('Label identity mismatch')
        labels[s] = word_labels(list(range(n)), counters[key]['e2e_correct_indices'], sample['detDontCare'], n)
    x = np.concatenate([tt[s]['x'] for s in fit_ids])
    y = np.concatenate([labels[s] for s in fit_ids])
    m = y >= 0
    print(dataset, 'fit words', int(m.sum()), 'positive rate', float(y[m].mean()), flush=True)
    models = dict(hgb=new_word_model('hgb_x', cfg), logistic=make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=11)))
    for mod in models.values():
        mod.fit(x[m], y[m])

    def probs(mod, tab, ids):
        return {s: (mod.predict_proba(tab[s]['x'])[:, 1] if len(tab[s]['lines']) else np.zeros(0)) for s in ids}

    def f1_of(c):
        return float(hmean_fraction(c))

    selection = {}
    # head cuts
    for name, mod in models.items():
        p = probs(mod, tt, sel_ids)
        rows = []
        for cut in CUTS:
            c = evaluate('sel_%s_%s' % (name, cut), sel_ids, tt, {s: (p[s] >= cut).tolist() for s in sel_ids}, base / 'selection_gt.zip', names)[0]
            rows.append(dict(cut=cut, f1=f1_of(c)))
        selection[name] = max(rows, key=lambda r: (r['f1'], -r['cut']))
        selection[name]['grid'] = rows
    # two-threshold grid (score-only / standard control)
    rows = []
    for st in SCORE_GRID:
        for rt in REC_GRID:
            keep = {s: ((tt[s]['score'] >= st) & (tt[s]['rec'] >= rt)).tolist() for s in sel_ids}
            c = evaluate('sel_grid_%s_%s' % (st, rt), sel_ids, tt, keep, base / 'selection_gt.zip', names)[0]
            rows.append(dict(score=st, rec=rt, f1=f1_of(c)))
    selection['grid'] = max(rows, key=lambda r: (r['f1'], r['score'], r['rec']))
    print(dataset, 'selection', json.dumps({k: {a: b for a, b in v.items() if a != 'grid'} for k, v in selection.items()}), flush=True)
    # audit64 (source only)
    audit = {}
    pa = {n: probs(mod, tt, aud_ids) for n, mod in models.items()}
    for name in models:
        c = evaluate('aud_' + name, aud_ids, tt, {s: (pa[name][s] >= selection[name]['cut']).tolist() for s in aud_ids}, base / 'audit_gt.zip', names)[0]
        audit[name] = f1_of(c)
    g = selection['grid']
    audit['grid'] = f1_of(evaluate('aud_grid', aud_ids, tt, {s: ((tt[s]['score'] >= g['score']) & (tt[s]['rec'] >= g['rec'])).tolist() for s in aud_ids}, base / 'audit_gt.zip', names)[0])
    for tag, spec in [('author_default', AUTHOR), ('qualified_calibrated', QUALIFIED)]:
        audit[tag] = f1_of(evaluate('aud_' + tag, aud_ids, tt, {s: ((tt[s]['score'] >= spec['score_threshold']) & (tt[s]['rec'] >= spec['recognition_threshold'])).tolist() for s in aud_ids}, base / 'audit_gt.zip', names)[0])
    print(dataset, 'audit64', json.dumps(audit), flush=True)
    # TEST: one pass
    gt = AUDIT / ('gt_%s.zip' % dataset)
    pt = {n: probs(mod, te, test_ids) for n, mod in models.items()}
    keeps = {
        'author_default': {s: ((te[s]['score'] >= AUTHOR['score_threshold']) & (te[s]['rec'] >= 0)).tolist() for s in test_ids},
        'qualified_calibrated': {s: ((te[s]['score'] >= QUALIFIED['score_threshold']) & (te[s]['rec'] >= QUALIFIED['recognition_threshold'])).tolist() for s in test_ids},
        'selection_tuned_thresholds': {s: ((te[s]['score'] >= g['score']) & (te[s]['rec'] >= g['rec'])).tolist() for s in test_ids},
        'logistic_head': {s: (pt['logistic'][s] >= selection['logistic']['cut']).tolist() for s in test_ids},
        'hgb_head': {s: (pt['hgb'][s] >= selection['hgb']['cut']).tolist() for s in test_ids},
    }
    counts, results = {}, {}
    for tag, keep in keeps.items():
        counts[tag] = evaluate('test_' + tag, test_ids, te, keep, gt, test_names)[0]
        c = counts[tag]
        results[tag] = dict(f1_pct=100 * f1_of(c), correct=int(c[:, 0].sum()), predictions=int(c[:, 1].sum()), gt=int(c[:, 2].sum()), deepsolo_images='%d/%d (100%%)' % (len(test_ids), len(test_ids)), ticls_images='0/%d (0%%)' % len(test_ids))
    rng = np.random.default_rng(20261104)
    draws = rng.integers(0, len(test_ids), size=(10000, len(test_ids)))

    def pooled(c):
        return 2 * c[..., 0].sum(-1) / (c[..., 1].sum(-1) + c[..., 2].sum(-1))
    for ref in ['selection_tuned_thresholds', 'author_default', 'qualified_calibrated']:
        for tag in ['hgb_head', 'logistic_head']:
            d = pooled(counts[tag][:, :3][draws]) - pooled(counts[ref][:, :3][draws])
            point = 100 * (pooled(counts[tag][:, :3]) - pooled(counts[ref][:, :3]))
            results[tag]['vs_' + ref] = dict(delta_pp=float(point), ci95_pp=[float(100 * np.percentile(d, 2.5)), float(100 * np.percentile(d, 97.5))],
                                              images_more_correct=int(((counts[tag][:, 0] - counts[ref][:, 0]) > 0).sum()), images_fewer_correct=int(((counts[tag][:, 0] - counts[ref][:, 0]) < 0).sum()))
    write_json(out / 'results.json', dict(status='COMPLETE', mode='approximate', claim_label='approximate reproduction (cached DeepSolo outputs, CPU only)', dataset=dataset,
        floor=FLOOR, selection={k: {a: b for a, b in v.items() if a != 'grid'} for k, v in selection.items()}, audit64_source_f1=audit, test=results,
        feature_names=NAMES, new_neural_calls=0, sklearn=sklearn.__version__, source_sha256=sha256(__file__), config_sha256=sha256(CFG)))
    print(json.dumps(results, indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', choices=['totaltext', 'icdar2015'], required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
