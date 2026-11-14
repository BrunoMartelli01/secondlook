"""E15: cross-expert agreement as a word-admission signal. CPU only, no neural call.

For every candidate word of expert A (DeepSolo or TiCLS) the other expert B's candidates on the same image are
matched by axis-aligned bounding-box IoU (>= 0.5, ASSUMPTION fixed in advance; best IoU wins). Four agreement features:
  iou        best IoU with a B candidate (0 if none >= 0.5)
  text_equal 1 if the normalized transcripts (lower case, alphanumerics only) of the matched pair are identical
  edit_sim   1 - Levenshtein / max length of the normalized transcripts (0 if unmatched)
  other_score  B's detector score of the matched candidate (0 if unmatched)
Arms per (expert A, task, partition) with the E10 recipe (HGB, cut on the selection images, common11 base):
  author, tuned_all            references
  c11_base                     common11 only, TiCLS augmented rows NOT used (like-for-like with the agreement arms)
  c11_agree                    common11 + 4 agreement features
  score_agree                  detector logit + 4 agreement features only
Using agreement requires inference of BOTH experts, so its cost is the sum of both; this is a separate question from
the single-expert head. Same partitions/images as E10-E14, exact pinned evaluator, per-image counts saved.
Approximate reproduction; TT/IC15 TEST was explored before.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from secondlook.data.tasks import Task, TiclsTask, evaluate, pick_thresholds, AUTHOR
from secondlook.studies.ablation_controls import hgb
from secondlook.studies.ablation_learners import proba, cut_from
from secondlook.evaluation.protocol import write_json, read_json


def parse(line):
    parts = line.rstrip('\n').split(',')
    nums = []
    for p in parts:
        try:
            nums.append(int(float(p)))
        except ValueError:
            break
    n = len(nums) - (len(nums) % 2)
    xy = np.asarray(nums[:n], float).reshape(-1, 2)
    text = ','.join(parts[n:])
    box = (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()) if len(xy) else (0, 0, 0, 0)
    return box, ''.join(ch for ch in text.lower() if ch.isalnum())


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def agreement(tab_a, tab_b, ids):
    """Fill tab_a[s]['ag'] (n_a, 4) from the other expert's candidates; returns match statistics."""
    matched = total = 0
    for s in ids:
        A = [parse(l) for l in tab_a[s]['lines']]
        B = [parse(l) for l in tab_b[s]['lines']]
        sb = np.asarray(tab_b[s]['score'], float)
        f = np.zeros((len(A), 4))
        for i, (ba, ta) in enumerate(A):
            best, j = 0.0, -1
            for k, (bb, _) in enumerate(B):
                v = iou(ba, bb)
                if v > best:
                    best, j = v, k
            total += 1
            if best >= 0.5:
                matched += 1
                tb = B[j][1]
                f[i] = [best, float(ta == tb), 1.0 - lev(ta, tb) / max(1, max(len(ta), len(tb))), sb[j]]
        tab_a[s]['ag'] = f
        tab_a[s]['xca'] = np.hstack([tab_a[s]['xc'], f]) if len(f) else np.zeros((0, 15))
    return dict(candidates=total, matched=matched)


def stack_noaug(T, ids, cols):
    x = np.concatenate([T.tab[s]['xca'][:, cols] for s in ids])
    y = np.concatenate([T.lab[s] for s in ids])
    m = y >= 0
    return x[m], y[m]


def job(args):
    out = Path(args.out) / args.task / ('seed%d' % args.seed)
    out.mkdir(parents=True, exist_ok=False)
    work = out / 'work'
    work.mkdir()
    (work / 'ds').mkdir()
    (work / 'ti').mkdir()
    D = Task(args.task, work / 'ds')
    L = TiclsTask(args.task, work / 'ti')
    if set(D.ids) != set(L.ids) or set(D.eval_ids) != set(L.eval_ids):
        raise ValueError('Expert populations differ')
    meta = dict(task=args.task, seed=args.seed, stats={})
    meta['stats']['train_ds_vs_ticls'] = agreement(D.tab, L.tab, D.ids)
    meta['stats']['train_ticls_vs_ds'] = agreement(L.tab, D.tab, L.ids)
    meta['stats']['test_ds_vs_ticls'] = agreement(D.eval_tab, L.eval_tab, D.eval_ids)
    meta['stats']['test_ticls_vs_ds'] = agreement(L.eval_tab, D.eval_tab, L.eval_ids)
    fit, sel, aud = D.partition(args.seed)
    if (fit, sel) != L.partition(args.seed)[:2]:
        raise ValueError('Partition mismatch')
    pool = list(fit) + list(sel)
    all_counts = {}
    for name, T in (('deepsolo', D), ('ticls', L)):
        tab, ids, names, gt = T.eval_tab, T.eval_ids, T.eval_names, T.eval_gt
        counts = {}

        def exact(tag, keep, T=T, tab=tab, ids=ids, names=names, gt=gt, counts=counts, name=name):
            c = np.asarray(evaluate(None, work / ('ds' if name == 'deepsolo' else 'ti'), tag, ids, tab, {s: np.asarray(keep[s], bool).tolist() for s in ids}, gt, names, T.metric)[0])
            for f in (work / ('ds' if name == 'deepsolo' else 'ti')).glob(tag + '.zip'):
                f.unlink()
            counts[tag] = c.astype(np.int64)
        exact('author', {s: [True] * len(tab[s]['lines']) for s in ids} if name == 'ticls' else {s: (tab[s]['score'] >= AUTHOR['score_threshold']) for s in ids})
        st, rt, _ = pick_thresholds(T, pool)
        exact('tuned_all', {s: (tab[s]['score'] >= st) & (tab[s]['rec'] >= rt) for s in ids})
        for arm, cols in (('c11_base', list(range(11))), ('c11_agree', list(range(15))), ('score_agree', [0, 11, 12, 13, 14])):
            x, y = stack_noaug(T, fit, cols)
            m = hgb()
            m.fit(x, y)
            spec = ('xca', cols)
            cut = cut_from(T, proba(m, T.tab, sel, spec), sel)
            meta.setdefault('cuts', {})['%s_%s' % (name, arm)] = cut
            pe = proba(m, tab, ids, spec)
            exact(arm, {s: pe[s] >= cut for s in ids})
        for k, v in counts.items():
            all_counts['%s__%s' % (name, k)] = v
        meta['eval_ids_' + name] = [str(s) for s in ids]
    np.savez_compressed(out / 'counts.npz', **all_counts)
    write_json(out / 'meta.json', meta)
    shutil.rmtree(work, ignore_errors=True)
    f = {k: round(200.0 * v[:, 0].sum() / (v[:, 1].sum() + v[:, 2].sum()), 2) for k, v in all_counts.items()}
    print('E15', args.task, 'seed', args.seed, json.dumps(f), flush=True)


def f1(c):
    return 200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())


def report(args):
    out = Path(args.out)
    rng = np.random.RandomState(20261108)
    summary = {}
    for task in args.tasks:
        runs = [dict(np.load(out / task / ('seed%d' % s) / 'counts.npz')) for s in args.seeds]
        stats = read_json(out / task / 'seed0' / 'meta.json')['stats']
        n = len(runs[0]['deepsolo__author'])
        draws = [rng.randint(0, n, n) for _ in range(args.draws)]
        t = dict(eval_images=n, match_stats=stats)
        for who in ('deepsolo', 'ticls'):
            rows = {}
            for arm in ('author', 'tuned_all', 'c11_base', 'c11_agree', 'score_agree'):
                k = '%s__%s' % (who, arm)
                row = dict(mean_f1=float(np.mean([f1(r[k]) for r in runs])))
                for ref in ('tuned_all', 'c11_base'):
                    kr = '%s__%s' % (who, ref)
                    if kr == k:
                        continue
                    diff = np.array([f1(r[k]) - f1(r[kr]) for r in runs])
                    d = np.array([np.mean([f1(r[k][i]) - f1(r[kr][i]) for r in runs]) for i in draws])
                    row['minus_' + ref] = float(diff.mean())
                    row['splits_above_' + ref] = int((diff > 0).sum())
                    row['ci_' + ref] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
                rows[arm] = row
            t[who] = rows
        summary[task] = t
    write_json(out / 'agree_summary.json', summary, exclusive=False)
    for task, t in summary.items():
        print(task, 'imgs', t['eval_images'], 'matched cand TEST ds->ticls %(matched)d/%(candidates)d' % t['match_stats']['test_ds_vs_ticls'], '| ticls->ds %(matched)d/%(candidates)d' % t['match_stats']['test_ticls_vs_ds'])
        for who in ('deepsolo', 'ticls'):
            for arm, r in t[who].items():
                s = ' | vs tuned_all %+.2f (%d/11) CI [%+.2f,%+.2f]' % (r['minus_tuned_all'], r['splits_above_tuned_all'], *r['ci_tuned_all']) if 'minus_tuned_all' in r else ''
                s += ' | vs c11_base %+.2f (%d/11) CI [%+.2f,%+.2f]' % (r['minus_c11_base'], r['splits_above_c11_base'], *r['ci_c11_base']) if 'minus_c11_base' in r else ''
                print('  %-8s %-12s F1 %.2f%s' % (who, arm, r['mean_f1'], s))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--stage', choices=['job', 'report'], required=True)
    p.add_argument('--task', choices=['totaltext', 'icdar2015'])
    p.add_argument('--tasks', nargs='+', default=['totaltext', 'icdar2015'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(11)))
    p.add_argument('--draws', type=int, default=2000)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    job(a) if a.stage == 'job' else report(a)
