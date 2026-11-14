"""E49 shared helpers for the report: formatting in the manuscript's style, F1/P/R from per-image counts, the shared bootstrap index matrix, readers of the run directories.
Result class: approximate reproduction. Nothing here fits a model."""
import json
from pathlib import Path

import numpy as np
from secondlook.paths import RUNS

RUNS = RUNS
RUN = RUNS / 'rerun-fast'
E18 = RUNS / 'candidate-banks'
E42 = RUNS / 'fulltrain'
E44 = RUNS / 'detection-crossdata'
ROWS9 = [('deepsolo', 'totaltext'), ('deepsolo', 'icdar2015'), ('ticls', 'totaltext'), ('ticls', 'icdar2015'), ('ppocr', 'totaltext'), ('ppocr', 'icdar2015'),
         ('deepsolo', 'ctw1500'), ('ticls', 'ctw1500'), ('ppocr', 'ctw1500')]
ROWS4 = ROWS9[:4]
NAMES = {'deepsolo_totaltext': 'DeepSolo, TT', 'deepsolo_icdar2015': 'DeepSolo, IC15', 'ticls_totaltext': 'TiCLS, TT', 'ticls_icdar2015': 'TiCLS, IC15', 'ppocr_totaltext': 'PP-OCR, TT',
         'ppocr_icdar2015': 'PP-OCR, IC15', 'deepsolo_ctw1500': 'DeepSolo R50, CTW', 'ticls_ctw1500': 'TiCLS ‡, CTW', 'ppocr_ctw1500': 'PP-OCR, CTW'}
import os
KEYS = ['%s_%s' % r for r in ROWS9]
if os.environ.get('TABLE_ROWS'):   # testing only: restrict the report to some rows
    KEYS = [k for k in KEYS if k in os.environ['TABLE_ROWS'].split(',')]
BOOT, SEED = 2000, 20261106
TEST_N = {'totaltext': 300, 'icdar2015': 500, 'ctw1500': 500}
ROUTER = {'deepsolo': 'all', 'ticls': 'none', 'ppocr': 'none'}


def router_text(e, t):
    n = TEST_N[t]
    return '%d/%d, 100.00%%' % (n, n) if e == 'deepsolo' else '0/%d, 0.00%%' % n


def sg(x, nd=2):
    return ('+' if x >= 0 else '−') + ('%.*f' % (nd, abs(x)))


def ub(x, nd=2):
    return ('' if x >= 0 else '−') + ('%.*f' % (nd, abs(x)))


def fm(v, nd=2):
    """[diff, lo, hi] -> '+0.55 [0.15, 0.97]' (manuscript style)."""
    return '%s [%s, %s]' % (sg(v[0], nd), ub(v[1], nd), ub(v[2], nd))


def f1(c):
    return float(200.0 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum()))


def prf(c):
    tp, pr, gt = (float(x) for x in c[:, :3].sum(0))
    return 100.0 * tp / pr if pr else 0.0, 100.0 * tp / gt, 200.0 * tp / (pr + gt), int(pr), int(gt)


def idx_matrix(n):
    return np.random.default_rng(SEED).integers(0, n, (BOOT, n))


def f1_boot(c, M):
    return 200.0 * c[M, 0].sum(1) / (c[M, 1].sum(1) + c[M, 2].sum(1))


def diff(ca, cb, M, fn=f1, fb=f1_boot):
    d = fb(ca, M) - fb(cb, M)
    return [fn(ca) - fn(cb), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def det_boot(c, M):
    k, p, g = (c[M, i].sum(1).astype(float) for i in (4, 5, 6))
    return dict(p=100.0 * k / p, r=100.0 * k / g, f=200.0 * k / (p + g))


def det_rates(c):
    k, p, g = (float(c[:, i].sum()) for i in (4, 5, 6))
    return dict(p=100.0 * k / p if p else 0.0, r=100.0 * k / g, f=200.0 * k / (p + g), n=int(p), g=int(g))


def jload(p):
    return json.load(open(p))


def md_table(header, rows, align=None):
    L = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(('---:' if (align and align[i] == 'r') else '---') for i in range(len(header))) + '|']
    for r in rows:
        L.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(L)
