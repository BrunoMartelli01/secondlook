#!/usr/bin/env python3
"""Redraw Figs. 5 (learning curve) and 6 (band-width sensitivity) from the tables.md of the exact-selection rerun."""
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# usage: python -m secondlook.figures.make_fig56 TABLES_MD OUT_DIR   (tables.md written by secondlook.reruns.exact.report)
TABLES = Path(sys.argv[1]).read_text(encoding='utf-8')
OUT = Path(sys.argv[2])
num = lambda s: float(s.replace('−', '-').replace('+', ''))


def unified_block(prefix):
    sec = next(s for s in re.split(r'(?m)^(?=#### )', TABLES) if s.startswith('#### ' + prefix))
    m = re.search(r'\*\*(?:UNIFIED|EXACT)[^\n]*\n(.*?)(?=\n\*\*DIFF|\Z)', sec, re.S)
    rows = [l for l in m.group(1).strip().split('\n') if l.startswith('|')]
    return [[c.strip() for c in r.strip('|').split('|')] for r in rows[2:]]


plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.linewidth': 1.4, 'xtick.direction': 'in',
                     'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'xtick.major.width': 1.3, 'ytick.major.width': 1.3, 'xtick.major.size': 6, 'ytick.major.size': 6})
COL = {'DeepSolo': '#1f77b4', 'TiCLS': '#d62728', 'PP-OCR': '#2ca02c'}
MARK = {'DeepSolo': 'o', 'TiCLS': 's', 'PP-OCR': '^'}

# ---- Fig. 5: learning curve (mean over eleven draws; last column = complete training pool)
lc = unified_block('Table 8 of rev 48')
data = {}
for r in lc:
    name = r[0]
    sp = name.split(',')[0].replace(' R50', '').replace(' ‡', '').strip()
    ds = 'CTW' if 'CTW' in name else ('TT' if name.endswith('TT') else 'IC15')
    means = [num(c.split()[0]) for c in r[1:6]] + [num(r[6].split()[0])]
    data[(sp, ds)] = means
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
BUDGETS = [30, 60, 120, 240, 480]
for ax, (ds, title, full) in zip(axes, [('TT', '(a) Total-Text', 1251), ('IC15', '(b) ICDAR2015', 1000), ('CTW', '(c) CTW1500', 1000)]):
    xs = BUDGETS + [full]
    for sp in COL:
        y = data[(sp, ds)]
        ax.plot(BUDGETS, y[:5], '-', color=COL[sp], lw=2.4, marker=MARK[sp], ms=8)
        ax.plot([480, full], [y[4], y[5]], '--', color=COL[sp], lw=1.6)
        ax.plot([full], [y[5]], marker=MARK[sp], ms=9, mfc='white', mec=COL[sp], mew=2, ls='')
    ax.axhline(0, color='k', lw=1.5)
    ax.set_xscale('log', base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.minorticks_off()
    ax.set_xlim(24, full * 1.2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('labelled training images $N$ (log scale)', fontsize=12)
    ax.grid(color='#dddddd', lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel('F1: classifier − thresholds', fontsize=12)
handles = [plt.Line2D([], [], color=COL[s], marker=MARK[s], lw=2.4, ms=8, label=s) for s in COL]
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=True, edgecolor='k', fancybox=False, fontsize=12,
           bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / 'fig5.png', dpi=200)
plt.close(fig)

# ---- Fig. 6: gain over first stage versus share read, four settings
bs = unified_block('Table 15 / Fig. 6')
rows = {}
for r in bs:
    m = re.match(r'(.+?) \(([\d.]+)\)$', r[0])
    rows[m.group(1)] = (m.group(2), r[1:])
fig, ax = plt.subplots(figsize=(7.2, 4.3))
sets = [('DeepSolo, TT', 'DeepSolo', 'o', '-'), ('DeepSolo, IC15', 'DeepSolo', 's', '--'),
        ('TiCLS, TT', 'TiCLS', 'o', '-'), ('TiCLS, IC15', 'TiCLS', 's', '--')]
KS = ['1.2', '1.6', '1.9', '2.2', '2.6', 'all']
for name, sp, mk, ls in sets:
    chosen, cells = rows[name]
    xs, ys, ring = [], [], None
    for k, c in zip(KS, cells):
        c = c.replace('**', '')
        m = re.match(r'([\d.]+)%, ([+−-][\d.]+)', c)
        x, y = float(m.group(1)), num(m.group(2))
        xs.append(x); ys.append(y)
        if k == chosen:
            ring = (x, y)
    # drop the duplicated "all candidates" point when it coincides with the previous one
    pts = []
    for x, y in zip(xs, ys):
        if not pts or (x, y) != pts[-1]:
            pts.append((x, y))
    pts.sort()
    ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, color=COL[sp], marker=mk, ms=7, lw=2.2, label=name)
    ax.plot([ring[0]], [ring[1]], marker='o', ms=17, mfc='none', mec=COL[sp], mew=2, ls='')
ax.set_xscale('log')
ax.set_xticks([2, 5, 10, 20, 50, 100])
ax.set_xticklabels(['2', '5', '10', '20', '50', '100'])
ax.set_xlim(2, 100)
ax.set_xlabel('candidates read again (% of test candidates)', fontsize=12)
ax.set_ylabel('gain over first stage (F1)', fontsize=12)
ax.grid(color='#dddddd', lw=0.8)
ax.set_axisbelow(True)
ax.legend(loc='upper left', frameon=True, edgecolor='k', fancybox=False, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / 'fig6.png', dpi=200)
print('ok', OUT)
