"""E50-exact report assembler (approximate reproduction): tables.md and results.json with, for every table, FAST (E49 unified features, label-based selection) | EXACT (pinned evaluator for c, k*, baseline pair)
| DIFFERENCE (EXACT - FAST).
The table builders are the unchanged E49 modules (results/unified_features_full_rerun/e49_rep_*.py): they read two arms named 'old' and 'unified' below a run root. Here the root is a VIEW directory
(RUN50/_view) whose 'old' is a symlink to the E49 UNIFIED run (= the FAST protocol, the manuscript-to-be numbers of E49) and whose 'unified' is a symlink to the E50 `exact` arm; the labels of the output are
renamed to FAST / EXACT afterwards (the renaming is text only, see RENAMES). Cells that the exact arm did not produce are never silently dropped: a missing input is reported as MISSING in tables.md and in
results.json['missing'].
    python -m secondlook.reruns.exact.report --out results/exact_selection_full_rerun"""
import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from secondlook.paths import REPO_ROOT, RUNS

R50 = (RUNS / 'rerun-exact')
R49 = (RUNS / 'rerun-fast')
VIEW = R50 / '_view'
VIEW.mkdir(exist_ok=True)
for name, target in (('old', R49 / 'unified'), ('unified', R50 / 'exact')):
    link = VIEW / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)

import secondlook.reruns.fast.lib as fast_lib   # noqa: E402
fast_lib.RUN = VIEW
import secondlook.reruns.fast.report_ablations as rep_abl   # noqa: E402
import secondlook.reruns.fast.report_cross as rep_cross   # noqa: E402
import secondlook.reruns.fast.report_main as rep_main   # noqa: E402
import secondlook.reruns.fast.report_readers as rep_readers   # noqa: E402
import secondlook.reruns.fast.report_stability as rep_stab   # noqa: E402
from secondlook.reruns.fast.lib import md_table   # noqa: E402

ROUTER_NOTE = ' [Router: DeepSolo rows N/N = 100.00% of the test images to DeepSolo (TT 300/300, IC15 500/500, CTW 500/500), TiCLS and PP-OCR rows 0/N = 0.00%; logical decisions on stored outputs, no online inference, no latency]'
RENAMES = [
    ('OLD features (rerun with the same code path; compared with the manuscript, see gates)', 'FAST selection (E49 unified features; c, k* and the baseline pair chosen with the label-based F1 2TP/(P+G))'),
    ('UNIFIED features', 'EXACT selection (E50; unified features; c, k* and the baseline pair chosen with the pinned evaluator)'),
    ('DIFFERENCE (unified − old)', 'DIFFERENCE (EXACT − FAST)'),
    ('unified − old', 'EXACT − FAST'), ('unified - old', 'EXACT - FAST'),
    ('difference of the means (unified − old)', 'difference of the means (EXACT − FAST)'),
    ('FAST selection for the classifier and the baseline alike', 'selection protocol as named in each block (FAST: label-based F1; EXACT: pinned evaluator), the same on the classifier and the baseline side'),
    ('FAST selection throughout', 'selection protocol as named in each block'), ('FAST selection', 'selection protocol as named in each block'),
]


LABELS = {'old': 'FAST selection (E49 unified features; c, k* and the baseline pair chosen with the label-based F1 2TP/(P+G))',
          'unified': 'EXACT selection (E50; unified features; c, k* and the baseline pair chosen with the pinned evaluator)', 'delta': 'DIFFERENCE (EXACT − FAST)'}


def rename(text, caption=False):
    for a, b in (RENAMES if caption else [('unified − old', 'EXACT − FAST')]):
        text = text.replace(a, b)
    text = re.sub(r'\bold ([+−])', r'FAST \1', text)
    text = re.sub(r'→ unified ([+−])', r'→ EXACT \1', text)
    return text


def sections():
    s = []
    s.append(('main', rep_main.build))
    s.append(('stability', rep_stab.build))
    s.append(('learning_curve', rep_cross.build_lc))
    s.append(('cross', rep_cross.build_cross))
    s.append(('ablation', rep_abl.build))
    s.append(('readers', rep_readers.build))
    return s


def main(a):
    out_dir = Path(a.out)
    md = ['# E50-exact: every manuscript analysis with the EXACT selection protocol, unified features (approximate reproduction)\n',
          'Columns: **FAST** = the E49 run (unified features, label-based F1 for c, k* and the baseline pair); **EXACT** = the same code with every selection made by the pinned evaluator; **DIFFERENCE** = EXACT − FAST. '
          'Every EXACT number is an approximate reproduction (not a strict one). Test labels are never used for a choice.\n']
    results = dict(claim_label='approximate reproduction', sections={}, missing={})
    only = set(a.only.split(',')) if a.only else None
    for name, builder in sections():
        if only and name not in only:
            continue
        try:
            tables, data = builder()
        except Exception as e:   # a missing input is reported, never skipped silently
            results['missing'][name] = '%s: %s' % (type(e).__name__, e)
            md.append('### %s\n\n**MISSING** (%s: %s)\n' % (name, type(e).__name__, e))
            traceback.print_exc()
            continue
        results['sections'][name] = data
        for title, hdr, blocks in tables:
            md.append('#### %s\n' % rename(title + ROUTER_NOTE, True))
            if isinstance(blocks, list):
                md.append(rename(md_table(hdr, blocks)) + '\n')
                continue
            for blk in ('old', 'unified', 'delta'):
                md.append('**%s**\n\n%s\n' % (LABELS[blk], rename(md_table(hdr, blocks[blk]))))
    (out_dir / (a.name + '.md')).write_text('\n'.join(md))
    (out_dir / ('results.json' if a.name == 'tables' else a.name + '.json')).write_text(json.dumps(results, indent=1, default=float))
    print('written', out_dir / (a.name + '.md'), 'missing:', list(results['missing']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--only')
    p.add_argument('--name', default='tables')
    main(p.parse_args())
