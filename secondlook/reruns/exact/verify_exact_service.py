"""E50-exact gate G1: the per-image cached exact service (secondlook.reruns.exact.exact_service.Store) equals the whole-pool pinned evaluator on several kept sets (all kept, none kept, random masks).
    python -m secondlook.reruns.exact.launch fast secondlook.reruns.exact.verify_exact_service --expert E --task T --out DIR   (the arm is irrelevant for this gate)"""
import argparse
import json
import tempfile
from pathlib import Path

from secondlook.selection.operating_point import load


def main(a):
    from secondlook.reruns.exact.exact_service import Store
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='verify_', dir=str(out)))
    T = load(a.expert, a.task, work)
    st = Store(T, work, '%s_%s' % (a.expert, a.task))
    r = st.verify(n_sets=a.sets)
    (out / ('g1_%s_%s.json' % (a.expert, a.task))).write_text(json.dumps(r, indent=1))
    print('G1', json.dumps(r), flush=True)
    if not r['identical']:
        raise SystemExit('GATE G1 FAILED')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--expert', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--sets', type=int, default=4)
    main(p.parse_args())
