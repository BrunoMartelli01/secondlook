"""E49: write job files (one shell command per line) for `xargs -P` and the generic job runner. Usage:
    python -m secondlook.reruns.fast.make_jobs KIND --out RUNDIR > jobs_KIND.txt
KIND in: stab, lc, abl, abl_argmax, cross_dump, cross_eval.   Every line is self-contained: `run_job.sh ARM LOGNAME MODULE args...`."""
import argparse

ROWS9 = [('deepsolo', 'totaltext'), ('deepsolo', 'icdar2015'), ('ticls', 'totaltext'), ('ticls', 'icdar2015'), ('ppocr', 'totaltext'), ('ppocr', 'icdar2015'),
         ('deepsolo', 'ctw1500'), ('ticls', 'ctw1500'), ('ppocr', 'ctw1500')]
EXPERTS = ['deepsolo', 'ticls', 'ppocr']
TASKS = ['totaltext', 'icdar2015', 'ctw1500']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind')
    p.add_argument('--out', required=True)
    p.add_argument('--arms', nargs='*', default=['old', 'unified'])
    p.add_argument('--seeds', type=int, default=11)
    a = p.parse_args()
    O = a.out
    for arm in a.arms:
        for e, t in ROWS9:
            r = '%s_%s' % (e, t)
            if a.kind == 'stab':
                print('%s stab_%s_%s secondlook.reruns.fast.stability --expert %s --task %s --out %s' % (arm, arm, r, e, t, O))
            elif a.kind == 'lc':
                for s in range(a.seeds):
                    print('%s lc_%s_%s_s%d secondlook.studies.learning_curve --expert %s --task %s --seed %d --out %s/%s/lc/out' % (arm, arm, r, s, e, t, s, O, arm))
            elif a.kind in ('abl', 'abl_argmax'):
                stages = ['all'] if a.kind == 'abl' else ['table10_argmax']
                for st in stages:
                    print('%s %s_%s_%s secondlook.studies.ablations --expert %s --task %s --stage %s --out %s/%s/ablations' % (arm, a.kind, arm, r, e, t, st, O, arm))
            elif a.kind == 'cross_dump':
                print('%s cdump_%s_%s secondlook.reruns.fast.cross_dataset dump --expert %s --task %s --out %s/%s/crossdata' % (arm, arm, r, e, t, O, arm))
            elif a.kind == 'cross_eval':
                print('%s ceval_%s_%s secondlook.reruns.fast.cross_dataset evaluate --expert %s --task %s --out %s/%s/crossdata' % (arm, arm, r, e, t, O, arm))
            else:
                raise SystemExit('unknown kind')


if __name__ == '__main__':
    main()
